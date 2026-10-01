"""Rebuild the bsk_rag_v2 collection from the ingested source files.

    python manage.py rebuild_rag_v2 [--dry-run] [--no-backup]

For each asset whose latest ingestion job is done:
  1. find its source file in INGESTION_RAW_BASE (data/incoming/)
  2. (once) back up the Chroma dir, drop bsk_rag_v2 and recreate it (cosine)
  3. replace the asset's jobs with a new job at the current
     INGESTION_CONFIG_VERSION and run the pipeline synchronously

Only bsk_rag_v2 is dropped; other collections in the Chroma dir (e.g. the
legacy bsk_rag with plc_historian summaries) are left untouched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from django.conf import settings
from django.core.management.base import BaseCommand

from ingestion.models import IngestionJob
from ingestion.pipeline import process_job
from ingestion.vector_store import VectorStore


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Command(BaseCommand):
    help = "Drop and re-ingest bsk_rag_v2 from the source files of all done documents."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show what would be rebuilt, change nothing.")
        parser.add_argument("--no-backup", action="store_true", help="Skip copying the Chroma dir first.")

    def handle(self, *args, dry_run=False, no_backup=False, **options):
        # Latest job per asset (jobs are ordered newest first); rebuild only done ones.
        latest: dict[str, IngestionJob] = {}
        for job in IngestionJob.objects.all():
            latest.setdefault(job.asset_id, job)

        plan, skipped = [], []
        for asset_id, job in latest.items():
            path = os.path.join(settings.INGESTION_RAW_BASE, job.source_filename)
            if job.status != IngestionJob.Status.DONE:
                skipped.append((asset_id, f"latest job is {job.status}"))
            elif not os.path.exists(path):
                skipped.append((asset_id, f"source file missing: {path}"))
            else:
                plan.append((job, path))

        chroma_dir = str(settings.CHROMA_DIR)
        self.stdout.write(f"Collection {settings.RAG_COLLECTION_V2} in {chroma_dir}")
        self.stdout.write(f"Config version -> {settings.INGESTION_CONFIG_VERSION}")
        for job, path in plan:
            self.stdout.write(f"  rebuild {job.asset_id}: {job.source_filename} (rev {job.document_revision}, "
                              f"{job.chunk_count} chunks @ {job.config_version})")
        for asset_id, reason in skipped:
            self.stdout.write(self.style.WARNING(f"  skip    {asset_id}: {reason}"))

        if dry_run or not plan:
            self.stdout.write("Dry run, nothing changed." if dry_run else "Nothing to rebuild.")
            return

        if not no_backup:
            backup = f"{chroma_dir.rstrip('/')}.bak-{datetime.now():%Y%m%d-%H%M%S}"
            shutil.copytree(chroma_dir, backup)
            self.stdout.write(f"Backed up Chroma dir to {backup}")

        client = chromadb.PersistentClient(path=chroma_dir, settings=ChromaSettings(anonymized_telemetry=False))
        try:
            client.delete_collection(settings.RAG_COLLECTION_V2)
        except Exception:  # not found (exception type varies by Chroma version)
            pass
        collection = VectorStore()._get_collection()  # recreated with cosine
        space = (collection.configuration_json or {}).get("hnsw", {}).get("space")
        self.stdout.write(f"Recreated {settings.RAG_COLLECTION_V2} (space={space})")

        failures = 0
        for old_job, path in plan:
            IngestionJob.objects.filter(asset_id=old_job.asset_id).delete()
            job = IngestionJob.objects.create(
                source_filename=old_job.source_filename,
                source_sha256=_sha256(path),
                document_id=old_job.document_id or old_job.asset_id,
                document_revision=old_job.document_revision,
                asset_id=old_job.asset_id,
                status=IngestionJob.Status.QUEUED,
                config_version=settings.INGESTION_CONFIG_VERSION,
            )
            self.stdout.write(f"Ingesting {job.asset_id} ...")
            process_job(job.id)
            job.refresh_from_db()
            if job.status == IngestionJob.Status.DONE:
                self.stdout.write(self.style.SUCCESS(f"  done: {job.chunk_count} chunks"))
            else:
                failures += 1
                self.stdout.write(self.style.ERROR(f"  {job.status}: {(job.error or '').splitlines()[0][:200]}"))

        self.stdout.write(f"Collection now has {collection.count()} chunks; {failures} failure(s).")
