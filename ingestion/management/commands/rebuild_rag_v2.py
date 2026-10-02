"""Rebuild the bsk_rag_v2 collection from the document library.

    python manage.py rebuild_rag_v2 [--dry-run] [--no-backup]

Re-embeds every library document whose RAG status is done, with the chunking
strategy/params it was embedded with:
  1. (once) back up the Chroma dir, drop bsk_rag_v2 and recreate it (cosine)
  2. per document, run an embed job synchronously (parses first if needed)

Only bsk_rag_v2 is dropped; other collections in the Chroma dir (e.g. the
legacy bsk_rag with plc_historian summaries) are left untouched. Run
`library_import_legacy` first if documents were ingested by the old pipeline.
"""

from __future__ import annotations

import shutil
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from ingestion import library
from ingestion.chroma_client import describe, get_client
from ingestion.models import Document, IngestionJob, Job
from ingestion.vector_store import VectorStore


class Command(BaseCommand):
    help = "Drop and re-embed bsk_rag_v2 from the document library."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show what would be rebuilt, change nothing.")
        parser.add_argument("--no-backup", action="store_true", help="Skip copying the Chroma dir first.")

    def handle(self, *args, dry_run=False, no_backup=False, **options):
        plan, skipped = [], []
        for doc in Document.objects.filter(rag_status=Document.PipelineStatus.DONE).order_by("uploaded_at"):
            if library.file_path(doc).exists():
                plan.append(doc)
            else:
                skipped.append((doc.doc_key, f"source file missing: {library.file_path(doc)}"))
        library_keys = set(Document.objects.values_list("doc_key", flat=True))
        for asset_id in IngestionJob.objects.exclude(asset_id__in=library_keys).values_list("asset_id", flat=True).distinct():
            skipped.append((asset_id, "legacy document not in the library (run library_import_legacy)"))

        chroma_dir = str(settings.CHROMA_DIR)
        self.stdout.write(f"Collection {settings.RAG_COLLECTION_V2} at {describe()}")
        for doc in plan:
            self.stdout.write(f"  rebuild {doc.doc_key}: {doc.filename} ({doc.rag_strategy or 'structure'} "
                              f"{doc.rag_params}, {doc.rag_chunk_count} chunks)")
        for key, reason in skipped:
            self.stdout.write(self.style.WARNING(f"  skip    {key}: {reason}"))
        if dry_run or not plan:
            self.stdout.write("Dry run, nothing changed." if dry_run else "Nothing to rebuild.")
            return

        if not no_backup:
            backup = f"{chroma_dir.rstrip('/')}.bak-{datetime.now():%Y%m%d-%H%M%S}"
            shutil.copytree(chroma_dir, backup)
            self.stdout.write(f"Backed up Chroma dir to {backup}")

        try:
            get_client().delete_collection(settings.RAG_COLLECTION_V2)
        except Exception:  # not found (exception type varies by Chroma version)
            pass
        collection = VectorStore()._get_collection()  # recreated with cosine
        space = (collection.configuration_json or {}).get("hnsw", {}).get("space")
        self.stdout.write(f"Recreated {settings.RAG_COLLECTION_V2} (space={space})")

        failures = 0
        for doc in plan:
            job = Job.objects.create(
                document=doc, kind=Job.Kind.EMBED, status=Job.Status.RUNNING, started_at=timezone.now(),
                heartbeat_at=timezone.now(), attempts=1,
                params={"strategy": doc.rag_strategy or "structure", "params": doc.rag_params or {}},
            )
            self.stdout.write(f"Embedding {doc.doc_key} ...")
            library.run_job(job)
            job.refresh_from_db()
            doc.refresh_from_db()
            if job.status == Job.Status.DONE:
                self.stdout.write(self.style.SUCCESS(f"  done: {doc.rag_chunk_count} chunks"))
            else:
                failures += 1
                self.stdout.write(self.style.ERROR(f"  {job.status}: {job.error.splitlines()[0][:200] if job.error else ''}"))

        self.stdout.write(f"Collection now has {collection.count()} chunks; {failures} failure(s).")
