"""Bring documents ingested by the legacy pipeline into the document library.

    python manage.py library_import_legacy [--dry-run] [--no-parse]

For each asset whose latest legacy IngestionJob is done: create a library
Document (doc_key = the legacy asset_id, RAG status from its existing chunks,
strategy "structure"), copy the source file into LIBRARY_BASE, stamp its
Chroma chunks with document_id / chunk_strategy, and queue a parse job so the
parsed text is cached for re-embedding and GraphLab. Idempotent: documents
already in the library are skipped.
"""

import hashlib
import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ingestion import library
from ingestion.chunking import resolve_params, variant_key
from ingestion.models import Document, IngestionJob, Job
from ingestion.vector_store import VectorStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Command(BaseCommand):
    help = "Import legacy ingested documents into the document library."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--no-parse", action="store_true", help="Don't queue parse jobs.")

    def handle(self, *args, dry_run=False, no_parse=False, **options):
        latest: dict[str, IngestionJob] = {}
        for job in IngestionJob.objects.all():
            latest.setdefault(job.asset_id, job)

        params = resolve_params("structure", {})
        collection = VectorStore()._get_collection()
        imported = 0
        for asset_id, job in latest.items():
            src = Path(settings.INGESTION_RAW_BASE) / job.source_filename
            if Document.objects.filter(doc_key=asset_id).exists():
                self.stdout.write(f"  skip {asset_id}: already in the library")
                continue
            if job.status != IngestionJob.Status.DONE or not src.exists():
                self.stdout.write(self.style.WARNING(f"  skip {asset_id}: job {job.status}, file present={src.exists()}"))
                continue
            sha = _sha256(src)
            if Document.objects.filter(sha256=sha).exists():
                self.stdout.write(f"  skip {asset_id}: same file already in the library")
                continue
            data = collection.get(where={"asset_id": asset_id}, include=["metadatas"])
            ids, metas = data.get("ids") or [], data.get("metadatas") or []
            self.stdout.write(f"  import {asset_id}: {job.source_filename} ({len(ids)} chunks)")
            imported += 1
            if dry_run:
                continue

            doc = Document.objects.create(
                filename=job.source_filename, doc_key=asset_id, sha256=sha, size=src.stat().st_size,
                document_revision=job.document_revision,
                rag_status=Document.PipelineStatus.DONE if ids else Document.PipelineStatus.NONE,
                rag_strategy="structure" if ids else "", rag_params=params if ids else {},
                rag_chunk_count=len(ids), rag_updated_at=job.completed_at,
            )
            library.doc_dir(doc).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, library.file_path(doc))
            if ids:
                stamp = {"document_id": str(doc.id), "chunk_strategy": "structure",
                         "chunk_variant": variant_key("structure", params)}
                collection.update(ids=ids, metadatas=[{**(m or {}), **stamp} for m in metas])
            if not no_parse:
                library.enqueue(doc, Job.Kind.PARSE)
        self.stdout.write(f"{'Would import' if dry_run else 'Imported'} {imported} document(s).")
