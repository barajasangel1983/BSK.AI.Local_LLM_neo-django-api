"""Background ingestion pipeline.

process_job(job_id) runs the full cycle:
  parse (Docling) → chunk → embed → store (Chroma) → archive

Uses a module-level lock to enforce single-job concurrency (v1).
On failure: job.status=failed, file quarantined to data/failed/.
"""

from __future__ import annotations

import json
import os
import threading
import traceback

from django.conf import settings
from django.utils import timezone

from .chunker import chunk_document
from .docling_client import DoclingClient, DoclingError
from .embedder import Embedder
from .models import IngestionJob
from .vector_store import VectorStore

_job_lock = threading.Lock()


def process_job(job_id) -> None:
    """Run the full ingestion pipeline for a single job (thread-safe)."""
    if not _job_lock.acquire(blocking=False):
        # Another job is running; re-queue by just waiting (v1: block)
        _job_lock.acquire()

    try:
        _run_pipeline(job_id)
    finally:
        _job_lock.release()


def _run_pipeline(job_id) -> None:
    job = IngestionJob.objects.get(id=job_id)

    try:
        # 1. Load file from incoming dir
        file_path = os.path.join(settings.INGESTION_RAW_BASE, job.source_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # 2. Parse with Docling
        job.status = IngestionJob.Status.PARSING
        job.save(update_fields=["status"])

        client = DoclingClient()
        task_id = client.submit_file(file_bytes, job.source_filename)
        status = client.poll_status(task_id, timeout_s=600)
        doc_json = client.fetch_result(task_id)

        # 3. Chunk
        job.status = IngestionJob.Status.CHUNKING
        job.save(update_fields=["status"])

        chunks = chunk_document(doc_json)
        if not chunks:
            raise ValueError("Chunker produced no chunks — document may be empty")

        # 4. Embed
        job.status = IngestionJob.Status.EMBEDDING
        job.save(update_fields=["status"])

        embedder = Embedder()
        embeddings = embedder.embed_passages([c.text for c in chunks])

        # 5. Store in Chroma
        metadatas = [
            {
                "source": job.source_filename,
                "asset_id": job.asset_id,
                "document_revision": job.document_revision,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "content_type": c.content_type,
            }
            for c in chunks
        ]

        store = VectorStore()
        written = store.write_chunks(
            chunks=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
            source_sha256=job.source_sha256,
            document_revision=job.document_revision,
            config_version=settings.INGESTION_CONFIG_VERSION,
        )

        # 6. Archive JSONL to data/processed/
        output_dir = settings.INGESTION_OUTPUT_BASE
        os.makedirs(output_dir, exist_ok=True)
        archive_path = os.path.join(output_dir, f"{job.id}.jsonl")
        with open(archive_path, "w") as f:
            for c in chunks:
                f.write(
                    json.dumps(
                        {
                            "text": c.text,
                            "section_path": c.section_path,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "content_type": c.content_type,
                            "token_count": c.token_count,
                        }
                    )
                    + "\n"
                )

        # 7. Mark done
        job.status = IngestionJob.Status.DONE
        job.chunk_count = written
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "chunk_count", "completed_at"])

    except Exception as e:
        # Failure: mark failed, quarantine file
        job.status = IngestionJob.Status.FAILED
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:2000]}"
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error", "completed_at"])

        failed_dir = os.path.join(settings.INGESTION_RAW_BASE, "..", "failed")
        os.makedirs(failed_dir, exist_ok=True)
        src = os.path.join(settings.INGESTION_RAW_BASE, job.source_filename)
        if os.path.exists(src):
            os.replace(src, os.path.join(failed_dir, job.source_filename))
