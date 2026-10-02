"""Shared document library: upload once, parse once, run pipelines per file.

    upload ──► Document (+ parse job) ──► worker: Docling → LIBRARY_BASE/<id>/parsed.json
    "Generate embeddings" ──► embed job ──► worker: chunk (strategy) → DGX embed → bsk_rag_v2

Long steps run in `manage.py library_worker` (separate from runserver, so
reloads can't kill them); jobs record progress and a heartbeat, and stale
running jobs are re-queued.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from .chunker import PAGE_BREAK
from .chunking import DEFAULT_STRATEGY, chunk_parsed, resolve_params, variant_key
from .docling_client import convert_file
from .embedder import Embedder
from .models import Document, IngestionJob, Job
from .vector_store import VectorStore

logger = logging.getLogger("chat")

EMBED_BATCH = 32
ACTIVE = (Job.Status.QUEUED, Job.Status.RUNNING)


class LibraryError(ValueError):
    pass


class JobCancelled(Exception):
    pass


# --- storage -------------------------------------------------------------------

def doc_dir(doc: Document) -> Path:
    return Path(settings.LIBRARY_BASE) / str(doc.id)


def file_path(doc: Document) -> Path:
    return doc_dir(doc) / doc.filename


def parsed_path(doc: Document) -> Path:
    return doc_dir(doc) / "parsed.json"


def doc_key_for(filename: str) -> str:
    """Unique document key from the filename: 'Attention is all you need.pdf' -> 'ATTENTION-IS-ALL-YOU-NEED'."""
    base = re.sub(r"[^A-Z0-9]+", "-", Path(filename).stem.upper()).strip("-")[:120] or "DOCUMENT"
    key, n = base, 2
    while Document.objects.filter(doc_key=key).exists():
        key, n = f"{base}-{n}", n + 1
    return key


def add_document(uploaded_file, doc_key: str | None = None) -> tuple[Document, bool]:
    """Store an upload (de-duplicated by SHA-256) and queue its parse. Returns (doc, created)."""
    if uploaded_file.size > settings.LIBRARY_MAX_UPLOAD_BYTES:
        raise LibraryError(f"{uploaded_file.name} is larger than {settings.LIBRARY_MAX_UPLOAD_BYTES} bytes")
    filename = Path(uploaded_file.name).name
    digest = hashlib.sha256()
    Path(settings.LIBRARY_BASE).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=settings.LIBRARY_BASE, delete=False) as tmp:
        for chunk in uploaded_file.chunks():
            digest.update(chunk)
            tmp.write(chunk)
    sha = digest.hexdigest()

    existing = Document.objects.filter(sha256=sha).first()
    if existing:
        Path(tmp.name).unlink(missing_ok=True)
        return existing, False

    if doc_key and Document.objects.filter(doc_key=doc_key).exists():
        raise LibraryError(f"document key {doc_key!r} is already used")
    doc = Document.objects.create(
        filename=filename,
        doc_key=doc_key or doc_key_for(filename),
        sha256=sha,
        size=uploaded_file.size,
    )
    doc_dir(doc).mkdir(parents=True, exist_ok=True)
    shutil.move(tmp.name, file_path(doc))
    enqueue(doc, Job.Kind.PARSE)
    return doc, True


def load_parsed(doc: Document) -> dict:
    return json.loads(parsed_path(doc).read_text())


# --- jobs -------------------------------------------------------------------

@transaction.atomic
def enqueue(doc: Document, kind: str, params: dict | None = None) -> Job:
    """Queue a job (or return the document's active job of the same kind).

    Parameters are validated first, so invalid input is rejected even when a
    job is already active.
    """
    if kind == Job.Kind.EMBED:
        strategy = (params or {}).get("strategy", DEFAULT_STRATEGY)
        params = {"strategy": strategy, "params": resolve_params(strategy, (params or {}).get("params"))}
    elif kind == Job.Kind.EXTRACT:
        from context_graph.extraction import validate_params  # graph app depends on the library, not vice versa
        params = validate_params(params or {})
    elif kind != Job.Kind.PARSE:
        raise LibraryError(f"unknown job kind {kind!r}")

    active = doc.jobs.filter(kind=kind, status__in=ACTIVE).first()
    if active:
        return active
    if kind == Job.Kind.EMBED:
        Document.objects.filter(pk=doc.pk).update(rag_status=Document.PipelineStatus.QUEUED, rag_error="")
    elif kind == Job.Kind.EXTRACT:
        Document.objects.filter(pk=doc.pk).update(graph_status=Document.PipelineStatus.QUEUED, graph_error="")
    else:
        Document.objects.filter(pk=doc.pk).update(parse_status=Document.ParseStatus.QUEUED, parse_error="")
    return Job.objects.create(document=doc, kind=kind, params=params or {})


def cancel(job: Job) -> bool:
    """Cancel a queued job now; a running job stops at its next checkpoint."""
    updated = Job.objects.filter(pk=job.pk, status__in=ACTIVE).update(status=Job.Status.CANCELLED)
    if updated and job.status == Job.Status.QUEUED:
        _settle_document(job, cancelled=True)
    return bool(updated)


def requeue_stale() -> int:
    """Running jobs without a recent heartbeat (worker died) go back to the queue."""
    cutoff = timezone.now() - timedelta(seconds=settings.LIBRARY_JOB_STALE_SECONDS)
    return Job.objects.filter(status=Job.Status.RUNNING, heartbeat_at__lt=cutoff).update(
        status=Job.Status.QUEUED, message="re-queued after worker interruption")


def claim_next() -> Job | None:
    """Atomically take the oldest queued job (safe with several workers)."""
    for job in Job.objects.filter(status=Job.Status.QUEUED).order_by("created_at")[:5]:
        now = timezone.now()
        claimed = Job.objects.filter(pk=job.pk, status=Job.Status.QUEUED).update(
            status=Job.Status.RUNNING, started_at=now, heartbeat_at=now, attempts=F("attempts") + 1)
        if claimed:
            return Job.objects.select_related("document").get(pk=job.pk)
    return None


def heartbeat(job: Job, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
    """Record progress; raises JobCancelled if the job was cancelled meanwhile."""
    fields = {"heartbeat_at": timezone.now()}
    if done is not None:
        fields["progress_done"] = done
    if total is not None:
        fields["progress_total"] = total
    if message is not None:
        fields["message"] = message[:255]
    Job.objects.filter(pk=job.pk).update(**fields)
    if Job.objects.filter(pk=job.pk, status=Job.Status.CANCELLED).exists():
        raise JobCancelled()


def run_job(job: Job) -> None:
    """Run a claimed job to completion, recording the outcome on the job and document."""
    try:
        if job.kind == Job.Kind.PARSE:
            run_parse(job)
        elif job.kind == Job.Kind.EMBED:
            run_embed(job)
        elif job.kind == Job.Kind.EXTRACT:
            from context_graph.extraction import run_extract
            run_extract(job)
        else:
            raise LibraryError(f"unknown job kind {job.kind!r}")
    except JobCancelled:
        Job.objects.filter(pk=job.pk).update(finished_at=timezone.now(), message="cancelled")
        _settle_document(job, cancelled=True)
        logger.info("library job cancelled job=%s kind=%s doc=%s", job.id, job.kind, job.document.doc_key)
        return
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        Job.objects.filter(pk=job.pk).update(status=Job.Status.FAILED, error=error[:4000], finished_at=timezone.now())
        _settle_document(job, error=error)
        logger.exception("library job failed job=%s kind=%s doc=%s", job.id, job.kind, job.document.doc_key)
        return
    Job.objects.filter(pk=job.pk).update(status=Job.Status.DONE, finished_at=timezone.now(),
                                         progress_done=F("progress_total"))
    logger.info("library job done job=%s kind=%s doc=%s", job.id, job.kind, job.document.doc_key)


def _settle_document(job: Job, error: str = "", cancelled: bool = False) -> None:
    doc = Document.objects.get(pk=job.document_id)
    if job.kind == Job.Kind.PARSE:
        doc.parse_status = Document.ParseStatus.FAILED if error else (
            Document.ParseStatus.PENDING if cancelled else doc.parse_status)
        doc.parse_error = error
        doc.save(update_fields=["parse_status", "parse_error"])
    elif job.kind == Job.Kind.EXTRACT:
        if cancelled:  # staged triples from earlier runs are untouched
            doc.graph_status = Document.PipelineStatus.DONE if doc.triples.exists() else Document.PipelineStatus.NONE
        else:
            doc.graph_status = Document.PipelineStatus.FAILED
        doc.graph_error = error
        doc.save(update_fields=["graph_status", "graph_error"])
    else:
        if cancelled:  # back to what's actually in Chroma
            doc.rag_status = Document.PipelineStatus.DONE if doc.rag_chunk_count else Document.PipelineStatus.NONE
        else:
            doc.rag_status = Document.PipelineStatus.FAILED
        doc.rag_error = error
        doc.save(update_fields=["rag_status", "rag_error"])


# --- pipelines -----------------------------------------------------------------

def run_parse(job: Job) -> None:
    doc = job.document
    Document.objects.filter(pk=doc.pk).update(parse_status=Document.ParseStatus.PARSING)
    heartbeat(job, 0, 1, "parsing with Docling")
    result = convert_file(file_path(doc).read_bytes(), doc.filename)
    md = (result.get("document") or {}).get("md_content") or ""
    parsed_path(doc).write_text(json.dumps(result))
    Document.objects.filter(pk=doc.pk).update(
        parse_status=Document.ParseStatus.PARSED, parse_error="", parsed_at=timezone.now(),
        page_count=md.count(PAGE_BREAK) + 1 if md else 0,
    )
    heartbeat(job, 1, 1, "parsed")


def run_embed(job: Job) -> None:
    doc = Document.objects.get(pk=job.document_id)
    strategy = job.params.get("strategy", DEFAULT_STRATEGY)
    params = resolve_params(strategy, job.params.get("params"))
    Document.objects.filter(pk=doc.pk).update(rag_status=Document.PipelineStatus.RUNNING)

    if doc.parse_status != Document.ParseStatus.PARSED or not parsed_path(doc).exists():
        heartbeat(job, message="parsing first")
        run_parse(job)
        doc.refresh_from_db()

    chunks = chunk_parsed(load_parsed(doc), strategy, params)
    if not chunks:
        raise LibraryError("chunking produced no chunks; the document may be empty")
    heartbeat(job, 0, len(chunks), f"embedding {len(chunks)} chunks ({strategy})")

    embedder = Embedder()
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        embeddings += embedder.embed_passages([c.text for c in chunks[start:start + EMBED_BATCH]])
        heartbeat(job, min(start + EMBED_BATCH, len(chunks)))

    variant = variant_key(strategy, params)
    metadatas = [
        {
            "source": doc.filename,
            "asset_id": doc.doc_key,
            "document_id": str(doc.id),
            "document_revision": doc.document_revision,
            "section_path": c.section_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "content_type": c.content_type,
            "chunk_strategy": strategy,
            "chunk_variant": variant,
        }
        for c in chunks
    ]
    # Replace this document's chunks. Chroma isn't transactional: record the
    # deletion first so a failed write never leaves a stale chunk count.
    store = VectorStore()
    store.delete_document(doc.doc_key)
    Document.objects.filter(pk=doc.pk).update(rag_chunk_count=0)
    written = store.write_chunks(
        chunks=[c.text for c in chunks], embeddings=embeddings, metadatas=metadatas,
        source_sha256=doc.sha256, document_revision=doc.document_revision,
        config_version=settings.INGESTION_CONFIG_VERSION, variant=variant,
    )
    Document.objects.filter(pk=doc.pk).update(
        rag_status=Document.PipelineStatus.DONE, rag_strategy=strategy, rag_params=params,
        rag_chunk_count=written, rag_error="", rag_updated_at=timezone.now(),
    )


def remove_embeddings(doc: Document) -> int:
    removed = VectorStore().delete_document(doc.doc_key)
    Document.objects.filter(pk=doc.pk).update(
        rag_status=Document.PipelineStatus.NONE, rag_chunk_count=0, rag_strategy="", rag_params={},
        rag_error="", rag_updated_at=timezone.now(),
    )
    return removed


def delete_document(doc: Document) -> dict:
    """Remove a document everywhere: graph triples, chunks, files, legacy ingestion jobs and library jobs."""
    for job in doc.jobs.filter(status__in=ACTIVE):
        cancel(job)
    committed = list(doc.triples.filter(status="approved"))
    if committed:  # take its approved triples back out of the Context Graph first
        from context_graph import triples
        triples.delete(committed)
    chunks = VectorStore().delete_document(doc.doc_key)
    shutil.rmtree(doc_dir(doc), ignore_errors=True)
    legacy, _ = IngestionJob.objects.filter(asset_id=doc.doc_key).delete()
    doc.delete()
    return {"chunks_deleted": chunks, "legacy_jobs_deleted": legacy}
