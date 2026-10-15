"""Shared document library: upload once, parse once, run pipelines per file.

    upload ──► Document (+ parse job) ──► worker: Docling → LIBRARY_BASE/<id>/parsed.json
    "Generate embeddings" ──► embed job ──► worker: chunk (strategy) → DGX embed → bsk_rag_v2
    "Describe figures"    ──► figures job ──► worker: crop each figure → VLM on BSK → figures.json

Long steps run in `manage.py library_worker` (separate from runserver, so
reloads can't kill them); jobs record progress and a heartbeat, and stale
running jobs are re-queued.

Unattended (P14d-2): a PDF longer than LIBRARY_PARSE_PART_PAGES is parsed in parts
of that many pages, one Docling request (and one GPU lock) each; finished parts are
kept, so a retry resumes. When BSK (Docling) or the DGX (embeddings) can't be
reached, the job waits and tries again (LIBRARY_RETRY_MINUTES, up to
LIBRARY_RETRY_DAYS) instead of failing. An upload can ask for its embeddings to
follow the parse (`then_embed`).
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
from django.db.models import F, Q
from django.utils import timezone

import io

import requests

from gpu import orchestrator as gpu

from .chunker import PAGE_BREAK
from .chunking import DEFAULT_STRATEGY, chunk_parsed, resolve_params, variant_key
from .docling_client import DoclingError, convert_file
from .embedder import Embedder
from . import figures
from .models import Document, IngestionJob, Job
from .vector_store import VectorStore

logger = logging.getLogger("chat")

EMBED_BATCH = 32
ACTIVE = (Job.Status.QUEUED, Job.Status.RUNNING)


class LibraryError(ValueError):
    pass


class JobCancelled(Exception):
    pass


class ServiceUnavailable(Exception):
    """BSK (Docling) or the DGX (embeddings) can't be reached: the job waits and tries again."""


def _is_unreachable(exc: BaseException) -> bool:
    """A service that is off or unreachable (not a document that fails)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (requests.ConnectionError, gpu.GpuError)):   # ConnectTimeout is a ConnectionError
            return True
        exc = exc.__cause__ or exc.__context__
    return False


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


def add_document(uploaded_file, doc_key: str | None = None, folder=None, then_embed: bool = False,
                 title: str = "", source_url: str = "", fetched_at=None, plant_asset: str = "",
                 then_figures: bool = False) -> tuple[Document, bool]:
    """Store an upload (de-duplicated by SHA-256) and queue its parse. Returns (doc, created).

    `folder` is a Folder (None: Unfiled); `then_embed` queues the embeddings when the parse is done;
    `then_figures` then describes the figures (PDFs and images) and embeds again;
    `title`, `source_url`, `fetched_at`, `plant_asset` record where the document comes from."""
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
        folder=folder, title=title[:512], source_url=source_url, fetched_at=fetched_at, plant_asset=plant_asset,
    )
    doc_dir(doc).mkdir(parents=True, exist_ok=True)
    shutil.move(tmp.name, file_path(doc))
    enqueue(doc, Job.Kind.PARSE, {"then_embed": then_embed or then_figures, "then_figures": then_figures})
    return doc, True


def load_parsed(doc: Document, with_figures: bool = True) -> dict:
    """The cached Docling result. With `with_figures`, figure descriptions (Describe
    figures) replace their image placeholders in the Markdown."""
    parsed = json.loads(parsed_path(doc).read_text())
    document = parsed.get("document")
    if with_figures and isinstance(document, dict) and document.get("md_content"):
        document["md_content"] = figures.apply_to_markdown(document["md_content"], figures.load(doc))
    return parsed


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
    elif kind == Job.Kind.FIGURES:
        if not (figures.is_pdf(doc) or figures.is_image(doc)):
            raise LibraryError("figures need a PDF or an image")
        only = (params or {}).get("figures")
        if only is not None and not (isinstance(only, list) and all(isinstance(i, int) for i in only)):
            raise LibraryError("figures must be a list of figure numbers")
        params = {"figures": only, "force": bool((params or {}).get("force")),
                  **({"then_embed": True} if (params or {}).get("then_embed") else {})}
    elif kind == Job.Kind.PARSE:
        params = {k: True for k in ("then_embed", "then_figures") if (params or {}).get(k)}
    else:
        raise LibraryError(f"unknown job kind {kind!r}")

    active = doc.jobs.filter(kind=kind, status__in=ACTIVE).first()
    if active:
        return active
    if kind == Job.Kind.EMBED:
        Document.objects.filter(pk=doc.pk).update(rag_status=Document.PipelineStatus.QUEUED, rag_error="")
    elif kind == Job.Kind.EXTRACT:
        Document.objects.filter(pk=doc.pk).update(graph_status=Document.PipelineStatus.QUEUED, graph_error="")
    elif kind == Job.Kind.FIGURES:
        Document.objects.filter(pk=doc.pk).update(figures_status=Document.FiguresStatus.QUEUED, figures_error="")
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
    now = timezone.now()
    ready = Job.objects.filter(status=Job.Status.QUEUED).filter(Q(run_after__isnull=True) | Q(run_after__lte=now))
    # Parse jobs first: Docling and the VLM share one GPU, so this avoids switching back and forth.
    # Then embeddings (the DGX; text becomes searchable soon), then the rest (figures, triples).
    queue = (list(ready.filter(kind=Job.Kind.PARSE).order_by("created_at")[:5])
             or list(ready.filter(kind=Job.Kind.EMBED).order_by("created_at")[:5])
             or list(ready.order_by("created_at")[:5]))
    for job in queue:
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
        elif job.kind == Job.Kind.FIGURES:
            run_figures(job)
        else:
            raise LibraryError(f"unknown job kind {job.kind!r}")
    except JobCancelled:
        Job.objects.filter(pk=job.pk).update(finished_at=timezone.now(), message="cancelled")
        _settle_document(job, cancelled=True)
        logger.info("library job cancelled job=%s kind=%s doc=%s", job.id, job.kind, job.document.doc_key)
        return
    except figures.BskUnavailable as exc:
        if job.params.get("then_embed"):   # an unattended pass waits for BSK like the parse does
            _postpone(job, str(exc))
        else:
            _figures_postponed(job, str(exc))
        return
    except ServiceUnavailable as exc:
        _postpone(job, str(exc))
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
    doc = Document.objects.get(pk=job.document_id)
    if job.kind == Job.Kind.PARSE and job.params.get("then_embed"):
        enqueue(doc, Job.Kind.EMBED)
    if job.kind == Job.Kind.PARSE and job.params.get("then_figures") and (figures.is_pdf(doc) or figures.is_image(doc)):
        enqueue(doc, Job.Kind.FIGURES, {"then_embed": True})     # after the embeddings (queued first)
    if job.kind == Job.Kind.FIGURES and job.params.get("then_embed") and doc.figures_status == Document.FiguresStatus.DONE:
        enqueue(doc, Job.Kind.EMBED)                             # the descriptions become searchable


def _postpone(job: Job, reason: str) -> None:
    """A service is off: queue the job again with back-off, for up to LIBRARY_RETRY_DAYS."""
    job.refresh_from_db()
    if timezone.now() - job.created_at > timedelta(days=settings.LIBRARY_RETRY_DAYS):
        error = f"gave up after {settings.LIBRARY_RETRY_DAYS} days: {reason}"
        Job.objects.filter(pk=job.pk).update(status=Job.Status.FAILED, error=error[:4000], finished_at=timezone.now())
        _settle_document(job, error=error)
        logger.warning("library job gave up job=%s doc=%s: %s", job.id, job.document.doc_key, reason)
        return
    delays = settings.LIBRARY_RETRY_MINUTES
    minutes = delays[min(job.attempts, len(delays)) - 1] if job.attempts else delays[0]
    service = "the DGX" if job.kind == Job.Kind.EMBED else "BSK"
    Job.objects.filter(pk=job.pk).update(
        status=Job.Status.QUEUED, run_after=timezone.now() + timedelta(minutes=minutes),
        message=f"waiting for {service} (not reachable); trying again in {minutes} min"[:255])
    if job.kind == Job.Kind.PARSE:
        Document.objects.filter(pk=job.document_id).update(parse_status=Document.ParseStatus.QUEUED)
    elif job.kind == Job.Kind.EMBED:
        Document.objects.filter(pk=job.document_id).update(rag_status=Document.PipelineStatus.QUEUED)
    elif job.kind == Job.Kind.FIGURES:
        Document.objects.filter(pk=job.document_id).update(figures_status=Document.FiguresStatus.QUEUED)
    logger.warning("library job postponed job=%s kind=%s doc=%s attempt=%d: %s",
                   job.id, job.kind, job.document.doc_key, job.attempts, reason)


def _settle_document(job: Job, error: str = "", cancelled: bool = False) -> None:
    doc = Document.objects.get(pk=job.document_id)
    if job.kind == Job.Kind.PARSE:
        doc.parse_status = Document.ParseStatus.FAILED if error else (
            Document.ParseStatus.PENDING if cancelled else doc.parse_status)
        doc.parse_error = error
        doc.save(update_fields=["parse_status", "parse_error"])
    elif job.kind == Job.Kind.FIGURES:
        if cancelled:  # figures described so far are kept
            counts = figures.summary(figures.load(doc))
            doc.figures_status = (Document.FiguresStatus.DONE if counts["described"] and not counts["pending"]
                                  else Document.FiguresStatus.NONE)
        else:
            doc.figures_status = Document.FiguresStatus.FAILED
        doc.figures_error = error
        doc.save(update_fields=["figures_status", "figures_error"])
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

def _convert(data: bytes, filename: str) -> dict:
    try:
        return convert_file(data, filename, with_layout=True)
    except Exception as exc:
        if _is_unreachable(exc):
            raise ServiceUnavailable(f"Docling: {exc}") from exc
        raise


def pdf_pages(doc: Document) -> int:
    """Pages of a PDF (0 when it is not a PDF or can't be read)."""
    if not figures.is_pdf(doc):
        return 0
    import pypdfium2 as pdfium
    try:
        pdf = pdfium.PdfDocument(str(file_path(doc)))
    except Exception:
        return 0
    try:
        return len(pdf)
    finally:
        pdf.close()


def _pdf_part(doc: Document, start: int, end: int) -> bytes:
    """Pages [start, end) (0-based) as a PDF of their own."""
    import pypdfium2 as pdfium
    src, out = pdfium.PdfDocument(str(file_path(doc))), pdfium.PdfDocument.new()
    try:
        out.import_pages(src, list(range(start, end)))
        buf = io.BytesIO()
        out.save(buf)
        return buf.getvalue()
    finally:
        out.close()
        src.close()


def _parts_dir(doc: Document) -> Path:
    return doc_dir(doc) / "parts"


def run_parse(job: Job) -> None:
    doc = job.document
    Document.objects.filter(pk=doc.pk).update(parse_status=Document.ParseStatus.PARSING)
    pages, size = pdf_pages(doc), settings.LIBRARY_PARSE_PART_PAGES
    if pages > size:
        return _run_parse_parts(job, doc, pages, size)
    heartbeat(job, 0, 1, "parsing with Docling")
    result = _convert(file_path(doc).read_bytes(), doc.filename)
    document = result.get("document") or {}
    md = document.get("md_content") or ""
    # Docling's JSON is only needed for the figure positions; it isn't kept (it is large).
    layout = document.pop("json_content", None) if isinstance(document, dict) else None
    figures.save(doc, figures.from_docling(doc, layout))
    parsed_path(doc).write_text(json.dumps(result))
    Document.objects.filter(pk=doc.pk).update(
        parse_status=Document.ParseStatus.PARSED, parse_error="", parsed_at=timezone.now(),
        page_count=md.count(PAGE_BREAK) + 1 if md else 0,
    )
    heartbeat(job, 1, 1, "parsed")


def _run_parse_parts(job: Job, doc: Document, pages: int, size: int) -> None:
    """A long PDF in parts of `size` pages: each part is saved, so a retry resumes; the GPU is
    held per part (a chat with an image gets in between); pages keep their numbers in the PDF."""
    parts_dir = _parts_dir(doc)
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + size, pages)) for start in range(0, pages, size)]
    done = 0
    for start, end in ranges:
        part_file = parts_dir / f"{doc.sha256[:12]}-{start + 1:04d}-{end:04d}.json"
        if part_file.exists():
            done += end - start
            continue
        heartbeat(job, done, pages, f"parsing pages {start + 1}–{end} of {pages}")
        data = _pdf_part(doc, start, end)
        for attempt in (1, 2):
            try:
                result = _convert(data, f"{Path(doc.filename).stem} p{start + 1}-{end}.pdf")
                break
            except ServiceUnavailable:
                raise
            except Exception as exc:
                if attempt == 2:
                    raise LibraryError(f"pages {start + 1}–{end}: {type(exc).__name__}: {exc}") from exc
                logger.warning("parse part failed, trying once more doc=%s pages=%d-%d: %s",
                               doc.doc_key, start + 1, end, exc)
        document = result.get("document") or {}
        md = document.get("md_content") or ""
        missing = (end - start - 1) - md.count(PAGE_BREAK)   # keep later pages numbered right
        if missing > 0:
            md += f"\n\n{PAGE_BREAK}\n\n" * missing
        part = {"pages": [start + 1, end], "md": md,
                "figures": figures.pictures(document.get("json_content"), offset=start)}
        tmp = part_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(part))
        tmp.replace(part_file)
        done += end - start
        heartbeat(job, done, pages)

    parts = [json.loads((parts_dir / f"{doc.sha256[:12]}-{s + 1:04d}-{e:04d}.json").read_text()) for s, e in ranges]
    md = f"\n\n{PAGE_BREAK}\n\n".join(p["md"] for p in parts)
    figures.save(doc, figures.from_docling(doc, None, pictures=[f for p in parts for f in p["figures"]]))
    result = {"status": "success", "errors": [], "parts": len(parts),
              "document": {"filename": doc.filename, "md_content": md}}
    parsed_path(doc).write_text(json.dumps(result))
    shutil.rmtree(parts_dir, ignore_errors=True)
    Document.objects.filter(pk=doc.pk).update(
        parse_status=Document.ParseStatus.PARSED, parse_error="", parsed_at=timezone.now(), page_count=pages)
    heartbeat(job, pages, pages, f"parsed {pages} pages in {len(parts)} parts")


def run_embed(job: Job) -> None:
    doc = Document.objects.get(pk=job.document_id)
    strategy = job.params.get("strategy", DEFAULT_STRATEGY)
    params = resolve_params(strategy, job.params.get("params"))
    Document.objects.filter(pk=doc.pk).update(rag_status=Document.PipelineStatus.RUNNING)

    if doc.parse_status != Document.ParseStatus.PARSED or not parsed_path(doc).exists():
        heartbeat(job, message="parsing first")
        run_parse(job)
        doc.refresh_from_db()

    parsed = load_parsed(doc)
    chunks = chunk_parsed(parsed, strategy, params)
    page_lang: list[str] = []
    if doc.languages:   # a multilingual manual keeps only some languages (pages)
        from . import languages
        page_lang = languages.page_languages(parsed["document"].get("md_content") or "")
        keep = set(doc.languages)

        def kept(c) -> bool:
            lang = page_lang[c.page_start - 1] if c.page_start and c.page_start <= len(page_lang) else ""
            return lang == languages.NEUTRAL or lang in keep
        chunks = [c for c in chunks if kept(c)]
    if not chunks:
        raise LibraryError("chunking produced no chunks; the document may be empty")
    heartbeat(job, 0, len(chunks), f"embedding {len(chunks)} chunks ({strategy})")

    embedder = Embedder()
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        try:
            embeddings += embedder.embed_passages([c.text for c in chunks[start:start + EMBED_BATCH]])
        except Exception as exc:
            if _is_unreachable(exc):
                raise ServiceUnavailable(f"embeddings: {exc}") from exc
            raise
        heartbeat(job, min(start + EMBED_BATCH, len(chunks)))
    from .folders import chunk_metadata
    place = chunk_metadata(doc)

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
            **place,
            **({"language": page_lang[c.page_start - 1] or "neutral"}
               if page_lang and c.page_start and c.page_start <= len(page_lang) else {}),
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


def run_figures(job: Job) -> None:
    """Describe the document's figures with the VLM on BSK, one GPU lock per figure
    (a waiting chat image gets in between two figures). Results are saved after each
    figure, so nothing is sent twice."""
    doc = Document.objects.get(pk=job.document_id)
    Document.objects.filter(pk=doc.pk).update(figures_status=Document.FiguresStatus.RUNNING, figures_error="")

    if doc.parse_status != Document.ParseStatus.PARSED or figures.load(doc) is None:
        heartbeat(job, message="parsing first (finding the figures)")
        run_parse(job)
        doc.refresh_from_db()

    data = figures.load(doc)
    if doc.languages:   # figures on pages in other languages are not described
        from . import languages
        kept = languages.kept_pages(load_parsed(doc, with_figures=False)["document"].get("md_content") or "", doc.languages)
        for figure in data.get("figures", []):
            if figure["status"] == "pending" and figure.get("page") and figure["page"] not in kept:
                figure.update(status="skipped", skip_reason=f"page in another language (kept: {', '.join(doc.languages)})")
        figures.save(doc, data)
    todo = figures.targets(data, job.params.get("figures"), bool(job.params.get("force")))
    heartbeat(job, 0, len(todo), f"describing {len(todo)} figure(s)")
    for n, figure in enumerate(todo):
        figures.describe(doc, figure)       # BskUnavailable → the job is postponed (see run_job)
        figures.save(doc, data)
        heartbeat(job, n + 1, len(todo), f"figure {n + 1} of {len(todo)} (p.{figure['page']})")

    counts = figures.summary(data)
    Document.objects.filter(pk=doc.pk).update(
        figures_status=Document.FiguresStatus.DONE, figures_updated_at=timezone.now(),
        figures_error="" if not counts["failed"] else f"{counts['failed']} figure(s) could not be described",
    )


def _figures_postponed(job: Job, reason: str) -> None:
    """BSK is asleep or unreachable: retry later with back-off, then leave the figures pending."""
    delays = settings.FIGURE_RETRY_MINUTES
    job.refresh_from_db()
    if job.attempts <= len(delays):
        minutes = delays[job.attempts - 1]
        Job.objects.filter(pk=job.pk).update(
            status=Job.Status.QUEUED, run_after=timezone.now() + timedelta(minutes=minutes),
            message=f"BSK not reachable; trying again in {minutes} min"[:255])
        Document.objects.filter(pk=job.document_id).update(figures_status=Document.FiguresStatus.QUEUED)
        logger.warning("figures postponed job=%s doc=%s attempt=%d: %s", job.id, job.document.doc_key, job.attempts, reason)
        return
    Job.objects.filter(pk=job.pk).update(status=Job.Status.DONE, finished_at=timezone.now(),
                                         message="visual interpretation pending: BSK not reachable")
    Document.objects.filter(pk=job.document_id).update(
        figures_status=Document.FiguresStatus.PENDING, figures_updated_at=timezone.now(),
        figures_error=f"BSK was not reachable: {reason}"[:1000])
    logger.warning("figures pending job=%s doc=%s: %s", job.id, job.document.doc_key, reason)


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
