"""Document library API (shared by RAG Lab and GraphLab).

    GET    /api/documents/                      list (with parse / RAG status and active jobs)
    POST   /api/documents/                      upload one or more files (multipart "files")
    GET    /api/documents/<id>/                 detail
    DELETE /api/documents/<id>/                 delete everywhere (chunks, files, jobs)
    POST   /api/documents/<id>/embed/           Generate embeddings {strategy, params}
    DELETE /api/documents/<id>/embeddings/      remove the document's chunks from RAG
    POST   /api/documents/embed/                bulk: {ids: [...], strategy, params}
    GET    /api/documents/<id>/figures/         figures found by Docling, with their descriptions
    GET    /api/documents/<id>/figures/<n>/image/   the cropped figure (JPEG)
    POST   /api/documents/<id>/figures/describe/    Describe figures {figures?: [n], force?: bool}
    GET    /api/jobs/?document=<id>&active=1    jobs
    GET    /api/jobs/<id>/                      job status / progress
    POST   /api/jobs/<id>/cancel/               cancel a queued or running job
"""

from django.db.models import Count
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from context_graph.driver import GraphUnavailable

from django.http import FileResponse

from gpu import images

from . import figures, library
from .chunking import ChunkingError
from .models import Document, Job


def job_json(job: Job) -> dict:
    return {
        "id": str(job.id),
        "document_id": str(job.document_id),
        "kind": job.kind,
        "status": job.status,
        "params": job.params,
        "progress_done": job.progress_done,
        "progress_total": job.progress_total,
        "message": job.message,
        "error": job.error,
        "attempts": job.attempts,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def document_json(doc: Document, active_jobs=None) -> dict:
    if active_jobs is None:
        active_jobs = doc.jobs.filter(status__in=library.ACTIVE)
    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "doc_key": doc.doc_key,
        "size": doc.size,
        "sha256": doc.sha256,
        "document_revision": doc.document_revision,
        "uploaded_at": doc.uploaded_at.isoformat(),
        "parse": {
            "status": doc.parse_status,
            "error": doc.parse_error,
            "parsed_at": doc.parsed_at.isoformat() if doc.parsed_at else None,
            "page_count": doc.page_count,
        },
        "rag": {
            "status": doc.rag_status,
            "strategy": doc.rag_strategy,
            "params": doc.rag_params,
            "chunk_count": doc.rag_chunk_count,
            "error": doc.rag_error,
            "updated_at": doc.rag_updated_at.isoformat() if doc.rag_updated_at else None,
        },
        "graph": {
            "status": doc.graph_status,
            "error": doc.graph_error,
            "updated_at": doc.graph_updated_at.isoformat() if doc.graph_updated_at else None,
            **graph_counts(doc),
        },
        "figures": figures_json(doc),
        "active_jobs": [job_json(j) for j in active_jobs],
    }


def figures_json(doc: Document, data=...) -> dict:
    if data is ...:
        data = figures.load(doc) if doc.parse_status == Document.ParseStatus.PARSED else None
    return {
        "status": doc.figures_status,
        "error": doc.figures_error,
        "updated_at": doc.figures_updated_at.isoformat() if doc.figures_updated_at else None,
        "supported": figures.is_pdf(doc) or figures.is_image(doc),
        # Parsed before figures existed: Describe figures parses the document again first.
        "needs_parse": data is None,
        **figures.summary(data),
    }


def graph_counts(doc: Document) -> dict:
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for row in doc.triples.values("status").annotate(n=Count("id")):
        counts[row["status"]] = row["n"]
    return {"triples": counts}


def _get_document(doc_id):
    return Document.objects.filter(pk=doc_id).first()


def _embed_params(data) -> dict:
    return {"strategy": data.get("strategy") or "structure", "params": data.get("params") or {}}


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def documents(request):
    if request.method == "GET":
        docs = list(Document.objects.all())
        active = {}
        for job in Job.objects.filter(status__in=library.ACTIVE, document__in=docs):
            active.setdefault(job.document_id, []).append(job)
        return Response({"documents": [document_json(d, active.get(d.id, [])) for d in docs]})

    files = request.FILES.getlist("files")
    if not files:
        return Response({"error": "no files uploaded (multipart field 'files')"}, status=status.HTTP_400_BAD_REQUEST)
    results = []
    for f in files:
        try:
            doc, created = library.add_document(f)
            results.append({"filename": f.name, "created": created, "document": document_json(doc)})
        except library.LibraryError as exc:
            results.append({"filename": f.name, "created": False, "error": str(exc)})
    return Response({"results": results}, status=status.HTTP_201_CREATED)


@api_view(["GET", "DELETE"])
def document_detail(request, doc_id):
    doc = _get_document(doc_id)
    if doc is None:
        return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        try:
            return Response({"document_id": str(doc_id), **library.delete_document(doc)})
        except GraphUnavailable as exc:
            return Response({"error": f"document has approved graph triples and the graph is unavailable: {exc}"},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response(document_json(doc))


@api_view(["POST"])
def document_embed(request, doc_id):
    doc = _get_document(doc_id)
    if doc is None:
        return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)
    try:
        job = library.enqueue(doc, Job.Kind.EMBED, _embed_params(request.data))
    except ChunkingError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(job_json(job), status=status.HTTP_202_ACCEPTED)


@api_view(["POST"])
def documents_embed(request):
    ids = request.data.get("ids") or []
    if not isinstance(ids, list) or not ids:
        return Response({"error": "ids must be a non-empty list"}, status=status.HTTP_400_BAD_REQUEST)
    docs = {str(d.id): d for d in Document.objects.filter(pk__in=ids)}
    missing = [i for i in ids if str(i) not in docs]
    if missing:
        return Response({"error": f"unknown document(s): {', '.join(map(str, missing))}"},
                        status=status.HTTP_404_NOT_FOUND)
    params = _embed_params(request.data)
    try:
        jobs = [library.enqueue(docs[str(i)], Job.Kind.EMBED, params) for i in ids]
    except ChunkingError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response({"jobs": [job_json(j) for j in jobs]}, status=status.HTTP_202_ACCEPTED)


@api_view(["DELETE"])
def document_embeddings(request, doc_id):
    doc = _get_document(doc_id)
    if doc is None:
        return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)
    for job in doc.jobs.filter(kind=Job.Kind.EMBED, status__in=library.ACTIVE):
        library.cancel(job)
    return Response({"document_id": str(doc.id), "chunks_deleted": library.remove_embeddings(doc)})


@api_view(["GET"])
def document_figures(request, doc_id):
    doc = _get_document(doc_id)
    if doc is None:
        return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)
    data = figures.load(doc)
    items = [{k: f.get(k) for k in ("index", "page", "caption", "area", "status", "skip_reason", "kind",
                                    "description", "error", "model", "described_at")}
             for f in (data or {}).get("figures", [])]
    return Response({"document_id": str(doc.id), **figures_json(doc, data), "figures": items})


@api_view(["GET"])
def document_figure_image(request, doc_id, index: int):
    doc = _get_document(doc_id)
    data = figures.load(doc) if doc else None
    figure = next((f for f in (data or {}).get("figures", []) if f["index"] == index), None)
    if figure is None:
        return Response({"error": "figure not found"}, status=status.HTTP_404_NOT_FOUND)
    path = figures.image_path(doc, index)
    if not path.exists():
        if figure.get("bbox") is None and not figures.is_image(doc):
            return Response({"error": "the figure has no position"}, status=status.HTTP_404_NOT_FOUND)
        try:
            figures.crop(doc, figure)       # CPU only: thumbnails exist before anything is described
        except images.ImageError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    return FileResponse(path.open("rb"), content_type="image/jpeg")


@api_view(["POST"])
def document_figures_describe(request, doc_id):
    doc = _get_document(doc_id)
    if doc is None:
        return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)
    try:
        job = library.enqueue(doc, Job.Kind.FIGURES, {"figures": request.data.get("figures"),
                                                     "force": request.data.get("force")})
    except library.LibraryError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(job_json(job), status=status.HTTP_202_ACCEPTED)


@api_view(["GET"])
def jobs(request):
    qs = Job.objects.all().order_by("-created_at")
    if request.query_params.get("document"):
        qs = qs.filter(document_id=request.query_params["document"])
    if request.query_params.get("active") in ("1", "true"):
        qs = qs.filter(status__in=library.ACTIVE)
    return Response({"jobs": [job_json(j) for j in qs[:200]]})


@api_view(["GET"])
def job_detail(request, job_id):
    job = Job.objects.filter(pk=job_id).first()
    if job is None:
        return Response({"error": "job not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(job_json(job))


@api_view(["POST"])
def job_cancel(request, job_id):
    job = Job.objects.filter(pk=job_id).first()
    if job is None:
        return Response({"error": "job not found"}, status=status.HTTP_404_NOT_FOUND)
    if not library.cancel(job):
        return Response({"error": f"job is already {job.status}"}, status=status.HTTP_409_CONFLICT)
    job.refresh_from_db()
    return Response(job_json(job))
