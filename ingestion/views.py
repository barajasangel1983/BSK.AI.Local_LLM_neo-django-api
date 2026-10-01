"""Ingestion API views.

Endpoints:
- POST /api/rag/ingest/         — upload a file, start pipeline
- GET  /api/rag/jobs/           — list jobs
- GET  /api/rag/jobs/<id>/      — job detail
- GET  /api/rag/health/         — check Docling + DGX + Chroma
"""

import hashlib
import os
import threading
from datetime import timedelta

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import IngestionJob


IN_FLIGHT_STATUSES = (
    IngestionJob.Status.QUEUED,
    IngestionJob.Status.PARSING,
    IngestionJob.Status.CHUNKING,
    IngestionJob.Status.EMBEDDING,
    IngestionJob.Status.STORED,
)


def should_retry(job: IngestionJob) -> bool:
    """A failed job, or an in-flight one whose worker thread is presumably dead."""
    if job.status == IngestionJob.Status.FAILED:
        return True
    stale_before = timezone.now() - timedelta(minutes=settings.INGESTION_STALE_MINUTES)
    return job.status in IN_FLIGHT_STATUSES and job.created_at < stale_before


@require_http_methods(["POST"])
@csrf_exempt
def ingest_document(request):
    """Accept a multipart file upload and kick off the ingestion pipeline."""
    file = request.FILES.get("file")
    if not file:
        return JsonResponse({"error": "No file provided"}, status=400)

    asset_id = request.POST.get("asset_id", "").strip()
    if not asset_id:
        return JsonResponse({"error": "asset_id is required"}, status=400)

    document_revision = request.POST.get("document_revision", "1").strip()

    # Compute SHA256 for idempotency
    sha256 = hashlib.sha256()
    for chunk in file.chunks():
        sha256.update(chunk)
    source_sha256 = sha256.hexdigest()

    # Idempotency: return existing job if same hash + revision + config,
    # unless it failed or got stuck — then re-run it below.
    existing = IngestionJob.objects.filter(
        source_sha256=source_sha256,
        document_revision=document_revision,
        config_version=settings.INGESTION_CONFIG_VERSION,
    ).first()
    if existing and not should_retry(existing):
        return JsonResponse(
            {
                "job_id": str(existing.id),
                "status": existing.status,
                "message": "Existing job returned (idempotency match)",
            }
        )

    # Save original to data/incoming/
    incoming_dir = settings.INGESTION_RAW_BASE
    os.makedirs(incoming_dir, exist_ok=True)
    save_path = os.path.join(incoming_dir, file.name)
    with open(save_path, "wb") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    if existing:
        # Retry: reuse the row (unique per sha/revision/config). created_at is
        # reset to the start of this attempt so the stale check measures from now.
        job = existing
        job.source_filename = file.name
        job.asset_id = asset_id
        job.status = IngestionJob.Status.QUEUED
        job.error = None
        job.chunk_count = None
        job.completed_at = None
        job.created_at = timezone.now()
        job.save()
        message = "Retrying previous job"
    else:
        # Create job record
        job = IngestionJob.objects.create(
            source_filename=file.name,
            source_sha256=source_sha256,
            document_revision=document_revision,
            asset_id=asset_id,
            status=IngestionJob.Status.QUEUED,
            config_version=settings.INGESTION_CONFIG_VERSION,
        )
        message = "Ingestion started"

    # Launch background processing
    from .pipeline import process_job

    thread = threading.Thread(target=process_job, args=(job.id,), daemon=True)
    thread.start()

    return JsonResponse(
        {
            "job_id": str(job.id),
            "status": job.status,
            "message": message,
        },
        status=202,
    )


def list_jobs(request):
    """List ingestion jobs, latest first. Optional ?status= filter."""
    status_filter = request.GET.get("status")
    jobs = IngestionJob.objects.all()
    if status_filter:
        jobs = jobs.filter(status=status_filter)
    jobs = jobs.order_by("-created_at")

    return JsonResponse(
        {
            "jobs": [
                {
                    "job_id": str(j.id),
                    "filename": j.source_filename,
                    "asset_id": j.asset_id,
                    "status": j.status,
                    "chunk_count": j.chunk_count,
                    "error": j.error,
                    "created_at": j.created_at.isoformat(),
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                for j in jobs
            ]
        }
    )


def job_detail(request, job_id):
    """Get a single job's status."""
    try:
        job = IngestionJob.objects.get(id=job_id)
    except IngestionJob.DoesNotExist:
        return JsonResponse({"error": "Job not found"}, status=404)

    return JsonResponse(
        {
            "job_id": str(job.id),
            "filename": job.source_filename,
            "asset_id": job.asset_id,
            "document_revision": job.document_revision,
            "status": job.status,
            "chunk_count": job.chunk_count,
            "error": job.error,
            "config_version": job.config_version,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
    )


def ingestion_health(request):
    """Check reachability of Docling, DGX embeddings, and Chroma."""
    import requests

    checks = {}

    # Docling
    try:
        resp = requests.get(f"{settings.DOCLING_URL}/docs", timeout=10)
        checks["docling"] = {"status": "ok" if resp.status_code == 200 else f"http_{resp.status_code}", "detail": f"{settings.DOCLING_URL}"}
    except Exception as e:
        checks["docling"] = {"status": "error", "detail": str(e)}

    # DGX Embeddings
    try:
        resp = requests.get(settings.DGX_EMBED_URL.replace("/v1/embeddings", "/health"), timeout=5)
        if resp.status_code == 404:
            # Fallback: just check connectivity
            resp = requests.get(settings.DGX_EMBED_URL, timeout=5)
        checks["embeddings"] = {"status": "ok" if resp.status_code < 500 else f"http_{resp.status_code}", "detail": settings.DGX_EMBED_URL}
    except Exception as e:
        checks["embeddings"] = {"status": "error", "detail": str(e)}

    # Chroma (local)
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_path = str(getattr(settings, "CHROMA_DIR", "/home/barajas_angel/repos/BSK.AI.Local_LLM_neo4j-graphrag/data/chroma_index"))
        client = chromadb.PersistentClient(path=chroma_path, settings=ChromaSettings(anonymized_telemetry=False))
        coll = client.get_or_create_collection(name=settings.RAG_COLLECTION_V2)
        checks["chroma"] = {"status": "ok", "detail": f"{settings.RAG_COLLECTION_V2}: {coll.count()} chunks"}
    except Exception as e:
        checks["chroma"] = {"status": "error", "detail": str(e)}

    all_ok = all(c["status"] == "ok" for c in checks.values())
    return JsonResponse({"healthy": all_ok, "checks": checks}, status=200 if all_ok else 503)
