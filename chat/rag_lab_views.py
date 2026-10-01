"""RAG Lab endpoints over bsk_rag_v2 (Phase 5).

Endpoints:
- GET    /api/rag/docs/              — ingested documents (grouped by asset_id) + in-flight/failed jobs
- DELETE /api/rag/docs/<asset_id>/   — remove a document's chunks and ingestion jobs
- GET    /api/rag/chunks/            — chunk browser (?asset_id=&offset=&limit=)
- GET    /api/rag/config/            — chunker / retrieval settings (read-only UI)

Chunks are written only by the ingestion pipeline; these views read them, and
delete them on request.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import tiktoken
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ingestion import chunker
from ingestion.models import IngestionJob

from .retrieval import distance_space, get_v2_collection

logger = logging.getLogger("chat")

CHUNKS_DEFAULT_LIMIT = 20
CHUNKS_MAX_LIMIT = 100


@lru_cache(maxsize=1)
def _encoding():
    # Same tokenizer the chunker uses for its token budgets.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text or ""))


def chunk_index(chunk_id: str) -> int:
    """Chunk position within its document: ids are '<ingest_key>:<index>'."""
    try:
        return int(str(chunk_id).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _file_size(filename: str) -> int:
    path = os.path.join(settings.INGESTION_RAW_BASE, filename)
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _job_fields(job: IngestionJob | None) -> dict:
    if job is None:
        return {"status": "done", "job_id": None, "ingested_at": None, "error": None}
    return {
        "status": job.status,
        "job_id": str(job.id),
        "ingested_at": (job.completed_at or job.created_at).isoformat(),
        "error": job.error,
    }


@api_view(["GET"])
def rag_docs(request):
    """GET /api/rag/docs/

    Documents in bsk_rag_v2 grouped by asset_id, joined with their latest
    IngestionJob, plus jobs that haven't produced chunks yet (queued, in
    progress, failed). `chunks` / `tokens` / `size` are kept for the
    pre-Phase-5 RAG Lab UI.
    """

    docs: dict[str, dict] = {}
    collection = get_v2_collection()
    if collection is not None:
        data = collection.get(include=["documents", "metadatas"])
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            meta = meta or {}
            asset_id = str(meta.get("asset_id") or meta.get("source") or "unknown")
            doc = docs.setdefault(
                asset_id,
                {
                    "asset_id": asset_id,
                    "name": str(meta.get("source", "")),
                    "document_revision": str(meta.get("document_revision", "")),
                    "config_version": str(meta.get("config_version", "")),
                    "chunk_count": 0,
                    "token_count": 0,
                },
            )
            doc["chunk_count"] += 1
            doc["token_count"] += count_tokens(text)

    # Latest job per asset (jobs are ordered newest first).
    latest_jobs: dict[str, IngestionJob] = {}
    for job in IngestionJob.objects.all():
        latest_jobs.setdefault(job.asset_id, job)

    for asset_id, job in latest_jobs.items():
        if asset_id not in docs:
            docs[asset_id] = {
                "asset_id": asset_id,
                "name": job.source_filename,
                "document_revision": job.document_revision,
                "config_version": job.config_version,
                "chunk_count": 0,
                "token_count": 0,
            }

    documents = []
    for asset_id, doc in docs.items():
        doc.update(_job_fields(latest_jobs.get(asset_id)))
        doc["size"] = _file_size(doc["name"])
        doc["chunks"] = doc["chunk_count"]
        doc["tokens"] = doc["token_count"]
        documents.append(doc)

    documents.sort(key=lambda d: d["ingested_at"] or "", reverse=True)
    return Response({"documents": documents})


@api_view(["DELETE"])
def rag_delete_doc(request, asset_id: str):
    """DELETE /api/rag/docs/<asset_id>/

    Removes the document's chunks from bsk_rag_v2 and its IngestionJob rows,
    so re-uploading the same file ingests it again instead of returning the
    old job (ingest idempotency is keyed on the job table).
    """

    chunks_deleted = 0
    collection = get_v2_collection()
    if collection is not None:
        ids = collection.get(where={"asset_id": asset_id}, include=[]).get("ids") or []
        if ids:
            collection.delete(ids=ids)
        chunks_deleted = len(ids)

    jobs_deleted, _ = IngestionJob.objects.filter(asset_id=asset_id).delete()

    if not chunks_deleted and not jobs_deleted:
        return Response({"error": f"Unknown asset_id: {asset_id}"}, status=status.HTTP_404_NOT_FOUND)

    logger.info("rag_delete_doc asset_id=%s chunks_deleted=%d jobs_deleted=%d",
                asset_id, chunks_deleted, jobs_deleted)
    return Response({"asset_id": asset_id, "chunks_deleted": chunks_deleted, "jobs_deleted": jobs_deleted})


@api_view(["GET"])
def rag_chunks(request):
    """GET /api/rag/chunks/?asset_id=X&offset=0&limit=20

    Chunks of one document in reading order, for the RAG Lab chunk browser.
    """

    asset_id = request.query_params.get("asset_id", "").strip()
    if not asset_id:
        return Response({"error": "asset_id is required"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        offset = max(0, int(request.query_params.get("offset", 0)))
        limit = int(request.query_params.get("limit", CHUNKS_DEFAULT_LIMIT))
    except ValueError:
        return Response({"error": "offset and limit must be integers"}, status=status.HTTP_400_BAD_REQUEST)
    limit = max(1, min(limit, CHUNKS_MAX_LIMIT))

    rows = []
    collection = get_v2_collection()
    if collection is not None:
        data = collection.get(where={"asset_id": asset_id}, include=["documents", "metadatas"])
        rows = sorted(
            zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []),
            key=lambda row: chunk_index(row[0]),
        )

    chunks = []
    for chunk_id, text, meta in rows[offset : offset + limit]:
        meta = meta or {}
        chunks.append(
            {
                "id": chunk_id,
                "index": chunk_index(chunk_id),
                "text": text or "",
                "asset_id": asset_id,
                "source": str(meta.get("source", "")),
                "section_path": list(meta.get("section_path") or []),
                "content_type": str(meta.get("content_type", "")),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "token_count": count_tokens(text),
            }
        )

    return Response({"asset_id": asset_id, "total": len(rows), "offset": offset, "limit": limit, "chunks": chunks})


@api_view(["GET"])
def rag_config(request):
    """GET /api/rag/config/ — current server-side RAG settings (read-only)."""

    collection = get_v2_collection()
    embedding_dims = None
    if collection is not None and collection.count():
        sample = collection.get(limit=1, include=["embeddings"])["embeddings"]
        embedding_dims = len(sample[0]) if sample is not None and len(sample) else None

    return Response(
        {
            "collection": {
                "name": settings.RAG_COLLECTION_V2,
                "chunk_count": collection.count() if collection is not None else 0,
                "distance_space": distance_space(collection) if collection is not None else None,
            },
            "chunker": {
                "target_tokens": chunker.TARGET_TOKENS,
                "max_tokens": chunker.MAX_TOKENS,
                "overlap_sentences": chunker.OVERLAP_SENTENCES,
                "tokenizer": "cl100k_base",
                "config_version": settings.INGESTION_CONFIG_VERSION,
            },
            "embedding": {"model": settings.DGX_EMBED_MODEL, "dimensions": embedding_dims},
            "reranker": {"model": settings.DGX_RERANK_MODEL},
            "retrieval": {
                "candidates": settings.RAG_V2_CANDIDATES,
                "chat_top_n": settings.RAG_CHAT_TOP_N,
                "query_default_top_k": settings.RAG_QUERY_DEFAULT_TOP_K,
                "query_max_top_k": settings.RAG_QUERY_MAX_TOP_K,
                "min_rerank_score": settings.RAG_MIN_RERANK_SCORE,
                "context_share": settings.RAG_CONTEXT_SHARE,
            },
        }
    )
