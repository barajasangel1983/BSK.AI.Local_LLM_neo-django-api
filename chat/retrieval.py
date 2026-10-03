"""Retrieval over the bsk_rag_v2 collection (Phase 5).

Flow: embed the query on the DGX (Nemotron, 2048-dim, "query: " prefix)
-> vector search in Chroma bsk_rag_v2 (distance space read from the collection)
   for RAG_V2_CANDIDATES
-> rerank with the DGX cross-encoder and keep the top N.

If the reranker is unavailable, results fall back to vector order so chat
keeps working. The embedder is required: without it v2 cannot be queried
(its vectors are not compatible with Chroma's default embedding function).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import requests

from usage import recorder as usage
from django.conf import settings

from ingestion.chroma_client import get_client
from ingestion.embedder import Embedder

logger = logging.getLogger("chat")


@dataclass
class V2Chunk:
    id: str
    text: str
    source: str
    asset_id: str
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    content_type: str = ""
    vector_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Best available relevance score (higher is better)."""
        return self.rerank_score if self.rerank_score is not None else self.vector_score

    def to_dict(self) -> dict:
        return {**asdict(self), "score": self.score}


@dataclass
class SearchResult:
    chunks: list[V2Chunk]
    candidates: int
    reranker: str  # "ok" | "fallback" | "skipped"
    latency_ms: int


def get_v2_collection():
    """Return the bsk_rag_v2 collection, or None if it hasn't been created yet.

    Read-only: never creates the collection (the ingestion pipeline owns it).
    """
    try:
        return get_client().get_collection(name=settings.RAG_COLLECTION_V2)
    except Exception:  # chromadb raises NotFoundError (type varies by version)
        return None


def distance_space(collection) -> str:
    """HNSW distance space of a collection: 'l2' (Chroma default), 'cosine' or 'ip'."""
    config = getattr(collection, "configuration_json", None) or {}
    return ((config.get("hnsw") or {}).get("space")) or "l2"


def to_similarity(distance: float, space: str) -> float:
    """Convert a Chroma distance to a similarity in [-1, 1] (higher is better).

    Assumes unit-length embeddings (the DGX embedder returns normalized vectors),
    for which squared L2 distance = 2 - 2 * cosine.
    """
    if space == "l2":
        return 1.0 - distance / 2.0
    return 1.0 - distance  # cosine and ip distances are 1 - similarity


def _vector_search(query: str, n_candidates: int, where: dict | None = None) -> list[V2Chunk]:
    collection = get_v2_collection()
    if collection is None or collection.count() == 0:
        return []
    space = distance_space(collection)

    embedding = Embedder().embed_queries([query])[0]
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_candidates, collection.count()),
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )

    chunks: list[V2Chunk] = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        meta = meta or {}
        section_path = meta.get("section_path") or []
        chunks.append(
            V2Chunk(
                id=str(chunk_id),
                text=text or "",
                source=str(meta.get("source", "")),
                asset_id=str(meta.get("asset_id", "")),
                section_path=list(section_path) if isinstance(section_path, (list, tuple)) else [str(section_path)],
                page_start=meta.get("page_start"),
                page_end=meta.get("page_end"),
                content_type=str(meta.get("content_type", "")),
                vector_score=round(to_similarity(float(distance), space), 4),
            )
        )
    return chunks


def _rerank(query: str, chunks: list[V2Chunk], top_n: int) -> list[V2Chunk]:
    """Reorder chunks with the DGX reranker; raises on any failure."""

    with usage.track("rerank", f"rerank:{settings.DGX_RERANK_MODEL}") as call:
        data = _post_rerank(query, chunks, top_n).json()
        call.from_response(data)

    ranked: list[V2Chunk] = []
    for item in data["results"]:
        chunk = chunks[item["index"]]
        chunk.rerank_score = round(float(item["relevance_score"]), 4)
        ranked.append(chunk)
    return ranked[:top_n]


def _post_rerank(query: str, chunks: list[V2Chunk], top_n: int):
    resp = requests.post(
        settings.DGX_RERANK_URL,
        json={
            "model": settings.DGX_RERANK_MODEL,
            "query": query,
            "documents": [c.text for c in chunks],
            "top_n": top_n,
        },
        timeout=settings.RAG_RERANK_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def search_v2(query: str, top_n: int, rerank: bool = True, doc_keys: list[str] | None = None) -> SearchResult:
    """Vector search + rerank over bsk_rag_v2. Raises if embedding/search fails.

    `doc_keys` limits the search to those documents (chunk metadata `asset_id` holds the document key).
    """

    start = time.time()
    where = {"asset_id": {"$in": list(doc_keys)}} if doc_keys else None
    candidates = _vector_search(query, max(settings.RAG_V2_CANDIDATES, top_n), where)

    reranker = "skipped"
    chunks = candidates
    if rerank and candidates:
        try:
            chunks = _rerank(query, candidates, top_n)
            reranker = "ok"
        except Exception as exc:
            logger.warning("reranker unavailable, using vector order: %s", exc)
            reranker = "fallback"

    if reranker != "ok":
        chunks = sorted(candidates, key=lambda c: c.vector_score, reverse=True)[:top_n]

    return SearchResult(
        chunks=chunks,
        candidates=len(candidates),
        reranker=reranker,
        latency_ms=round((time.time() - start) * 1000),
    )
