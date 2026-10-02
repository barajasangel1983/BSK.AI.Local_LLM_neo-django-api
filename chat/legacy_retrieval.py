"""Retrieval over the legacy `bsk_rag` collection (plc_historian shift summaries).

Replaces the GraphRAG repo's `rag.retrieval.query_chunks` so every Chroma
access goes through ingestion.chroma_client (server mode). Embeddings for this
collection are Chroma's default MiniLM model, computed client-side.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingestion.chroma_client import legacy_collection


@dataclass
class LegacyChunk:
    id: str
    text: str
    document_path: str
    source: str
    score: float


def query_chunks(query: str, top_k: int = 5, where: dict | None = None) -> list[LegacyChunk]:
    if not query.strip():
        return []
    kwargs = {"query_texts": [query], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    result = legacy_collection().query(**kwargs)
    return [
        LegacyChunk(
            id=str(chunk_id),
            text=text or "",
            document_path=str((meta or {}).get("document_path", "")),
            source=str((meta or {}).get("source", "")),
            score=float(distance),
        )
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
