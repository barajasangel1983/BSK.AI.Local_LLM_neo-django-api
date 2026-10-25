"""Chroma vector store writer for bsk_rag_v2.

Idempotency key = (source_sha256, document_revision, config_version).
Before writing, we check if a document with the same key already exists.
If so, we skip (or optionally replace).

Collection: bsk_rag_v2 (created on first use, cosine distance).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from django.conf import settings

from .chroma_client import get_client


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        # An explicit path opens that folder embedded (tests); otherwise the
        # shared client (Chroma server in normal operation).
        self.chroma_path = chroma_path
        self.collection_name = collection_name or settings.RAG_COLLECTION_V2
        self._client = None
        self._collection: Optional[chromadb.Collection] = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = (
                chromadb.PersistentClient(path=self.chroma_path, settings=ChromaSettings(anonymized_telemetry=False))
                if self.chroma_path else get_client()
            )
            # Chroma 1.x reads the distance space from `configuration`; the old
            # metadata={"hnsw:space": ...} form is ignored (collections ended up L2).
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
        return self._collection

    def exists(self, source_sha256: str, document_revision: str, config_version: str) -> bool:
        """Return True if a document with this idempotency key already exists."""
        coll = self._get_collection()
        # We store the key in metadata under a single field for fast lookup
        key = f"{source_sha256}:{document_revision}:{config_version}"
        res = coll.get(where={"ingest_key": key}, limit=1)
        return bool(res.get("ids"))

    def write_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        source_sha256: str,
        document_revision: str,
        config_version: str,
        variant: str = "",
    ) -> int:
        """Write chunks + embeddings. Returns number of chunks written.

        Skips the entire batch if the idempotency key already exists. `variant`
        (chunking strategy + params) is part of the key, so the same file can be
        re-embedded with different chunking.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise VectorStoreError("chunks, embeddings and metadatas must have same length")

        key = f"{source_sha256}:{document_revision}:{config_version}"
        if variant:
            key = f"{key}:{variant}"
        coll = self._get_collection()
        if coll.get(where={"ingest_key": key}, limit=1).get("ids"):
            return 0  # idempotent skip

        ids = [f"{key}:{i}" for i in range(len(chunks))]

        # Add the ingest_key to every metadata record for future lookups
        enriched = []
        for m in metadatas:
            m = dict(m)  # copy
            m["ingest_key"] = key
            m["source_sha256"] = source_sha256
            m["config_version"] = config_version
            enriched.append(m)

        # In batches: one Chroma call takes a limited number of records (5,461 in 1.5).
        step = self.max_batch()
        for start in range(0, len(ids), step):
            end = start + step
            coll.add(ids=ids[start:end], documents=chunks[start:end],
                     embeddings=embeddings[start:end], metadatas=enriched[start:end])
        return len(chunks)

    def update_document_metadata(self, asset_id: str, fields: Dict[str, Any]) -> int:
        """Set metadata fields on every chunk of one document (no new embeddings)."""
        coll = self._get_collection()
        found = coll.get(where={"asset_id": asset_id}, include=["metadatas"])
        ids = found.get("ids") or []
        step = self.max_batch()
        for start in range(0, len(ids), step):
            metas = [{**(m or {}), **fields} for m in found["metadatas"][start:start + step]]
            coll.update(ids=ids[start:start + step], metadatas=metas)
        return len(ids)

    def document_chunks(self, asset_id: str) -> dict:
        """One document's chunks: {ids, documents, metadatas} (no embeddings)."""
        return self._get_collection().get(where={"asset_id": asset_id}, include=["documents", "metadatas"])

    def update_chunk_metadata(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> int:
        """Replace the metadata of these chunks (P25b spans): ids and embeddings stay."""
        coll = self._get_collection()
        step = self.max_batch()
        for start in range(0, len(ids), step):
            coll.update(ids=ids[start:start + step], metadatas=metadatas[start:start + step])
        return len(ids)

    def max_batch(self) -> int:
        """Records per Chroma call: the client's limit, capped at 2000 (1000 if it can't say)."""
        self._get_collection()
        try:
            return max(1, min(int(self._client.get_max_batch_size()), 2000))
        except Exception:
            return 1000

    def delete_document(self, asset_id: str) -> int:
        """Remove every chunk of one document (by its `asset_id` / doc key)."""
        coll = self._get_collection()
        ids = coll.get(where={"asset_id": asset_id}, include=[]).get("ids") or []
        step = self.max_batch()
        for start in range(0, len(ids), step):
            coll.delete(ids=ids[start:start + step])
        return len(ids)


# Module-level convenience (used by pipeline)
_store = VectorStore()


def write_chunks(chunks, embeddings, metadatas, source_sha256, document_revision, config_version):
    return _store.write_chunks(
        chunks, embeddings, metadatas, source_sha256, document_revision, config_version
    )
