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


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.chroma_path = chroma_path or str(
            getattr(settings, "CHROMA_DIR", "/home/barajas_angel/repos/BSK.AI.Local_LLM_neo4j-graphrag/data/chroma_index")
        )
        self.collection_name = collection_name or settings.RAG_COLLECTION_V2
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False),
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
    ) -> int:
        """Write chunks + embeddings. Returns number of chunks written.

        Skips the entire batch if the idempotency key already exists.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise VectorStoreError("chunks, embeddings and metadatas must have same length")

        key = f"{source_sha256}:{document_revision}:{config_version}"
        if self.exists(source_sha256, document_revision, config_version):
            return 0  # idempotent skip

        coll = self._get_collection()
        ids = [f"{key}:{i}" for i in range(len(chunks))]

        # Add the ingest_key to every metadata record for future lookups
        enriched = []
        for m in metadatas:
            m = dict(m)  # copy
            m["ingest_key"] = key
            m["source_sha256"] = source_sha256
            m["config_version"] = config_version
            enriched.append(m)

        coll.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=enriched,
        )
        return len(chunks)


# Module-level convenience (used by pipeline)
_store = VectorStore()


def write_chunks(chunks, embeddings, metadatas, source_sha256, document_revision, config_version):
    return _store.write_chunks(
        chunks, embeddings, metadatas, source_sha256, document_revision, config_version
    )
