"""The one place that opens Chroma.

Server mode (CHROMA_HOST set, the normal setup): chromadb.HttpClient to the
Chroma server (deploy/chroma/docker-compose.yml). Required because several
processes use Chroma — the API (runserver), the library worker and management
commands — and the embedded PersistentClient is not process-safe (a process
keeps a stale cache of data written by another one). The client is created
once per process and reused.

Embedded mode (CHROMA_HOST empty): PersistentClient(CHROMA_DIR). Only for
tests / single-process use. Never run it against the folder the server uses.
"""

from __future__ import annotations

import threading

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from django.conf import settings

LEGACY_COLLECTION = "bsk_rag"  # plc_historian shift summaries (Chroma's default MiniLM embeddings)


def server_mode() -> bool:
    return bool(settings.CHROMA_HOST)


_http_clients: dict = {}        # (host, port) → HttpClient, one per process
_http_lock = threading.Lock()


def get_client():
    if server_mode():
        # One client per server and process: every HttpClient keeps its own connection pool and never
        # closes it, so a new one per call leaks a socket each time (the API ran out of file handles).
        key = (settings.CHROMA_HOST, settings.CHROMA_PORT)
        with _http_lock:
            if key not in _http_clients:
                _http_clients[key] = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            return _http_clients[key]
    return chromadb.PersistentClient(path=str(settings.CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False))


def describe() -> str:
    return f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}" if server_mode() else str(settings.CHROMA_DIR)


def legacy_collection():
    """The legacy `bsk_rag` collection; queries embed text client-side with Chroma's default model."""
    return get_client().get_or_create_collection(name=LEGACY_COLLECTION, embedding_function=DefaultEmbeddingFunction())
