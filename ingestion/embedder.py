"""Embedding client for DGX Nemotron endpoint.

Calls the OpenAI-compatible /v1/embeddings endpoint on the DGX Spark
(Nemotron-3-Embed-1B-NVFP4, 2048-dim). Uses the `passage: ` prefix
for indexing and `query: ` for retrieval (per DGX spec).

Config keys (from settings):
- DGX_EMBED_URL (default http://100.74.225.3:8010/v1/embeddings)
- DGX_EMBED_MODEL (default nemotron-3-embed-1b)
"""

from __future__ import annotations

from typing import List

import requests

from usage import recorder as usage
from django.conf import settings


class EmbedderError(Exception):
    pass


class Embedder:
    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.url = url or settings.DGX_EMBED_URL
        self.model = model or settings.DGX_EMBED_MODEL
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed document chunks (passage: prefix)."""
        if not texts:
            return []
        prefixed = [f"passage: {t}" for t in texts]
        return self._embed(prefixed)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embed query strings (query: prefix)."""
        if not texts:
            return []
        prefixed = [f"query: {t}" for t in texts]
        return self._embed(prefixed)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            with usage.track("embed", f"embed:{self.model}") as call:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                if resp.status_code >= 300:
                    raise EmbedderError(
                        f"Embedding failed: HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                data = resp.json()
                call.from_response(data)
            for item in data.get("data", []):
                out.append(item["embedding"])
        return out


# Convenience module-level functions (used by pipeline)
_embedder = Embedder()


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    return _embedder.embed_passages(chunks)
