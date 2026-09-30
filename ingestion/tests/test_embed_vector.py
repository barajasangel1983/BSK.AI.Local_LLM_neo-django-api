"""Unit tests for embedder + vector_store (mocked HTTP + in-memory Chroma)."""

import os
from unittest.mock import patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "neo_llm_api.settings")
django.setup()

from unittest.mock import patch

import pytest

from ingestion.embedder import Embedder, embed_chunks
from ingestion.vector_store import VectorStore, write_chunks


def test_embedder_batches_and_prefix():
    with patch("ingestion.embedder.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": [0.1] * 2048}, {"embedding": [0.2] * 2048}]
        }
        emb = Embedder(batch_size=2)
        result = emb.embed_passages(["hello", "world"])
        assert len(result) == 2
        # Verify the prefix was added
        call_args = mock_post.call_args
        assert "passage: hello" in str(call_args)


def test_vector_store_idempotency(tmp_path):
    store = VectorStore(chroma_path=str(tmp_path / "chroma"), collection_name="test_v2")
    chunks = ["chunk one", "chunk two"]
    embs = [[0.1] * 2048, [0.2] * 2048]
    metas = [{"page": 1}, {"page": 2}]

    # First write should succeed
    n = store.write_chunks(chunks, embs, metas, "abc123", "v1", "v1-2026-09-29")
    assert n == 2

    # Second write with same key should be skipped
    n2 = store.write_chunks(chunks, embs, metas, "abc123", "v1", "v1-2026-09-29")
    assert n2 == 0


def test_module_level_write_chunks():
    # Smoke test that the convenience wrapper works
    with patch("ingestion.vector_store._store.write_chunks") as mock_write:
        mock_write.return_value = 3
        result = write_chunks(
            ["a", "b", "c"],
            [[0.0]] * 3,
            [{"p": i} for i in range(3)],
            "sha",
            "rev",
            "cfg",
        )
        assert result == 3
