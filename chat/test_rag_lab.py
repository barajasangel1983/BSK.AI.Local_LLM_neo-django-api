from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from ingestion.models import IngestionJob

from .rag_lab_views import chunk_index

KEY = "sha:1:v1"


def _chunk_rows(asset_id, n, source="paper.pdf", start=0):
    ids = [f"{asset_id}-{KEY}:{i}" for i in range(start, start + n)]
    docs = [f"text of chunk {i}" for i in range(start, start + n)]
    metas = [
        {"asset_id": asset_id, "source": source, "document_revision": "1", "config_version": "v1",
         "section_path": ["Paper", f"S{i}"], "content_type": "paragraph", "page_start": 1, "page_end": 1}
        for i in range(start, start + n)
    ]
    return ids, docs, metas


def _collection(*groups):
    """Fake Chroma collection holding the given (ids, docs, metas) groups."""
    ids, docs, metas = [], [], []
    for g in groups:
        ids += g[0]
        docs += g[1]
        metas += g[2]

    coll = MagicMock()
    coll.count.return_value = len(ids)
    coll.configuration_json = {"hnsw": {"space": "l2"}}

    def get(where=None, include=None, limit=None):
        rows = list(zip(ids, docs, metas))
        if where:
            rows = [r for r in rows if all(r[2].get(k) == v for k, v in where.items())]
        if limit:
            rows = rows[:limit]
        result = {"ids": [r[0] for r in rows]}
        include = include or []
        if "documents" in include:
            result["documents"] = [r[1] for r in rows]
        if "metadatas" in include:
            result["metadatas"] = [r[2] for r in rows]
        if "embeddings" in include:
            result["embeddings"] = [[0.0] * 8 for _ in rows]
        return result

    coll.get.side_effect = get
    return coll


def _job(asset_id, status=IngestionJob.Status.DONE, sha="sha", **extra):
    return IngestionJob.objects.create(
        source_filename=extra.pop("source_filename", f"{asset_id}.pdf"),
        source_sha256=sha, document_id=asset_id, asset_id=asset_id, status=status, **extra,
    )


class ChunkIndexTests(SimpleTestCase):
    def test_parses_trailing_index(self):
        self.assertEqual(chunk_index("abc:1:v1:12"), 12)
        self.assertEqual(chunk_index("no-index"), 0)


@patch("chat.rag_lab_views.get_v2_collection")
class RagDocsTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_groups_chunks_by_asset_and_joins_jobs(self, mock_coll):
        mock_coll.return_value = _collection(_chunk_rows("PAPER-1", 3), _chunk_rows("MANUAL-2", 2, "manual.pdf"))
        _job("PAPER-1", sha="s1", source_filename="paper.pdf", chunk_count=3)
        _job("FAILED-3", status=IngestionJob.Status.FAILED, sha="s3", error="docling 500")

        docs = {d["asset_id"]: d for d in self.client.get("/api/rag/docs/").data["documents"]}

        self.assertEqual(set(docs), {"PAPER-1", "MANUAL-2", "FAILED-3"})
        paper = docs["PAPER-1"]
        self.assertEqual((paper["name"], paper["chunk_count"], paper["status"]), ("paper.pdf", 3, "done"))
        self.assertGreater(paper["token_count"], 0)
        self.assertEqual((paper["chunks"], paper["tokens"]), (paper["chunk_count"], paper["token_count"]))
        self.assertIsNotNone(paper["job_id"])
        self.assertEqual(docs["MANUAL-2"]["job_id"], None)  # chunks without a job record
        self.assertEqual((docs["FAILED-3"]["chunk_count"], docs["FAILED-3"]["status"]), (0, "failed"))
        self.assertEqual(docs["FAILED-3"]["error"], "docling 500")

    def test_missing_collection_lists_jobs_only(self, mock_coll):
        mock_coll.return_value = None
        _job("QUEUED-1", status=IngestionJob.Status.QUEUED)
        docs = self.client.get("/api/rag/docs/").data["documents"]
        self.assertEqual([(d["asset_id"], d["status"]) for d in docs], [("QUEUED-1", "queued")])


@patch("chat.rag_lab_views.get_v2_collection")
class RagDeleteDocTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_deletes_chunks_and_jobs_for_asset_only(self, mock_coll):
        coll = _collection(_chunk_rows("PAPER-1", 3), _chunk_rows("OTHER-2", 2))
        mock_coll.return_value = coll
        _job("PAPER-1", sha="s1")
        _job("OTHER-2", sha="s2")

        with self.assertLogs("chat", level="INFO"):
            r = self.client.delete("/api/rag/docs/PAPER-1/")

        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.data["chunks_deleted"], r.data["jobs_deleted"]), (3, 1))
        deleted_ids = coll.delete.call_args.kwargs["ids"]
        self.assertTrue(all(i.startswith("PAPER-1-") for i in deleted_ids))
        self.assertEqual(list(IngestionJob.objects.values_list("asset_id", flat=True)), ["OTHER-2"])

    def test_unknown_asset_is_404(self, mock_coll):
        mock_coll.return_value = _collection(_chunk_rows("PAPER-1", 1))
        self.assertEqual(self.client.delete("/api/rag/docs/NOPE/").status_code, 404)


@patch("chat.rag_lab_views.get_v2_collection")
class RagChunksTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    def test_returns_chunks_in_reading_order_with_paging(self, mock_coll):
        # stored out of order: indexes 5..9 then 0..4
        mock_coll.return_value = _collection(_chunk_rows("PAPER-1", 5, start=5), _chunk_rows("PAPER-1", 5))

        d = self.client.get("/api/rag/chunks/", {"asset_id": "PAPER-1", "offset": 3, "limit": 4}).data

        self.assertEqual(d["total"], 10)
        self.assertEqual([c["index"] for c in d["chunks"]], [3, 4, 5, 6])
        first = d["chunks"][0]
        self.assertEqual(first["section_path"], ["Paper", "S3"])
        self.assertEqual(first["content_type"], "paragraph")
        self.assertGreater(first["token_count"], 0)

    def test_limit_is_clamped(self, mock_coll):
        mock_coll.return_value = _collection(_chunk_rows("PAPER-1", 2))
        d = self.client.get("/api/rag/chunks/", {"asset_id": "PAPER-1", "limit": 500}).data
        self.assertEqual(d["limit"], 100)

    def test_validation(self, mock_coll):
        mock_coll.return_value = _collection()
        self.assertEqual(self.client.get("/api/rag/chunks/").status_code, 400)
        r = self.client.get("/api/rag/chunks/", {"asset_id": "X", "offset": "abc"})
        self.assertEqual(r.status_code, 400)


@patch("chat.rag_lab_views.get_v2_collection")
class RagConfigTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    def test_reports_chunker_and_retrieval_settings(self, mock_coll):
        mock_coll.return_value = _collection(_chunk_rows("PAPER-1", 2))
        d = self.client.get("/api/rag/config/").data
        self.assertEqual(
            (d["chunker"]["target_tokens"], d["chunker"]["max_tokens"], d["chunker"]["overlap_sentences"]),
            (600, 800, 1),
        )
        self.assertEqual((d["retrieval"]["query_default_top_k"], d["retrieval"]["query_max_top_k"]), (5, 10))
        self.assertEqual(d["embedding"]["dimensions"], 8)
        self.assertEqual((d["collection"]["chunk_count"], d["collection"]["distance_space"]), (2, "l2"))

    def test_missing_collection(self, mock_coll):
        mock_coll.return_value = None
        d = self.client.get("/api/rag/config/").data
        self.assertEqual((d["collection"]["chunk_count"], d["embedding"]["dimensions"]), (0, None))
