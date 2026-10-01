from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from .models import Message
from .retrieval import SearchResult, V2Chunk, search_v2
from .views import build_rag_context, citation_label


def _collection(texts, distances, space="cosine"):
    coll = MagicMock()
    coll.configuration_json = {"hnsw": {"space": space}}
    coll.count.return_value = len(texts)
    coll.query.return_value = {
        "ids": [[f"id{i}" for i in range(len(texts))]],
        "documents": [texts],
        "metadatas": [[
            {"source": "paper.pdf", "asset_id": "PAPER-1", "section_path": ["Paper", f"S{i}"],
             "page_start": i + 1, "page_end": i + 1, "content_type": "paragraph"}
            for i in range(len(texts))
        ]],
        "distances": [distances],
    }
    return coll


def _rerank_response(order_scores):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"results": [{"index": i, "relevance_score": s} for i, s in order_scores]}
    return resp


@patch("chat.retrieval.Embedder")
@patch("chat.retrieval.get_v2_collection")
class SearchV2Tests(SimpleTestCase):
    def test_reranks_and_keeps_both_scores(self, mock_coll, mock_embedder):
        mock_coll.return_value = _collection(["a", "b", "c"], [0.1, 0.2, 0.3])
        mock_embedder.return_value.embed_queries.return_value = [[0.0] * 4]
        with patch("chat.retrieval.requests.post", return_value=_rerank_response([(2, 0.9), (0, 0.5)])):
            result = search_v2("q", top_n=2)

        self.assertEqual(result.reranker, "ok")
        self.assertEqual(result.candidates, 3)
        self.assertEqual([c.text for c in result.chunks], ["c", "a"])
        self.assertEqual(result.chunks[0].rerank_score, 0.9)
        self.assertEqual(result.chunks[0].vector_score, 0.7)  # 1 - cosine distance
        self.assertEqual(result.chunks[0].section_path, ["Paper", "S2"])
        mock_embedder.return_value.embed_queries.assert_called_once_with(["q"])

    def test_falls_back_to_vector_order_when_reranker_fails(self, mock_coll, mock_embedder):
        mock_coll.return_value = _collection(["a", "b", "c"], [0.3, 0.1, 0.2])
        mock_embedder.return_value.embed_queries.return_value = [[0.0] * 4]
        with patch("chat.retrieval.requests.post", side_effect=TimeoutError("down")), \
                self.assertLogs("chat", level="WARNING"):
            result = search_v2("q", top_n=2)

        self.assertEqual(result.reranker, "fallback")
        self.assertEqual([c.text for c in result.chunks], ["b", "c"])
        self.assertIsNone(result.chunks[0].rerank_score)
        self.assertEqual(result.chunks[0].score, result.chunks[0].vector_score)

    def test_l2_distance_converted_for_unit_vectors(self, mock_coll, mock_embedder):
        # squared L2 of unit vectors = 2 - 2cos -> d=0.5 means cos=0.75
        mock_coll.return_value = _collection(["a"], [0.5], space="l2")
        mock_embedder.return_value.embed_queries.return_value = [[0.0] * 4]
        result = search_v2("q", top_n=1, rerank=False)
        self.assertEqual(result.chunks[0].vector_score, 0.75)

    def test_missing_collection_returns_nothing(self, mock_coll, mock_embedder):
        mock_coll.return_value = None
        self.assertEqual(search_v2("q", top_n=5).chunks, [])

    def test_empty_collection_skips_embedding(self, mock_coll, mock_embedder):
        mock_coll.return_value = _collection([], [])
        result = search_v2("q", top_n=5)
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.reranker, "skipped")
        mock_embedder.return_value.embed_queries.assert_not_called()


class BuildRagContextTests(SimpleTestCase):
    def test_adds_entries_in_order_within_budget(self):
        entries = [{"text": "x" * 300, "source": "a"}, {"text": "y" * 300, "source": "b"},
                   {"text": "z" * 50, "source": "c"}]
        prompt, used = build_rag_context(entries, budget_chars=600)
        # second entry doesn't fit, the smaller third one does
        self.assertEqual([e["source"] for e in used], ["a", "c"])
        self.assertLessEqual(len(prompt), 600)
        self.assertIn("[2] (source: c)", prompt)

    def test_truncates_top_entry_when_it_alone_exceeds_budget(self):
        prompt, used = build_rag_context([{"text": "x" * 5000, "source": "a"}], budget_chars=1000)
        self.assertEqual(len(used), 1)
        self.assertLessEqual(len(prompt), 1000)

    def test_returns_none_without_entries_or_room(self):
        self.assertEqual(build_rag_context([], 5000), (None, []))
        self.assertEqual(build_rag_context([{"text": "x" * 500, "source": "a"}], 150), (None, []))

    def test_citation_label(self):
        self.assertEqual(
            citation_label({"source": "p.pdf", "section_path": ["Paper", "3 Model"], "page_start": 3}),
            "p.pdf — 3 Model, p.3",
        )
        self.assertEqual(citation_label({"source": "plc_historian"}), "plc_historian")


def _v2_result(n=3, text_len=100):
    chunks = [
        V2Chunk(id=f"id{i}", text=f"chunk{i} " + "t" * text_len, source="paper.pdf", asset_id="PAPER-1",
                section_path=["Paper", f"S{i}"], page_start=i + 1, vector_score=0.6, rerank_score=0.9 - i / 10)
        for i in range(n)
    ]
    return SearchResult(chunks=chunks, candidates=20, reranker="ok", latency_ms=5)


def _llm_response(content="answer"):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class ChatViewRagTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _chat(self, message, **extra):
        body = {"conversation_id": None, "message": message, "model": "dgx-qwen38-27b-fp8", **extra}
        with self.assertLogs("chat", level="INFO"):
            return self.client.post("/api/chat/", body, format="json")

    @patch("chat.views.requests.post")
    @patch("chat.views.search_v2")
    def test_rag_on_by_default_stores_and_returns_citations(self, mock_search, mock_post):
        mock_search.return_value = _v2_result(3)
        mock_post.return_value = _llm_response()

        r = self._chat("What is attention?")  # no use_rag -> defaults to True

        self.assertEqual(r.status_code, 200)
        mock_search.assert_called_once_with("What is attention?", top_n=5)
        sources = r.data["messages"][-1]["sources"]
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0]["asset_id"], "PAPER-1")
        self.assertEqual(sources[0]["section_path"], ["Paper", "S0"])
        self.assertEqual(sources[0]["rerank_score"], 0.9)
        self.assertEqual(Message.objects.get(role="assistant").sources, sources)
        system = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("chunk0", system)
        self.assertIn("(source: paper.pdf — S0, p.1)", system)

    @patch("chat.views.requests.post")
    @patch("chat.views.search_v2")
    def test_ollama_budget_limits_rag_chunks(self, mock_search, mock_post):
        mock_search.return_value = _v2_result(5, text_len=1200)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = resp

        r = self._chat("What is attention?", model="ollama-qwen3-8b")

        # 6000 * 0.5 = 3000 chars of RAG -> only 2 of the ~1.2k-char chunks fit
        self.assertEqual(len(r.data["messages"][-1]["sources"]), 2)

    @patch("chat.views.requests.post")
    @patch("chat.views.search_v2")
    def test_drops_chunks_below_min_rerank_score(self, mock_search, mock_post):
        result = _v2_result(3)
        result.chunks[2].rerank_score = 0.0
        mock_search.return_value = result
        mock_post.return_value = _llm_response()

        r = self._chat("What is attention?")

        self.assertEqual([s["section_path"][-1] for s in r.data["messages"][-1]["sources"]], ["S0", "S1"])

    @patch("chat.views.requests.post")
    @patch("chat.views.search_v2")
    def test_rag_off_sends_no_context(self, mock_search, mock_post):
        mock_post.return_value = _llm_response()
        r = self._chat("hello", use_rag=False)
        mock_search.assert_not_called()
        self.assertEqual(r.data["messages"][-1]["sources"], [])

    @patch("chat.views.requests.post")
    @patch("chat.views.search_v2", side_effect=RuntimeError("embedder down"))
    def test_retrieval_failure_does_not_fail_chat(self, mock_search, mock_post):
        mock_post.return_value = _llm_response()
        r = self._chat("What is attention?")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["messages"][-1]["sources"], [])

    @patch("chat.views.requests.post")
    @patch("chat.views.query_chunks")
    @patch("chat.views.search_v2")
    def test_factory_question_uses_historian_first(self, mock_search, mock_historian, mock_post):
        mock_search.return_value = _v2_result(1)
        mock_historian.return_value = [MagicMock(text="EXTR01 OEE 81%", source="plc_historian")]
        mock_post.return_value = _llm_response()

        r = self._chat("What was the OEE on extruder 1?")

        mock_historian.assert_called_once()
        self.assertEqual(mock_historian.call_args.kwargs["where"], {"source": "plc_historian"})
        mock_search.assert_called_once_with("What was the OEE on extruder 1?", top_n=3)
        self.assertEqual([s["source"] for s in r.data["messages"][-1]["sources"]], ["plc_historian", "paper.pdf"])


class RagQueryViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("chat.views.search_v2")
    def test_clamps_top_k_and_returns_scores(self, mock_search):
        mock_search.return_value = _v2_result(2)
        with self.assertLogs("chat", level="INFO"):
            r = self.client.post("/api/rag/query/", {"query": "attention", "top_k": 50}, format="json")

        self.assertEqual(r.status_code, 200)
        mock_search.assert_called_once_with("attention", top_n=10)
        self.assertEqual(r.data["top_k"], 10)
        self.assertEqual(r.data["reranker"], "ok")
        first = r.data["results"][0]
        for key in ("id", "text", "source", "document_path", "asset_id", "section_path",
                    "page_start", "score", "vector_score", "rerank_score"):
            self.assertIn(key, first)
        self.assertEqual(first["score"], first["rerank_score"])

    def test_empty_query_is_400(self):
        r = self.client.post("/api/rag/query/", {"query": "  "}, format="json")
        self.assertEqual(r.status_code, 400)

    @patch("chat.views.search_v2", side_effect=RuntimeError("chroma down"))
    def test_search_failure_is_502(self, mock_search):
        with self.assertLogs("chat", level="ERROR"):
            r = self.client.post("/api/rag/query/", {"query": "attention"}, format="json")
        self.assertEqual(r.status_code, 502)
