"""Model-call recording and the Analytics summary."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import requests
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from usage import recorder as usage
from usage.models import ModelCall
from usage.summary import _p95


def ok(data):
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    return resp


class RecorderTests(TestCase):
    def test_token_formats(self):
        call = usage.Call()
        call.from_response({"usage": {"prompt_tokens": 120, "completion_tokens": 30}})
        self.assertEqual((call.prompt_tokens, call.completion_tokens, call.estimated), (120, 30, False))
        call = usage.Call()
        call.from_response({"prompt_eval_count": 50, "eval_count": 7})          # Ollama
        self.assertEqual((call.prompt_tokens, call.completion_tokens), (50, 7))
        call = usage.Call()
        call.from_response({"choices": []})                                      # no usage reported
        call.estimate("hello world, how are you?", "fine")
        self.assertTrue(call.estimated)
        self.assertGreater(call.prompt_tokens, 0)
        reported = usage.Call()
        reported.tokens(10, 2)
        reported.estimate("long text " * 50, "x")                                # never overrides reported usage
        self.assertEqual((reported.prompt_tokens, reported.estimated), (10, False))

    def test_track_records_ok_error_and_timeout_and_reraises(self):
        with usage.scope(purpose="compare", conversation_id="5f0b3c9e-3a8a-4c0e-9d7e-2d1f0a1b2c3d"):
            with usage.track(None, "dgx-qwen") as call:
                call.tokens(100, 20)
            with self.assertRaises(requests.Timeout):
                with usage.track("title", "dgx-qwen"):
                    raise requests.Timeout("read timed out")
        with self.assertRaises(ValueError):
            with usage.track("extract", "dgx-qwen"):
                raise ValueError("bad json")
        rows = list(ModelCall.objects.order_by("id").values("purpose", "status", "prompt_tokens", "conversation_id", "error"))
        self.assertEqual([(r["purpose"], r["status"]) for r in rows], [("compare", "ok"), ("title", "timeout"), ("extract", "error")])
        self.assertEqual(rows[0]["prompt_tokens"], 100)
        self.assertEqual(str(rows[0]["conversation_id"]), "5f0b3c9e-3a8a-4c0e-9d7e-2d1f0a1b2c3d")
        self.assertIsNone(rows[2]["conversation_id"])                            # scope ended
        self.assertEqual(rows[2]["error"], "ValueError: bad json")

    def test_a_broken_recorder_never_breaks_the_call(self):
        with patch("usage.models.ModelCall.objects.create", side_effect=RuntimeError("db locked")):
            with self.assertLogs("usage", level="ERROR"):
                with usage.track("chat", "m") as call:
                    call.tokens(1, 1)
                    result = "reply"
        self.assertEqual(result, "reply")

    def test_p95(self):
        self.assertIsNone(_p95([]))
        self.assertEqual(_p95([5]), 5)
        self.assertEqual(_p95(list(range(1, 101))), 95)
        self.assertEqual(_p95(list(range(1, 21))), 19)


class ChatRecordingTests(TestCase):
    """The chat view labels every model call with its purpose and conversation."""

    def setUp(self):
        self.client = APIClient()

    @patch("chat.views.generate_title", return_value=("Title", "llm"))
    @patch("chat.views.requests.post")
    def test_chat_and_compare_calls(self, post, title):
        post.return_value = ok({"choices": [{"message": {"content": "hi"}}],
                                "usage": {"prompt_tokens": 40, "completion_tokens": 3}})
        r = self.client.post("/api/chat/", {"conversation_id": None, "message": "hello", "model": "dgx-qwen38-27b-fp8",
                                            "use_rag": False}, format="json")
        self.client.post("/api/chat/", {"conversation_id": None, "message": "hello", "model": "dgx-qwen38-27b-fp8",
                                        "use_rag": False, "purpose": "compare"}, format="json")
        post.return_value = ok({"message": {"content": "hey"}, "prompt_eval_count": 12, "eval_count": 4})
        self.client.post("/api/chat/", {"conversation_id": None, "message": "hello", "model": "ollama-qwen3-8b",
                                        "use_rag": False, "purpose": "bogus"}, format="json")
        rows = list(ModelCall.objects.order_by("id"))
        self.assertEqual([(c.purpose, c.model_id, c.prompt_tokens, c.completion_tokens) for c in rows], [
            ("chat", "dgx-qwen38-27b-fp8", 40, 3), ("compare", "dgx-qwen38-27b-fp8", 40, 3),
            ("chat", "ollama-qwen3-8b", 12, 4)])
        self.assertEqual(str(rows[0].conversation_id), r.data["id"])

    @patch("chat.views.generate_title", return_value=("Title", "llm"))
    @patch("chat.views.requests.post", side_effect=requests.Timeout("slow"))
    def test_failed_chat_is_recorded(self, post, title):
        r = self.client.post("/api/chat/", {"conversation_id": None, "message": "hello", "model": "dgx-qwen38-27b-fp8",
                                            "use_rag": False}, format="json")
        self.assertEqual(r.status_code, 502)
        self.assertEqual(ModelCall.objects.get().status, "timeout")

    @patch("chat.titles.requests.post")
    def test_title_calls(self, post):
        from chat.titles import generate_title
        post.return_value = ok({"choices": [{"message": {"content": "Die plug help"}}],
                                "usage": {"prompt_tokens": 90, "completion_tokens": 4}})
        self.assertEqual(generate_title("q", "a"), ("Die plug help", "llm"))
        self.assertEqual((ModelCall.objects.get().purpose, ModelCall.objects.get().completion_tokens), ("title", 4))


class SummaryApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def add(self, days_ago, purpose, model, prompt, completion, latency, status="ok", estimated=False):
        call = ModelCall.objects.create(purpose=purpose, model_id=model, prompt_tokens=prompt,
                                        completion_tokens=completion, latency_ms=latency, status=status,
                                        tokens_estimated=estimated)
        ModelCall.objects.filter(pk=call.pk).update(created_at=timezone.now() - timedelta(days=days_ago))

    def test_summary(self):
        self.add(0, "chat", "dgx-qwen", 100, 20, 1000)
        self.add(0, "chat", "dgx-qwen", 200, 40, 3000)
        self.add(1, "extract", "dgx-qwen", 500, 300, 9000, estimated=True)
        self.add(1, "chat", "ollama-qwen3-8b", None, None, 60000, status="timeout")
        self.add(2, "embed", "embed:e5", None, None, 50)
        self.add(20, "chat", "dgx-qwen", 999, 999, 1)                  # outside 7 days

        body = self.client.get("/api/usage/summary/?days=7").json()
        self.assertEqual(body["totals"]["requests"], 5)
        self.assertEqual((body["totals"]["prompt_tokens"], body["totals"]["completion_tokens"]), (800, 360))
        self.assertEqual((body["totals"]["errors"], body["totals"]["error_rate"]), (1, 0.2))
        self.assertEqual(body["totals"]["estimated_calls"], 1)
        dgx = next(m for m in body["per_model"] if m["model_id"] == "dgx-qwen")
        self.assertEqual((dgx["requests"], dgx["avg_latency_ms"], dgx["p95_latency_ms"]), (3, 4333, 9000))
        ollama = next(m for m in body["per_model"] if m["model_id"] == "ollama-qwen3-8b")
        self.assertIsNone(ollama["avg_latency_ms"])                     # failed calls don't count as latency
        self.assertEqual({p["purpose"] for p in body["per_purpose"]}, {"chat", "extract", "embed"})
        self.assertEqual(len(body["daily"]), 7)
        today = body["daily"][-1]
        self.assertEqual(today["models"]["dgx-qwen"], {"requests": 2, "prompt_tokens": 300, "completion_tokens": 60})
        self.assertIsNotNone(body["measured_since"])
        self.assertIn("conversations_per_model", body)

        chat_only = self.client.get("/api/usage/summary/?days=30&purpose=chat").json()
        self.assertEqual(chat_only["totals"]["requests"], 4)
        self.assertEqual(set(chat_only["purposes"]), {"chat", "extract", "embed"})
        self.assertEqual(self.client.get("/api/usage/summary/?days=3").status_code, 400)
