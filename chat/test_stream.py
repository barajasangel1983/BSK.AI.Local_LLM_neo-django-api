"""P24-1 (2026-11-08): one answer engine, streamed.

Every text backend streams; POST /api/chat/stream/ sends the pieces as server-sent events and saves the reply;
POST /api/chat/ joins the same pieces. The model servers are mocked.
"""

import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from chat import views
from chat.models import Conversation


def _streamed(lines: list[bytes], status_ok: bool = True):
    resp = MagicMock()
    resp.iter_lines.side_effect = lambda *a, **k: iter(lines)
    resp.json.side_effect = ValueError("streamed")
    resp.raise_for_status.return_value = None
    return resp


def _openai(*pieces: str, usage: dict | None = None) -> list[bytes]:
    out = [b"data: " + json.dumps({"choices": [{"delta": {"content": p}}]}).encode() for p in pieces]
    if usage:
        out.append(b"data: " + json.dumps({"choices": [], "usage": usage}).encode())
    return [*out, b"data: [DONE]"]


def _ollama(*pieces: str) -> list[bytes]:
    out = [json.dumps({"message": {"role": "assistant", "content": p}, "done": False}).encode() for p in pieces]
    return [*out, json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 7}).encode()]


def _events(response) -> list[dict]:
    body = b"".join(response.streaming_content).decode()
    return [json.loads(chunk[len("data: "):]) for chunk in body.split("\n\n") if chunk.startswith("data: ")]


class StreamBackendTests(TestCase):
    @patch("chat.views.requests.post")
    def test_vllm_pieces_and_usage(self, post):
        post.return_value = _streamed(_openai("Wait ", "5 minutes.", usage={"prompt_tokens": 900, "completion_tokens": 4}))
        pieces, sent = views.stream_reply_backend("Is it safe?", "dgx-qwen38-27b-fp8", True, [], "SYSTEM")
        self.assertEqual(list(pieces), ["Wait ", "5 minutes."])
        self.assertEqual(sent, 0)
        kwargs = post.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["messages"][0]["content"], "SYSTEM")
        from usage.models import ModelCall
        call = ModelCall.objects.latest("id")
        self.assertEqual((call.prompt_tokens, call.completion_tokens, call.tokens_estimated), (900, 4, False))

    @patch("chat.views.requests.post")
    def test_ollama_lines(self, post):
        post.return_value = _streamed(_ollama("Check ", "the fan."))
        pieces, _ = views.stream_reply_backend("fan?", "ollama-qwen3-8b", True, [])
        self.assertEqual("".join(pieces), "Check the fan.")
        self.assertTrue(post.call_args.kwargs["json"]["stream"])

    @patch("chat.views.requests.post")
    def test_a_server_that_ignores_stream_still_answers(self, post):
        resp = MagicMock()
        resp.iter_lines.side_effect = lambda *a, **k: iter([])
        resp.json.return_value = {"choices": [{"message": {"content": "Whole reply."}}]}
        post.return_value = resp
        reply, _ = views.generate_reply_backend("x", "dgx-qwen38-27b-fp8", False, [])
        self.assertEqual(reply, "Whole reply.")


@override_settings(CHAT_HISTORIAN_WINDOWS=False)
class ChatStreamViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        patcher = patch("chat.views.generate_title", return_value=("Title", "llm"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, path, **body):
        return self.client.post(path, {"conversation_id": None, "message": "Hello", "model": "dgx-qwen38-27b-fp8",
                                       "use_rag": False, **body}, format="json")

    @patch("chat.views.requests.post")
    def test_tokens_then_sources_then_done_and_the_reply_is_saved(self, post):
        post.return_value = _streamed(_openai("Hi ", "Angel."))
        r = self._post("/api/chat/stream/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "text/event-stream")
        events = _events(r)
        self.assertEqual([e["type"] for e in events], ["model", "token", "token", "sources", "done"])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "Hi Angel.")
        done = events[-1]["conversation"]
        self.assertEqual([m["role"] for m in done["messages"]], ["user", "assistant"])
        self.assertEqual(done["messages"][-1]["content"], "Hi Angel.")
        self.assertEqual(done["title"], "Title")
        self.assertEqual(Conversation.objects.get(id=done["id"]).messages.count(), 2)

    @patch("chat.views.requests.post")
    def test_the_same_request_unstreamed_gets_the_same_reply(self, post):
        post.side_effect = lambda *a, **k: _streamed(_openai("Same ", "answer."))
        streamed = _events(self._post("/api/chat/stream/"))[-1]["conversation"]["messages"][-1]["content"]
        whole = self._post("/api/chat/").data["messages"][-1]["content"]
        self.assertEqual(streamed, whole)
        # Both sent the model the same messages.
        self.assertEqual(post.call_args_list[0].kwargs["json"]["messages"], post.call_args_list[1].kwargs["json"]["messages"])

    @patch("chat.views.requests.post")
    def test_a_model_failure_is_an_error_event(self, post):
        post.side_effect = RuntimeError("down")
        events = _events(self._post("/api/chat/stream/"))
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["kind"], "model")
        self.assertIn("down", events[-1]["message"])

    def test_a_bad_request_is_refused_before_streaming(self):
        r = self.client.post("/api/chat/stream/", {"message": ""}, format="json")
        self.assertEqual(r.status_code, 400)
