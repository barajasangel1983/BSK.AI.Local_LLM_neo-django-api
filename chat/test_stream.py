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
        self.assertEqual([e["type"] for e in events], ["status", "model", "token", "token", "sources", "done"])
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

    @patch("chat.views.requests.post")
    def test_a_browser_asking_for_an_event_stream_is_answered(self, post):
        """The Studio sends Accept: text/event-stream; DRF refused it with 406 on 2026-11-08."""
        post.return_value = _streamed(_openai("ok"))
        r = self.client.post("/api/chat/stream/", {"message": "Hello", "model": "dgx-qwen38-27b-fp8", "use_rag": False},
                             format="json", HTTP_ACCEPT="text/event-stream")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(_events(r)[-1]["type"], "done")
        bad = self.client.post("/api/chat/stream/", {"message": ""}, format="json", HTTP_ACCEPT="text/event-stream")
        self.assertEqual((bad.status_code, json.loads(bad.content)["error"]), (400, "message is required"))

    def test_a_bad_request_is_refused_before_streaming(self):
        r = self.client.post("/api/chat/stream/", {"message": ""}, format="json")
        self.assertEqual(r.status_code, 400)


@override_settings(CHAT_HISTORIAN_WINDOWS=False, CHAT_QUESTION_AGENT=True)
class EngineFeatureTests(TestCase):
    """P24-2: EDGE's question agent, "nothing found" and "which machine?" in the one engine."""

    def setUp(self):
        self.client = APIClient()
        for target, value in (("chat.views.generate_title", ("Title", "llm")),
                              ("chat.question_agent.known_names", ["Extruder EXTR01", "BX80 manual"])):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # No machine named in these questions.
        patcher = patch("context_service.service.resolve", return_value={"asset": None, "focus": [], "ambiguous": []})
        self.resolve = patcher.start()
        self.addCleanup(patcher.stop)

    def _ask(self, message, conversation_id=None):
        return _events(self.client.post("/api/chat/stream/", {"conversation_id": conversation_id, "message": message,
                                                               "model": "dgx-qwen38-27b-fp8", "use_rag": True},
                                        format="json"))

    @patch("chat.views.requests.post")
    def test_a_name_almost_known_is_asked_about_without_a_model_call(self, post):
        events = self._ask("what is the temperature range of the BX81?")
        self.assertEqual(post.call_count, 0)
        reply = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn("Did you mean BX80?", reply)
        agent = [s for s in events[-2]["sources"] if s.get("kind") == "agent"][0]
        self.assertEqual(agent["action"], "confirm")

    @patch("chat.views.search_v2")
    @patch("chat.views.requests.post")
    def test_topics_are_searched_and_shown_and_nothing_found_says_where_it_looked(self, post, search):
        agent_reply = json.dumps({"action": "search", "queries": ["BX80 temperature range", "BX80 cleaning"], "message": ""})
        post.return_value = _streamed(_openai(agent_reply))
        empty = MagicMock(chunks=[], reranker="ok")
        search.return_value = empty
        events = self._ask("tell me the BX80 temperature range and how to clean it")
        self.assertEqual(post.call_count, 1)                  # the agent only: the answer model is not asked
        queries = [c.args[0] for c in search.call_args_list]
        self.assertEqual(queries, ["tell me the BX80 temperature range and how to clean it", "BX80 temperature range",
                                   "BX80 cleaning"])
        reply = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn("found nothing relevant enough", reply)
        self.assertIn("“BX80 cleaning”", reply)
        agent = [s for s in events[-2]["sources"] if s.get("kind") == "agent"][0]
        self.assertEqual((agent["action"], agent["queries"]), ("nothing_found", ["BX80 temperature range", "BX80 cleaning"]))

    @patch("chat.views.requests.post")
    def test_a_question_naming_two_machines_asks_which(self, post):
        self.resolve.return_value = {"asset": None, "focus": [], "ambiguous": [
            {"phrase": "asset", "candidates": [{"id": "bsk:asset:A", "name": "A"}, {"id": "bsk:asset:B", "name": "B"}]}]}
        events = self._ask("compare the drives of A and B")
        self.assertEqual(events[-1]["type"], "needs_choice")
        self.assertEqual([c["id"] for c in events[-1]["candidates"]], ["bsk:asset:A", "bsk:asset:B"])
        self.assertEqual(post.call_count, 0)
        unstreamed = self.client.post("/api/chat/", {"message": "compare the drives of A and B",
                                                     "model": "dgx-qwen38-27b-fp8", "use_rag": True}, format="json")
        self.assertEqual(unstreamed.status_code, 409)
        self.assertEqual(unstreamed.data["code"], "needs_choice")

    @patch("chat.views.requests.post")
    def test_the_agent_never_blocks_a_question(self, post):
        post.side_effect = [RuntimeError("agent down"), _streamed(_openai("An answer."))]
        with patch("chat.views.search_v2") as search:
            search.return_value = MagicMock(chunks=[], reranker="ok")
            events = self._ask("hello there")
        # The agent failed: the question was searched as written; nothing found → where it looked.
        self.assertIn("“hello there”", "".join(e["text"] for e in events if e["type"] == "token"))


@override_settings(CHAT_HISTORIAN_WINDOWS=False)
class PromptTests(TestCase):
    """P24-2: one system prompt — the standard editable part (Settings → Chat), a conversation's own, fixed rules."""

    def setUp(self):
        self.client = APIClient()
        patcher = patch("chat.views.generate_title", return_value=("Title", "llm"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _system(self, post, **body):
        post.return_value = _streamed(_openai("ok"))
        r = self.client.post("/api/chat/", {"message": "hello", "model": "dgx-qwen38-27b-fp8", "use_rag": False, **body},
                             format="json")
        return r, post.call_args.kwargs["json"]["messages"][0]["content"]

    @patch("chat.views.requests.post")
    def test_standard_then_settings_then_the_conversations_own(self, post):
        from chat.prompt import DEFAULT_EDITABLE
        r, system = self._system(post)
        self.assertEqual(system, DEFAULT_EDITABLE)
        self.client.put("/api/chat/settings/", {"system_prompt": "You are the plant's assistant. Be brief."}, format="json")
        r, system = self._system(post, conversation_id=r.data["id"])
        self.assertEqual(system, "You are the plant's assistant. Be brief.")
        patched = self.client.patch(f"/api/conversations/{r.data['id']}/", {"system_prompt": "Answer in Spanish."},
                                    format="json")
        self.assertEqual(patched.data["system_prompt"], "Answer in Spanish.")
        r, system = self._system(post, conversation_id=r.data["id"])
        self.assertEqual(system, "Answer in Spanish.")
        self.client.patch(f"/api/conversations/{r.data['id']}/", {"system_prompt": None}, format="json")
        r, system = self._system(post, conversation_id=r.data["id"])
        self.assertEqual(system, "You are the plant's assistant. Be brief.")
