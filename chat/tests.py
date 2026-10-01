from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from .models import Conversation
from .views import (
    build_chat_messages,
    format_rag_footer,
    strip_rag_footer,
)

SYSTEM = "sys"


def build(history, user="now", max_messages=20, max_chars=10_000):
    return build_chat_messages(history, user, SYSTEM, max_messages, max_chars)


class BuildChatMessagesTests(SimpleTestCase):
    def test_first_message_is_system_plus_user(self):
        self.assertEqual(
            build([]),
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "now"}],
        )

    def test_history_in_order_between_system_and_user(self):
        msgs = build([("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")])
        self.assertEqual(
            [(m["role"], m["content"]) for m in msgs],
            [("system", SYSTEM), ("user", "u1"), ("assistant", "a1"),
             ("user", "u2"), ("assistant", "a2"), ("user", "now")],
        )

    def test_max_messages_keeps_most_recent(self):
        history = [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")]
        msgs = build(history, max_messages=2)
        self.assertEqual([m["content"] for m in msgs[1:-1]], ["u2", "a2"])

    def test_char_budget_drops_oldest_whole_messages(self):
        history = [("user", "x" * 50), ("assistant", "y" * 50), ("user", "u2"), ("assistant", "a2")]
        # budget after system ("sys") + user ("now") = 20 -> only the last turn fits
        msgs = build(history, max_chars=26)
        self.assertEqual([m["content"] for m in msgs[1:-1]], ["u2", "a2"])

    def test_system_prompt_counts_against_budget(self):
        history = [("user", "u1"), ("assistant", "a1")]
        msgs = build_chat_messages(history, "now", "s" * 100, 20, 100)
        self.assertEqual(len(msgs), 2)  # no room left for history

    def test_window_never_starts_with_assistant(self):
        history = [("user", "x" * 50), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")]
        msgs = build(history, max_messages=3)
        self.assertEqual([m["role"] for m in msgs[1:-1]], ["user", "assistant"])

    def test_skips_unanswered_user_messages(self):
        history = [("user", "failed"), ("user", "u1"), ("assistant", "a1"), ("user", "failed-last")]
        msgs = build(history)
        self.assertEqual([m["content"] for m in msgs[1:-1]], ["u1", "a1"])

    def test_strips_rag_footer_from_assistant_history(self):
        reply = "answer" + format_rag_footer(["a.txt", "b.txt"])
        msgs = build([("user", "q"), ("assistant", reply)])
        self.assertEqual(msgs[2]["content"], "answer")


class RagFooterTests(SimpleTestCase):
    def test_round_trip_and_truncation(self):
        footer = format_rag_footer(["a", "b", "a", "c", "d"])
        self.assertTrue(footer.endswith("a, b, c, +1 more"))
        self.assertEqual(strip_rag_footer("text" + footer), "text")
        self.assertEqual(strip_rag_footer("no footer"), "no footer")


def _ok_response(content):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status.return_value = None
    return resp


class ChatViewHistoryTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _chat(self, message, conversation_id=None, model="dgx-qwen38-27b-fp8"):
        return self.client.post(
            "/api/chat/",
            {"conversation_id": conversation_id, "message": message, "model": model, "use_rag": False},
            format="json",
        )

    @patch("chat.views.requests.post")
    def test_second_turn_sends_first_exchange(self, mock_post):
        mock_post.side_effect = [_ok_response("Hi Angel"), _ok_response("Your name is Angel")]

        with self.assertLogs("chat", level="INFO") as logs:
            r1 = self._chat("My name is Angel")
            self.assertEqual(r1.status_code, 200)
            r2 = self._chat("What is my name?", conversation_id=r1.data["id"])
            self.assertEqual(r2.status_code, 200)
        self.assertIn("history_sent=2/2", logs.output[-1])

        sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual(
            [(m["role"], m["content"]) for m in sent[1:]],
            [("user", "My name is Angel"), ("assistant", "Hi Angel"), ("user", "What is my name?")],
        )
        self.assertEqual(sent[0]["role"], "system")

    @patch("chat.views.requests.post")
    def test_backend_failure_returns_502_and_next_turn_skips_orphan(self, mock_post):
        mock_post.side_effect = [RuntimeError("boom"), _ok_response("ok")]

        with self.assertLogs("chat", level="INFO") as logs:
            r1 = self._chat("lost message")
            self.assertEqual(r1.status_code, 502)
            conv = Conversation.objects.get()
            self.assertEqual(conv.messages.count(), 1)  # user message saved, no reply

            r2 = self._chat("retry", conversation_id=str(conv.id))
            self.assertEqual(r2.status_code, 200)
        self.assertIn("chat failed", logs.output[0])
        sent = mock_post.call_args_list[1].kwargs["json"]["messages"]
        self.assertEqual([m["content"] for m in sent[1:]], ["retry"])
