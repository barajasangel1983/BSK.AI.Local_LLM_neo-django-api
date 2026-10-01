import io
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from .models import Conversation, Message
from .titles import DEFAULT_TITLE, clean_title, fallback_title, generate_title


def _llm(content):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class CleanTitleTests(SimpleTestCase):
    def test_cleans_model_output(self):
        self.assertEqual(clean_title('"Multi-Head Attention Explained."'), "Multi-Head Attention Explained")
        self.assertEqual(clean_title("Title: **OEE on EXTR01**\nextra line"), "OEE on EXTR01")
        self.assertEqual(clean_title("<think>hmm, the user asks…</think>\n\nTransformer Positional Encoding"),
                         "Transformer Positional Encoding")

    def test_unusable_output_is_empty(self):
        self.assertEqual(clean_title(""), "")
        self.assertEqual(clean_title("<think>still reasoning when max_tokens hit"), "")
        self.assertEqual(clean_title('  "" '), "")

    def test_long_output_is_truncated_on_a_word(self):
        title = clean_title("word " * 30)
        self.assertLessEqual(len(title), 61)
        self.assertTrue(title.endswith("…"))


class FallbackTitleTests(SimpleTestCase):
    def test_uses_start_of_message(self):
        self.assertEqual(fallback_title("  What is   the OEE?\n"), "What is the OEE?")
        long = fallback_title("In the Attention paper, why did the authors use multi-head attention?")
        self.assertEqual(long, "In the Attention paper, why did the authors use…")

    def test_empty_message(self):
        self.assertEqual(fallback_title("   "), DEFAULT_TITLE)


@patch("chat.titles.requests.post")
class GenerateTitleTests(SimpleTestCase):
    def test_llm_title(self, mock_post):
        mock_post.return_value = _llm("Positional Encoding in Transformers")
        self.assertEqual(generate_title("how is positional encoding computed?", "Using sines..."),
                         ("Positional Encoding in Transformers", "llm"))
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("how is positional encoding computed?", body["messages"][1]["content"])
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 6)

    def test_falls_back_when_dgx_fails(self, mock_post):
        mock_post.side_effect = TimeoutError("stalled")
        with self.assertLogs("chat", level="WARNING"):
            self.assertEqual(generate_title("What is the OEE?", "81%"), ("What is the OEE?", "fallback"))

    def test_falls_back_on_empty_output(self, mock_post):
        mock_post.return_value = _llm("<think>never finished")
        self.assertEqual(generate_title("What is the OEE?", "81%"), ("What is the OEE?", "fallback"))


@patch("chat.views.requests.post", return_value=_llm("answer"))
@patch("chat.views.generate_title", return_value=("Attention Heads", "llm"))
class AutoTitleInChatTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _chat(self, message, conversation_id=None):
        with self.assertLogs("chat", level="INFO"):
            return self.client.post("/api/chat/", {"conversation_id": conversation_id, "message": message,
                                                   "model": "dgx-qwen38-27b-fp8", "use_rag": False}, format="json")

    def test_first_exchange_is_titled_once(self, mock_title, mock_post):
        r1 = self._chat("why multiple heads?")
        self.assertEqual(r1.data["title"], "Attention Heads")
        mock_title.assert_called_once_with("why multiple heads?", "answer")

        r2 = self._chat("and positional encoding?", conversation_id=r1.data["id"])
        self.assertEqual(r2.data["title"], "Attention Heads")
        mock_title.assert_called_once()  # not regenerated on later turns

    def test_renamed_conversation_is_never_retitled(self, mock_title, mock_post):
        from .views import get_current_user
        conv = Conversation.objects.create(owner=get_current_user(), title="My notes")
        r = self._chat("hello", conversation_id=str(conv.id))
        self.assertEqual(r.data["title"], "My notes")
        mock_title.assert_not_called()

    def test_titled_after_a_failed_first_attempt(self, mock_title, mock_post):
        mock_post.side_effect = [RuntimeError("down"), _llm("answer")]
        with self.assertLogs("chat", level="INFO"):
            r1 = self.client.post("/api/chat/", {"conversation_id": None, "message": "first try",
                                                 "model": "dgx-qwen38-27b-fp8", "use_rag": False}, format="json")
        self.assertEqual(r1.status_code, 502)
        conv = Conversation.objects.get()
        self.assertEqual(conv.title, DEFAULT_TITLE)

        r2 = self._chat("second try", conversation_id=str(conv.id))
        self.assertEqual(r2.data["title"], "Attention Heads")


class RenameConversationTests(TestCase):
    def setUp(self):
        from .views import get_current_user
        self.client = APIClient()
        self.conv = Conversation.objects.create(owner=get_current_user(), title=DEFAULT_TITLE)

    def _patch(self, title, conv_id=None):
        return self.client.patch(f"/api/conversations/{conv_id or self.conv.id}/", {"title": title}, format="json")

    def test_renames_and_normalizes_whitespace(self):
        r = self._patch("  Shift   report\nEXTR01 ")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["title"], "Shift report EXTR01")
        self.conv.refresh_from_db()
        self.assertEqual(self.conv.title, "Shift report EXTR01")

    def test_rejects_empty_title(self):
        self.assertEqual(self._patch("   ").status_code, 400)

    def test_caps_length(self):
        self.assertEqual(len(self._patch("x" * 300).data["title"]), 120)

    def test_unknown_conversation_is_404(self):
        self.assertEqual(self._patch("t", conv_id="00000000-0000-4000-8000-000000000000").status_code, 404)


@patch("chat.management.commands.backfill_conversation_titles.generate_title", return_value=("Generated", "llm"))
class BackfillTitlesTests(TestCase):
    def setUp(self):
        self.untitled = Conversation.objects.create(title=DEFAULT_TITLE)
        Message.objects.create(conversation=self.untitled, role="user", content="what is OEE?")
        Message.objects.create(conversation=self.untitled, role="assistant", content="Overall equipment effectiveness")
        self.only_user = Conversation.objects.create(title="")
        Message.objects.create(conversation=self.only_user, role="user", content="Shift report for line 2 please")
        self.empty = Conversation.objects.create(title=DEFAULT_TITLE)
        self.renamed = Conversation.objects.create(title="Keep me")
        Message.objects.create(conversation=self.renamed, role="user", content="x")

    def _titles(self):
        return {c.id: c.title for c in Conversation.objects.all()}

    def test_backfills_untitled_conversations(self, mock_generate):
        call_command("backfill_conversation_titles", stdout=io.StringIO())
        titles = self._titles()
        self.assertEqual(titles[self.untitled.id], "Generated")
        self.assertEqual(titles[self.only_user.id], "Shift report for line 2 please")  # no reply -> fallback
        self.assertEqual(titles[self.empty.id], DEFAULT_TITLE)  # no messages -> skipped
        self.assertEqual(titles[self.renamed.id], "Keep me")
        mock_generate.assert_called_once_with("what is OEE?", "Overall equipment effectiveness")

    def test_dry_run_saves_nothing(self, mock_generate):
        before = self._titles()
        out = io.StringIO()
        call_command("backfill_conversation_titles", "--dry-run", stdout=out)
        self.assertEqual(self._titles(), before)
        self.assertIn("would title 2 conversation(s), skipped 1", out.getvalue())
