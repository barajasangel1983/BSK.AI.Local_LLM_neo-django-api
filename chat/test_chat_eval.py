"""chat_eval (P16a): the reference set is scored by the facts an answer states, and leaves no conversation."""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from rest_framework.response import Response

from chat.management.commands.chat_eval import DEFAULT_SET, normalize, score
from chat.models import Conversation


class ScoreTests(TestCase):
    def test_numbers_and_spellings(self):
        self.assertEqual(normalize("Every 5 600 h, or 200,000 double strokes"), "every 5600 h, or 200000 double strokes")
        self.assertEqual(normalize("95.20 and 99.10"), "95.20 and 99.10")          # decimals are not thousands
        self.assertEqual(score("Regrease every 5,600 h with 70 g per bearing.", [["5600"], ["70 g", "70g"]]),
                         ("✓", [True, True]))
        self.assertEqual(score("About 5600 hours.", [["5600"], ["70 g"]]), ("◐", [True, False]))
        self.assertEqual(score("Not in the context.", [["5600"]]), ("✗", [False]))

    def test_the_reference_set_is_complete(self):
        ref = json.loads(DEFAULT_SET.read_text())
        self.assertEqual(len(ref["questions"]), 10)
        for q in ref["questions"]:
            self.assertTrue(q["question"] and q["answer"] and q["source"] and q["expect"], q["id"])
            self.assertEqual(score(q["answer"], q["expect"])[0], "✓", q["id"])   # the key's own answer passes


class CommandTests(TestCase):
    def test_runs_scores_and_deletes_its_conversations(self):
        def chat_view(request):
            conversation = Conversation.objects.create(title="eval")
            body = json.loads(request.body)
            content = "Check parameter 95.20, bit 14; the auxiliary code identifies the fan with missing feedback." \
                if "A581" in body["message"] else "The context does not say."
            return Response({"id": str(conversation.id), "messages": [
                {"role": "user", "content": body["message"]},
                {"role": "assistant", "content": content,
                 "sources": [{"kind": "graph", "snippet": "12 of 536 facts about Extruder EXTR01"}]}]})

        out = Path(tempfile.mkdtemp()) / "run.json"
        stdout = StringIO()
        with patch("chat.views.chat_view", side_effect=chat_view):
            call_command("chat_eval", "--model", "dgx-qwen38-27b-fp8", "--only", "Q6,Q7", "--out", str(out), stdout=stdout)
        run = json.loads(out.read_text())
        self.assertEqual([(r["id"], r["mark"]) for r in run["results"]], [("Q6", "✓"), ("Q7", "✗")])
        self.assertEqual(run["results"][0]["graph_facts"], "12 of 536 facts about Extruder EXTR01")
        self.assertIn("1 ✓, 0 ◐, 1 ✗ of 2", stdout.getvalue())
        self.assertFalse(Conversation.objects.exists())
