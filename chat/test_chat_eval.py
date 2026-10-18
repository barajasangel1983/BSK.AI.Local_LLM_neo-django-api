"""chat_eval (P16a): the reference set is scored by the facts an answer states, and leaves no conversation."""

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from rest_framework.response import Response

from chat.management.commands.chat_eval import DEFAULT_SET, normalize, score, score_data, states_measure, wrong
from chat.models import Conversation


class ScoreTests(TestCase):
    def test_numbers_and_spellings(self):
        self.assertEqual(normalize("Every 5 600 h, or 200,000 double strokes"), "every 5600 h, or 200000 double strokes")
        self.assertEqual(normalize("95.20 and 99.10"), "95.20 and 99.10")          # decimals are not thousands
        self.assertEqual(score("Regrease every 5,600 h with 70 g per bearing.", [["5600"], ["70 g", "70g"]]),
                         ("✓", [True, True]))
        self.assertEqual(score("About 5600 hours.", [["5600"], ["70 g"]]), ("◐", [True, False]))
        self.assertEqual(score("Not in the context.", [["5600"]]), ("✗", [False]))

    def test_abstain_and_forbid(self):
        forbid = [r"\d+\s*(n\s*·?\s*m|nm)\b"]
        self.assertEqual(score("The die plate torque is not documented in the manuals.", [["x"]], forbid, True)[0], "✓")
        self.assertEqual(score("Tighten them to 120 Nm.", [["x"]], forbid, True)[0], "✗")
        self.assertEqual(score("It's not in the documents; typically 120 N·m.", [["x"]], forbid, True)[0], "✗")
        self.assertEqual(score("The environment matters.", [["x"]], forbid, True)[0], "✗")   # no abstain, "nm" in a word
        self.assertEqual(score("The IP rating is not explicitly stated in the documents.", [["x"]], [], True)[0], "✓")
        self.assertEqual(wrong("Frame R9.", [r"\br9\b"]), [r"\br9\b"])
        self.assertEqual(score("Frame R9; fans at 9 and 18 years.", [["r11"], ["9 and 18"]], [r"\br9\b"])[0], "✗")
        self.assertEqual(score("315 °C; the K series takes 538 °C.", [["315"]], [r"\b538\b"])[0], "✓")   # context
        self.assertEqual(score("538 °C.", [["315"]], [r"\b538\b"])[0], "✗")

    def test_the_reference_sets_are_complete(self):
        for name, count in (("extr01_reference.json", 10), ("extr01_hard.json", 30)):
            ref = json.loads((DEFAULT_SET.parent / name).read_text())
            self.assertEqual(len(ref["questions"]), count, name)
            self.assertEqual(len({q["id"] for q in ref["questions"]}), count, name)
            for q in ref["questions"]:
                self.assertTrue(q["question"] and q["answer"] and q["source"] and q["expect"], q["id"])
                # the key's own answer passes
                self.assertEqual(score(q["answer"], q["expect"], q.get("forbid") or [], q.get("abstain", False))[0],
                                 "✓", q["id"])


class DataScoreTests(TestCase):
    def test_numbers_within_tolerance(self):
        self.assertTrue(states_measure("OEE was 87.8 % [H].", "oee", [87.6]))
        self.assertTrue(states_measure("OEE was 88 %.", "oee", [87.6]))
        self.assertFalse(states_measure("OEE was 86.9 %.", "oee", [87.6]))
        self.assertTrue(states_measure("It ran 1,035 minutes.", "running_min", [1035.3]))
        self.assertTrue(states_measure("It ran about 17.3 hours.", "running_min", [1035.3]))      # hours for minutes
        self.assertFalse(states_measure("It ran 17.3 minutes.", "running_min", [1035.3]))
        self.assertTrue(states_measure("9,151 kg/h", "throughput", [9135.4]))
        self.assertTrue(states_measure("Downtime 54 min.", "downtime_min", [34.7, 54.3]))       # before or after
        self.assertFalse(states_measure("Downtime 54 min.", "downtime_min", [None]))

    def test_alarms_and_ranges(self):
        top = [["LO_MOISTURE", "HI_MOISTURE", "OUT_OF_SPEC_DENSITY"]]
        self.assertTrue(states_measure("LO_MOISTURE (19), HI MOISTURE (10) and out-of-spec density (7).", "alarm_top", top))
        self.assertFalse(states_measure("Mostly LO_MOISTURE.", "alarm_top", top))
        self.assertTrue(states_measure("Feeder rate and motor power were out of range.", "out_of_range", [["Feeder rate", "Motor power"]]))
        self.assertTrue(states_measure("No signal stayed out of its normal range today [H].", "out_of_range", [[]]))
        self.assertTrue(states_measure("All signals stayed within their normal ranges.", "out_of_range", [[]]))
        self.assertFalse(states_measure("Die pressure was high.", "out_of_range", [[]]))
        self.assertEqual(score_data("OEE 87.8 %, running 899 min.", ["oee", "running_min"], {"oee": [87.8], "running_min": [600]}),
                         ("◐", [True, False]))


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

    def test_a_data_question_is_checked_against_the_historian(self):
        from chat.historian_window import Window
        from datetime import datetime, timezone
        now = datetime(2026, 10, 18, tzinfo=timezone.utc)
        stats = {"oee": 87.8, "running_min": 899, "states": {}, "alarms": {}, "out_of_range": []}

        def chat_view(request):
            conversation = Conversation.objects.create(title="eval")
            return Response({"id": str(conversation.id), "messages": [
                {"role": "assistant", "content": "OEE while running was 87.8 % over 899 min [H].",
                 "sources": [{"kind": "historian", "source": "Historian — last 24 hours"}]}]})

        out = Path(tempfile.mkdtemp()) / "run.json"
        with patch("chat.views.chat_view", side_effect=chat_view), \
                patch("chat.historian_window.stats", return_value=(Window("last 24 hours", now, now), stats)):
            call_command("chat_eval", "--model", "ollama-qwen3-8b", "--set", str(DEFAULT_SET.parent / "extr01_data.json"),
                         "--only", "D1", "--out", str(out), stdout=StringIO())
        r = json.loads(out.read_text())["results"][0]
        self.assertEqual((r["mark"], r["found"], r["window_ok"]), ("✓", [True, True], True))
        self.assertEqual(r["values"], {"oee": [87.8, 87.8], "running_min": [899, 899]})
        self.assertFalse(Conversation.objects.exists())
