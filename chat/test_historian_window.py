"""Historian windows in the asset chat (P18a): data questions, their windows, the block, the view."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.test import SimpleTestCase, TestCase, override_settings

from chat import historian_window as hw
from rest_framework.test import APIClient

from chat.retrieval import SearchResult
from chat.test_asset_chat import ASSET, CONTEXT, SNAPSHOT
from chat.tests import _ok_response
from monitor.plant_calendar import Calendar

LA = ZoneInfo("America/Los_Angeles")
CHI = ZoneInfo("America/Chicago")
# The PLC's shifts (Angel, 2026-10-17): every day, A 00-08, B 08-16, C 16-24, Los Angeles.
PLC = Calendar.from_dict("bsk:plant:demo", {
    "time_zone": "America/Los_Angeles", "working_days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    "shifts": [{"name": "A", "start": "00:00", "end": "08:00"}, {"name": "B", "start": "08:00", "end": "16:00"},
               {"name": "C", "start": "16:00", "end": "00:00"}]})
OFFICE = Calendar.from_dict("bsk:plant:demo", {
    "time_zone": "America/Chicago", "working_days": ["mon", "tue", "wed", "thu", "fri"],
    "shifts": [{"name": "Day", "start": "07:00", "end": "15:00"}]})


def la(*args) -> datetime:
    return datetime(*args, tzinfo=LA).astimezone(timezone.utc)


NOW = la(2026, 10, 17, 15, 37)          # Saturday afternoon, shift B


class DataQuestionTests(SimpleTestCase):
    def test_data_questions(self):
        for q in ("what was the OEE the last 24 hrs", "downtime yesterday?", "how did shift B do today",
                  "which alarms in the last 8 h", "OEE?", "throughput this week", "was die pressure out of range today",
                  "how many times did F30004 start today", "how is it going this shift",
                  "how much time did changeovers take yesterday", "minutes in cool down today"):
            self.assertTrue(hw.is_data_question(q), q)

    def test_not_data_questions(self):
        for q in ("what does alarm F30004 mean", "which alarms does the cutter drive have", "alarm F30004?",
                  "how do I reset the alarm from yesterday", "what is the replacement interval of the bearings",
                  "which motors are on the extruder", "what is the cool down procedure"):
            self.assertFalse(hw.is_data_question(q), q)


    def test_the_data_set_is_all_data_questions(self):
        from chat.management.commands.chat_eval import DEFAULT_SET
        for q in json.loads((DEFAULT_SET.parent / "extr01_data.json").read_text())["questions"]:
            self.assertTrue(hw.is_data_question(q["question"]), q["id"])


class ParseTests(SimpleTestCase):
    def window(self, q, cal=PLC, now=NOW):
        return hw.parse(q, now, cal)

    def test_durations(self):
        w = self.window("OEE the last 24 hrs")
        self.assertEqual((w.label, w.start, w.end), ("last 24 hrs", NOW - timedelta(hours=24), NOW))
        self.assertEqual(self.window("downtime in the past 90 min").minutes, 90)
        self.assertEqual(self.window("alarms over the last 2 days").minutes, 2880)
        self.assertEqual(self.window("OEE in the last hour").minutes, 60)
        self.assertEqual(self.window("OEE over the last 400 days").minutes, hw.MAX_DAYS * 1440)

    def test_days(self):
        w = self.window("downtime yesterday")
        self.assertEqual((w.start, w.end), (la(2026, 10, 16, 0, 0), la(2026, 10, 17, 0, 0)))
        self.assertIn("Fri 2026-10-16", w.label)
        w = self.window("OEE today")
        self.assertEqual((w.start, w.end), (la(2026, 10, 17, 0, 0), NOW))

    def test_shifts(self):
        w = self.window("how is it going this shift")
        self.assertEqual((w.start, w.end), (la(2026, 10, 17, 8, 0), NOW))
        self.assertTrue(w.notes)                                  # still running
        w = self.window("OEE last shift")
        self.assertEqual((w.start, w.end), (la(2026, 10, 17, 0, 0), la(2026, 10, 17, 8, 0)))
        w = self.window("how did shift C do yesterday")        # "did shift": not a shift name
        self.assertEqual((w.start, w.end), (la(2026, 10, 16, 16, 0), la(2026, 10, 17, 0, 0)))
        w = self.window("downtime on shift A")                   # the last A that ended
        self.assertEqual(w.start, la(2026, 10, 17, 0, 0))

    def test_last_working_day_and_weeks(self):
        monday = datetime(2026, 10, 19, 9, 0, tzinfo=CHI).astimezone(timezone.utc)
        w = self.window("OEE on the last working day", OFFICE, monday)
        self.assertEqual(w.start, datetime(2026, 10, 16, 0, 0, tzinfo=CHI).astimezone(timezone.utc))   # Friday
        w = self.window("throughput this week")
        self.assertEqual(w.start, la(2026, 10, 12, 0, 0))
        w = self.window("throughput last week")
        self.assertEqual((w.start, w.end), (la(2026, 10, 5, 0, 0), la(2026, 10, 12, 0, 0)))

    def test_no_time_is_homes_window(self):
        w = self.window("what is the OEE?")
        self.assertEqual((w.label, w.minutes), ("last 8 h", 480))
        self.assertIn("no time", w.notes[0])

    def test_without_a_calendar(self):
        w = self.window("downtime yesterday", None)
        self.assertEqual(w.start, datetime(2026, 10, 16, tzinfo=timezone.utc))
        self.assertIn("UTC", w.notes[0])
        w = self.window("OEE this shift", None)
        self.assertEqual((w.label, w.minutes), ("last 8 h", 480))
        self.assertIn("no plant calendar", w.notes[0])


SIGNALS = [{"key": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 80, "high": 100}]


def rows(start: datetime, minutes: int, state: str, **values) -> list[dict]:
    """20 s rows."""
    return [{"ts": start + timedelta(seconds=20 * (i + 1)), "machine_state": state, **values} for i in range(minutes * 3)]


def two_hours() -> tuple[hw.Window, list[dict]]:
    start = la(2026, 10, 17, 12, 0)
    data = (rows(start, 50, "RUNNING", oee_pct=80, availability_pct=90, performance_pct=95, quality_pct=93.6,
                 throughput_actual_kg_hr=9000, throughput_target_kg_hr=9400, die_pressure_bar=90, downtime_min=0)
            + rows(start + timedelta(minutes=50), 10, "RUNNING", oee_pct=80, die_pressure_bar=120, downtime_min=0)
            + rows(start + timedelta(minutes=60), 30, "STOPPED", downtime_min=1, alarm_code="HI_DIE_PRESSURE")
            + rows(start + timedelta(minutes=90), 30, "FLUSH", downtime_min=0))
    return hw.Window("last 2 h", start, start + timedelta(hours=2)), data


class BlockTests(SimpleTestCase):
    def test_totals(self):
        w, data = two_hours()
        text = hw.block("Extruder EXTR01", w, data, SIGNALS, 20, "historian_live", False, {"live": True}, PLC, 2500)
        self.assertIn("Sat 2026-10-17 12:00 – Sat 2026-10-17 14:00 (America/Los_Angeles, plant calendar)", text)
        self.assertIn("running 60 min, not running 60 min (downtime 30 min, other stops 30 min), no data 0 min", text)
        self.assertIn("OEE 80.0 %", text)
        self.assertIn("Throughput while running 9000 kg/h (target 9400 kg/h)", text)
        self.assertIn("HI_DIE_PRESSURE × 1", text)
        self.assertIn("Die pressure 10 min", text)
        self.assertIn("By hour", text)
        self.assertNotIn("Outside working hours", text)          # every day is a working day

    def test_no_data_minutes_and_buckets(self):
        w, data = two_hours()
        text = hw.block("Extruder EXTR01", w, data[:180], SIGNALS, 20, "historian_live", False, {"live": True}, None, 2500)
        self.assertIn("no data 60 min", text)
        self.assertIn("UTC", text)
        text = hw.block("Extruder EXTR01", w, data, SIGNALS, 300, "historian_live", True, {"live": True}, PLC, 2500)
        self.assertIn("5-minute means, so minutes are approximate", text)

    def test_outside_working_hours(self):
        w, data = two_hours()
        sat = Calendar.from_dict("bsk:plant:demo", {**{"time_zone": "America/Los_Angeles", "working_days": ["mon"]},
                                                    "shifts": [{"name": "A", "start": "00:00", "end": "08:00"}]})
        text = hw.block("Extruder EXTR01", w, data, SIGNALS, 20, "historian_live", False, {"live": True}, sat, 2500)
        self.assertIn("Outside working hours (plant calendar): 120 min, of which running 60 min", text)

    def test_a_small_budget_drops_the_breakdowns_first(self):
        w, data = two_hours()
        full = hw.block("Extruder EXTR01", w, data, SIGNALS, 20, "historian_live", False, {"live": True}, PLC, 2500)
        budget = full.index("\n- By hour") + 10
        text = hw.block("Extruder EXTR01", w, data, SIGNALS, 20, "historian_live", False, {"live": True}, PLC, budget)
        self.assertNotIn("By hour", text)
        self.assertIn("running 60 min", text)
        self.assertLessEqual(len(text), budget)


class ComputeTests(SimpleTestCase):
    def test_not_a_data_question_reads_nothing(self):
        with patch("chat.historian_window.agent.clock") as clock:
            self.assertIsNone(hw.compute(ASSET, "Extruder EXTR01", "what does alarm F30004 mean", 2500))
        clock.assert_not_called()

    def test_a_window(self):
        w, data = two_hours()
        with patch("chat.historian_window.agent.clock", return_value={"now": w.end, "live": True, "replay": False}), \
                patch("chat.historian_window.plant_calendar.for_asset", return_value=PLC), \
                patch("chat.historian_window.read", return_value=(data, SIGNALS + SIGNALS, 20, "historian_live", False)):
            r = hw.compute(ASSET, "Extruder EXTR01", "downtime in the last 2 h", 2500)
        self.assertEqual(r.citation["source"], "Historian — last 2 h")
        self.assertEqual(r.text.count("Die pressure 10 min"), 1)     # a signal listed twice is counted once
        self.assertIn("rows=360", r.log)

    def test_no_rows(self):
        with patch("chat.historian_window.agent.clock", return_value={"now": NOW, "live": True, "replay": False}), \
                patch("chat.historian_window.plant_calendar.for_asset", return_value=None), \
                patch("chat.historian_window.read", return_value=([], SIGNALS, 20, "historian_live", False)):
            r = hw.compute(ASSET, "Extruder EXTR01", "OEE yesterday", 2500)
        self.assertIn("no rows in this window", r.text)


WINDOW = hw.Result("HISTORIAN WINDOW for Extruder EXTR01, last 24 h: …\n- Of 1,440 min: running 909 min",
                   {"kind": "historian", "source": "Historian — last 24 h", "asset_id": ASSET, "section_path": [],
                    "page_start": None, "page_end": None, "score": None, "snippet": "", "facts": []}, "window=test")


@patch("chat.views.generate_title", return_value=("Title", "llm"))
@patch("chat.views.search_v2", return_value=SearchResult(chunks=[], candidates=0, reranker="ok", latency_ms=1))
@patch("chat.views.requests.post", return_value=_ok_response("OEE was 87.8 % [H]."))
class ChatViewTests(TestCase):
    """The view puts the block in the prompt and its source under the answer (the setting on)."""

    def setUp(self):
        self.client = APIClient()
        for target, value in (("context_graph.services.get_asset_context", CONTEXT),
                              ("chat.asset_context.historian_snapshot", SNAPSHOT),
                              ("context_graph.services.node_detail", CONTEXT["asset"])):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def chat(self, message, **extra):
        return self.client.post("/api/chat/", {"message": message, "model": "dgx-qwen38-27b-fp8", "use_rag": False, **extra},
                                format="json")

    @override_settings(CHAT_HISTORIAN_WINDOWS=True)
    def test_window_in_prompt_and_sources(self, post, search, title):
        with patch("chat.views.historian_window.compute", return_value=WINDOW) as compute:
            r = self.chat("what was the OEE the last 24 hrs", asset_id=ASSET)
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("HISTORIAN WINDOW for Extruder EXTR01", system)
        self.assertIn("cite it as [H]", system)
        self.assertIn("never give a live value or a latest sample", system)
        self.assertLess(system.index("ASSET FACTS"), system.index("HISTORIAN WINDOW"))
        self.assertIn("Historian — last 24 h", [s["source"] for s in r.data["messages"][-1]["sources"]])
        self.assertEqual(compute.call_args.args[:3], (ASSET, "Extruder EXTR01", "what was the OEE the last 24 hrs"))

    @override_settings(CHAT_HISTORIAN_WINDOWS=True)
    def test_a_failing_window_still_answers(self, post, search, title):
        with patch("chat.views.historian_window.compute", side_effect=RuntimeError("historian down")):
            r = self.chat("OEE yesterday", asset_id=ASSET)
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("HISTORIAN WINDOW", post.call_args.kwargs["json"]["messages"][0]["content"])

    @override_settings(CHAT_HISTORIAN_WINDOWS=False)
    def test_off(self, post, search, title):
        with patch("chat.views.historian_window.compute") as compute:
            self.chat("OEE yesterday", asset_id=ASSET)
        compute.assert_not_called()
