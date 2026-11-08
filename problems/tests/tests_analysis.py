"""P29e: the data summary — which signals moved first, the groups, the baseline, the report. The historian is a fake."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import TestCase

from problems import analysis, context, report, service
from problems.models import Ticket

ASSET = "bsk:asset:EXTR01"
START = datetime(2026, 11, 7, 16, 0, tzinfo=timezone.utc)
SIGNALS = [{"key": "machine_state", "name": "Machine state", "unit": "", "low": None, "high": None},
           {"key": "scrap_output_kg", "name": "Scrap output", "unit": "kg", "low": None, "high": None},
           {"key": "feeder_rate_actual_kg_hr", "name": "Feeder rate", "unit": "kg/h", "low": 7959, "high": 10391},
           {"key": "throughput_actual_kg_hr", "name": "Throughput", "unit": "kg/h", "low": 7972, "high": 10385},
           {"key": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 84.3, "high": 99.2},
           {"key": "steam_flow_kg_hr", "name": "Steam flow", "unit": "kg/h", "low": 96.6, "high": 132.9},
           {"key": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 84.3, "high": 99.2}]   # listed twice


def wiggle(i: int, size: float) -> float:
    return size * ((i * 7919) % 13 - 6) / 6           # deterministic noise


def row(t: datetime, i: int, state: str = "RUNNING") -> dict:
    m = (t - START).total_seconds() / 60
    feeder = 9500 + wiggle(i, 20) - (800 if m >= -10 else 0)          # falls 10 min before the start
    return {"machine_state": state, "scrap_output_kg": 2.5 + (2.0 if m >= 0 else 0) + wiggle(i, 0.05),
            "feeder_rate_actual_kg_hr": feeder,
            "throughput_actual_kg_hr": 9530 + wiggle(i + 3, 20) - (800 if m >= -9 else 0),   # a minute later: same group
            "die_pressure_bar": 92 + wiggle(i + 5, 0.3) + (4 if m >= 5 else 0),   # later, on its own
            "steam_flow_kg_hr": 115 + wiggle(i + 1, 1.5)}                     # never moves


class Fake:
    def __init__(self, test, state=lambda t: "RUNNING", fail=False):
        self.state, self.fail = state, fail
        p = patch("context_service.service.signal_history", side_effect=self.history)
        p.start()
        test.addCleanup(p.stop)

    def history(self, asset_id, start, end, signals=None, bucket_minutes=None):
        if self.fail:
            raise RuntimeError("historian down")
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
        n = int((end - start).total_seconds() // 20)
        stamps = [start + timedelta(seconds=20 * (k + 1)) for k in range(n)]
        rows = [row(t, int(t.timestamp() // 20), self.state(t)) for t in stamps]
        keys = list(dict.fromkeys(s["key"] for s in SIGNALS))
        return {"source": "historian_live", "period_seconds": 20, "signals": SIGNALS,
                "ts": [t.isoformat() for t in stamps], "series": {k: [r[k] for r in rows] for k in keys}}


TRIGGER = {"key": "scrap_output_kg", "name": "Scrap output", "unit": "kg",
           "limits": {"lcl": 1.2, "cl": 2.4, "ucl": 3.5, "lsl": 0.5, "usl": 4.0}}


class SummaryTests(TestCase):
    def build(self, **kw):
        return analysis.build(ASSET, START, START + timedelta(hours=1, minutes=20), kw.pop("trigger", TRIGGER),
                              now=START + timedelta(hours=3), **kw)

    def test_the_signal_that_moved_first_leads_its_group(self):
        Fake(self)
        a = self.build()
        names = [c["name"] for c in a["candidates"]]
        self.assertEqual(names, ["Feeder rate", "Die pressure"])           # steam never moved; throughput in the group
        feeder = a["candidates"][0]
        self.assertEqual(feeder["with"], ["Throughput"])
        self.assertEqual(feeder["direction"], "down")
        self.assertAlmostEqual(feeder["lead_min"], -10, delta=1)
        self.assertAlmostEqual(a["candidates"][1]["lead_min"], 5, delta=1)
        self.assertAlmostEqual(a["lead_min"], -10, delta=1)                 # the onset band starts with the feeder
        self.assertAlmostEqual(a["trigger"]["lead_min"], 0, delta=1)
        self.assertEqual(a["trigger"]["limits"]["usl"], 4.0)
        self.assertAlmostEqual(a["baseline"]["running_rows"], 180, delta=1)   # 1 h of 20 s rows
        self.assertLessEqual(len(feeder["points"]), analysis.MAX_POINTS)    # 4 h of rows, thinned

    def test_the_baseline_uses_running_rows_only(self):
        Fake(self, state=lambda t: "STOPPED" if t < START - timedelta(minutes=62) else "RUNNING")
        a = self.build()
        self.assertEqual(a["candidates"], [])
        self.assertIn("No running baseline", a["note"])
        self.assertIsNotNone(a["trigger"])                                  # the trigger's trend is still shown

    def test_nothing_moved_is_an_answer(self):
        Fake(self)
        a = analysis.build(ASSET, START + timedelta(hours=3), None, None, now=START + timedelta(hours=5))
        self.assertEqual(a["candidates"], [])
        self.assertIn("No other signal moved", a["note"])
        self.assertIsNone(a["trigger"])

    def test_a_historian_failure_is_kept_not_raised(self):
        Fake(self, fail=True)
        a = self.build()
        self.assertIn("historian down", a["error"])

    def test_the_states_band_and_a_long_problem_is_cut(self):
        Fake(self, state=lambda t: "ALARMED" if START <= t < START + timedelta(minutes=30) else "RUNNING")
        a = analysis.build(ASSET, START, None, None, now=START + timedelta(hours=20))
        self.assertTrue(a["cut"])
        self.assertEqual([s for _, s in a["states"]], ["RUNNING", "ALARMED", "RUNNING"])

    def test_trigger_for_an_spc_ticket(self):
        from spc.models import SpcEvent, SpcTool
        tool = SpcTool.objects.create(asset_id=ASSET, signal_id="bsk:signal:EXTR01/scrap_output_kg",
                                      column="scrap_output_kg", name="Scrap output", unit="kg", lsl=0.5, usl=4.0)
        ev = SpcEvent.objects.create(tool=tool, tool_version=1, type="control_violation", rule="beyond", started_at=START,
                                     last_at=START + timedelta(hours=1), value=4.1, extreme=5.0,
                                     limits={"lcl": 1.2, "cl": 2.4, "ucl": 3.5})
        t = analysis.trigger_for(f"spc:{ev.pk}")
        self.assertEqual(t["key"], "scrap_output_kg")
        self.assertEqual(t["limits"], {"lcl": 1.2, "cl": 2.4, "ucl": 3.5, "lsl": 0.5, "usl": 4.0})
        self.assertIsNone(analysis.trigger_for("downtime:x"))


class TicketTests(TestCase):
    def setUp(self):
        Fake(self)
        for target, value in (("problems.service.asset_name", "EXTR01"), ("problems.service._post", None)):
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def test_a_ticket_keeps_its_summary_and_shows_it_in_words_and_the_report(self):
        with patch("problems.analysis.trigger_for", return_value=TRIGGER), \
                patch("problems.context.build", return_value={"states": []}), \
                patch("problems.analysis.datetime") as dt:
            dt.now.return_value = START + timedelta(hours=3)
            dt.fromisoformat = datetime.fromisoformat
            t = service.open_ticket(ASSET, Ticket.Trigger.SPC_CONTROL, "Scrap output outside control limits", START,
                                    START + timedelta(hours=1), source_key="spc:1")
        t.refresh_from_db()
        a = t.context["analysis"]
        self.assertEqual(a["candidates"][0]["name"], "Feeder rate")
        text = context.as_text(t)
        self.assertIn("Data summary: Moved 10 min before the start: Feeder rate (down", text)
        self.assertIn("with Throughput", text)
        page = report.html(t)
        self.assertIn("<h2>Data summary</h2>", page)
        self.assertEqual(page.count("<polyline"), 3)                         # trigger + 2 candidates
        self.assertIn("USL 4", page)

    def test_a_failing_summary_never_blocks_the_ticket(self):
        with patch("problems.analysis.build", side_effect=ValueError("bad")), \
                patch("problems.context.build", return_value={"states": []}):
            t = service.open_ticket(ASSET, Ticket.Trigger.MANUAL, "By hand", START, START + timedelta(hours=1))
        self.assertIn("could not be made", t.context["analysis"]["error"])
        self.assertIn("could not be made", report.html(t))
