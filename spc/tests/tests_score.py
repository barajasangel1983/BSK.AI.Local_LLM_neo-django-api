"""The score against the injected anomalies (P12d): found, missed, delay, false alarms on a made-up record,
and the tools' events made again in memory without writing anything. The historian is a fake."""

from datetime import timedelta
from pathlib import Path

from django.test import SimpleTestCase, TestCase

from spc import engine, evaluate, score
from spc.models import SpcCursor, SpcEvent, SpcTool
from spc.tests.tests_api import PRESSURE
from spc.tests.tests_evaluate import EvaluatorTestCase, T0, at


def rows(event_id, event_type, signal, first, last, size=2.0):
    """The record's rows of one anomaly, one every 20 s from `first` to `last`."""
    out, ts = [], first
    while ts <= last:
        out.append({"ts": ts, "event_id": event_id, "event_type": event_type, "event_signal": signal, "event_size": size})
        ts += timedelta(seconds=20)
    return out


def event(tool, column, started, last=None, kind=engine.CONTROL, rule="beyond_limits"):
    return score.Event(tool=tool, column=column, type=kind, rule=rule, started_at=started, last_at=last or started)


class ScoreTests(SimpleTestCase):
    def setUp(self):
        self.anomalies = score.anomalies(
            rows(2, 1, "die_pressure_bar", at(10), at(20)) + rows(3, 4, "die_pressure_bar", at(60), at(70), 0.0)
            + rows(4, 2, "product_moisture_pct", at(100), at(130), -3.0) + rows(5, 3, "steam_flow_kg_hr", at(200), at(200), 5.8))
        self.tools = {"Die pressure": "die_pressure_bar", "Moisture": "product_moisture_pct"}

    def test_the_record_gives_one_anomaly_per_event_id_with_its_first_and_last_row(self):
        a = self.anomalies
        self.assertEqual([(x.event_id, x.type, x.signal, x.rows) for x in a],
                         [(2, "Step", "die_pressure_bar", 31), (3, "Stuck value", "die_pressure_bar", 31),
                          (4, "Drift", "product_moisture_pct", 91), (5, "Spike", "steam_flow_kg_hr", 1)])
        self.assertEqual((a[0].start, a[0].end, a[3].start, a[3].end), (at(10), at(20), at(200), at(200)))

    def test_found_missed_not_monitored_and_the_delay(self):
        events = [
            event("Die pressure", "die_pressure_bar", at(12), at(18)),                      # finds #2, 2 min late
            event("Die pressure", "die_pressure_bar", at(14), kind=engine.SPEC, rule="spec"),   # a later one: not counted
            event("Moisture", "product_moisture_pct", at(134, 40), at(135)),                # 4 min 40 s after #4's last row: found
        ]
        result = score.score(self.anomalies, events, self.tools)
        found = {f.anomaly.event_id: (f.monitored, f.event and f.event.rule, f.delay_minutes) for f in result.findings}
        self.assertEqual(found, {2: (True, "beyond_limits", 2.0), 3: (True, None, None),
                                 4: (True, "beyond_limits", 34 + 40 / 60), 5: (False, None, None)})
        self.assertEqual((len(result.monitored), len(result.found)), (3, 2))
        self.assertAlmostEqual(result.median_delay, (2.0 + 34 + 40 / 60) / 2)

    def test_an_event_before_the_anomaly_or_after_the_margin_does_not_find_it(self):
        events = [event("Die pressure", "die_pressure_bar", at(9, 40), at(30)),             # began before #2: not a finding
                  event("Moisture", "product_moisture_pct", at(135, 20))]                   # 5 min 20 s after #4: too late
        result = score.score(self.anomalies, events, self.tools)
        self.assertEqual(result.found, [])

    def test_false_alarms_are_violations_that_touch_no_anomaly_on_their_signal_and_warnings_are_counted_apart(self):
        events = [
            event("Die pressure", "die_pressure_bar", at(9, 40), at(30)),                   # touches #2: not false
            event("Die pressure", "die_pressure_bar", at(4, 50)),                           # 5 min 10 s before #2: false
            event("Die pressure", "die_pressure_bar", at(75)),                              # 5 min after #3: touches it
            event("Die pressure", "die_pressure_bar", at(150), kind=engine.SPEC, rule="spec"),          # false
            event("Moisture", "product_moisture_pct", at(20)),                              # #2 is not on moisture: false
            event("Moisture", "product_moisture_pct", at(110), kind=engine.WARNING, rule="no_variation"),   # on #4
            event("Moisture", "product_moisture_pct", at(300), kind=engine.WARNING, rule="six_trending"),   # false warning
        ]
        alarms = {t.tool: (t.control, t.spec, t.warnings, t.false_warnings)
                  for t in score.score(self.anomalies, events, self.tools).false_alarms}
        self.assertEqual(alarms, {"Die pressure": (1, 1, 0, 0), "Moisture": (1, 0, 2, 1)})

    def test_no_anomaly_and_no_event(self):
        result = score.score([], [], self.tools)
        self.assertEqual((result.findings, result.median_delay), ([], None))
        self.assertEqual([(t.tool, t.control) for t in result.false_alarms], [("Die pressure", 0), ("Moisture", 0)])


class JudgeWindowTests(EvaluatorTestCase):
    """The events of the present configuration, made again for a window in memory."""

    def test_the_window_is_judged_again_without_writing_or_changing_anything(self):
        tool = self.tool(usl=11.0)
        self.special.update({at(5): 10.5, at(5, 20): 10.6, at(8): 11.5})
        self.run_pass()                                                                 # the evaluator wrote its events
        SpcEvent.objects.filter(rule="spec").update(acknowledged_by="Angel")
        stored = [(e.id, e.rule, e.count, e.acknowledged_by) for e in SpcEvent.objects.order_by("started_at")]
        cursor = SpcCursor.objects.get(tool=tool).judged_until

        tool.rules = ["beyond_limits", "spec", "no_variation"]                          # changed in memory only
        limits = {lim.group: lim for lim in tool.limits.all()}
        made = evaluate.judge_window(tool, limits, T0, at(10))
        self.assertEqual([(e.rule, e.started_at, e.last_at, e.count) for e in made],
                         [("beyond_limits", at(5), at(8), 3), ("spec", at(8), at(8), 1)])      # 3 min apart: one event
        self.assertTrue(all(e._state.adding for e in made))                             # none of them is stored

        self.assertEqual([(e.id, e.rule, e.count, e.acknowledged_by) for e in SpcEvent.objects.order_by("started_at")], stored)
        self.assertEqual(SpcCursor.objects.get(tool=tool).judged_until, cursor)
        saved = SpcTool.objects.get(pk=tool.pk)
        self.assertEqual((saved.rules, saved.version), (["beyond_limits", "spec"], tool.version))

    def test_a_tool_that_is_not_saved_is_judged_with_limits_that_are_not_saved(self):
        tool = SpcTool(asset_id="bsk:asset:EXTR01", signal_id=PRESSURE, column="die_pressure_bar", name="Die pressure",
                       ignore_seconds=0, group_by="recipe_id", rules=["beyond_limits"])
        inside = {"RC-A": engine.fixed_limits("RC-A", 94, 92, 96, None, method=tool.method, subgroup_size=tool.subgroup_size, k=3)}
        self.assertEqual(evaluate.judge_window(tool, inside, T0, at(10)), [])           # the fake swings 93 to 95
        narrow = {"RC-A": engine.fixed_limits("RC-A", 94, 93.5, 94.5, None, method=tool.method, subgroup_size=tool.subgroup_size, k=3)}
        made = evaluate.judge_window(tool, narrow, T0, at(10))                          # every row beyond: one event
        self.assertEqual([(e.rule, e.group, e.started_at, e.last_at, e.count) for e in made],
                         [("beyond_limits", "RC-A", at(0, 20), at(10), 30)])
        self.assertEqual(SpcTool.objects.count(), 0)


class OnlyTheScoreReadsTheRecordTests(TestCase):
    def test_the_injected_anomaly_record_is_read_by_spc_score_only(self):
        folder = Path(__file__).resolve().parent.parent
        readers = [p.relative_to(folder).as_posix() for p in folder.rglob("*.py")
                   if "tests" not in p.parts and ("LiveSimEvent" in p.read_text() or "sim_events" in p.read_text())]
        self.assertEqual(readers, ["management/commands/spc_score.py"])            # score.py is given plain rows
        self.assertNotIn("LiveSimEvent", (folder / "score.py").read_text())
        self.assertNotIn("from historian", (folder / "score.py").read_text())
