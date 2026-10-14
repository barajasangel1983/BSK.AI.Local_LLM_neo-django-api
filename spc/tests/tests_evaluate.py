"""The SPC evaluator and its events (specification section 7 and 6.6). The historian is a fake."""

import math
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from chat.test_asset_chat import ASSET
from spc import evaluate, service
from spc.models import SpcCursor, SpcEvent, SpcTool
from spc.tests.tests_api import CONTEXT, MOISTURE, PRESSURE, TORQUE

T0 = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
at = lambda minute, second=0: T0 + timedelta(minutes=minute, seconds=second)    # noqa: E731


@override_settings(HISTORIAN_SOURCE="live")
class EvaluatorTestCase(TestCase):
    """A plant that runs steadily (moisture 9.4 and 9.6 in turn) except where a test says otherwise."""

    def setUp(self):
        self.client = APIClient()
        self.special: dict = {}                     # ts → moisture
        self.newest = at(10)                        # the newest row the historian has

        def rows(asset_key, columns, start, end, limit):
            first = start.replace(second=start.second // 20 * 20, microsecond=0) + timedelta(seconds=20)
            out = []
            for i in range(max(0, int((min(end, self.newest) - first).total_seconds() // 20) + 1)):
                ts = first + timedelta(seconds=20 * i)
                swing = 0.1 if (int(ts.timestamp()) // 20) % 2 else -0.1
                out.append({"ts": ts, "machine_state": "RUNNING", "recipe_id": "RC-A",
                            "product_moisture_pct": self.special.get(ts, self.wave(ts) if self.wave else 9.5 + swing),
                            "die_pressure_bar": 94 + 10 * swing, "screw_torque_pct": 95 + 10 * swing})
            return [{k: v for k, v in r.items() if k == "ts" or k in columns} for r in out]
        self.wave = None
        self.reads = patch("chat.asset_context.historian_rows", side_effect=rows).start()
        patch("context_graph.services.get_asset_context", return_value=CONTEXT).start()
        patch("spc.service.recent_states", return_value=["RUNNING"]).start()
        patch("spc.service.newest_row", side_effect=lambda asset_id: {"ts": self.newest, "machine_state": "RUNNING"}).start()
        self.addCleanup(patch.stopall)

    def tool(self, signal_id=MOISTURE, cl=9.5, lcl=9.0, ucl=10.0, **fields):
        """A tool with fixed limits (sigma 1/6), judging from T0."""
        fields = {"ignore_seconds": 0, "rules": ["beyond_limits", "spec"], **fields}
        tool = service.create_tool({"asset_id": ASSET, "signal_id": signal_id, **fields})
        service.set_fixed_limits(tool, [{"group": "RC-A" if tool.group_by else "", "cl": cl, "lcl": lcl, "ucl": ucl}])
        SpcCursor.objects.filter(tool=tool).update(judged_until=T0)
        return SpcTool.objects.prefetch_related("limits").get(pk=tool.pk)

    def run_pass(self, until=None, **options):
        if until is not None:
            self.newest = until
        return evaluate.evaluate(**options)

    def events(self):
        return [(e.rule, e.started_at, e.last_at, e.count) for e in SpcEvent.objects.order_by("started_at", "rule")]


class EvaluatorTests(EvaluatorTestCase):
    def test_a_violation_that_continues_is_one_event_and_nothing_is_written_twice(self):
        tool = self.tool(usl=11.0)
        self.special.update({at(5): 10.5, at(5, 20): 10.8, at(5, 40): 10.6, at(20): 10.4})
        (done,) = self.run_pass()
        self.assertEqual((done["judged"], done["created"], done["extended"], done["skipped"]), (30, 1, 0, ""))
        event = SpcEvent.objects.get()
        self.assertEqual((event.type, event.rule, event.group, event.machine_state), ("control_violation", "beyond_limits", "", "RUNNING"))
        self.assertEqual((event.started_at, event.last_at, event.count, event.value, event.extreme), (at(5), at(5, 40), 3, 10.5, 10.8))
        self.assertEqual((event.tool_version, event.limits), (tool.version, {"cl": 9.5, "lcl": 9.0, "ucl": 10.0, "lsl": None, "usl": 11.0}))
        self.assertEqual(SpcCursor.objects.get(tool=tool).judged_until, at(10))        # the newest row read, not the clock

        self.assertEqual(self.run_pass()[0]["skipped"], "no new row")                  # the same pass again
        self.assertEqual(SpcEvent.objects.count(), 1)

        self.run_pass(at(30))                                                          # 14 minutes later: another event
        self.assertEqual(self.events(), [("beyond_limits", at(5), at(5, 40), 3), ("beyond_limits", at(20), at(20), 1)])

        self.special.update({at(30, 20): 10.5, at(30, 40): 10.5, at(34): 11.5})        # it continues over three passes
        self.run_pass(at(30, 20))
        (done,) = self.run_pass(at(31))
        self.assertEqual((done["created"], done["extended"]), (0, 1))
        self.run_pass(at(36))                                                          # 3 min 20 s later: still the same event
        self.assertEqual(self.events()[2:], [("beyond_limits", at(30, 20), at(34), 3), ("spec", at(34), at(34), 1)])
        self.assertEqual(SpcEvent.objects.get(rule="spec").type, "spec_violation")
        self.assertEqual(SpcEvent.objects.get(started_at=at(30, 20)).extreme, 11.5)

    def test_rows_that_arrive_late_are_judged_on_a_later_pass(self):
        tool = self.tool()
        self.special[at(12)] = 10.5
        self.run_pass(at(10))                                                          # the collector has kept the rows after 12:10
        self.assertEqual((SpcEvent.objects.count(), SpcCursor.objects.get(tool=tool).judged_until), (0, at(10)))
        self.run_pass(at(20))                                                          # they are delivered with their own times
        self.assertEqual(self.events(), [("beyond_limits", at(12), at(12), 1)])

    def test_a_dry_run_writes_nothing(self):
        tool = self.tool()
        self.special[at(5)] = 10.5
        (done,) = self.run_pass(dry_run=True)
        self.assertEqual((done["created"], done["events"][0]["rule"], done["events"][0]["started_at"]), (1, "beyond_limits", at(5)))
        self.assertEqual((SpcEvent.objects.count(), SpcCursor.objects.get(tool=tool).judged_until), (0, T0))

    def test_tools_that_are_paused_or_have_no_limits_are_skipped_and_a_failing_tool_does_not_stop_the_others(self):
        self.tool()
        self.special[at(5)] = 10.5
        paused = self.tool(PRESSURE, cl=94, lcl=90, ucl=98, enabled=False)
        bare = service.create_tool({"asset_id": ASSET, "signal_id": TORQUE})
        by_name = {r["tool"]: r for r in self.run_pass()}
        self.assertEqual((by_name[paused.name]["skipped"], by_name[bare.name]["skipped"], by_name["Product moisture"]["created"]),
                         ("paused", "no control limits", 1))

        SpcTool.objects.filter(pk=paused.pk).update(enabled=True)
        real = service.read_rows

        def broken(asset_id, columns, start, end):
            if "die_pressure_bar" in columns:
                raise RuntimeError("column gone")
            return real(asset_id, columns, start, end)
        self.special[at(15)] = 10.5
        with patch("spc.service.read_rows", side_effect=broken), self.assertLogs("spc.evaluate", "ERROR"):
            by_name = {r["tool"]: r for r in self.run_pass(at(20))}
        self.assertIn("RuntimeError: column gone", by_name[paused.name]["error"])
        self.assertEqual(by_name["Product moisture"]["created"], 1)                     # the others were judged

    def test_the_recorded_month_is_not_judged(self):
        self.tool()
        self.special[at(5)] = 10.5
        with override_settings(HISTORIAN_SOURCE="march"):
            (done,) = self.run_pass()
        self.assertIn("not the present", done["skipped"])
        self.assertEqual((SpcEvent.objects.count(), self.reads.call_count), (0, 0))

    def test_judging_again_from_a_time_replaces_the_events_from_then_on(self):
        tool = self.tool()
        self.special.update({at(5): 10.5, at(15): 10.5})
        self.run_pass(at(20))
        self.assertEqual(SpcEvent.objects.count(), 2)
        service.set_fixed_limits(tool, [{"group": "", "cl": 9.5, "lcl": 8.0, "ucl": 11.0}])       # wider limits: no violation
        self.run_pass(tool_id=str(tool.id), since=at(10))
        self.assertEqual(self.events(), [("beyond_limits", at(5), at(5), 1)])           # the event before 12:10 stays
        self.assertEqual(SpcCursor.objects.get(tool=tool).judged_until, at(20))

    def test_an_event_type_that_is_not_recorded_writes_no_event(self):
        actions = {"statistical_warning": {"log": False}, "control_violation": {"log": True}, "spec_violation": {"log": True}}
        self.tool(rules=["beyond_limits", "eight_one_side"], event_actions=actions)
        self.special.update({at(5, 20 * i): 9.7 for i in range(9)})                    # nine in a row above the centre
        self.special[at(9)] = 10.5
        self.run_pass()
        self.assertEqual([e[0] for e in self.events()], ["beyond_limits"])

    def test_one_pass_and_many_short_passes_write_the_same_events(self):
        """The evaluator runs every minute; what it finds must not depend on where its reads begin."""
        self.wave = lambda ts: round(9.5 + 0.22 * math.sin(ts.timestamp() / 310) + 0.2 * math.sin(ts.timestamp() / 47), 3)
        rules = ["beyond_limits", "range_beyond_limits", "two_of_three", "four_of_five", "eight_one_side", "six_trending"]
        for method in ("imr", "xbar_r"):
            SpcTool.objects.all().delete()
            tool = self.tool(rules=rules, method=method, subgroup_size=3, ignore_seconds=120, ucl=9.9, lcl=9.1)
            self.run_pass(at(180))
            together = self.events()
            self.assertGreater(len(together), 5, method)
            self.assertGreater(len({e[0] for e in together}), 2, method)               # several rules fired

            SpcEvent.objects.all().delete()
            SpcCursor.objects.filter(tool=tool).update(judged_until=T0)
            minute = 0.0
            while minute < 180:
                minute = min(180.0, minute + 1 + (minute * 7) % 3)                     # passes of one to four minutes
                self.run_pass(T0 + timedelta(seconds=int(minute * 60) // 20 * 20))
            self.assertEqual(self.events(), together, method)

    def test_the_command_says_what_it_did(self):
        out = StringIO()
        call_command("spc_evaluate", stdout=out)
        self.assertIn("No SPC tool.", out.getvalue())
        tool = self.tool()
        self.special[at(5)] = 10.5
        out = StringIO()
        call_command("spc_evaluate", "--dry-run", stdout=out)
        self.assertIn("Product moisture: 30 samples judged, 1 event(s) started, 0 continued (dry run, nothing written)", out.getvalue())
        self.assertIn("12:05:00 control_violation beyond_limits", out.getvalue())
        out = StringIO()
        call_command("spc_evaluate", "--tool", str(tool.id), "--since", "2026-10-12T12:00:00Z", stdout=out)
        self.assertEqual(SpcEvent.objects.count(), 1)
        with patch("spc.service.read_rows", side_effect=RuntimeError("historian down")), self.assertLogs("spc.evaluate", "ERROR"):
            self.newest = at(20)
            with self.assertRaises(SystemExit):
                call_command("spc_evaluate", stdout=StringIO())                          # no tool could be judged: exit code 1


class EventApiTests(EvaluatorTestCase):
    def screening(self):
        return self.client.get("/api/spc/screening/", {"asset_id": ASSET}).json()

    def test_events_give_the_tool_its_status_until_they_are_five_minutes_old(self):
        self.tool(usl=11.0, rules=["beyond_limits", "spec", "six_trending", "eight_one_side"])
        self.special.update({at(1, 20 * i): 9.3 + 0.05 * i for i in range(6)})         # six rising, all inside the limits
        self.run_pass(at(3))
        view = self.screening()
        self.assertEqual((view["tools"][0]["status"], view["tools"][0]["status_reason"], view["summary"]["trending"]), ("trending", "six_trending", 1))

        self.special[at(8)] = 10.5
        self.run_pass(at(9))
        view = self.screening()
        self.assertEqual((view["tools"][0]["status"], view["tools"][0]["status_reason"]), ("out_of_control", "beyond_limits"))
        self.assertEqual((view["summary"]["out_of_control"], view["summary"]["active_violations"], view["tools"][0]["violation_count"]), (1, 1, 2))
        self.assertEqual(view["tools"][0]["last_violation_at"][:19], "2026-10-12T12:08:00")

        self.special[at(9, 20)] = 11.5                                                 # over the specification: that comes first
        self.run_pass(at(10))
        self.assertEqual(self.screening()["tools"][0]["status"], "spec_violation")

        event = SpcEvent.objects.get(rule="spec")
        url = f"/api/spc/events/{event.id}/acknowledge/"
        seen = self.client.post(url, {"by": "Angel"}, format="json").json()
        self.assertEqual((seen["acknowledged_by"], seen["active"] is not None, bool(seen["acknowledged_at"])), ("Angel", True, True))
        again = self.client.post(url, {"by": "someone else"}, format="json").json()
        self.assertEqual((again["acknowledged_by"], again["acknowledged_at"]), ("Angel", seen["acknowledged_at"]))
        self.assertEqual(self.screening()["tools"][0]["status"], "spec_violation")      # acknowledging does not change the status
        self.assertEqual(self.client.post("/api/spc/events/nope/acknowledge/").status_code, 404)

        self.run_pass(at(16))                                                          # five quiet minutes later
        view = self.screening()
        self.assertEqual((view["tools"][0]["status"], view["summary"]["active_violations"], view["tools"][0]["violation_count"]), ("stable", 0, 3))

    def test_the_events_of_a_tool_newest_first_in_pages(self):
        tool = self.tool()
        other = self.tool(PRESSURE, cl=94, lcl=80, ucl=108)
        self.special.update({at(2): 10.5, at(12): 10.6, at(28): 10.7})
        self.run_pass(at(30))
        body = self.client.get("/api/spc/events/", {"asset_id": ASSET}).json()
        self.assertEqual(([e["value"] for e in body["events"]], body["more"]), ([10.7, 10.6, 10.5], False))
        first = body["events"][0]
        self.assertEqual((first["tool_name"], first["unit"], first["type"], first["rule"], first["active"], first["tool_version"]),
                         ("Product moisture", "%", "control_violation", "beyond_limits", True, tool.version))
        self.assertEqual([e["active"] for e in body["events"]], [True, False, False])
        active = self.client.get("/api/spc/events/", {"tool": str(tool.id), "active": "true"}).json()
        self.assertEqual([e["value"] for e in active["events"]], [10.7])
        page = self.client.get("/api/spc/events/", {"tool": str(tool.id), "limit": 2}).json()
        self.assertEqual(([e["value"] for e in page["events"]], page["more"]), ([10.7, 10.6], True))
        rest = self.client.get("/api/spc/events/", {"tool": str(tool.id), "limit": 2, "before": page["events"][-1]["started_at"]}).json()
        self.assertEqual(([e["value"] for e in rest["events"]], rest["more"]), ([10.5], False))
        self.assertEqual(self.client.get("/api/spc/events/", {"tool": str(other.id)}).json(), {"events": [], "more": False})
        self.assertEqual(self.client.get("/api/spc/events/", {"tool": "nope"}).status_code, 404)
        self.client.delete(f"/api/spc/tools/{tool.id}/")                                # a deleted tool takes its events with it
        self.assertEqual(SpcEvent.objects.count(), 0)


class NoGroundTruthTests(TestCase):
    def test_the_evaluator_never_reads_the_injected_anomaly_record(self):
        folder = Path(__file__).resolve().parent.parent
        for name in ("evaluate.py", "management/commands/spc_evaluate.py", "service.py", "views.py"):
            text = (folder / name).read_text()
            self.assertNotIn("LiveSimEvent", text, name)
            self.assertNotIn("sim_events", text, name)
