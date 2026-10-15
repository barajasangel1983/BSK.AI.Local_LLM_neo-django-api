"""The monitoring agent and Home's API (P15a). Ported from BSKLAB EDGE's tests (P11g, P13e): the same data and
the same expectations, with the Context Service and the model replaced by fakes."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from monitor import agent
from monitor.models import MonitorCursor, MonitorPost, MonitorSettings

ASSET = "bsk:asset:EXTR01"
LATEST = datetime(2026, 3, 31, 23, 59, tzinfo=timezone.utc)                  # the recorded month ends here
REAL = datetime(2026, 10, 5, 14, 30, 20, tzinfo=timezone.utc)                # "today": replayed as 5 March, 14:30
SIGNALS = [{"key": k, "signal_id": f"bsk:signal:EXTR01/{k}", "name": n, "unit": u, "low": lo, "high": hi} for k, n, u, lo, hi in (
    ("machine_state", "Machine state", "", None, None), ("alarm_code", "Active alarm code", "", None, None),
    ("oee_pct", "OEE", "%", None, None), ("availability_pct", "Availability", "%", None, None),
    ("performance_pct", "Performance", "%", None, None), ("quality_pct", "Quality", "%", None, None),
    ("throughput_actual_kg_hr", "Throughput (actual)", "kg/h", 7972, 10385),
    ("throughput_target_kg_hr", "Throughput (target)", "kg/h", None, None),
    ("die_pressure_bar", "Die pressure", "bar", 84.3, 99.2))]
CONTEXT = {"context_text": "GRAPH FACTS [G]:\n- DIE_PLUG: die plugged; procedure: Die plug clearing"}


def sample(ts):
    """The recorded data: the machine runs, stops from 12:20 to 12:40, and the die pressure is high from 13:10 to 13:15."""
    minute = ts.hour * 60 + ts.minute
    stopped = 12 * 60 + 20 < minute <= 12 * 60 + 40
    return {"machine_state": "STOPPED" if stopped else "RUNNING", "alarm_code": "DIE_PLUG" if minute == 13 * 60 + 12 else "",
            "oee_pct": 0.0 if stopped else 80.0, "availability_pct": 95.0, "performance_pct": 90.0, "quality_pct": 93.0,
            "throughput_actual_kg_hr": 9000.0, "throughput_target_kg_hr": 10000.0,
            "die_pressure_bar": 104.0 if 13 * 60 + 10 < minute <= 13 * 60 + 15 else (101.0 if minute == 11 * 60 + 5 else 90.0)}


class Plant:
    """The Context Service the agent reads, and a model that writes one sentence (`reply`; None = the model fails)."""

    def __init__(self, test: TestCase, live: bool = False):
        self.reply, self.calls, self.live = "Written by the model.", [], live
        self.flow = {"age_seconds": 30, "stale": False, "no_data_reason": None}
        self.rows_until = None
        for target, fake in (("context_service.service.operational_state", self.state),
                             ("context_service.service.signal_history", self.history),
                             ("context_service.service.assemble", self.assemble),
                             ("context_service.service.list_assets", lambda: [{"id": ASSET, "name": "Extruder EXTR01"}]),
                             ("monitor.llm.pick", lambda: ("ollama-qwen3-8b", "qwen3:8b (Ollama)")),
                             ("monitor.llm.write", self.write)):
            p = patch(target, side_effect=fake)
            p.start()
            test.addCleanup(p.stop)
        agent._overviews.clear()

    def state(self, asset_id):
        if self.live:
            return {"source": "historian_live", "current_state": {"ts": "2026-10-12T14:30:00+00:00", "machine_state": "RUNNING", **self.flow}}
        return {"source": "plc_1_historian", "current_state": {"ts": LATEST.isoformat(), "machine_state": "STOPPED"}}

    def history(self, asset_id, start, end, signals=None, bucket_minutes=None):
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if self.live:
            end = min(end, self.rows_until) if self.rows_until else end
            first = start.replace(second=start.second // 20 * 20) + timedelta(seconds=20)
            stamps = [t for t in (first + timedelta(seconds=20 * i) for i in range(max(0, int((end - first).total_seconds() // 20) + 1)))
                      if live_sample(t)]
            rows, signals_, extra = [live_sample(t) for t in stamps], LIVE_SIGNALS, {"period_seconds": 20, "bucket_minutes": 0}
        else:
            stamps = [start + timedelta(minutes=i) for i in range(1, int((end - start).total_seconds() // 60) + 1)]
            rows, signals_, extra = [sample(t) for t in stamps], SIGNALS, {}
        self.calls.append(("history", start, end))
        return {"source": "historian_live" if self.live else "plc_1_historian", **extra, "signals": signals_,
                "ts": [t.isoformat() for t in stamps], "series": {s["key"]: [r[s["key"]] for r in rows] for s in signals_}}

    def assemble(self, query, asset_id=None, include_documents=True, budget_chars=None, search_queries=None):
        self.calls.append(("assemble", query, asset_id))
        return CONTEXT

    def write(self, model, system, body):
        return (self.reply, model[1]) if self.reply is not None else ("", "")


LIVE_REAL = datetime(2026, 10, 12, 14, 30, 10, tzinfo=timezone.utc)
LIVE_SIGNALS = SIGNALS + [{"key": "downtime_min", "signal_id": "bsk:signal:EXTR01/downtime_min", "name": "Downtime", "unit": "min",
                           "low": None, "high": None}]


def live_sample(ts):
    """The PLC's rows: a changeover 09:00–09:10, no rows 10:50–11:10, stopped 12:20–12:40, die pressure high 13:10–13:13
    (and for two rows at 08:05, which is not enough for an alert)."""
    second = ts.hour * 3600 + ts.minute * 60 + ts.second
    at = lambda h, m: h * 3600 + m * 60   # noqa: E731
    if at(10, 50) < second <= at(11, 10):
        return None
    state = "CHANGEOVER" if at(9, 0) < second <= at(9, 10) else "STOPPED" if at(12, 20) < second <= at(12, 40) else "RUNNING"
    high = at(13, 10) < second <= at(13, 13) or at(8, 5) < second <= at(8, 5) + 40
    return {"machine_state": state, "alarm_code": "DIE_PLUG" if at(13, 12) < second <= at(13, 13) else "",
            "oee_pct": 80.0 if state == "RUNNING" else 0.0, "availability_pct": 95.0, "performance_pct": 90.0, "quality_pct": 93.0,
            "throughput_actual_kg_hr": 9000.0, "throughput_target_kg_hr": 10000.0, "die_pressure_bar": 104.0 if high else 90.0,
            "downtime_min": 0.33 if state == "STOPPED" else 0.0}


class RecordedMonthTests(TestCase):
    def setUp(self):
        self.plant = Plant(self)

    def test_old_data_is_replayed_on_the_same_day_and_time_of_its_month(self):
        self.assertEqual(agent.replay_time(LATEST, REAL), datetime(2026, 3, 5, 14, 30, tzinfo=timezone.utc))
        february = datetime(2026, 2, 28, 23, 59, tzinfo=timezone.utc)
        self.assertEqual(agent.replay_time(february, REAL.replace(day=31)).day, 28)            # a shorter month

    def test_the_agent_posts_state_changes_alerts_and_hourly_summaries(self):
        written = agent.tick(ASSET, REAL)                                      # first pass: the last 8 hours, 06:00 to 14:30
        feed = agent.posts(ASSET, limit=50)["posts"]
        self.assertEqual((written, len(feed)), (11, 11))
        self.assertGreater(feed[0]["ts"], feed[-1]["ts"])                     # newest first
        by_kind = {k: [p for p in reversed(feed) if p["kind"] == k] for k in ("state", "alert", "summary")}

        self.assertEqual([p["title"] for p in by_kind["state"]], ["Machine state: RUNNING → STOPPED", "Machine state: STOPPED → RUNNING"])
        self.assertEqual((by_kind["state"][0]["text"], by_kind["state"][0]["model"]),
                         ("The machine went from RUNNING to STOPPED at 12:21.", ""))

        (alert,) = by_kind["alert"]                                            # one minute at 11:05 is not an alert; 3 minutes are
        self.assertEqual(alert["title"], "Die pressure above its normal range")
        self.assertTrue(alert["ts"].startswith("2026-03-05T13:13"))
        facts = "Die pressure has been above its normal range (84.3–99.2 bar) for 3 minutes; it is 104 bar at 13:13."
        self.assertEqual((alert["text"], alert["model"]), (facts + " Written by the model.", "qwen3:8b (Ollama)"))
        self.assertEqual((alert["data"]["value"], alert["data"]["low"], alert["data"]["high"]), (104.0, 84.3, 99.2))
        asked = next(c for c in self.plant.calls if c[0] == "assemble")
        self.assertEqual(asked[2], ASSET)
        self.assertIn("Die pressure above normal range", asked[1])

        titles = [p["title"] for p in by_kind["summary"]]
        self.assertEqual((titles[:2], len(titles)), (["Hourly summary 06:00–07:00", "Hourly summary 07:00–08:00"], 8))
        noon = next(p for p in by_kind["summary"] if p["title"] == "Hourly summary 12:00–13:00")
        self.assertEqual((noon["data"]["running_min"], noon["data"]["downtime_min"], noon["severity"]), (40, 20, "warning"))
        self.assertIn("Running 40 min, not running 20 min (STOPPED 20 min).", noon["data"]["facts"])
        self.assertIn("In order: RUNNING 20 min, then STOPPED 20 min, then RUNNING 20 min.", noon["data"]["facts"])
        eleven = next(p for p in by_kind["summary"] if p["title"] == "Hourly summary 11:00–12:00")
        self.assertIn("No signal stayed outside its normal range.", eleven["data"]["facts"])
        one = next(p for p in by_kind["summary"] if p["title"] == "Hourly summary 13:00–14:00")
        self.assertIn("Die pressure was outside its normal range (84.3–99.2 bar) for 5 min", one["data"]["facts"])
        self.assertIn("Alarms started: DIE_PLUG × 1.", one["data"]["facts"])

        self.assertEqual(agent.tick(ASSET, REAL), 0)                           # nothing new: nothing is written twice
        self.assertEqual(agent.tick(ASSET, REAL + timedelta(minutes=31)), 1)   # 15:01: the hour from 14:00 has ended
        self.assertEqual(agent.posts(ASSET, limit=1)["posts"][0]["title"], "Hourly summary 14:00–15:00")
        page = agent.posts(ASSET, limit=5)
        self.assertTrue(page["more"])
        self.assertEqual(len(agent.posts(ASSET, before_id=page["posts"][-1]["id"], limit=50)["posts"]), 7)

    def test_an_alert_says_so_when_the_studio_documents_no_cause(self):
        self.plant.reply = "NONE"
        agent.tick(ASSET, REAL)
        alert = next(p for p in agent.posts(ASSET, limit=50)["posts"] if p["kind"] == "alert")
        self.assertTrue(alert["text"].endswith("it is 104 bar at 13:13. The Context Studio has no documented cause or procedure for this signal yet."))

    def test_posts_are_still_written_when_the_model_cannot_be_used(self):
        self.plant.reply = None
        agent.tick(ASSET, REAL)
        alert = next(p for p in agent.posts(ASSET, limit=50)["posts"] if p["kind"] == "alert")
        self.assertEqual((alert["text"], alert["model"]),
                         ("Die pressure has been above its normal range (84.3–99.2 bar) for 3 minutes; it is 104 bar at 13:13.", ""))

    def test_home_shows_the_last_hours_of_the_machine(self):
        with patch("monitor.agent.now", return_value=REAL):
            o = APIClient().get("/api/monitor/overview/", {"asset_id": ASSET}).json()
        self.assertEqual((o["available"], o["replay"], o["now"]), (True, True, "2026-03-05T14:30:00+00:00"))
        self.assertEqual(o["state"]["machine_state"], "RUNNING")
        self.assertTrue(o["state"]["since"].startswith("2026-03-05T12:41"))
        self.assertEqual((o["last_hour"]["oee"], o["last_hour"]["running_min"], o["last_hour"]["throughput_target"]), (80.0, 60, 10000.0))
        self.assertEqual([h["start"][11:16] for h in o["hours"]], ["07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00"])
        noon = o["hours"][5]
        self.assertEqual((noon["running_min"], noon["downtime_min"], noon["complete"], o["hours"][-1]["complete"]), (40, 20, True, False))
        self.assertEqual((o["window"]["downtime_min"], o["window"]["alarms"], o["window"]["alarm_codes"]), (20, 1, {"DIE_PLUG": 1}))
        self.assertEqual((o["out_of_range"], len(o["trend"]), o["trend"][0]["oee"]), ([], 45, 80.0))
        self.assertEqual(APIClient().get("/api/monitor/posts/", {"asset_id": ASSET}).json(), {"posts": [], "more": False})
        self.assertEqual(APIClient().get("/api/monitor/assets/").json()["assets"][0]["id"], ASSET)


class LiveRowsTests(TestCase):
    def setUp(self):
        self.plant = Plant(self, live=True)
        self.plant.reply = None                                    # posts written from the data

    def test_on_the_plcs_rows_three_minutes_are_nine_rows_and_minutes_stay_minutes(self):
        written = agent.tick(ASSET, LIVE_REAL)                     # first pass: 06:00 to 14:30:10
        feed = agent.posts(ASSET, limit=50)["posts"]
        by_kind = {k: [p for p in reversed(feed) if p["kind"] == k] for k in ("state", "alert", "summary", "data")}
        self.assertEqual((written, by_kind["data"]), (len(feed), []))
        self.assertEqual([p["title"][15:] for p in by_kind["state"]],
                         ["RUNNING → CHANGEOVER", "CHANGEOVER → RUNNING", "RUNNING → STOPPED", "STOPPED → RUNNING"])
        (alert,) = by_kind["alert"]                                # two rows at 08:05 are not an alert; nine in a row are
        self.assertTrue(alert["ts"].startswith("2026-10-12T13:13:00"))
        self.assertEqual(alert["text"], "Die pressure has been above its normal range (84.3–99.2 bar) for 3 minutes; it is 104 bar at 13:13.")

        self.assertEqual([p["title"] for p in by_kind["summary"]], [f"Hourly summary {h:02d}:00–{h + 1:02d}:00" for h in range(6, 14)])
        hour = lambda h: next(p["data"] for p in by_kind["summary"] if p["title"].startswith(f"Hourly summary {h:02d}"))  # noqa: E731
        self.assertEqual((hour(12)["running_min"], hour(12)["downtime_min"], hour(12)["not_running_min"]), (40, 20, 20))
        self.assertEqual((hour(9)["downtime_min"], hour(9)["not_running_min"], hour(9)["states"]["CHANGEOVER"]), (0, 10, 10))
        self.assertEqual((hour(10)["running_min"], hour(10)["no_data_min"]), (50, 10))
        self.assertIn("No data for 10 min of the hour.", hour(10)["facts"])
        self.assertEqual((hour(11)["no_data_min"], hour(13)["no_data_min"]), (10, 0))
        self.assertIn("Die pressure was outside its normal range (84.3–99.2 bar) for 3 min", hour(13)["facts"])
        self.assertEqual((hour(13)["alarms"], hour(8)["out_of_range"]), ({"DIE_PLUG": 1}, []))

        self.assertEqual(agent.tick(ASSET, LIVE_REAL), 0)          # nothing new
        # A collector kept the rows of 14:30 to 15:01 and delivers them later: the next pass begins after the newest row read.
        self.plant.rows_until = datetime(2026, 10, 12, 14, 30, tzinfo=timezone.utc)
        self.assertEqual(agent.tick(ASSET, LIVE_REAL + timedelta(minutes=31)), 0)
        self.plant.rows_until = None
        self.assertEqual(agent.tick(ASSET, LIVE_REAL + timedelta(minutes=32)), 1)
        self.assertEqual(agent.posts(ASSET, limit=1)["posts"][0]["title"], "Hourly summary 14:00–15:00")

    def test_the_agent_says_when_the_rows_stop_and_when_they_are_back(self):
        agent.tick(ASSET, LIVE_REAL)
        self.plant.flow.update(age_seconds=400, stale=True, no_data_reason="no_rows_from_plc")
        later = LIVE_REAL + timedelta(seconds=400)
        self.assertEqual((agent.tick(ASSET, later), agent.tick(ASSET, later + timedelta(minutes=1))), (1, 0))   # said once
        stopped = agent.posts(ASSET, limit=1)["posts"][0]
        self.assertEqual((stopped["kind"], stopped["severity"], stopped["title"]), ("data", "warning", "No data from the PLC"))
        self.assertEqual(stopped["text"], "The last row is from 14:30:00; the state shown is the last known one. "
                                          "The collector reports but has no rows: the generator is paused, or the PLC or its link is down.")
        self.plant.flow.update(age_seconds=30, stale=False, no_data_reason=None)
        agent.tick(ASSET, later + timedelta(minutes=10))
        back = next(p for p in agent.posts(ASSET, limit=5)["posts"] if p["kind"] == "data")
        self.assertEqual(back["title"], "Data from the PLC is back")
        self.assertTrue(back["text"].startswith("Rows arrive again, 16 min after the row of 14:30:00."))

    def test_home_on_the_plcs_rows_is_the_present_and_says_when_no_row_arrives(self):
        o = agent.overview(ASSET, LIVE_REAL)
        self.assertEqual((o["available"], o["live"], o["replay"], o["now"], o["period_seconds"]),
                         (True, True, False, "2026-10-12T14:30:10+00:00", 20))
        self.assertEqual(o["data"], {"live": True, "stale": False, "age_seconds": 30, "reason": "", "text": ""})
        self.assertTrue(o["state"]["since"].startswith("2026-10-12T12:40:20"))
        self.assertEqual((o["last_hour"]["running_min"], o["last_hour"]["downtime_min"], o["last_hour"]["oee"]), (60, 0, 80.0))
        self.plant.flow.update(age_seconds=400, stale=True, no_data_reason="collector_not_reporting")
        o = agent.overview(ASSET, LIVE_REAL)
        self.assertTrue(o["data"]["stale"])
        self.assertTrue(o["data"]["text"].startswith("No row from the PLC for 6 min. The collector is not reporting"))
        self.assertEqual(o["out_of_range"], [])                    # a last known value is not "now"


class CommandAndSettingsTests(TestCase):
    def setUp(self):
        self.plant = Plant(self)

    def test_the_pass_runs_every_asset_and_the_switch_stops_it(self):
        with patch("monitor.agent.now", return_value=REAL):
            call_command("monitor_tick", stdout=open("/dev/null", "w"))
        self.assertEqual(MonitorPost.objects.count(), 11)
        self.assertTrue(MonitorCursor.objects.filter(asset_id=ASSET).exists())
        client = APIClient()
        self.assertEqual(client.put("/api/monitor/settings/", {"agent_enabled": "no"}, format="json").status_code, 400)
        self.assertEqual(client.put("/api/monitor/settings/", {"agent_enabled": False}, format="json").json(), {"agent_enabled": False})
        MonitorPost.objects.all().delete()
        MonitorCursor.objects.all().delete()
        with patch("monitor.agent.now", return_value=REAL):
            call_command("monitor_tick", stdout=open("/dev/null", "w"))
        self.assertEqual(MonitorPost.objects.count(), 0)
        self.assertFalse(MonitorSettings.get().agent_enabled)


class SpcPostsTests(TestCase):
    """P15a-2: SPC tools in the feed (on the live historian), and their events in the hourly summary."""

    def setUp(self):
        from spc.models import SpcCursor, SpcTool
        self.plant = Plant(self, live=True)
        self.plant.reply = None
        self.tool = SpcTool.objects.create(asset_id=ASSET, signal_id="bsk:signal:EXTR01/scrap_output_kg", column="scrap_output_kg",
                                           name="Scrap output", unit="kg")
        self.cursor = SpcCursor.objects.create(tool=self.tool, judged_until=LIVE_REAL)

    def event(self, kind, rule, at, value=9.6):
        from spc.models import SpcEvent
        return SpcEvent.objects.create(tool=self.tool, tool_version=1, type=kind, rule=rule, started_at=at, last_at=at,
                                       value=value, extreme=value, limits={"lcl": 0.2, "cl": 1.9, "ucl": 3.6, "lsl": 0.5, "usl": 4.0})

    def spc_posts(self):
        return [p for p in reversed(agent.posts(ASSET, limit=100)["posts"]) if p["kind"] == "spc"]

    def test_a_tool_that_gets_worse_is_posted_and_back_to_stable_after_calm(self):
        at = LIVE_REAL - timedelta(minutes=2)
        self.event("control_violation", "beyond_limits", at)
        self.event("statistical_warning", "two_of_three", at)          # a warning alone is never posted
        agent.tick(ASSET, LIVE_REAL)
        (first,) = self.spc_posts()
        self.assertEqual((first["title"], first["severity"]), ("SPC · Scrap output: stable → out of control", "warning"))
        self.assertEqual(first["text"], "Scrap output is out of control since 14:28: rule beyond_limits, value 9.6 kg "
                                        "(LSL 0.5, LCL 0.2, CL 1.9, UCL 3.6, USL 4).")
        self.assertEqual(first["data"]["tool_id"], str(self.tool.id))

        later = LIVE_REAL + timedelta(minutes=1)
        self.cursor.judged_until = later
        self.cursor.save()
        agent.tick(ASSET, later)                                         # still out of control: nothing new
        self.event("spec_violation", "spec", later)
        agent.tick(ASSET, later + timedelta(seconds=20))
        self.assertEqual([p["title"][6:] for p in self.spc_posts()],
                         ["Scrap output: stable → out of control", "Scrap output: out of control → specification violation"])

        calm = later + timedelta(minutes=10)                             # the events are no longer active
        self.cursor.judged_until = calm
        self.cursor.save()
        agent.tick(ASSET, calm)
        agent.tick(ASSET, calm + timedelta(minutes=29))
        self.assertEqual(len(self.spc_posts()), 2)                       # calm, but not for 30 minutes yet
        self.cursor.judged_until = calm + timedelta(minutes=31)
        self.cursor.save()
        agent.tick(ASSET, calm + timedelta(minutes=31))
        back = self.spc_posts()[-1]
        self.assertEqual((back["title"], back["severity"]), ("SPC · Scrap output: back to stable", "info"))

    def test_the_hourly_summary_counts_the_spc_events(self):
        for minute in (5, 25, 45):
            self.event("control_violation", "beyond_limits", datetime(2026, 10, 12, 13, minute, tzinfo=timezone.utc))
        self.event("spec_violation", "spec", datetime(2026, 10, 12, 13, 46, tzinfo=timezone.utc))
        agent.tick(ASSET, LIVE_REAL)
        one = next(p for p in agent.posts(ASSET, limit=100)["posts"] if p["title"] == "Hourly summary 13:00–14:00")
        self.assertIn("SPC: Scrap output 1 specification violation, 3 control-limit violations.", one["data"]["facts"])
        self.assertEqual(one["data"]["spc"], {"Scrap output": {"spec_violation": 1, "control_violation": 3}})
        twelve = next(p for p in agent.posts(ASSET, limit=100)["posts"] if p["title"] == "Hourly summary 12:00–13:00")
        self.assertNotIn("SPC:", twelve["data"]["facts"])

    def test_no_spc_posts_on_a_replay(self):
        self.plant.live = False
        self.event("spec_violation", "spec", LATEST - timedelta(minutes=1))
        agent.tick(ASSET, REAL)
        self.assertEqual(self.spc_posts(), [])
