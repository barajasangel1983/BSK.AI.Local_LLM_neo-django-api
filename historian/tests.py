"""The historian's read-only view (Health page → Historian Postgres).

The Postgres user cannot create a test database, so the samples table is built in the test
SQLite database and the historian router is pointed at it for the length of the tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from io import StringIO

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import OperationalError, connection
from django.test import SimpleTestCase, TestCase, override_settings

from chat import asset_context

from . import check, pipeline
from .models import ExtruderSample, LiveCollector, LiveRow, LiveRowMeta, LiveSimEvent

START = datetime(2026, 3, 1, tzinfo=timezone.utc)
LIVE_START = datetime(2026, 10, 11, 19, 50, 20, tzinfo=timezone.utc)


def _sample(minute: int, state: str = "RUNNING") -> ExtruderSample:
    numbers = {f.name: 1 for f in ExtruderSample._meta.fields if f.get_internal_type() == "DecimalField"}
    numbers["die_pressure_bar"] = 80 + minute / 2
    return ExtruderSample(ts=START + timedelta(minutes=minute), extruder_id="EXTR01", batch_id="B1", recipe_id="R1",
                          shift_code="A", machine_state=state, **numbers)


def _live(period: int, state: str = "RUNNING") -> LiveRow:
    numbers = {f.name: 1.0 for f in LiveRow._meta.fields if f.get_internal_type() == "FloatField"}
    numbers["die_pressure_bar"] = 90 + period / 10
    return LiveRow(ts=LIVE_START + timedelta(seconds=20 * period), extruder_id="EXTR01", batch_id="B1",
                   recipe_id="RC-A", shift_code="B", machine_state=state, alarm_active=False, **numbers)


class HistorianViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        with connection.schema_editor() as editor:
            for model in (ExtruderSample, LiveRow, LiveRowMeta, LiveSimEvent, LiveCollector):
                editor.create_model(model)
        cls._routes = [patch(f"neo_llm_api.db_routers.HistorianRouter.{name}", return_value="default")
                       for name in ("db_for_read", "db_for_write")]
        for route in cls._routes:
            route.start()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        for route in cls._routes:
            route.stop()
        with connection.schema_editor() as editor:
            for model in (ExtruderSample, LiveRow, LiveRowMeta, LiveSimEvent, LiveCollector):
                editor.delete_model(model)

    @classmethod
    def setUpTestData(cls):
        ExtruderSample.objects.bulk_create(
            [_sample(m, "STOPPED" if m >= 100 else "RUNNING") for m in range(120)])
        LiveRow.objects.bulk_create([_live(n, "ALARMED" if n >= 6 else "RUNNING") for n in range(9)])
        # Counters 947..955 with 950 and 951 never delivered (rows 3 and 4 arrive as 952, 953), then a PLC restart.
        # Row 2 reports its boot time a second later, as the PLC does within one run: not a restart.
        counters = [947, 948, 949, 952, 953, 954, 955, 1, 2]
        LiveRowMeta.objects.bulk_create([
            LiveRowMeta(extruder_id="EXTR01", ts=LIVE_START + timedelta(seconds=20 * n), period_counter=counter,
                        plc_boot_utc=LIVE_START - timedelta(hours=5 if n < 7 else 0, seconds=1 if n == 2 else 0),
                        quality="suspect" if n == 5 else "good", quality_notes="ts_off_grid" if n == 5 else "",
                        collected_at=LIVE_START, received_at=LIVE_START, collector_id="bsk-edge1")
            for n, counter in enumerate(counters)])
        LiveSimEvent.objects.bulk_create([
            LiveSimEvent(extruder_id="EXTR01", ts=LIVE_START + timedelta(seconds=20 * n), event_id=7, event_type=1,
                         event_signal="die_pressure_bar", event_size=2.5) for n in (1, 2)])
        LiveCollector.objects.create(collector_id="bsk-edge1", version="0.1.0", buffered=0,
                                     last_seen=LIVE_START + timedelta(seconds=200),
                                     last_row_ts=LIVE_START + timedelta(seconds=160))

    def test_summary(self):
        data = self.client.get("/api/historian/summary/").json()
        self.assertEqual((data["table"], data["rows"], data["assets"]), ("extruder_samples", 120, ["EXTR01"]))
        self.assertEqual((data["first_ts"], data["last_ts"]), ("2026-03-01T00:00:00Z", "2026-03-01T01:59:00Z"))
        self.assertEqual(data["states"], [{"state": "RUNNING", "rows": 100}, {"state": "STOPPED", "rows": 20}])
        columns = {c["key"]: c["type"] for c in data["columns"]}
        self.assertNotIn("id", columns)
        self.assertEqual((columns["ts"], columns["die_pressure_bar"], columns["alarm_active"], columns["machine_state"]),
                         ("time", "number", "boolean", "text"))

    def test_samples_are_paged_oldest_first(self):
        data = self.client.get("/api/historian/samples/").json()
        self.assertEqual((data["count"], data["page"], data["page_size"], data["pages"]), (120, 1, 50, 3))
        self.assertEqual(len(data["rows"]), 50)
        self.assertEqual(data["rows"][0]["ts"], "2026-03-01T00:00:00Z")
        self.assertEqual(data["rows"][1]["die_pressure_bar"], 80.5)      # a number, not Decimal text
        self.assertEqual([c["key"] for c in data["columns"]][:2], ["ts", "extruder_id"])

        last = self.client.get("/api/historian/samples/?page=3").json()
        self.assertEqual(len(last["rows"]), 20)
        self.assertEqual(last["rows"][-1]["ts"], "2026-03-01T01:59:00Z")

    def test_filters(self):
        stopped = self.client.get("/api/historian/samples/?state=STOPPED").json()
        self.assertEqual((stopped["count"], stopped["rows"][0]["ts"]), (20, "2026-03-01T01:40:00Z"))

        window = self.client.get("/api/historian/samples/?start=2026-03-01T00:10:00Z&end=2026-03-01T00:19").json()
        self.assertEqual((window["count"], window["pages"]), (10, 1))                 # both ends included; no zone = UTC
        self.assertEqual(window["rows"][-1]["ts"], "2026-03-01T00:19:00Z")

        none = self.client.get("/api/historian/samples/?state=NOPE").json()
        self.assertEqual((none["count"], none["pages"], none["rows"]), (0, 1, []))

    def test_newest_first(self):
        data = self.client.get("/api/historian/samples/?order=desc").json()
        self.assertEqual((data["order"], data["rows"][0]["ts"]), ("desc", "2026-03-01T01:59:00Z"))
        self.assertEqual(self.client.get("/api/historian/samples/").json()["order"], "asc")
        self.assertEqual(self.client.get("/api/historian/samples/?order=up").status_code, 400)

    def test_page_size_is_capped(self):
        self.assertEqual(self.client.get("/api/historian/samples/?page_size=5000").json()["page_size"], 200)

    def test_bad_requests(self):
        for query in ("page=x", "start=yesterday"):
            self.assertEqual(self.client.get(f"/api/historian/samples/?{query}").status_code, 400, query)

    def test_database_down(self):
        with patch("django.db.models.query.QuerySet.count", side_effect=OperationalError("down")):
            response = self.client.get("/api/historian/samples/")
        self.assertEqual((response.status_code, response.json()["error"]), (503, "the historian is not available"))
        with patch("django.db.models.query.QuerySet.aggregate", side_effect=OperationalError("down")):
            self.assertEqual(self.client.get("/api/historian/summary/").status_code, 503)

    # --- the live historian (the PLC's rows, one every 20 s) ---

    def test_march_is_served_unless_asked(self):
        data = self.client.get("/api/historian/summary/").json()
        self.assertEqual((data["source"], data["serving"], data["table"]), ("march", "march", "extruder_samples"))
        self.assertEqual(self.client.get("/api/historian/samples/").json()["source"], "march")
        self.assertEqual(self.client.get("/api/historian/summary/?source=other").status_code, 400)
        self.assertEqual(self.client.get("/api/historian/samples/?source=other").status_code, 400)

    def test_live_source(self):
        data = self.client.get("/api/historian/summary/?source=live").json()
        self.assertEqual((data["source"], data["serving"], data["table"], data["rows"]), ("live", "march", "historian_rows", 9))
        self.assertEqual((data["first_ts"], data["last_ts"]), ("2026-10-11T19:50:20Z", "2026-10-11T19:53:00Z"))
        self.assertEqual(data["states"], [{"state": "RUNNING", "rows": 6}, {"state": "ALARMED", "rows": 3}])
        keys = [c["key"] for c in data["columns"]]
        self.assertEqual(keys[:32], [c["key"] for c in self.client.get("/api/historian/summary/").json()["columns"]])
        self.assertEqual(keys[32:], ["period_counter", "quality", "anomaly"])

        rows = self.client.get("/api/historian/samples/?source=live&state=ALARMED").json()
        self.assertEqual((rows["source"], rows["count"]), ("live", 3))
        self.assertEqual((rows["rows"][0]["ts"], rows["rows"][0]["die_pressure_bar"]), ("2026-10-11T19:52:20Z", 90.6))

    @override_settings(HISTORIAN_SOURCE="live")
    def test_the_setting_switches_what_the_studio_serves(self):
        self.assertEqual(self.client.get("/api/historian/summary/").json()["table"], "historian_rows")
        self.assertEqual(self.client.get("/api/historian/summary/?source=march").json()["rows"], 120)

        snapshot = asset_context.historian_snapshot("EXTR01", ["die_pressure_bar"])
        self.assertEqual((snapshot["latest"]["machine_state"], snapshot["latest"]["die_pressure_bar"]), ("ALARMED", 90.8))
        self.assertEqual(snapshot["running"]["die_pressure_bar"], 90.5)                  # the newest RUNNING row
        self.assertEqual(snapshot["state_since"], LIVE_START + timedelta(seconds=100))   # the last row before ALARMED

        rows = asset_context.historian_rows("EXTR01", ["die_pressure_bar"], LIVE_START, LIVE_START + timedelta(seconds=60), 100)
        self.assertEqual([r["die_pressure_bar"] for r in rows], [90.1, 90.2, 90.3])       # start < ts <= end

    def test_asset_chat_reads_march_by_default(self):
        snapshot = asset_context.historian_snapshot("EXTR01", ["die_pressure_bar"])
        self.assertEqual(snapshot["latest"]["ts"], START + timedelta(minutes=119))


    def test_live_rows_carry_counter_quality_and_anomaly(self):
        data = self.client.get("/api/historian/samples/?source=live").json()
        self.assertEqual([c["key"] for c in data["columns"]][-3:], ["period_counter", "quality", "anomaly"])
        rows = data["rows"]
        self.assertEqual([r["period_counter"] for r in rows], [947, 948, 949, 952, 953, 954, 955, 1, 2])
        self.assertEqual((rows[0]["quality"], rows[5]["quality"]), ("good", "suspect (ts_off_grid)"))
        self.assertEqual([r["anomaly"] for r in rows[:4]],
                         ["", "#7 Step on die_pressure_bar (2.5 σ)", "#7 Step on die_pressure_bar (2.5 σ)", ""])
        self.assertNotIn("quality", self.client.get("/api/historian/samples/").json()["rows"][0])     # March has none

    def test_pipeline_status(self):
        healthy = {"ok": True, "version": "0.1.0", "schema_version": 2}
        with patch("historian.pipeline.ingest_health", return_value=healthy), \
                patch("historian.pipeline.timezone.now", return_value=LIVE_START + timedelta(seconds=200)):
            data = self.client.get("/api/historian/pipeline/").json()
        self.assertEqual((data["state"], data["database"], data["newest_row_age_seconds"]), ("online", True, 40))
        self.assertEqual(data["newest_row"]["machine_state"], "ALARMED")
        self.assertEqual(data["ingest_api"], healthy)
        self.assertEqual(data["window"], {"hours": 24, "rows": 9, "missing_rows": 2, "plc_restarts": 1,
                                          "suspect_rows": 1, "anomaly_rows": 2, "anomalies": 1})
        self.assertEqual((data["collectors"][0]["collector_id"], data["collectors"][0]["buffered"],
                          data["collectors"][0]["last_seen_age_seconds"]), ("bsk-edge1", 0, 0))

        with patch("historian.pipeline.ingest_health", return_value=healthy), \
                patch("historian.pipeline.timezone.now", return_value=LIVE_START + timedelta(minutes=30)):
            self.assertEqual(pipeline.status()["state"], "stale")           # the database answers, nothing arrives
        with patch("historian.pipeline.ingest_health", return_value=healthy), \
                patch("django.db.models.query.QuerySet.first", side_effect=OperationalError("down")):
            down = pipeline.status()
        self.assertEqual((down["state"], down["database"]), ("offline", False))

    def test_health_page_line(self):
        from chat.views import TRACKED_ENDPOINTS, _check_single_endpoint
        ep = next(e for e in TRACKED_ENDPOINTS if e["id"] == "historian-live")
        for state, shown in (("online", "online"), ("stale", "degraded"), ("offline", "offline")):
            with patch("historian.pipeline.status", return_value={"state": state}):
                self.assertEqual(_check_single_endpoint(ep)["status"], shown)

    def test_check_command(self):
        # The nine rows of this class are all 1.0, so the relations fail; the counters miss two periods.
        out = StringIO()
        with patch("historian.management.commands.historian_check.timezone.now",
                   return_value=LIVE_START + timedelta(seconds=200)), self.assertRaises(SystemExit):
            call_command("historian_check", "--hours", "1", stdout=out)
        report = out.getvalue()
        self.assertIn("Rows: 9, from 2026-10-11 19:50:20 to 2026-10-11 19:53:00 UTC", report)
        self.assertIn("FAIL  the period counter advances by 1 (no period lost): 2 missing: 2 before 2026-10-11 19:51:20",
                      report)
        self.assertIn("PLC restarts: 1 at 2026-10-11 19:52:40", report)
        self.assertIn("FAIL  every row marked good by the collector: 1 suspect", report)
        self.assertIn("#7   Step            die_pressure_bar", report)
        self.assertIn("transfer check(s) FAILED", report)
        with self.assertRaises(CommandError):
            call_command("historian_check", "--extruder-id", "EXTR99", stdout=StringIO())


def _good(period: int, state: str = "RUNNING", **changes) -> dict:
    """A row as the PLC computes it: every relation of the brief holds."""
    down = state in check.DOWN_STATES
    actual = 900.0 if down else 9500.0
    row = {
        "ts": LIVE_START + timedelta(seconds=20 * period), "extruder_id": "EXTR01", "batch_id": "EXTR01-20261011-001",
        "recipe_id": "RC-A", "shift_code": "B", "machine_state": state,
        "screw_speed_rpm": 425.0, "screw_torque_pct": 95.0, "motor_power_kw": 350.0, "feeder_rate_actual_kg_hr": 9500.0,
        "barrel_zone_1_temp_c": 90.0, "barrel_zone_2_temp_c": 97.0, "barrel_zone_3_temp_c": 104.0, "head_temp_c": 105.0,
        "steam_flow_kg_hr": 115.0, "water_injection_l_hr": 92.0, "die_pressure_bar": 91.0,
        "product_temp_discharge_c": 106.0, "product_moisture_pct": 9.5, "bulk_density_g_l": 355.0,
        "kibble_diameter_mm": 8.5, "throughput_actual_kg_hr": actual, "throughput_target_kg_hr": 10000.0,
        "good_output_kg": 4.75 if down else 50.14, "scrap_output_kg": 0.25 if down else 2.64,
        "downtime_min": 0.33 if down else 0.0, "availability_pct": 65.0 if down else 95.0,
        "performance_pct": 9.0 if down else 95.0, "quality_pct": 96.0, "oee_pct": 5.62 if down else 86.64,
        "alarm_active": False, "alarm_code": "",
    }
    return {**row, **changes}


def _meta(rows: list[dict], counters=None, **changes) -> dict:
    """row_meta for `rows`, keyed by ts: counters from 947 unless given, delivered a second after the period."""
    counters = counters or range(947, 947 + len(rows))
    return {r["ts"]: {"period_counter": c, "plc_boot_utc": LIVE_START - timedelta(hours=5), "quality": "good",
                      "quality_notes": "", "received_at": r["ts"] + timedelta(seconds=21), **changes}
            for r, c in zip(rows, counters)}


def _event(event_id: int = 7, signal: str = "die_pressure_bar") -> dict:
    return {"event_id": event_id, "event_type": 1, "event_signal": signal, "event_size": 2.5}


class TransferCheckTests(SimpleTestCase):
    def failed(self, result) -> list[str]:
        return [c["name"] for c in result["checks"] if not c["ok"]]

    def test_rows_as_the_plc_computes_them_pass(self):
        rows = [_good(n, "STOPPED" if n >= 9 else "RUNNING") for n in range(12)]
        result = check.transfer(rows, _meta(rows), {})
        self.assertEqual(self.failed(result), [])
        self.assertEqual(len(result["checks"]), 6)
        self.assertIn("PLC restarts: 0; pauses of the generator: 0 (0 s without rows)", result["info"])

    def test_a_lost_period_fails_and_a_pause_or_a_restart_does_not(self):
        rows = [_good(n) for n in (0, 1, 2, 5, 6)]                              # periods 3 and 4 are not there
        lost = check.transfer(rows, _meta(rows, [947, 948, 949, 952, 953]), {})
        self.assertEqual(self.failed(lost), ["the period counter advances by 1 (no period lost)"])
        self.assertIn("2 missing: 2 before 2026-10-11 19:52:00", lost["checks"][2]["detail"])

        paused = check.transfer(rows, _meta(rows), {})                          # the counter held: a pause
        self.assertEqual(self.failed(paused), [])
        self.assertIn("PLC restarts: 0; pauses of the generator: 1 (40 s without rows)", paused["info"])

        restarted = check.transfer(rows, _meta(rows, [947, 948, 949, 1, 2]), {})
        self.assertEqual(self.failed(restarted), [])
        self.assertIn("PLC restarts: 1 at 2026-10-11 19:52:00", restarted["info"][0])

    def test_damaged_rows_fail(self):
        rows = [_good(0), _good(1, oee_pct=80.0), _good(2, throughput_target_kg_hr=9000.0, performance_pct=100.0, oee_pct=91.2),
                _good(3, machine_state="Running"), _good(4, shift_code="A"), _good(5, screw_speed_rpm=425.123),
                _good(6, alarm_active=True), _good(7, barrel_zone_2_temp_c=93.0), _good(8, recipe_id="")]
        detail = check.transfer(rows, _meta(rows), {})["checks"][3]
        self.assertFalse(detail["ok"])
        for fault in ("oee is not availability x performance x quality: 1 rows, first 2026-10-11 19:50:40",
                      "the target is not the row's recipe's: 1 rows", "machine_state is not in the vocabulary: 1 rows",
                      "shift_code does not follow the hour of ts: 1 rows", "a number has more than two decimals: 1 rows",
                      "alarm_code and alarm_active disagree: 1 rows", "scrap is not 5 % (20 % with an alarm): 1 rows",
                      "zone 2 - zone 1 is outside 5 to 10 °C: 1 rows", "zone 3 - zone 2 is outside 5 to 10 °C: 1 rows",
                      "a column is empty: 1 rows"):
            self.assertIn(fault, detail["detail"])

    def test_grid_record_recipe_and_quality(self):
        rows = [_good(0), _good(1, ts=LIVE_START + timedelta(seconds=25)), _good(2, recipe_id="RC-B",
                                                                               throughput_target_kg_hr=9000.0,
                                                                               performance_pct=100.0, oee_pct=91.2)]
        meta = _meta(rows)
        meta[rows[1]["ts"]].update(quality="suspect", quality_notes="ts_off_grid")
        del meta[rows[2]["ts"]]
        result = check.transfer(rows, meta, {})
        self.assertEqual(self.failed(result), [
            "every ts on the 20 s grid (second 00, 20 or 40)",
            "every row has its transfer record (row_meta), and the reverse",
            "a batch keeps one recipe", "every row marked good by the collector"])
        self.assertIn("collector notes: ts_off_grid 1", result["info"])

    def test_a_relation_broken_inside_an_anomaly_is_not_a_failure(self):
        rows = [_good(0), _good(1, barrel_zone_2_temp_c=93.0)]
        result = check.transfer(rows, _meta(rows), {rows[1]["ts"]: _event(signal="barrel_zone_2_temp_c")})
        self.assertEqual(self.failed(result), [])
        self.assertTrue(any(line.startswith("relations broken inside an injected anomaly") for line in result["info"]))

    def test_rows_that_waited_in_the_buffer(self):
        rows = [_good(n) for n in range(6)]
        meta = _meta(rows)
        for n in (1, 2, 3):                                                      # delivered together, after an outage
            meta[rows[n]["ts"]]["received_at"] = LIVE_START + timedelta(minutes=10)
        info = check.transfer(rows, meta, {})["info"]
        self.assertIn("waited in the collector's buffer: 3 rows, 2026-10-11 19:50:40 to 2026-10-11 19:51:20, "
                      "delivered 2026-10-11 20:00:20 (the first after 9 min)", info)

    def test_likeness(self):
        alarm = {"alarm_active": True, "alarm_code": "HI_MOISTURE", "good_output_kg": 42.22, "scrap_output_kg": 10.56}
        rows = ([_good(n) for n in range(3)] + [_good(n, **alarm) for n in range(3, 6)]      # one alarm, three rows
                + [_good(6, screw_speed_rpm=450.0), _good(7, die_pressure_bar=120.0), _good(8)]
                + [_good(n, "STOPPED") for n in range(9, 12)])
        like = check.likeness(rows, {rows[7]["ts"]: _event()})
        running, stopped = like["states"][0], like["states"][1]
        self.assertEqual((running["rows"], running["share_pct"], running["blocks"], running["block_minutes"]), (9, 75.0, 1, 3.0))
        self.assertEqual((stopped["rows"], stopped["oee_pct"], like["states"][2]["rows"]), (3, 5.62, 0))
        self.assertAlmostEqual(running["alarm_rows_pct"], 100 / 3)
        self.assertEqual((like["alarm_starts"], like["alarm_starts_per_hour"]), (1, 15.0))
        # Row 7 belongs to an injected anomaly, so only row 6 counts as outside its normal range.
        self.assertEqual((like["running_rows"], like["outside_normal_pct"]), (8, 12.5))
        self.assertEqual(like["outside_by_signal"], {"screw_speed_rpm": 12.5})

    def test_anomaly_record(self):
        rows = [_good(n, "STOPPED" if n == 5 else "RUNNING") for n in range(12)]
        events = {rows[n]["ts"]: _event(7) for n in (1, 2, 3)}
        events.update({rows[n]["ts"]: _event(8, "screw_speed_rpm") for n in (5, 6)})
        found = check.anomalies(rows, events)
        self.assertEqual([(e["event_id"], e["signal"], e["rows"], e["start"]) for e in found["events"]],
                         [(7, "die_pressure_bar", 3, rows[1]["ts"]), (8, "screw_speed_rpm", 2, rows[5]["ts"])])
        self.assertEqual((found["rows"], found["by_type"], found["per_day"]), (5, {"Step": 2}, 720.0))
        self.assertEqual([(c["ok"], c["detail"]) for c in found["checks"]],
                         [(False, "1 rows in another state"), (False, "2026-10-11 19:52:00")])
        self.assertEqual(check.anomalies(rows, {})["events"], [])


class RouterTests(TestCase):
    def test_live_models_go_to_the_live_database_and_are_never_migrated(self):
        from neo_llm_api.db_routers import HistorianRouter
        from .models import LiveCollector, LiveRowMeta, LiveSimEvent
        router = HistorianRouter()
        for model in (LiveRow, LiveRowMeta, LiveSimEvent, LiveCollector):
            self.assertEqual((router.db_for_read(model), router.db_for_write(model)), ("historian_live", "historian_live"))
            self.assertFalse(model._meta.managed)
        self.assertEqual(router.db_for_read(ExtruderSample), "historian")
        self.assertFalse(router.allow_migrate("historian_live", "historian"))
        self.assertFalse(router.allow_migrate("historian_live", "chat"))
        self.assertTrue(router.allow_migrate("historian", "historian"))

