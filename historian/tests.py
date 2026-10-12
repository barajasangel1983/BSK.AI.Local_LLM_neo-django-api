"""The historian's read-only view (Health page → Historian Postgres).

The Postgres user cannot create a test database, so the samples table is built in the test
SQLite database and the historian router is pointed at it for the length of the tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.db import OperationalError, connection
from django.test import TestCase, override_settings

from chat import asset_context

from .models import ExtruderSample, LiveRow

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
            editor.create_model(ExtruderSample)
            editor.create_model(LiveRow)
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
            editor.delete_model(ExtruderSample)
            editor.delete_model(LiveRow)

    @classmethod
    def setUpTestData(cls):
        ExtruderSample.objects.bulk_create(
            [_sample(m, "STOPPED" if m >= 100 else "RUNNING") for m in range(120)])
        LiveRow.objects.bulk_create([_live(n, "ALARMED" if n >= 6 else "RUNNING") for n in range(9)])

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
        self.assertEqual(keys, [c["key"] for c in self.client.get("/api/historian/summary/").json()["columns"]])
        self.assertEqual(len(keys), 32)

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

