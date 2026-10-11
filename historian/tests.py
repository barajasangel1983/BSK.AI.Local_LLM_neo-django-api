"""The historian's read-only view (Health page → Historian Postgres).

The Postgres user cannot create a test database, so the samples table is built in the test
SQLite database and the historian router is pointed at it for the length of the tests.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.db import OperationalError, connection
from django.test import TestCase

from .models import ExtruderSample

START = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _sample(minute: int, state: str = "RUNNING") -> ExtruderSample:
    numbers = {f.name: 1 for f in ExtruderSample._meta.fields if f.get_internal_type() == "DecimalField"}
    numbers["die_pressure_bar"] = 80 + minute / 2
    return ExtruderSample(ts=START + timedelta(minutes=minute), extruder_id="EXTR01", batch_id="B1", recipe_id="R1",
                          shift_code="A", machine_state=state, **numbers)


class HistorianViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        with connection.schema_editor() as editor:
            editor.create_model(ExtruderSample)
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

    @classmethod
    def setUpTestData(cls):
        ExtruderSample.objects.bulk_create(
            [_sample(m, "STOPPED" if m >= 100 else "RUNNING") for m in range(120)])

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
