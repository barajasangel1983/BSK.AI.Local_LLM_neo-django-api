"""The plant calendar (P18a0): validation, the shift and day helpers, and the API."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from monitor import plant_calendar
from monitor.models import PlantCalendar
from monitor.plant_calendar import Calendar, validate

CHI = ZoneInfo("America/Chicago")
THREE = [{"name": "A", "start": "06:00", "end": "14:00"},
         {"name": "B", "start": "14:00", "end": "22:00"},
         {"name": "C", "start": "22:00", "end": "06:00"}]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]
PLANT = "bsk:plant:bsklab-demo"


def at(*args) -> datetime:
    """A local Chicago time, as UTC."""
    return datetime(*args, tzinfo=CHI).astimezone(timezone.utc)


def calendar(shifts=THREE, days=WEEKDAYS, tz="America/Chicago") -> Calendar:
    return Calendar.from_dict(PLANT, {"time_zone": tz, "working_days": days, "shifts": shifts})


class ValidateTests(SimpleTestCase):
    def check(self, **changes):
        data = {"time_zone": "America/Chicago", "working_days": WEEKDAYS, "shifts": THREE, **changes}
        return validate(data)

    def test_a_valid_calendar(self):
        clean, errors = self.check(working_days=["FRI", "mon"])
        self.assertEqual(errors, {})
        self.assertEqual(clean["working_days"], ["mon", "fri"])   # lower case, week order

    def test_time_zone(self):
        self.assertIn("time_zone", self.check(time_zone="Mars/Olympus")[1])
        self.assertIn("time_zone", self.check(time_zone="")[1])

    def test_working_days(self):
        self.assertIn("working_days", self.check(working_days=[])[1])
        self.assertIn("working_days", self.check(working_days=["monday"])[1])

    def test_times(self):
        errors = self.check(shifts=[{"name": "A", "start": "6:00", "end": "24:00"}])[1]
        self.assertIn("shifts.0.start", errors)
        self.assertIn("shifts.0.end", errors)
        self.assertIn("shifts.0.end", self.check(shifts=[{"name": "A", "start": "06:00", "end": "06:00"}])[1])

    def test_names(self):
        errors = self.check(shifts=[{"name": "A", "start": "06:00", "end": "10:00"},
                                    {"name": "a", "start": "10:00", "end": "14:00"}])[1]
        self.assertIn("shifts.1.name", errors)
        self.assertIn("shifts.0.name", self.check(shifts=[{"name": " ", "start": "06:00", "end": "10:00"}])[1])

    def test_overlap(self):
        errors = self.check(shifts=[{"name": "A", "start": "06:00", "end": "14:00"},
                                    {"name": "B", "start": "13:00", "end": "22:00"}])[1]
        self.assertEqual(errors, {"shifts.1.start": "overlaps shift A"})

    def test_overlap_across_midnight(self):
        errors = self.check(shifts=[{"name": "C", "start": "22:00", "end": "06:00"},
                                    {"name": "A", "start": "05:00", "end": "13:00"}])[1]
        self.assertIn("shifts.1.start", errors)

    def test_gaps_and_touching_shifts_are_allowed(self):
        _, errors = self.check(shifts=[{"name": "Day", "start": "07:00", "end": "15:30"},
                                       {"name": "Late", "start": "15:30", "end": "23:00"}])
        self.assertEqual(errors, {})

    def test_at_least_one_shift(self):
        self.assertIn("shifts", self.check(shifts=[])[1])


class HelperTests(SimpleTestCase):
    """Tue 2026-10-13 … in Chicago (CDT, UTC-5); Mon-Fri working; A 06-14, B 14-22, C 22-06."""

    def test_shift_at(self):
        cal = calendar()
        self.assertEqual(cal.shift_at(at(2026, 10, 13, 10, 0)).name, "A")
        late = cal.shift_at(at(2026, 10, 13, 23, 0))
        self.assertEqual((late.name, late.day.isoformat()), ("C", "2026-10-13"))
        early = cal.shift_at(at(2026, 10, 14, 3, 0))             # the C that started on Tuesday
        self.assertEqual((early.name, early.day.isoformat()), ("C", "2026-10-13"))
        self.assertEqual(early.start, at(2026, 10, 13, 22, 0))
        self.assertEqual(early.end, at(2026, 10, 14, 6, 0))

    def test_between_shifts(self):
        cal = calendar(shifts=[{"name": "Day", "start": "07:00", "end": "15:00"}])
        self.assertIsNone(cal.shift_at(at(2026, 10, 13, 18, 0)))

    def test_last_shift(self):
        cal = calendar()
        last = cal.last_shift(at(2026, 10, 14, 7, 0))
        self.assertEqual((last.name, last.day.isoformat()), ("C", "2026-10-13"))
        b = cal.last_shift(at(2026, 10, 14, 7, 0), name="b")
        self.assertEqual((b.name, b.start), ("B", at(2026, 10, 13, 14, 0)))

    def test_days(self):
        cal = calendar()
        self.assertEqual(cal.day(at(2026, 10, 14, 3, 0)), (at(2026, 10, 14, 0, 0), at(2026, 10, 15, 0, 0)))
        self.assertEqual(cal.day(at(2026, 10, 14, 3, 0), -1)[0], datetime(2026, 10, 13, 5, 0, tzinfo=timezone.utc))

    def test_last_working_day_over_a_weekend(self):
        cal = calendar()
        start, end = cal.last_working_day(at(2026, 10, 19, 9, 0))   # Monday
        self.assertEqual((start, end), (at(2026, 10, 16, 0, 0), at(2026, 10, 17, 0, 0)))  # Friday

    def test_is_working(self):
        cal = calendar()
        self.assertTrue(cal.is_working(at(2026, 10, 16, 23, 0)))   # Friday's C
        self.assertTrue(cal.is_working(at(2026, 10, 17, 3, 0)))    # still Friday's C, on Saturday morning
        self.assertFalse(cal.is_working(at(2026, 10, 17, 10, 0)))  # Saturday's A: not a working day

    def test_daylight_saving_end(self):
        # Sun 2026-11-01 02:00 CDT -> 01:00 CST: Saturday's C lasts 9 real hours
        c = calendar().shift_at(at(2026, 11, 1, 4, 0))
        self.assertEqual(c.day.isoformat(), "2026-10-31")
        self.assertEqual(c.end - c.start, timedelta(hours=9))

    def test_occurrences(self):
        cal = calendar()
        found = cal.occurrences(*cal.day(at(2026, 10, 13, 12, 0)))
        self.assertEqual([(o.name, o.day.isoformat()) for o in found],
                         [("C", "2026-10-12"), ("A", "2026-10-13"), ("B", "2026-10-13"), ("C", "2026-10-13")])


class StoredTests(TestCase):
    def test_for_asset_without_graph_or_calendar(self):
        with patch("monitor.plant_calendar.plant_of", side_effect=RuntimeError("graph down")):
            self.assertIsNone(plant_calendar.for_asset("bsk:asset:EXTR01"))
        with patch("monitor.plant_calendar.plant_of", return_value=PLANT):
            self.assertIsNone(plant_calendar.for_asset("bsk:asset:EXTR01"))
            PlantCalendar.objects.create(plant_id=PLANT, time_zone="America/Chicago",
                                         working_days=WEEKDAYS, shifts=THREE)
            self.assertEqual(plant_calendar.for_asset("bsk:asset:EXTR01").tz, CHI)


@patch("monitor.plant_calendar.plants", return_value=[{"id": PLANT, "name": "BSKLAB Demo Plant"}])
class ApiTests(TestCase):
    url = "/api/monitor/calendar/"

    def setUp(self):
        self.client = APIClient()

    def test_get_without_a_calendar(self, _):
        body = self.client.get(self.url).json()
        self.assertEqual(body["plant_id"], PLANT)
        self.assertIsNone(body["calendar"])
        self.assertIsNone(body["now"])

    def test_put_and_get(self, _):
        data = {"plant_id": PLANT, "time_zone": "America/Chicago", "working_days": WEEKDAYS, "shifts": THREE,
                "updated_by": "Admin User"}
        r = self.client.put(self.url, data, format="json")
        self.assertEqual(r.status_code, 200)
        body = self.client.get(self.url).json()
        self.assertEqual(body["calendar"]["shifts"], THREE)
        self.assertEqual(body["calendar"]["updated_by"], "Admin User")
        self.assertIn("working", body["now"])
        r = self.client.put(self.url, {**data, "working_days": ["mon"]}, format="json")   # updates, not adds
        self.assertEqual(PlantCalendar.objects.get().working_days, ["mon"])

    def test_put_with_errors(self, _):
        r = self.client.put(self.url, {"plant_id": PLANT, "time_zone": "Nowhere", "working_days": [],
                                       "shifts": THREE[:1] + [{"name": "B", "start": "13:00", "end": "22:00"}]},
                            format="json")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(set(r.json()["errors"]), {"time_zone", "working_days", "shifts.1.start"})
        self.assertFalse(PlantCalendar.objects.exists())

    def test_put_unknown_plant(self, _):
        r = self.client.put(self.url, {"plant_id": "bsk:plant:other", "time_zone": "UTC",
                                       "working_days": WEEKDAYS, "shifts": THREE}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("plant_id", r.json()["errors"])
