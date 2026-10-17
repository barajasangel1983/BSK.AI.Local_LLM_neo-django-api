"""The plant calendar (P18a0): shifts, working days and time zone, and the windows they give.

    validate(data)              -> (clean, errors): errors is {field: message}, empty when valid
    Calendar.from_model(row)    the helpers below, on one plant's calendar
        .local(ts)                      ts in the plant's time zone
        .shift_at(ts)                   the shift occurrence ts falls in, or None (between shifts)
        .this_shift(ts) / .last_shift(ts, name=None)
        .day(ts, offset=0)              the local day (0 = today, -1 = yesterday) as a UTC window
        .last_working_day(ts)           the most recent working day before ts's day
        .is_working(ts)                 inside a shift that starts on a working day
        .occurrences(start, end)        the shift occurrences overlapping a window
    for_plant(plant_id) / for_asset(asset_id)  the stored calendar, or None (callers fall back: UTC days, 8 h)

A shift that crosses midnight belongs to the day it starts: C 22:00-06:00 on Monday runs into Tuesday morning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
MAX_SHIFTS = 6
DAY_MIN = 24 * 60


def _minutes(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def _span(start: int, end: int) -> set[int]:
    """The minutes of the day a shift covers (wrapping past midnight)."""
    length = (end - start) % DAY_MIN
    return {(start + i) % DAY_MIN for i in range(length)}


def validate(data: dict) -> tuple[dict, dict]:
    errors: dict[str, str] = {}
    tz = str(data.get("time_zone") or "").strip()
    if not tz:
        errors["time_zone"] = "choose a time zone"
    elif tz not in available_timezones():
        errors["time_zone"] = f"unknown time zone {tz!r}"

    days = data.get("working_days")
    if not isinstance(days, list) or not all(isinstance(d, str) and d.lower() in DAYS for d in days):
        errors["working_days"] = "working days must be a list of mon … sun"
        days = []
    days = [d for d in DAYS if d in {x.lower() for x in days}]
    if not days and "working_days" not in errors:
        errors["working_days"] = "tick at least one working day"

    shifts = data.get("shifts")
    clean_shifts = []
    if not isinstance(shifts, list) or not shifts:
        errors["shifts"] = "add at least one shift"
        shifts = []
    elif len(shifts) > MAX_SHIFTS:
        errors["shifts"] = f"at most {MAX_SHIFTS} shifts"
    names: set[str] = set()
    for i, s in enumerate(shifts):
        if not isinstance(s, dict):
            errors[f"shifts.{i}"] = "a shift needs a name, a start and an end"
            continue
        name = str(s.get("name") or "").strip()
        start, end = str(s.get("start") or "").strip(), str(s.get("end") or "").strip()
        if not name:
            errors[f"shifts.{i}.name"] = "name the shift"
        elif len(name) > 20:
            errors[f"shifts.{i}.name"] = "at most 20 characters"
        elif name.lower() in names:
            errors[f"shifts.{i}.name"] = f"another shift is already called {name}"
        names.add(name.lower())
        for key, value in (("start", start), ("end", end)):
            if not TIME_RE.match(value):
                errors[f"shifts.{i}.{key}"] = "use HH:MM, e.g. 06:00"
        if TIME_RE.match(start) and TIME_RE.match(end) and start == end:
            errors[f"shifts.{i}.end"] = "the end must differ from the start"
        clean_shifts.append({"name": name, "start": start, "end": end})

    if not any(k.startswith("shifts") for k in errors):
        spans = [_span(_minutes(s["start"]), _minutes(s["end"])) for s in clean_shifts]
        for i in range(len(spans)):
            for j in range(i):
                if spans[i] & spans[j]:
                    errors[f"shifts.{i}.start"] = f"overlaps shift {clean_shifts[j]['name']}"
                    break

    clean = {"time_zone": tz, "working_days": days, "shifts": clean_shifts}
    return clean, errors


@dataclass(frozen=True)
class Shift:
    name: str
    start: int   # minutes after local midnight
    end: int

    @property
    def length(self) -> timedelta:
        return timedelta(minutes=(self.end - self.start) % DAY_MIN)

    @property
    def crosses_midnight(self) -> bool:
        return self.end < self.start


@dataclass(frozen=True)
class Occurrence:
    """One shift on one day: `day` is the local day it starts; start and end are UTC."""
    name: str
    day: date
    start: datetime
    end: datetime
    scheduled: bool   # the day it starts is a working day

    def as_dict(self) -> dict:
        return {"name": self.name, "day": self.day.isoformat(), "start": self.start.isoformat(),
                "end": self.end.isoformat(), "scheduled": self.scheduled}


@dataclass(frozen=True)
class Calendar:
    plant_id: str
    tz: ZoneInfo
    working_days: frozenset[int]   # 0 = Monday
    shifts: tuple[Shift, ...]

    @classmethod
    def from_dict(cls, plant_id: str, data: dict) -> "Calendar":
        return cls(
            plant_id=plant_id,
            tz=ZoneInfo(data["time_zone"]),
            working_days=frozenset(DAYS.index(d) for d in data["working_days"]),
            shifts=tuple(Shift(s["name"], _minutes(s["start"]), _minutes(s["end"])) for s in data["shifts"]),
        )

    @classmethod
    def from_model(cls, row) -> "Calendar":
        return cls.from_dict(row.plant_id, {"time_zone": row.time_zone, "working_days": row.working_days,
                                            "shifts": row.shifts})

    # --- helpers ------------------------------------------------------------

    def local(self, ts: datetime) -> datetime:
        return ts.astimezone(self.tz)

    def _at(self, day: date, minutes: int) -> datetime:
        local = datetime.combine(day, time(minutes // 60, minutes % 60), tzinfo=self.tz)
        return local.astimezone(timezone.utc)

    def _occurrence(self, shift: Shift, day: date) -> Occurrence:
        start = self._at(day, shift.start)
        end = self._at(day + timedelta(days=1) if shift.crosses_midnight else day, shift.end)
        return Occurrence(shift.name, day, start, end, day.weekday() in self.working_days)

    def _around(self, ts: datetime, days_back: int, days_ahead: int = 0) -> list[Occurrence]:
        today = self.local(ts).date()
        found = [self._occurrence(s, today + timedelta(days=d))
                 for d in range(-days_back, days_ahead + 1) for s in self.shifts]
        return sorted(found, key=lambda o: o.start)

    def shift_at(self, ts: datetime) -> Occurrence | None:
        for o in self._around(ts, 1):
            if o.start <= ts < o.end:
                return o
        return None

    def this_shift(self, ts: datetime) -> Occurrence | None:
        return self.shift_at(ts)

    def last_shift(self, ts: datetime, name: str | None = None) -> Occurrence | None:
        """The most recent shift occurrence that has ended by ts (of that name, if given)."""
        ended = [o for o in self._around(ts, 8) if o.end <= ts
                 and (name is None or o.name.lower() == name.lower())]
        return ended[-1] if ended else None

    def day(self, ts: datetime, offset: int = 0) -> tuple[datetime, datetime]:
        d = self.local(ts).date() + timedelta(days=offset)
        return self._at(d, 0), self._at(d + timedelta(days=1), 0)

    def last_working_day(self, ts: datetime) -> tuple[datetime, datetime]:
        d = self.local(ts).date() - timedelta(days=1)
        for _ in range(7):
            if d.weekday() in self.working_days:
                break
            d -= timedelta(days=1)
        return self._at(d, 0), self._at(d + timedelta(days=1), 0)

    def is_working(self, ts: datetime) -> bool:
        o = self.shift_at(ts)
        return bool(o and o.scheduled)

    def occurrences(self, start: datetime, end: datetime) -> list[Occurrence]:
        days = (end - start).days + 2
        found = self._around(end, days)
        return [o for o in found if o.end > start and o.start < end]


# --- stored calendars -------------------------------------------------------

def for_plant(plant_id: str) -> Calendar | None:
    from .models import PlantCalendar
    row = PlantCalendar.objects.filter(plant_id=plant_id).first()
    if row is None:
        return None
    try:
        return Calendar.from_model(row)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return None


def plant_of(asset_id: str) -> str | None:
    """The Plant above an asset in the graph (Plant → Area → Line → Asset)."""
    from context_graph import services
    with services.session() as s:
        chain = s.execute_read(services._hierarchy, asset_id)
    return next((n["id"] for n in chain if n["label"] == "Plant"), None)


def for_asset(asset_id: str) -> Calendar | None:
    try:
        plant_id = plant_of(asset_id)
    except Exception:        # the graph is down: no calendar, the caller falls back
        return None
    return for_plant(plant_id) if plant_id else None


def plants() -> list[dict]:
    from context_graph import repository, services
    with services.session() as s:
        items, _ = s.execute_read(repository.list_nodes, "Plant", None, 100, 0)
    return [{"id": p["id"], "name": p["name"]} for p in items]
