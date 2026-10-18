"""Historian windows in the asset chat (P18a): "what was the OEE in the last 24 h?"

A data question (a measure, and a time or none) gets the numbers of its window computed from the historian's
rows with Home's own stats (monitor.agent.hour_stats) and put in the prompt as one block, cited [H]. One model
call, as before; the model never writes a query. The live historian is only read.

    is_data_question(question)                    -> bool
    parse(question, now, cal)                     -> Window (no time in the question: the last 8 h, as Home)
    compute(asset_id, question, budget, real)     -> Result | None   (the block, its citation, a log text)

Windows use the plant calendar (P18a0) for "today", "yesterday", shifts and working days; without one, days are
UTC and "this shift" is the last 8 h, and the block says so.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from monitor import agent, plant_calendar
from monitor.plant_calendar import Calendar

logger = logging.getLogger("chat")

DEFAULT_HOURS = agent.WINDOW_HOURS      # a measure without a time: Home's window
ROWS_HOURS = 24                         # up to this every row is read; longer windows use buckets
MAX_DAYS = 31

# Measures that make a data question on their own …
MEASURE = re.compile(
    r"\b(oee|availability|performance|quality rate|throughput|output|production|produced|downtime|down time|"
    r"stopp(?:ed|ages?)|stops|run ?time|uptime|running time|time running|out of range|scrap|rejects?|good output|"
    r"efficiency|utili[sz]ation|machine states?|how (?:did|was|has) (?:it|the line|the machine|the extruder|extr01)|"
    r"kg/h|kg per hour)\b")
# … and ones that do only with a time ("which alarms does the drive have" is a graph question).
WITH_TIME = re.compile(r"\b(alarms?|faults?|warnings?|quality|running|stopped|state|temperatures?|pressure|speed|"
                       r"moisture|density|flow|torque|power|how (?:did|was|were|has|is)|go|going|perform\w*|times|how many|how often|"
                       r"start(?:ed|s)?|occur\w*|happen\w*)\b")
# Questions about a procedure or a meaning, even when they name a measure or a time.
NOT_DATA = re.compile(r"\b(mean|means|meaning|stands? for|how (?:do|to|can|should) (?:i|we|you)|procedure|steps?|"
                      r"fix|replace|reset|troubleshoot\w*|interval|maintenance|manual|datasheet)\b")
CODE = re.compile(r"\b(?:[A-F]?\d{4}|[A-F][0-9A-F]{3}|[A-Z]\d{4,5})\b")

UNITS = {"m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
         "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
         "d": 1440, "day": 1440, "days": 1440, "w": 10080, "wk": 10080, "week": 10080, "weeks": 10080}
DURATION = re.compile(r"\b(?:last|past|previous|over the last|in the last|over the past)\s+(\d+(?:\.\d+)?)\s*"
                      r"(minutes?|mins?|m|hours?|hrs?|hr|h|days?|d|weeks?|wk|w)\b")
ONE = re.compile(r"\b(?:last|past|previous)\s+(minute|hour|day|24 ?h)\b")
# "shift B" or "B shift"; the lookahead leaves "shift" to match again ("did shift c": "did", then "c").
NAMED_SHIFT = re.compile(r"\b(?:shift\s+([a-z0-9]{1,3})|([a-z0-9]{1,3})\s+(?=shift\b))")


def _time_phrase(q: str) -> bool:
    return bool(DURATION.search(q) or ONE.search(q) or NAMED_SHIFT.search(q) or re.search(
        r"\b(today|yesterday|tonight|this (?:shift|week|month)|current shift|last (?:shift|week|working day)|"
        r"previous (?:shift|week|working day)|past week)\b", q))


def is_data_question(question: str) -> bool:
    q = question.lower()
    if NOT_DATA.search(q):
        return False
    timed = _time_phrase(q)
    if CODE.search(question) and not timed:
        return False                         # "alarm F30004?" asks what it is
    return bool(MEASURE.search(q) or (timed and WITH_TIME.search(q)))


@dataclass
class Window:
    label: str                 # "last 24 h", "yesterday", "shift B (Fri 2026-10-16)"
    start: datetime            # UTC
    end: datetime
    notes: list[str] = field(default_factory=list)

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


def _utc_day(ts: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    d = ts.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=offset)
    return d, d + timedelta(days=1)


def _shift_label(o, cal: Calendar) -> str:
    return f"shift {o.name} ({o.day:%a %Y-%m-%d}, {cal.local(o.start):%H:%M}–{cal.local(o.end):%H:%M})"


def parse(question: str, now: datetime, cal: Calendar | None) -> Window:
    """The window a question asks about, ending at the latest `now` (a window still running ends now)."""
    q = question.lower()
    no_cal = "no plant calendar is saved, so days are UTC"
    m = DURATION.search(q)
    if m:
        minutes = float(m.group(1)) * UNITS[m.group(2)]
        minutes = min(minutes, MAX_DAYS * 1440)
        n = m.group(1).rstrip("0").rstrip(".") if "." in m.group(1) else m.group(1)
        return Window(f"last {n} {m.group(2)}", now - timedelta(minutes=minutes), now)
    m = ONE.search(q)
    if m:
        minutes = {"minute": 1, "hour": 60}.get(m.group(1), 1440)
        return Window(f"last {'24 h' if minutes == 1440 else m.group(1)}", now - timedelta(minutes=minutes), now)

    names = [(m.group(1) or m.group(2)) for m in NAMED_SHIFT.finditer(q)]     # "did shift b": "did" is not a shift
    known = {s.name.lower() for s in cal.shifts} if cal else set()
    named = next((n.upper() for n in names if n in known), None)
    if named and cal:
        name = named
        if "yesterday" in q or "today" in q:
            day = cal.local(now).date() - timedelta(days=1 if "yesterday" in q else 0)
            found = [o for o in cal.occurrences(now - timedelta(days=3), now + timedelta(days=1))
                     if o.name.lower() == name.lower() and o.day == day]
            o = found[0] if found else None
        else:
            current = cal.shift_at(now)
            o = current if current and current.name.lower() == name.lower() else cal.last_shift(now, name)
        if o:
            end = min(o.end, now)
            w = Window(_shift_label(o, cal), o.start, end)
            if end < o.end:
                w.notes.append(f"shift {o.name} is still running: the numbers are up to now")
            return w

    if re.search(r"\b(this shift|current shift|the shift)\b", q) or (names and not cal and re.search(r"\bshift [a-z0-9]\b", q)):
        if cal and (o := cal.shift_at(now)):
            return Window(_shift_label(o, cal), o.start, now, ["the shift is still running: the numbers are up to now"])
        note = "no plant calendar is saved, so a shift is taken as the last 8 h" if not cal else "no shift is running now; the last 8 h"
        return Window("last 8 h", now - timedelta(hours=8), now, [note])
    if re.search(r"\b(last|previous) shift\b", q):
        if cal and (o := cal.last_shift(now)):
            return Window(_shift_label(o, cal), o.start, o.end)
        return Window("last 8 h", now - timedelta(hours=8), now, ["no plant calendar is saved, so a shift is taken as the last 8 h"])
    if re.search(r"\b(last|previous) working day\b", q):
        if cal:
            start, end = cal.last_working_day(now)
            return Window(f"last working day ({cal.local(start):%a %Y-%m-%d})", start, end)
        start, end = _utc_day(now, -1)
        return Window("yesterday (UTC)", start, end, [no_cal + "; the last working day is taken as yesterday"])
    if "yesterday" in q:
        start, end = cal.day(now, -1) if cal else _utc_day(now, -1)
        w = Window(f"yesterday ({(cal.local(start) if cal else start):%a %Y-%m-%d})", start, end)
        if not cal:
            w.notes.append(no_cal)
        return w
    if re.search(r"\b(today|tonight)\b", q):
        start, _ = cal.day(now) if cal else _utc_day(now)
        w = Window(f"today ({(cal.local(start) if cal else start):%a %Y-%m-%d}, up to now)", start, now)
        if not cal:
            w.notes.append(no_cal)
        return w
    if re.search(r"\b(last|previous) week\b", q) or re.search(r"\bthis week\b", q):
        local = cal.local(now) if cal else now.astimezone(timezone.utc)
        monday = (cal.day(now, -local.weekday())[0] if cal else _utc_day(now, -local.weekday())[0])
        if "this week" in q:
            w = Window(f"this week (since Mon {(cal.local(monday) if cal else monday):%Y-%m-%d})", monday, now)
        else:
            start = (cal.day(monday, -7)[0] if cal else monday - timedelta(days=7))
            w = Window(f"last week (Mon {(cal.local(start) if cal else start):%Y-%m-%d} – Sun)", start, monday)
        if not cal:
            w.notes.append(no_cal)
        return w
    if re.search(r"\bpast week\b", q):
        return Window("last 7 days", now - timedelta(days=7), now)
    if re.search(r"\bthis month\b", q):
        local = cal.local(now) if cal else now.astimezone(timezone.utc)
        first = cal.day(now, -(local.day - 1))[0] if cal else _utc_day(now, -(local.day - 1))[0]
        return Window(f"this month (since {(cal.local(first) if cal else first):%Y-%m-%d})", first, now)
    return Window(f"last {DEFAULT_HOURS} h", now - timedelta(hours=DEFAULT_HOURS), now,
                  [f"the question gives no time, so this is the last {DEFAULT_HOURS} h, as on Home"])


# --- reading and the numbers --------------------------------------------------------------

def read(asset_id: str, w: Window) -> tuple[list[dict], list[dict], int, str, bool]:
    """(rows, signals, period_seconds, source, bucketed) for the window."""
    if w.end - w.start <= timedelta(hours=ROWS_HOURS):
        h = agent.history(asset_id, w.start, w.end)
        return agent._rows(h), h["signals"], agent._period(h), h.get("source") or "", False
    from context_service import service
    bucket = 5 if w.end - w.start <= timedelta(days=7) else 15
    piece = timedelta(minutes=bucket * 1200)                     # 1,200 points a request (the limit is 1,500)
    ts: list = []
    series: dict = {}
    signals, source, start = [], "", w.start
    while start < w.end:
        until = min(start + piece, w.end)
        part = service.signal_history(asset_id, start.isoformat(), until.isoformat(), bucket_minutes=bucket)
        signals, source = part["signals"], part.get("source") or source
        ts += part["ts"]
        for k, v in part["series"].items():
            series.setdefault(k, []).extend(v)
        start = until
    return agent._rows({"ts": ts, "series": series}), signals, bucket * 60, source, True


def _pct(v) -> str:
    return "–" if v is None else f"{v:.1f} %"


def _min(v) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) and float(v).is_integer() else f"{v:,}"


def _local(ts: datetime, cal: Calendar | None) -> str:
    return f"{cal.local(ts):%a %Y-%m-%d %H:%M}" if cal else f"{ts.astimezone(timezone.utc):%a %Y-%m-%d %H:%M} UTC"


def _parts(rows, signals, period, cal, w: Window) -> dict[str, list[tuple[str, dict]]]:
    """The window's breakdowns: by shift (with a calendar), by hour (up to a day), by day (longer)."""
    out: dict[str, list[tuple[str, dict]]] = {}
    if cal and w.minutes >= 4 * 60:
        shifts = []
        for o in cal.occurrences(w.start, w.end):
            part = [r for r in rows if max(o.start, w.start) < r["ts"] <= min(o.end, w.end)]
            if part:
                shifts.append((f"{o.name} {o.day:%a}", agent.hour_stats(part, signals, period)))
        if len(shifts) > 1:
            out["shift"] = shifts
    if w.minutes <= 24 * 60 and w.minutes > 90:
        hours, begin = [], w.start.replace(minute=0, second=0, microsecond=0)
        while begin < w.end:
            part = [r for r in rows if max(begin, w.start) < r["ts"] <= begin + timedelta(hours=1)]
            if part:
                hours.append((f"{(cal.local(begin) if cal else begin):%H:%M}", agent.hour_stats(part, signals, period)))
            begin += timedelta(hours=1)
        out["hour"] = hours
    elif w.minutes > 24 * 60:
        days, day = [], (cal.day(w.start)[0] if cal else _utc_day(w.start)[0])
        while day < w.end:
            nxt = cal.day(day, 1)[0] if cal else day + timedelta(days=1)
            part = [r for r in rows if max(day, w.start) < r["ts"] <= min(nxt, w.end)]
            if part:
                days.append((f"{(cal.local(day) if cal else day):%a %m-%d}", agent.hour_stats(part, signals, period)))
            day = nxt
        out["day"] = days
    return out


def block(name: str, w: Window, rows, signals, period, source, bucketed, clk: dict, cal: Calendar | None,
          budget: int) -> str:
    total = agent.hour_stats(rows, signals, period, w.minutes)
    where = (f"{cal.tz.key}, plant calendar" if cal else "UTC")
    kind = "PLC rows every 20 s" if clk.get("live") else "simulated data"
    head = (f"HISTORIAN WINDOW for {name}, {w.label}: {_local(w.start, cal)} – {_local(w.end, cal)} ({where}); "
            f"source {source or 'historian'}, {kind}"
            + (f", read as {period // 60}-minute means, so minutes are approximate" if bucketed else "") + ".")
    if clk.get("replay"):
        head += " Recorded data played back: 'now' is the replay time."
    lines = [head]
    lines += [f"- Note: {n}." for n in w.notes]
    other = round(total["not_running_min"] - total["downtime_min"], 1)
    lines.append(f"- Of {_min(round(w.minutes))} min: running {_min(total['running_min'])} min, not running "
                 f"{_min(total['not_running_min'])} min (downtime {_min(total['downtime_min'])} min, other stops "
                 f"{_min(max(other, 0))} min), no data {_min(total['no_data_min'])} min.")
    stops = {k: v for k, v in total["states"].items() if k != "RUNNING"}
    if stops:
        lines.append("- Not running, by state: " + ", ".join(f"{k} {_min(v)} min" for k, v in sorted(stops.items(), key=lambda x: -x[1])) + ".")
    if total["oee"] is not None:
        lines.append(f"- While running: OEE {_pct(total['oee'])} (availability {_pct(total['availability'])}, performance "
                     f"{_pct(total['performance'])}, quality {_pct(total['quality'])}). Each is the PLC's own value averaged over the "
                     f"running rows, as on Home; the stopped time is not in them.")
    else:
        lines.append("- No running time in the window, so no OEE.")
    if total["throughput"] is not None:
        lines.append(f"- Throughput while running {total['throughput']:.0f} kg/h"
                     + (f" (target {total['throughput_target']:.0f} kg/h)." if total["throughput_target"] else "."))
    if cal:
        outside = [r for r in rows if not cal.is_working(r["ts"])]
        if outside:
            run = sum(1 for r in outside if r.get("machine_state") == "RUNNING")
            lines.append(f"- Outside working hours (plant calendar): {_min(agent._minutes(len(outside), period))} min, "
                         f"of which running {_min(agent._minutes(run, period))} min.")
    alarms = Counter(total["alarms"]).most_common(8)
    lines.append("- Alarms started: " + (", ".join(f"{c} × {n}" for c, n in alarms) if alarms else "none") + ".")
    if total["out_of_range"]:
        lines.append(f"- Out of normal range while running ({agent.OUT_MINUTES} min or more in a row): " + "; ".join(
            f"{o['name']} {_min(o['minutes'])} min (between {agent._fmt(o['min'])} and {agent._fmt(o['max'])} {o['unit']}; normal "
            f"{agent._range_text(o['low'], o['high'], o['unit'])})" for o in total["out_of_range"][:6]) + ".")
    else:
        lines.append("- No signal stayed out of its normal range while running.")

    parts = _parts(rows, signals, period, cal, w)
    extra = []
    titles = {"shift": "By shift", "hour": "By hour", "day": "By day"}
    for key in ("shift", "day", "hour"):           # the hour-by-hour line is dropped first
        if parts.get(key):
            extra.append(f"- {titles[key]} (OEE while running, running min, downtime min): " + " | ".join(
                f"{label} {_pct(s['oee'])}, {_min(s['running_min'])}, {_min(s['downtime_min'])}" for label, s in parts[key]))
    text = "\n".join(lines)
    for e in extra:
        if len(text) + len(e) + 1 <= budget:
            text += "\n" + e
    return text[:budget]


@dataclass
class Result:
    text: str
    citation: dict
    log: str


def compute(asset_id: str, name: str, question: str, budget: int, real: datetime | None = None) -> Result | None:
    """The window block for a data question about an asset, or None (not a data question, or no data)."""
    if not is_data_question(question):
        return None
    clk = agent.clock(asset_id, real)
    if clk is None:
        return None
    cal = plant_calendar.for_asset(asset_id)
    w = parse(question, clk["now"], cal)
    rows, signals, period, source, bucketed = read(asset_id, w)
    signals = list({s["key"]: s for s in signals}.values())   # a signal monitored by two parts is listed twice
    if not rows:
        text = (f"HISTORIAN WINDOW for {name}, {w.label}: {_local(w.start, cal)} – {_local(w.end, cal)}: "
                f"the historian has no rows in this window.")
    else:
        text = block(name, w, rows, signals, period, source, bucketed, clk, cal, budget)
    citation = {"kind": "historian", "source": f"Historian — {w.label}", "asset_id": asset_id, "section_path": [],
                "page_start": None, "page_end": None, "score": None,
                "snippet": text.split("\n", 1)[1][:400] if "\n" in text else text[:400],
                "facts": text.split("\n")[1:]}
    return Result(text, citation, f"window={w.label!r} rows={len(rows)} bucketed={bucketed} calendar={bool(cal)} chars={len(text)}")
