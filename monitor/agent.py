"""Asset monitoring for the Studio's Home (P15a): the numbers of the last hours, and an agent that writes posts.

Ported from BSKLAB EDGE's monitor (P11g, P13e). The rules are the same; the data comes straight from the
Context Service (`operational_state`, `signal_history`, `assemble`), not over MCP. Once EDGE reads these
posts (P15e), this is the only monitoring agent.

**Live or replay.** `historian_live` holds the PLC's rows, one every 20 s: the clock is the present, and when
no row arrives the page shows the last known state and says which part stopped. The recorded month
(`plc_1_historian`, one row a minute) is *replayed*: "now" is the same day of the month and time of day
inside that month.

**Cadence.** The history says how far apart its rows are (`period_seconds`). Durations are counted in rows
and shown in minutes, so "3 minutes" is 3 rows of the recorded month and 9 rows from the PLC.

**The agent** (`manage.py monitor_tick`, once a minute) reads each asset's new rows and writes a post:

    state     the machine state changed                       (written from the data, no model)
    alert     a signal left its normal range for 3 minutes    (the value from the data; the model adds what the
              while RUNNING; armed again after 10 minutes      Studio documents about it, or "not documented")
              in range
    summary   an hour in which the machine ran has ended      (the model writes it from the hour's numbers)
    data      rows from the PLC stopped arriving, or arrive again (live only; written from the data)
    spc       an SPC tool got worse (stable → out of control → specification violation), or has been calm
              again for 30 minutes (P15a-2; written from the SPC events, no model)

The hourly summary also counts the SPC events that started in the hour, per tool. SPC posts follow the
live historian only (the SPC evaluator judges the rows the Studio serves; a replay is not judged again).

Titles and numbers are always computed here; the model only writes the sentences, and a post is still
written (from a template) when the model cannot be used.
"""

from __future__ import annotations

import calendar
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from context_service import service

from . import llm
from .models import MonitorCursor, MonitorPost

logger = logging.getLogger("chat")

WINDOW_HOURS = 8                # what Home shows, and how far back the agent starts
HISTORY_HOURS = 4               # one history question (720 of the PLC's rows; the limit is 1,500 points)
STALE_HOURS = 2                 # recorded data: newest sample older than this → replay
LIVE_SOURCE = "historian_live"  # the PLC's rows; this source is never replayed
NO_DATA_MINUTES = 2             # missing minutes in an hour below this are not reported (rows arrive up to 40 s late)
OUT_MINUTES = 3                 # minutes out of range before an alert
REARM_MINUTES = 10              # minutes back in range before the same signal can alert again
MODEL_ALERTS_PER_TICK = 4       # further alerts in one pass are written from the template
KEEP_POSTS = 2000               # per asset
SPC_CALM_MINUTES = 30           # an SPC tool is "back to stable" after this long without an active event
SPC_LEVELS = {"spec_violation": 2, "control_violation": 1}      # SpcEvent.type → how bad (warnings are not posted)
SPC_STATUS = {0: "stable", 1: "out of control", 2: "specification violation"}
SPC_IS = {1: "is out of control", 2: "violates its specification"}
SPC_TYPE_TEXT = {"spec_violation": "specification violation", "control_violation": "control-limit violation",
                 "statistical_warning": "statistical warning"}
OVERVIEW_TTL = 30

_overviews: dict[str, tuple[float, dict]] = {}


def now() -> datetime:
    return datetime.now(timezone.utc)


def _when(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --- the clock --------------------------------------------------------------------------------

def replay_time(latest: datetime, real: datetime) -> datetime:
    """The moment inside the recorded month that stands for `real`: same day of month, same time."""
    days = calendar.monthrange(latest.year, latest.month)[1]
    return real.replace(year=latest.year, month=latest.month, day=min(real.day, days), second=0, microsecond=0)


NO_DATA_TEXT = {
    "collector_not_reporting": "The collector is not reporting: it or its link to the historian is down. "
                               "It keeps the rows and delivers them when it is back.",
    "no_rows_from_plc": "The collector reports but has no rows: the generator is paused, or the PLC or its link is down.",
}


def _age(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds} s" if seconds < 120 else f"{seconds // 60} min" if seconds < 7200 else f"{seconds // 3600} h"


def clock(asset_id: str, real: datetime | None = None) -> dict | None:
    """{"now", "replay", "latest", "live", "data"} for an asset, or None when it has no recorded data.

    `data` is what Home says about the flow of rows: {"live", "stale", "age_seconds", "reason", "text"}."""
    real = (real or now()).astimezone(timezone.utc)
    state = service.operational_state(asset_id)
    current = state.get("current_state") or {}
    latest = _when(current["ts"]) if current.get("ts") else None
    if latest is None:
        return None
    if state.get("source") == LIVE_SOURCE:
        stale, reason = bool(current.get("stale")), current.get("no_data_reason") or ""
        age = current.get("age_seconds")
        data = {"live": True, "stale": stale, "age_seconds": age, "reason": reason if stale else "",
                "text": (f"No row from the PLC for {_age(age or 0)}. " + NO_DATA_TEXT.get(reason, "")).strip() if stale else ""}
        return {"now": real.replace(microsecond=0), "replay": False, "latest": latest, "live": True, "data": data}
    data = {"live": False, "stale": False, "age_seconds": None, "reason": "", "text": ""}
    if real - latest <= timedelta(hours=STALE_HOURS):
        return {"now": real.replace(second=0, microsecond=0), "replay": False, "latest": latest, "live": False, "data": data}
    return {"now": min(replay_time(latest, real), latest), "replay": True, "latest": latest, "live": False, "data": data}


# --- reading the data ---------------------------------------------------------------------------

def history(asset_id: str, start: datetime, end: datetime) -> dict:
    """Every stored row in (start, end], asked in pieces (at most 1,500 points an answer)."""
    out: dict | None = None
    while start < end:
        until = min(start + timedelta(hours=HISTORY_HOURS), end)
        part = service.signal_history(asset_id, start.isoformat(), until.isoformat())
        if out is None:
            out = part
        else:
            out["ts"] = out["ts"] + part["ts"]
            out["series"] = {key: out["series"][key] + part["series"].get(key, []) for key in out["series"]}
        start = until
    return out or {"signals": [], "ts": [], "series": {}}


def _rows(h: dict) -> list[dict]:
    keys = list(h["series"])
    return [{"ts": _when(ts), **{k: h["series"][k][i] for k in keys}} for i, ts in enumerate(h["ts"])]


def _period(h: dict) -> int:
    """Seconds between two rows of a history answer (60 when it does not say)."""
    return int(h.get("period_seconds") or 60)


def _minutes(rows: int, period: int = 60):
    """How long `rows` rows last, in minutes: whole when it is whole, else to one decimal."""
    minutes = rows * period / 60
    return round(minutes) if abs(minutes - round(minutes)) < 1e-9 else round(minutes, 1)


def _mean(values: list) -> float | None:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(numbers) / len(numbers), 1) if numbers else None


def _alarm_onsets(rows: list[dict], before: dict | None = None) -> Counter:
    """How often each alarm code started."""
    out: Counter = Counter()
    previous = (before or {}).get("alarm_code") or ""
    for row in rows:
        code = row.get("alarm_code") or ""
        if code and code != previous:
            out[code] += 1
        previous = code
    return out


def _sequence(rows: list[dict], period: int = 60) -> list[list]:
    """The machine states in the order they happened: [[state, minutes], …]."""
    out: list[list] = []
    for row in rows:
        state = row.get("machine_state") or "UNKNOWN"
        if out and out[-1][0] == state:
            out[-1][1] += 1
        else:
            out.append([state, 1])
    return [[state, _minutes(n, period)] for state, n in out]


def _outside(s: dict, v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and (
        (s["low"] is not None and v < s["low"]) or (s["high"] is not None and v > s["high"]))


def hour_stats(rows: list[dict], signals: list[dict], period: int = 60, span_min: float | None = None) -> dict:
    """The numbers of one stretch (an hour): running and stopped minutes, OEE, throughput, alarms, signals out of range.

    Downtime is the historian's own: the rows whose `downtime_min` is above zero (a changeover is not running and
    not downtime). Without that column it is every row that is not RUNNING. `span_min` is how long the stretch is;
    the minutes without a row are `no_data_min`."""
    running = [r for r in rows if r.get("machine_state") == "RUNNING"]
    states = Counter(r.get("machine_state") or "UNKNOWN" for r in rows)
    if any(isinstance(r.get("downtime_min"), (int, float)) for r in rows):
        down = sum(1 for r in rows if (r.get("downtime_min") or 0) > 0)
    else:
        down = len(rows) - len(running)
    need = max(1, round(OUT_MINUTES * 60 / period))                 # rows in a row that make an alert
    out_of_range = []
    for s in signals:
        if s["low"] is None and s["high"] is None:
            continue
        values = [r[s["key"]] for r in running if isinstance(r.get(s["key"]), (int, float))]
        outside = [v for v in values if _outside(s, v)]
        longest = stretch = 0                       # the alert's rule: a stretch of OUT_MINUTES in a row
        for r in rows:
            stretch = stretch + 1 if r.get("machine_state") == "RUNNING" and _outside(s, r.get(s["key"])) else 0
            longest = max(longest, stretch)
        if longest >= need:
            out_of_range.append({"name": s["name"], "unit": s["unit"], "minutes": _minutes(len(outside), period), "low": s["low"],
                                 "high": s["high"], "min": round(min(values), 2), "max": round(max(values), 2)})
    covered = len(rows) * period / 60
    missing = max(0.0, span_min - covered) if span_min is not None else 0.0
    return {
        "minutes": _minutes(len(rows), period), "running_min": _minutes(len(running), period),
        "downtime_min": _minutes(down, period), "not_running_min": _minutes(len(rows) - len(running), period),
        "no_data_min": round(missing) if missing >= NO_DATA_MINUTES else 0,
        "states": {state: _minutes(n, period) for state, n in states.items()}, "sequence": _sequence(rows, period),
        "oee": _mean([r.get("oee_pct") for r in running]), "availability": _mean([r.get("availability_pct") for r in running]),
        "performance": _mean([r.get("performance_pct") for r in running]), "quality": _mean([r.get("quality_pct") for r in running]),
        "throughput": _mean([r.get("throughput_actual_kg_hr") for r in running]),
        "throughput_target": _mean([r.get("throughput_target_kg_hr") for r in running]),
        "alarms": dict(_alarm_onsets(rows)), "out_of_range": sorted(out_of_range, key=lambda o: -o["minutes"]),
    }


def assets() -> list[dict]:
    return service.list_assets()


def overview(asset_id: str, real: datetime | None = None) -> dict:
    """What Home shows for an asset: the state now, the last hours hour by hour, and the OEE trend."""
    cached = _overviews.get(asset_id)
    if real is None and cached and time.monotonic() - cached[0] < OVERVIEW_TTL:
        return cached[1]
    clk = clock(asset_id, real)
    if clk is None:
        return {"asset_id": asset_id, "available": False}
    end = clk["now"]
    start = end.replace(minute=0, second=0) - timedelta(hours=WINDOW_HOURS - 1)        # whole hours, the current one last
    h = history(asset_id, start, end)
    rows, signals, period = _rows(h), h["signals"], _period(h)
    if not rows:            # live: no row in the whole window; the page says why
        return {"asset_id": asset_id, "available": False, "live": clk["live"], "data": clk["data"],
                "latest": clk["latest"].isoformat(), "window_hours": WINDOW_HOURS}

    last = rows[-1]
    since = last["ts"]
    for row in reversed(rows):
        if row.get("machine_state") != last.get("machine_state"):
            break
        since = row["ts"]
    hours = []
    for n in range(WINDOW_HOURS):
        begin = start + timedelta(hours=n)
        until = min(begin + timedelta(hours=1), end)
        part = [r for r in rows if begin < r["ts"] <= begin + timedelta(hours=1)]
        stats = hour_stats(part, signals, period, (until - begin).total_seconds() / 60)
        hours.append({"start": begin.isoformat(), "complete": begin + timedelta(hours=1) <= end, "running_min": stats["running_min"],
                      "downtime_min": stats["downtime_min"], "no_data_min": stats["no_data_min"], "oee": stats["oee"],
                      "alarms": sum(stats["alarms"].values())})
    total = hour_stats(rows, signals, period)
    recent = hour_stats([r for r in rows if r["ts"] > end - timedelta(hours=1)], signals, period)
    step = max(1, round(600 / period))                                  # 10-minute means
    trend = []
    for i in range(0, len(rows), step):
        part = [r for r in rows[i:i + step] if r.get("machine_state") == "RUNNING"]
        trend.append({"ts": rows[i]["ts"].isoformat(), "oee": _mean([r.get("oee_pct") for r in part]),
                      "throughput": _mean([r.get("throughput_actual_kg_hr") for r in part])})
    running_now = last.get("machine_state") == "RUNNING" and not clk["data"]["stale"]      # a last known value is not "now"
    now_out = [{"name": s["name"], "value": last[s["key"]], "unit": s["unit"], "low": s["low"], "high": s["high"]}
               for s in signals if running_now and _outside(s, last.get(s["key"]))]
    result = {
        "asset_id": asset_id, "available": True, "now": end.isoformat(), "replay": clk["replay"], "live": clk["live"],
        "data": clk["data"], "latest": last["ts"].isoformat(), "period_seconds": period,
        "window_hours": WINDOW_HOURS, "source": h.get("source"),
        "state": {"machine_state": last.get("machine_state"), "since": since.isoformat(),
                  "alarm": last.get("alarm_code") or "", "recipe": last.get("recipe_id") or "", "shift": last.get("shift_code") or ""},
        "last_hour": {k: recent[k] for k in ("oee", "availability", "performance", "quality", "throughput", "throughput_target",
                                             "running_min", "downtime_min")},
        "window": {"oee": total["oee"], "running_min": total["running_min"], "downtime_min": total["downtime_min"],
                   "no_data_min": sum(hour["no_data_min"] for hour in hours),
                   "alarms": sum(total["alarms"].values()), "alarm_codes": dict(Counter(total["alarms"]).most_common(5))},
        "out_of_range": now_out, "hours": hours, "trend": trend,
    }
    if real is None:
        _overviews[asset_id] = (time.monotonic(), result)
    return result


# --- posts ----------------------------------------------------------------------------------

def post_json(p: MonitorPost) -> dict:
    return {"id": p.id, "asset_id": p.asset_id, "kind": p.kind, "severity": p.severity, "title": p.title, "text": p.text,
            "data": p.data or {}, "ts": p.ts.astimezone(timezone.utc).isoformat(), "model": p.model}


def posts(asset_id: str, before_id: int | None = None, limit: int = 30) -> dict:
    """Newest first; `before_id` continues an earlier page."""
    limit = max(1, min(int(limit), 100))
    q = MonitorPost.objects.filter(asset_id=asset_id)
    if before_id:
        q = q.filter(id__lt=before_id)
    found = list(q.order_by("-id")[:limit + 1])
    return {"posts": [post_json(p) for p in found[:limit]], "more": len(found) > limit}


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _range_text(low, high, unit: str) -> str:
    unit = f" {unit}" if unit else ""
    if low is not None and high is not None:
        return f"{_fmt(low)}–{_fmt(high)}{unit}"
    return f"at least {_fmt(low)}{unit}" if low is not None else f"at most {_fmt(high)}{unit}"


SUMMARY_PROMPT = (
    "You are the monitoring agent of a production line. Write the hourly note for the operators from the NUMBERS below: "
    "2 to 4 short sentences, plain text, no headings or lists. Say how the hour went, name what stands out "
    "(downtime, signals out of range, frequent alarms, a low figure) and what to watch next. Use only these numbers; never invent one, "
    "and do not compare a figure with a target unless the target is given."
)
ALERT_PROMPT = (
    "You are the monitoring agent of a production line. A signal has left its normal range (WHAT HAPPENED). "
    "From the CONTEXT, write 1 or 2 short sentences, plain text: the documented cause and what to check. "
    "Use ONLY context facts about this signal, the component it is on, and that component's alarms and procedures; cite them as [G]. "
    "Lines about 'last running value' are from another time: ignore them. Do not repeat the value or the range. "
    "Never suggest a cause, a part or a step that is not written in the CONTEXT. "
    "If the CONTEXT has no such fact, reply with exactly: NONE"
)
NOT_DOCUMENTED = "The Context Studio has no documented cause or procedure for this signal yet."


def summary_facts(stats: dict) -> str:
    lines = [f"Running {stats['running_min']} min, not running {stats['not_running_min']} min "
             f"({', '.join(f'{k} {v} min' for k, v in stats['states'].items() if k != 'RUNNING') or 'no stops'})."]
    if stats["no_data_min"]:
        lines.append(f"No data for {stats['no_data_min']} min of the hour.")
    if len(stats["sequence"]) > 1:
        lines.append("In order: " + ", then ".join(f"{state} {minutes} min" for state, minutes in stats["sequence"]) + ".")
    if stats["oee"] is not None:
        lines.append(f"While running: OEE {stats['oee']} %, availability {stats['availability']} %, performance {stats['performance']} %, "
                     f"quality {stats['quality']} %.")
    if stats["throughput"] is not None:
        lines.append(f"Throughput {stats['throughput']:.0f} kg/h" + (f" against a target of {stats['throughput_target']:.0f} kg/h."
                                                                    if stats["throughput_target"] else "."))
    if stats["alarms"]:
        lines.append("Alarms started: " + ", ".join(f"{code} × {n}" for code, n in Counter(stats["alarms"]).most_common(6)) + ".")
    else:
        lines.append("No alarms.")
    for o in stats["out_of_range"][:5]:
        lines.append(f"{o['name']} was outside its normal range ({_range_text(o['low'], o['high'], o['unit'])}) for {o['minutes']} min "
                     f"(between {_fmt(o['min'])} and {_fmt(o['max'])}).")
    if not stats["out_of_range"]:
        lines.append("No signal stayed outside its normal range.")
    if stats.get("spc"):
        lines.append("SPC: " + "; ".join(
            f"{tool} " + ", ".join(f"{n} {SPC_TYPE_TEXT.get(kind, kind)}{'s' if n > 1 else ''}" for kind, n in kinds.items())
            for tool, kinds in stats["spc"].items()) + ".")
    return "\n".join(lines)


# --- SPC (P15a-2) ---------------------------------------------------------------------------------

def spc_counts(asset_id: str, begin: datetime, until: datetime) -> dict[str, dict[str, int]]:
    """SPC events that started in (begin, until], per tool name and type, worst first."""
    from spc.models import SpcEvent
    out: dict[str, Counter] = {}
    for name, kind in SpcEvent.objects.filter(tool__asset_id=asset_id, started_at__gt=begin, started_at__lte=until) \
            .values_list("tool__name", "type"):
        out.setdefault(name, Counter())[kind] += 1
    order = list(SPC_TYPE_TEXT)
    return {name: {k: kinds[k] for k in sorted(kinds, key=order.index)}
            for name, kinds in sorted(out.items(), key=lambda kv: -sum(kv[1].values()))}


def _spc_posts(asset_id: str, now_: datetime, tracking: dict) -> int:
    """A post when an SPC tool gets worse, and one when it has been calm for SPC_CALM_MINUTES. `tracking` is
    {tool id: {"level", "calm_since"}}, kept with the cursor. The first pass posts the tools that are bad now."""
    from spc import service as spc_service
    from spc.models import SpcTool
    written = 0
    for tool in SpcTool.objects.filter(asset_id=asset_id, enabled=True):
        active = spc_service.active_events(tool)
        level = max((SPC_LEVELS.get(e["type"], 0) for e in active), default=0)
        known = tracking.get(str(tool.id))
        old = known["level"] if known else 0
        if level > old:
            event = tool.events.filter(type=next(k for k, v in SPC_LEVELS.items() if v == level)).order_by("-last_at").first()
            limits = event.limits if event else {}
            bounds = ", ".join(f"{k.upper()} {limits[k]:.4g}" for k in ("lsl", "lcl", "cl", "ucl", "usl")
                               if isinstance(limits.get(k), (int, float)))
            unit = f" {tool.unit}" if tool.unit else ""
            text = (f"{tool.name} {SPC_IS[level]}" + (f" since {event.started_at:%H:%M}" if event else "") +
                    (f": rule {event.rule}, value {event.value:.4g}{unit}" if event else "") +
                    (f" ({bounds})." if bounds else ".") +
                    (f" It fired {event.count} time{'s' if event.count != 1 else ''} so far." if event and event.count > 1 else ""))
            _add(asset_id, "spc", "warning", f"SPC · {tool.name}: {SPC_STATUS[old]} → {SPC_STATUS[level]}", text,
                 event.started_at if event else now_,
                 {"tool_id": str(tool.id), "tool": tool.name, "from": SPC_STATUS[old], "to": SPC_STATUS[level],
                  "type": event.type if event else "", "rule": event.rule if event else "",
                  "value": event.value if event else None, "limits": limits, "unit": tool.unit})
            tracking[str(tool.id)] = {"level": level, "calm_since": None}
            written += 1
        elif level == 0 and old > 0:
            calm = known.get("calm_since")
            if calm is None:
                tracking[str(tool.id)] = {"level": old, "calm_since": now_.isoformat()}
            elif now_ - _when(calm) >= timedelta(minutes=SPC_CALM_MINUTES):
                _add(asset_id, "spc", "info", f"SPC · {tool.name}: back to stable",
                     f"{tool.name} has had no SPC violation for {SPC_CALM_MINUTES} minutes (since {_when(calm):%H:%M}).",
                     now_, {"tool_id": str(tool.id), "tool": tool.name, "from": SPC_STATUS[old], "to": "stable"})
                tracking[str(tool.id)] = {"level": 0, "calm_since": None}
                written += 1
        elif level > 0:                     # still bad (maybe less): the next rise is posted again
            tracking[str(tool.id)] = {"level": level, "calm_since": None}
    return written


def _add(asset_id: str, kind: str, severity: str, title: str, text: str, ts: datetime, data: dict, model: str = "") -> None:
    MonitorPost.objects.create(asset_id=asset_id, kind=kind, severity=severity, title=title[:200], text=text, ts=ts,
                               data=data, model=model)


# --- the agent ----------------------------------------------------------------------------------

def _save(asset_id: str, ts: datetime, tracking: dict) -> None:
    MonitorCursor.objects.update_or_create(asset_id=asset_id, defaults={"ts": ts, "state": tracking})
    old = MonitorPost.objects.filter(asset_id=asset_id).order_by("-id").values_list("id", flat=True)[KEEP_POSTS:KEEP_POSTS + 1]
    if old:
        MonitorPost.objects.filter(asset_id=asset_id, id__lte=old[0]).delete()


def _data_post(asset_id: str, clk: dict, flow: dict) -> int:
    """A post when the PLC's rows stop arriving, and one when they arrive again. `flow` is kept with the cursor."""
    data = clk["data"]
    if data["stale"] and not flow.get("stopped"):
        flow.update(stopped=True, since=clk["latest"].isoformat())
        _add(asset_id, "data", "warning", "No data from the PLC",
             f"The last row is from {clk['latest']:%H:%M:%S}; the state shown is the last known one. "
             f"{NO_DATA_TEXT.get(data['reason'], '')}".strip(), clk["now"], {"reason": data["reason"], "latest": flow["since"]})
        return 1
    if flow.get("stopped") and not data["stale"]:
        since = _when(flow["since"])
        flow.update(stopped=False, since="")
        _add(asset_id, "data", "info", "Data from the PLC is back",
             f"Rows arrive again, {_age((clk['now'] - since).total_seconds())} after the row of {since:%H:%M:%S}. "
             "Rows a collector kept are delivered with their own times and are read now.", clk["now"], {"latest": since.isoformat()})
        return 1
    return 0


def tick(asset_id: str, real: datetime | None = None) -> int:
    """Read one asset's new rows and write the posts they call for. Returns how many were written."""
    clk = clock(asset_id, real)
    if clk is None:
        return 0
    end = clk["now"]
    cursor = MonitorCursor.objects.filter(asset_id=asset_id).first()
    start = cursor.ts.astimezone(timezone.utc) if cursor else None
    tracking = dict(cursor.state or {}) if cursor else {}
    flow = dict(tracking.get("flow") or {})                 # whether the PLC's rows have stopped; outlives a new window
    spc = dict(tracking.get("spc") or {})                   # SPC tools' levels; outlive a new window too
    saved_spc = dict(spc)
    if start is None or start > end or end - start > timedelta(hours=WINDOW_HOURS):
        # First run, a long time without a row, or the replay clock went back to the start of the month:
        # begin a window ago, on the hour.
        start, tracking = (end - timedelta(hours=WINDOW_HOURS)).replace(minute=0, second=0), {}
    written = _data_post(asset_id, clk, flow) if clk["live"] else 0
    if clk["live"]:
        written += _spc_posts(asset_id, end, spc)
    if end - start < timedelta(minutes=1):
        if written or spc != saved_spc:
            _save(asset_id, start, {**tracking, "flow": flow, "spc": spc})
            _overviews.pop(asset_id, None)
        return written

    h = history(asset_id, start, end)
    rows, signals, period = _rows(h), h["signals"], _period(h)
    if tracking.get("period", period) != period:    # another historian answers: the counts were in other rows
        tracking = {}
    out_rows, rearm_rows = max(1, round(OUT_MINUTES * 60 / period)), max(1, round(REARM_MINUTES * 60 / period))
    limited = [s for s in signals if s["low"] is not None or s["high"] is not None]
    previous_state = tracking.get("machine_state")
    out = tracking.get("out", {})                   # signal key → {"n": rows out, "ok": rows back in, "alerted": bool}
    model = llm.pick()
    model_alerts = 0

    def summary(until: datetime) -> int:
        """The post for the hour (until − 1 h, until], when the machine ran in it."""
        begin = until - timedelta(hours=1)
        hour = [r for r in rows if begin < r["ts"] <= until]
        if begin < start:                                           # the hour began before this pass: read it whole
            hour = _rows(history(asset_id, begin, until))
        stats = hour_stats(hour, signals, period, 60)
        if stats["running_min"] <= 0:                               # "every hour of running"
            return 0
        if clk["live"]:
            stats["spc"] = spc_counts(asset_id, begin, until)
        facts = summary_facts(stats)
        text, used = llm.write(model, SUMMARY_PROMPT, f"NUMBERS for {begin:%H:%M}–{until:%H:%M}\n{facts}")
        quiet = stats["not_running_min"] == 0 and not stats["out_of_range"] and not stats["no_data_min"]
        _add(asset_id, "summary", "info" if quiet else "warning", f"Hourly summary {begin:%H:%M}–{until:%H:%M}",
             text or facts, until, {**stats, "facts": facts}, used)
        return 1

    before = start
    for row in rows:
        # Hours that ended between two rows (the PLC's rows can stop for a while): their summary comes first.
        hour_end = before.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        while hour_end < row["ts"]:
            written += summary(hour_end)
            hour_end += timedelta(hours=1)
        before = row["ts"]

        state = row.get("machine_state") or "UNKNOWN"
        if previous_state and state != previous_state:
            alarm = f" Active alarm: {row['alarm_code']}." if row.get("alarm_code") else ""
            _add(asset_id, "state", "info" if state == "RUNNING" else "warning", f"Machine state: {previous_state} → {state}",
                 f"The machine went from {previous_state} to {state} at {row['ts']:%H:%M}.{alarm}", row["ts"],
                 {"from": previous_state, "to": state, "alarm": row.get("alarm_code") or ""})
            written += 1
        previous_state = state

        for sig in limited:
            value, track = row.get(sig["key"]), out.setdefault(sig["key"], {"n": 0, "ok": 0, "alerted": False})
            if state != "RUNNING" or not isinstance(value, (int, float)):
                track["n"] = 0                                      # the normal ranges apply while running
                continue
            above = sig["high"] is not None and value > sig["high"]
            below = sig["low"] is not None and value < sig["low"]
            if not (above or below):
                track["n"], track["ok"] = 0, track["ok"] + 1
                if track["ok"] >= rearm_rows:
                    track["alerted"] = False
                continue
            track["n"], track["ok"] = track["n"] + 1, 0
            if track["n"] == out_rows and not track["alerted"]:
                track["alerted"] = True
                side = "above" if above else "below"
                unit = f" {sig['unit']}" if sig["unit"] else ""
                title = f"{sig['name']} {side} its normal range"
                facts = (f"{sig['name']} has been {side} its normal range ({_range_text(sig['low'], sig['high'], sig['unit'])}) "
                         f"for {OUT_MINUTES} minutes; it is {_fmt(value)}{unit} at {row['ts']:%H:%M}.")
                text, used = "", ""
                if model is not None and model_alerts < MODEL_ALERTS_PER_TICK:
                    model_alerts += 1
                    try:
                        packet = service.assemble(f"{sig['name']} {side} normal range: causes and what to check",
                                                  asset_id=asset_id, include_documents=True, budget_chars=3000)
                        text, used = llm.write(model, ALERT_PROMPT, f"WHAT HAPPENED\n{facts}\n\nCONTEXT\n{packet.get('context_text', '')}")
                    except service.ContextError as exc:
                        logger.warning("monitor: no context for the alert (%s)", exc)
                # The value and the range are always written from the data; the model only adds what the Studio documents.
                documented = bool(text) and text.strip().strip(".").upper() != "NONE"
                text = f"{facts} {text if documented else NOT_DOCUMENTED}" if used else facts
                _add(asset_id, "alert", "warning", title, text, row["ts"],
                     {"signal": sig["name"], "key": sig["key"], "value": value, "unit": sig["unit"], "low": sig["low"],
                      "high": sig["high"], "side": side}, used)
                written += 1

        if row["ts"].minute == 0 and row["ts"].second == 0:        # an hour has ended with this row: (ts − 1 h, ts]
            written += summary(row["ts"])

    # The next pass begins after the newest row read, not at the clock: the PLC's rows arrive up to 40 s late,
    # and rows a collector kept arrive much later with their own times.
    _save(asset_id, rows[-1]["ts"] if rows else start,
          {"machine_state": previous_state, "out": out, "period": period, "flow": flow, "spc": spc})
    if written:
        _overviews.pop(asset_id, None)
        logger.info("monitor: %s %d post(s) up to %s%s", asset_id, written, end.isoformat(), " (replay)" if clk["replay"] else "")
    return written
