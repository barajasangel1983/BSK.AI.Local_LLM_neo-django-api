"""The data summary on a ticket (P29e): the trigger's trend, the signals that moved first, and when it started.

Shown between "What happened" and the 5-Why. Taken with the snapshot (problems.context) and stored on the ticket, so
it stays as it was after the historian moves on.

- **Layer 1, the trigger:** the SPC tool's signal with its limits (an SPC ticket); a downtime or a hand-opened ticket
  has no trigger signal, its state band (running / stopped / alarmed) is the first layer.
- **Layer 2, candidates:** every process signal with a normal range, compared with a **baseline of running rows only**
  (2 h to 1 h before the start). A signal "moved" when the mean of 5 running rows in a row stays more than 3 σ from its
  baseline for 2 minutes, between 1 h before and 30 min after the start. Signals that moved within 2 minutes of each other and go
  together (|r| ≥ 0.8) are one group, led by the one that moved first. The top 3 groups are kept: earlier first, then
  further. "Nothing moved" is a valid answer.
- **Onset:** from the first signal that moved (when before the start) to the start.

The data picks the candidates; nothing here is called a cause. Never blocks a ticket: when the historian can't be
read the summary says so.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("chat")

VERSION = 1
BASELINE_FROM_MIN = 120       # the baseline: 2 h …
BASELINE_UNTIL_MIN = 60       # … to 1 h before the start
SEARCH_BEFORE_MIN = 60        # a signal may move from 1 h before the start …
SEARCH_AFTER_MIN = 30         # … to 30 min after it
CHART_MAX_HOURS = 12          # the trend shows at most 12 h after the start
ROLL = 5                      # running rows in the moving mean
PERSIST = 6                   # … beyond SIGMA for this many rows in a row (2 min at 20 s): a spike is not a move
SIGMA = 3.0                   # how far is "moved"
MIN_BASELINE_ROWS = 15        # 5 minutes at 20 s
GROUP_MINUTES = 2.0           # moved this close together …
GROUP_R = 0.8                 # … and go together: one group
TOP = 3                       # groups shown
MAX_POINTS = 300              # points a trend keeps (min and max of each bucket)
MAX_STATES = 200              # state changes kept for the band
RUNNING = "RUNNING"


def _iso(ts) -> str | None:
    return ts.isoformat() if ts else None


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _epoch(ts: datetime) -> int:
    return int(ts.timestamp())


def points(rows: list[dict], key: str, limit: int = MAX_POINTS) -> list[list]:
    """[epoch seconds, value] pairs; more rows than `limit` keep each bucket's lowest and highest value, so a spike
    stays visible."""
    pts = [(r["ts"], r.get(key)) for r in rows if _num(r.get(key))]
    if len(pts) <= limit:
        return [[_epoch(t), round(v, 4)] for t, v in pts]
    buckets = max(1, limit // 2)
    size = len(pts) / buckets
    out = []
    for b in range(buckets):
        part = pts[int(b * size):int((b + 1) * size)]
        if not part:
            continue
        lo, hi = min(part, key=lambda p: p[1]), max(part, key=lambda p: p[1])
        for t, v in sorted({lo, hi}, key=lambda p: p[0]):
            out.append([_epoch(t), round(v, 4)])
    return out


def states(rows: list[dict]) -> list[list]:
    """[epoch seconds, state] at each change, for the state band."""
    out, last = [], None
    for r in rows:
        s = r.get("machine_state") or ""
        if s != last:
            out.append([_epoch(r["ts"]), s])
            last = s
    return out[-MAX_STATES:]


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def moved(signal: dict, baseline: list[dict], search: list[dict]) -> dict | None:
    """When a signal's moving mean (running rows) first left SIGMA × its baseline σ, and how far it went."""
    key = signal["key"]
    base = [r[key] for r in baseline if _num(r.get(key))]
    if len(base) < MIN_BASELINE_ROWS:
        return None
    mean = sum(base) / len(base)
    sd = math.sqrt(sum((v - mean) ** 2 for v in base) / len(base))
    span = (signal.get("high") - signal.get("low")) if _num(signal.get("high")) and _num(signal.get("low")) else 0
    sd = max(sd, span / 100 if span > 0 else 0, 1e-9)       # a flat baseline: 1 % of the normal range
    vals = [(r["ts"], r[key]) for r in search if _num(r.get(key))]
    first, peak, sign, run, run_first = None, 0.0, 0, 0, None
    for i in range(ROLL - 1, len(vals)):
        z = (sum(v for _, v in vals[i - ROLL + 1:i + 1]) / ROLL - mean) / sd
        if abs(z) > peak:
            peak, sign = abs(z), (1 if z > 0 else -1)
        if first is not None:
            continue
        if abs(z) > SIGMA and (run == 0 or (z > 0) == run_up):
            if run == 0:
                run_up = z > 0
                # the first row of the crossing window that is itself beyond SIGMA (a step: the step's first row)
                window = vals[i - ROLL + 1:i + 1]
                run_first = next((t for t, v in window if abs((v - mean) / sd) > SIGMA and (v > mean) == run_up),
                                 vals[i][0])
            run += 1
            if run >= PERSIST:                       # it stayed there: a spike is not a move
                first = run_first
        else:
            run, run_first = 0, None
    if first is None:
        return None
    return {"first": first, "peak": peak, "direction": "up" if sign > 0 else "down", "baseline_mean": mean,
            "baseline_sd": sd}


def build(asset_id: str, start: datetime, end: datetime | None, trigger: dict | None = None,
          now: datetime | None = None) -> dict:
    """The summary for a problem that started at `start`. `trigger`: {"key", "limits"} for an SPC ticket, else None."""
    from monitor import agent

    now = now or datetime.now(timezone.utc)
    until = min(end or now, now, start + timedelta(hours=CHART_MAX_HOURS))
    read_from = start - timedelta(minutes=BASELINE_FROM_MIN)
    out: dict = {"version": VERSION, "from": _iso(read_from), "until": _iso(until), "start": _iso(start),
                 "end": _iso(end), "cut": bool((end or now) > until), "onset": _iso(start), "lead_min": 0,
                 "trigger": None, "states": [], "candidates": [], "note": ""}
    try:
        h = agent.history(asset_id, read_from, until)
    except Exception as exc:
        logger.warning("problems: no data summary for %s (%s)", asset_id, exc)
        out["error"] = f"The historian could not be read: {type(exc).__name__}: {exc}"[:300]
        return out
    rows = agent._rows(h)
    seen, signals = set(), []
    for s in h.get("signals") or []:                      # the historian may list a signal twice
        if s["key"] not in seen:
            seen.add(s["key"])
            signals.append(s)
    out["states"] = states(rows)
    out["period_s"] = agent._period(h)

    def between(a_min: float, b_min: float) -> list[dict]:
        a, b = start + timedelta(minutes=a_min), start + timedelta(minutes=b_min)
        return [r for r in rows if a <= r["ts"] < b and r.get("machine_state") == RUNNING]

    baseline = between(-BASELINE_FROM_MIN, -BASELINE_UNTIL_MIN)
    search = between(-SEARCH_BEFORE_MIN, SEARCH_AFTER_MIN)
    out["baseline"] = {"from": _iso(start - timedelta(minutes=BASELINE_FROM_MIN)),
                       "until": _iso(start - timedelta(minutes=BASELINE_UNTIL_MIN)), "running_rows": len(baseline)}

    def lead(ts: datetime) -> float:
        return round((ts - start).total_seconds() / 60, 1)

    by_key = {s["key"]: s for s in signals}
    tkey = (trigger or {}).get("key")
    if tkey:
        s = by_key.get(tkey, {"key": tkey, "name": tkey, "unit": "", "low": None, "high": None})
        m = moved(s, baseline, search)
        out["trigger"] = {"key": tkey, "name": (trigger or {}).get("name") or s.get("name") or tkey,
                          "unit": s.get("unit") or (trigger or {}).get("unit") or "", "low": s.get("low"),
                          "high": s.get("high"), "limits": (trigger or {}).get("limits") or {},
                          "first_moved": _iso(m["first"]) if m else None, "lead_min": lead(m["first"]) if m else None,
                          "points": points(rows, tkey)}

    if len(baseline) < MIN_BASELINE_ROWS:
        out["note"] = ("No running baseline in the 2 h before the start (the machine was not running long enough), "
                       "so the signals could not be compared.")
        return out

    movers = []
    for s in signals:
        if s["key"] == tkey or not (_num(s.get("low")) or _num(s.get("high"))):
            continue
        m = moved(s, baseline, search)
        if m:
            movers.append((s, m))
    movers.sort(key=lambda x: (x[1]["first"], -x[1]["peak"]))

    groups: list[dict] = []
    for s, m in movers:
        for g in groups:
            head_s, head_m = g["head"]
            close = abs((m["first"] - head_m["first"]).total_seconds()) <= GROUP_MINUTES * 60
            if close:
                pairs = [(r[head_s["key"]], r[s["key"]]) for r in search
                         if _num(r.get(head_s["key"])) and _num(r.get(s["key"]))]
                if pairs and abs(_corr([a for a, _ in pairs], [b for _, b in pairs])) >= GROUP_R:
                    g["with"].append(s)
                    break
        else:
            groups.append({"head": (s, m), "with": []})
    groups.sort(key=lambda g: (g["head"][1]["first"], -g["head"][1]["peak"]))

    for g in groups[:TOP]:
        s, m = g["head"]
        out["candidates"].append({
            "key": s["key"], "name": s.get("name") or s["key"], "unit": s.get("unit") or "", "low": s.get("low"),
            "high": s.get("high"), "first_moved": _iso(m["first"]), "lead_min": lead(m["first"]),
            "peak_sigma": round(m["peak"], 1), "direction": m["direction"],
            "baseline_mean": round(m["baseline_mean"], 4), "with": [w.get("name") or w["key"] for w in g["with"]],
            "points": points(rows, s["key"])})
    out["more"] = max(0, len(groups) - TOP)

    firsts = [c["first_moved"] for c in out["candidates"]]
    if out["trigger"] and out["trigger"]["first_moved"]:
        firsts.append(out["trigger"]["first_moved"])
    earliest = min((datetime.fromisoformat(f) for f in firsts), default=start)
    if earliest < start:
        out["onset"], out["lead_min"] = _iso(earliest), lead(earliest)
    if not out["candidates"]:
        out["note"] = "No other signal moved more than 3 σ from its baseline between 1 h before and 30 min after the start."
    return out


def trigger_for(source_key: str) -> dict | None:
    """The trigger signal of an SPC ticket (`spc:<event id>`): its column, name, unit and limits."""
    if not source_key.startswith("spc:"):
        return None
    from spc.models import SpcEvent
    event = SpcEvent.objects.filter(pk=source_key.removeprefix("spc:")).select_related("tool").first()
    if event is None:
        return None
    t = event.tool
    limits = {k: v for k, v in (event.limits or {}).items() if _num(v)}
    for k in ("lsl", "usl"):
        if k not in limits and _num(getattr(t, k, None)):
            limits[k] = getattr(t, k)
    return {"key": t.column, "name": t.name, "unit": t.unit, "limits": limits}


def as_lines(a: dict) -> list[str]:
    """The summary in words, for a model and the report."""
    if not a:
        return []
    lines = []
    if a.get("trigger"):
        t = a["trigger"]
        if t.get("lead_min") is not None:
            lines.append(f"Trigger signal {t['name']} first moved {_minutes(t['lead_min'])} the start")
    for c in a.get("candidates") or []:
        follow = f"; with {', '.join(c['with'])}" if c.get("with") else ""
        lines.append(f"Moved {_minutes(c['lead_min'])} the start: {c['name']} ({c['direction']} {c['peak_sigma']} σ from "
                     f"its running baseline {c['baseline_mean']:.4g} {c['unit']}){follow}".replace(" )", ")"))
    if a.get("note"):
        lines.append(a["note"])
    return lines


def _minutes(m: float) -> str:
    if m < 0:
        return f"{abs(m):g} min before"
    if m > 0:
        return f"{m:g} min after"
    return "at"
