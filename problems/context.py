"""What a ticket opens with (P29a): the machine's states, alarms, signals and SPC events in the problem's window.

Taken when the ticket opens and again when its window closes, so the ticket still shows what happened after the
historian moves on. Built from the monitor's own readers (the same numbers as Home's posts). Never blocks a ticket:
when the historian can't be read, the snapshot says so.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("chat")

MAX_HOURS = 12            # the longest window read; a longer problem shows its last 12 hours
MAX_SEQUENCE = 40         # state changes kept
MAX_SPC = 20              # SPC events kept


def _iso(ts) -> str | None:
    return ts.isoformat() if ts else None


def spc_events(asset_id: str, start: datetime, end: datetime) -> list[dict]:
    from spc.models import SpcEvent
    found = (SpcEvent.objects.filter(tool__asset_id=asset_id, started_at__lte=end, last_at__gte=start)
             .exclude(type="statistical_warning").select_related("tool").order_by("started_at")[:MAX_SPC])
    return [{"tool": e.tool.name, "unit": e.tool.unit, "type": e.type, "rule": e.rule,
             "started_at": _iso(e.started_at), "last_at": _iso(e.last_at),
             "hours": round((e.last_at - e.started_at).total_seconds() / 3600, 2), "value": e.value,
             "extreme": e.extreme, "machine_state": e.machine_state, "limits": e.limits} for e in found]


def build(asset_id: str, start: datetime, end: datetime | None, now: datetime | None = None) -> dict:
    """The snapshot for (start, end]; `end` None means the problem is still going (read up to now)."""
    from monitor import agent

    now = (now or datetime.now(timezone.utc))
    until = min(end or now, now)
    begin = max(start, until - timedelta(hours=MAX_HOURS))
    out: dict = {"captured_at": now.isoformat(), "window": {"start": _iso(start), "end": _iso(end),
                                                            "read_from": _iso(begin), "read_until": _iso(until),
                                                            "cut": begin > start}}
    try:
        before = agent.history(asset_id, begin - timedelta(minutes=10), begin - timedelta(seconds=1))
        last_before = agent._rows(before)[-1] if before.get("ts") else {}
        h = agent.history(asset_id, begin - timedelta(seconds=1), until)        # the window's first row too
        rows, signals, period = agent._rows(h), h["signals"], agent._period(h)
        span = (until - begin).total_seconds() / 60
        stats = agent.hour_stats(rows, signals, period, span)
        sequence = stats["sequence"]
        out.update({
            "state_before": last_before.get("machine_state") or "",
            "states": sequence[-MAX_SEQUENCE:], "states_cut": len(sequence) > MAX_SEQUENCE,
            "state_minutes": stats["states"], "running_min": stats["running_min"],
            "downtime_min": stats["downtime_min"], "no_data_min": stats["no_data_min"],
            "alarms": stats["alarms"], "out_of_range": stats["out_of_range"],
            "oee": stats["oee"], "throughput": stats["throughput"], "throughput_target": stats["throughput_target"],
            "signals_at_start": _values(rows[0] if rows else {}, signals),
            "signals_at_end": _values(rows[-1] if rows else {}, signals),
        })
    except Exception as exc:          # the historian is down: the ticket still opens
        logger.warning("problems: no historian context for %s (%s)", asset_id, exc)
        out["error"] = f"The historian could not be read: {type(exc).__name__}: {exc}"[:300]
    try:
        out["spc"] = spc_events(asset_id, start, until)
    except Exception as exc:
        out["spc"] = []
        out.setdefault("error", f"The SPC events could not be read: {exc}"[:300])
    return out


def _values(row: dict, signals: list[dict]) -> list[dict]:
    """The signals that have a normal range, with their value in one row."""
    out = []
    for s in signals:
        if s.get("low") is None and s.get("high") is None:
            continue
        v = row.get(s["key"])
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append({"name": s["name"], "value": round(v, 3), "unit": s.get("unit") or "", "low": s["low"],
                        "high": s["high"],
                        "outside": (s["low"] is not None and v < s["low"]) or (s["high"] is not None and v > s["high"])})
    return out


def as_text(ticket) -> str:
    """The snapshot as plain lines, for a model (P29b, P29c) and the published report (P29d)."""
    c = ticket.context or {}
    lines = [f"Machine: {ticket.asset_name or ticket.asset_id}", f"Trigger: {ticket.get_trigger_display()}",
             f"Window: {ticket.window_start:%Y-%m-%d %H:%M} UTC to "
             + (f"{ticket.window_end:%Y-%m-%d %H:%M} UTC" if ticket.window_end else "still going")]
    if c.get("state_before"):
        lines.append(f"State before: {c['state_before']}")
    if c.get("states"):
        lines.append("States in order: " + " → ".join(f"{s} ({m} min)" for s, m in c["states"]))
    if c.get("alarms"):
        lines.append("Alarm codes: " + ", ".join(f"{k} ×{v}" for k, v in c["alarms"].items()))
    for o in c.get("out_of_range") or []:
        lines.append(f"Out of range while running: {o['name']} {o['min']}–{o['max']} {o['unit']} "
                     f"(normal {o['low']}–{o['high']}) for {o['minutes']} min")
    for e in c.get("spc") or []:
        lim = e.get("limits") or {}
        bounds = ", ".join(f"{k.upper()} {lim[k]:.4g}" for k in ("lsl", "lcl", "cl", "ucl", "usl")
                           if isinstance(lim.get(k), (int, float)))
        lines.append(f"SPC {e['type'].replace('_', ' ')}: {e['tool']} for {e['hours']} h, value {e['value']:.4g} "
                     f"{e['unit']}, worst {e['extreme']:.4g} ({bounds})")
    if c.get("oee") is not None:
        lines.append(f"OEE while running: {c['oee']} %")
    from .analysis import as_lines
    lines += [f"Data summary: {line}" for line in as_lines(c.get("analysis") or {})]
    users = c.get("edge_users") or []
    if users:
        lines.append("Signed in on EDGE: " + ", ".join(f"{u.get('name') or u.get('username')} ({u.get('role', '')}, "
                                                        f"signed in {u.get('signed_in_at', '')[:16]})" for u in users))
    if c.get("error"):
        lines.append(f"Note: {c['error']}")
    return "\n".join(lines)
