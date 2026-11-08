"""Alarm episodes from the historian, their categories, and what the Studio knows about each code (P31)."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from django.db import transaction

from .models import AlarmCode, AlarmCursor, AlarmEpisode

logger = logging.getLogger("chat")

MERGE_MINUTES = 5          # the same code again within this: the same episode, one more repeat
FIRST_HOURS = 2            # a cursor that doesn't exist yet starts this far back (the backfill is a command)
CHUNK_HOURS = 12           # rows read at a time
CATEGORIES = [c for c, _ in AlarmCode.CATEGORIES]
SEVERITY_ORDER = {"high": 0, "medium": 1, "warning": 2, "low": 3}


class AlarmError(ValueError):
    """Shown to the client as is."""


def default_category(code: str) -> str:
    """The category from the code's name, until someone sets it in the Studio."""
    c = code.upper()
    if "SENSOR" in c:
        return "sensor"
    if c.startswith("OUT_OF_SPEC") or "MOISTURE" in c or c in ("HIGH_SCRAP", "LOW_OEE"):
        return "quality"
    if any(w in c for w in ("OVERLOAD", "BLOCK", "PLUG", "STARVE", "STOP", "FAIL", "TRIP")):
        return "equipment"
    return "process"


def categories(asset_id: str) -> dict[str, str]:
    return {a.code: a.category for a in AlarmCode.objects.filter(asset_id=asset_id)}


def category_of(code: str, set_: dict[str, str]) -> str:
    return set_.get(code) or default_category(code)


# --- building episodes -------------------------------------------------------------------------------------

def update(asset_id: str, until: datetime | None = None, since: datetime | None = None) -> dict:
    """Read the asset's rows after the cursor (or `since`) into episodes. Live historian only."""
    from historian import source as historian_source
    from monitor import agent
    if historian_source.current() != "live":
        return {"skipped": "the Studio serves the recorded month, not the present"}
    now = until or datetime.now(timezone.utc)
    cursor = AlarmCursor.objects.filter(asset_id=asset_id).first()
    start = since or (cursor.read_until if cursor else now - timedelta(hours=FIRST_HOURS))
    merge = timedelta(minutes=MERGE_MINUTES)
    recent: dict[str, AlarmEpisode] = {}
    for e in AlarmEpisode.objects.filter(asset_id=asset_id, ended_at__gte=start - merge).order_by("ended_at"):
        recent[e.code] = e                                  # the newest episode of each code that may still grow
    dirty: dict[int, AlarmEpisode] = {}
    new: list[AlarmEpisode] = []
    rows_read, last_ts = 0, None
    chunk = start
    while chunk < now:
        stop = min(chunk + timedelta(hours=CHUNK_HOURS), now)
        h = agent.history(asset_id, chunk, stop)
        period = timedelta(seconds=agent._period(h))
        for r in agent._rows(h):
            ts = r["ts"]
            if ts <= start:
                continue
            rows_read += 1
            last_ts = ts
            code = str(r.get("alarm_code") or "").strip()
            if not code:
                continue
            e = recent.get(code)
            if e is not None and ts - e.ended_at <= merge:
                if ts > e.ended_at + period / 2:            # it went off and came back
                    e.repeats += 1
                e.ended_at = max(e.ended_at, ts + period)
                e.alarm_seconds += period.total_seconds()
            else:
                e = AlarmEpisode(asset_id=asset_id, code=code, started_at=ts, ended_at=ts + period,
                                 alarm_seconds=period.total_seconds(), state_at_start=str(r.get("machine_state") or ""))
                recent[code] = e
                new.append(e)
            if e.pk:
                dirty[e.pk] = e
        chunk = stop
    with transaction.atomic():
        for e in dirty.values():
            e.save(update_fields=["ended_at", "alarm_seconds", "repeats"])
        AlarmEpisode.objects.bulk_create(new)
        if last_ts is not None or cursor is None:
            AlarmCursor.objects.update_or_create(asset_id=asset_id, defaults={"read_until": last_ts or start})
    return {"rows": rows_read, "new": len(new), "extended": len(dirty)}


def rebuild(asset_id: str, days: int) -> dict:
    """Delete the asset's episodes and read the last `days` again (the backfill)."""
    with transaction.atomic():
        AlarmEpisode.objects.filter(asset_id=asset_id).delete()
        AlarmCursor.objects.filter(asset_id=asset_id).delete()
    return update(asset_id, since=datetime.now(timezone.utc) - timedelta(days=days))


# --- reading ------------------------------------------------------------------------------------------------

def _newest(asset_id: str) -> datetime | None:
    c = AlarmCursor.objects.filter(asset_id=asset_id).first()
    return c.read_until if c else None


def knowledge(asset_id: str) -> dict[str, dict]:
    """The asset's PLC alarms in the graph (`bsk:alarm:<machine>/<CODE>`, subtype "PLC alarm"): node id, description,
    severity, by code. {} when the graph is down."""
    try:
        from context_graph.driver import session
        short = asset_id.rsplit(":", 1)[-1]
        with session() as s:
            rows = s.run("MATCH (n:Alarm) WHERE n.id STARTS WITH $prefix AND n.subtype = 'PLC alarm' "
                         "RETURN n.id AS id, coalesce(n.code, n.name) AS code, n.description AS description, "
                         "n.severity AS severity", prefix=f"bsk:alarm:{short}/").data()
        return {r["code"]: r for r in rows if r.get("code")}
    except Exception as exc:
        logger.warning("alarms: no graph knowledge for %s (%s)", asset_id, exc)
        return {}


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def episode_json(e: AlarmEpisode, cats: dict, know: dict, newest: datetime | None) -> dict:
    k = know.get(e.code) or {}
    return {"id": e.id, "code": e.code, "description": k.get("description") or "", "severity": k.get("severity") or "",
            "category": category_of(e.code, cats), "started_at": _iso(e.started_at), "ended_at": _iso(e.ended_at),
            "minutes": round((e.ended_at - e.started_at).total_seconds() / 60, 1),
            "alarm_minutes": round(e.alarm_seconds / 60, 1), "repeats": e.repeats, "state_at_start": e.state_at_start,
            "active": bool(newest and e.ended_at >= newest)}


def summary(asset_id: str, hours: int = 24, code: str = "", category: str = "", active_only: bool = False,
            limit: int = 200) -> dict:
    """The episodes in the last `hours` (newest first), what is active now, and per code: episodes, repeats, minutes,
    last seen (the Pareto)."""
    hours = max(1, min(int(hours), 24 * 31))
    newest = _newest(asset_id)
    end = newest or datetime.now(timezone.utc)
    since = end - timedelta(hours=hours)
    cats, know = categories(asset_id), knowledge(asset_id)
    found = list(AlarmEpisode.objects.filter(asset_id=asset_id, ended_at__gt=since).order_by("-started_at"))
    if code:
        found = [e for e in found if e.code == code]
    if category:
        if category not in CATEGORIES:
            raise AlarmError(f"category must be one of {', '.join(CATEGORIES)}")
        found = [e for e in found if category_of(e.code, cats) == category]
    items = [episode_json(e, cats, know, newest) for e in found]
    per = defaultdict(lambda: {"episodes": 0, "repeats": 0, "alarm_minutes": 0.0, "last_seen": None})
    for e in found:
        p = per[e.code]
        p["episodes"] += 1
        p["repeats"] += e.repeats
        p["alarm_minutes"] += e.alarm_seconds / 60
        p["last_seen"] = max(p["last_seen"] or e.ended_at, e.ended_at)
    codes = sorted(({"code": c, "description": (know.get(c) or {}).get("description") or "",
                     "severity": (know.get(c) or {}).get("severity") or "", "category": category_of(c, cats),
                     "episodes": p["episodes"], "repeats": p["repeats"], "alarm_minutes": round(p["alarm_minutes"], 1),
                     "last_seen": _iso(p["last_seen"])} for c, p in per.items()),
                   key=lambda x: (-x["episodes"], -x["alarm_minutes"], x["code"]))
    active = [i for i in items if i["active"]]
    if active_only:
        items = active
    limit = max(0, min(int(limit), 1000))
    return {"asset_id": asset_id, "hours": hours, "since": _iso(since), "newest_row": _iso(newest),
            "merge_minutes": MERGE_MINUTES, "active": active, "codes": codes, "episodes": items[:limit],
            "more": len(items) > limit, "categories": CATEGORIES}


def code_detail(asset_id: str, code: str, days: int = 30) -> dict:
    """One code: its episodes over `days`, the Studio's knowledge (procedures, parts, signals from the graph), and the
    problem-solving tickets whose window covers an episode."""
    if not code:
        raise AlarmError("an alarm code is required")
    newest = _newest(asset_id)
    end = newest or datetime.now(timezone.utc)
    cats, know = categories(asset_id), knowledge(asset_id)
    found = list(AlarmEpisode.objects.filter(asset_id=asset_id, code=code, ended_at__gt=end - timedelta(days=days))
                 .order_by("-started_at")[:200])
    k = know.get(code) or {}
    related: list[dict] = []
    if k.get("id"):
        try:
            from context_service import service as context
            ctx = context.entity_context(k["id"])
            for rel in ctx.get("relationships") or []:
                n = rel.get("node") or {}
                p = n.get("properties") or {}
                related.append({"type": rel.get("type"), "label": n.get("label"), "id": n.get("id"), "name": n.get("name"),
                                "text": p.get("summary") or p.get("description") or ""})
        except Exception as exc:
            logger.warning("alarms: no context for %s (%s)", k["id"], exc)
    tickets = []
    if found:
        from problems.models import Ticket
        first = min(e.started_at for e in found)
        for t in Ticket.objects.filter(asset_id=asset_id, window_start__lte=end).exclude(window_end__lt=first):
            t_end = t.window_end or end
            hits = [e.id for e in found if e.started_at <= t_end and e.ended_at >= t.window_start]
            if hits:
                tickets.append({"id": t.id, "number": t.number, "title": t.title, "status": t.status, "episodes": hits})
    order = {"Procedure": 0, "FailureMode": 1, "Component": 2, "Signal": 3}
    related.sort(key=lambda r: (order.get(r["label"], 9), r["name"] or ""))
    return {"asset_id": asset_id, "code": code, "node_id": k.get("id"), "description": k.get("description") or "",
            "severity": k.get("severity") or "", "category": category_of(code, cats), "category_set": code in cats,
            "days": days, "episodes": [episode_json(e, cats, know, newest) for e in found], "related": related,
            "tickets": tickets}


def codes(asset_id: str) -> list[dict]:
    """Every code seen or known, with its category (set or default), for the Studio's settings."""
    cats, know = categories(asset_id), knowledge(asset_id)
    seen = set(AlarmEpisode.objects.filter(asset_id=asset_id).values_list("code", flat=True).distinct())
    out = [{"code": c, "description": (know.get(c) or {}).get("description") or "", "severity": (know.get(c) or {}).get("severity") or "",
            "category": category_of(c, cats), "default": default_category(c), "set": c in cats, "seen": c in seen}
           for c in sorted(seen | set(know) | set(cats))]
    out.sort(key=lambda x: (SEVERITY_ORDER.get(x["severity"], 9), x["code"]))
    return out


def set_category(asset_id: str, code: str, category: str, by: str = "") -> dict:
    if category not in CATEGORIES:
        raise AlarmError(f"category must be one of {', '.join(CATEGORIES)}")
    if not code:
        raise AlarmError("an alarm code is required")
    AlarmCode.objects.update_or_create(asset_id=asset_id, code=code, defaults={"category": category, "updated_by": by[:200]})
    return next(c for c in codes(asset_id) if c["code"] == code)
