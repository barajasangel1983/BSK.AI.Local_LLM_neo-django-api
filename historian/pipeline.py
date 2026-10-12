"""State of the live historian's pipeline (PLC → OPC UA → collector → ingest API → TimescaleDB).

Read from the database the pipeline writes to, plus the ingest API's /health. Nothing here writes.
"""

from datetime import timedelta

import requests
from django.conf import settings
from django.db import Error as DatabaseError
from django.utils import timezone

from .models import LiveCollector, LiveRow, LiveRowMeta, LiveSimEvent

PERIOD_SECONDS = 20
STALE_SECONDS = 90          # a row arrives up to two periods after its `ts`; beyond this nothing is coming
WINDOW_HOURS = 24
BOOT_JITTER_SECONDS = 5     # `plc_boot_utc` is derived from the PLC's uptime and moves by a second within one run

EVENT_TYPES = {1: "Step", 2: "Drift", 3: "Spike", 4: "Stuck value", 5: "More noise", 6: "Broken relation"}


def ingest_health() -> dict:
    """The ingest API's own /health (no token needed)."""
    try:
        response = requests.get(f"{settings.HIST_INGEST_URL.rstrip('/')}/health", timeout=3)
        body = response.json() if response.ok else {}
        return {"ok": response.ok and body.get("status") == "ok", "version": body.get("version"),
                "schema_version": body.get("schema_version")}
    except (requests.RequestException, ValueError):
        return {"ok": False, "version": None, "schema_version": None}


def is_restart(counter, boot, previous, previous_boot) -> bool:
    """True when a row is the first of another PLC run: its counter began again, or its boot time moved."""
    if counter <= previous:
        return True
    return bool(boot and previous_boot) and abs((boot - previous_boot).total_seconds()) > BOOT_JITTER_SECONDS


def _window(now) -> dict:
    """Rows of the last 24 hours: how many, how many are missing, PLC restarts, suspect rows, anomalies."""
    since = now - timedelta(hours=WINDOW_HOURS)
    meta = list(LiveRowMeta.objects.filter(ts__gte=since).order_by("ts")
                .values_list("period_counter", "plc_boot_utc", "quality"))
    missing = restarts = 0
    for (counter, boot, _), (previous, previous_boot, _) in zip(meta[1:], meta):
        if is_restart(counter, boot, previous, previous_boot):
            restarts += 1                       # the counter began again: another PLC run
        else:
            missing += counter - previous - 1   # the counter moved on without a row reaching us
    events = LiveSimEvent.objects.filter(ts__gte=since)
    return {
        "hours": WINDOW_HOURS,
        "rows": len(meta),
        "missing_rows": missing,
        "plc_restarts": restarts,
        "suspect_rows": sum(1 for _, _, quality in meta if quality != "good"),
        "anomaly_rows": events.count(),
        "anomalies": events.values("event_id").distinct().count(),
    }


def status() -> dict:
    """{state: online | stale | offline, ...}. `stale` = the database answers but no recent row
    (generator paused, PLC or its link down, or the collector not delivering)."""
    now = timezone.now()
    out = {"state": "offline", "database": False, "ingest_api": ingest_health(), "checked_at": now,
           "period_seconds": PERIOD_SECONDS, "stale_after_seconds": STALE_SECONDS}
    try:
        newest = LiveRow.objects.order_by("-ts").values("ts", "extruder_id", "machine_state").first()
        collectors = list(LiveCollector.objects.order_by("collector_id")
                          .values("collector_id", "version", "last_seen", "buffered", "last_row_ts"))
        window = _window(now)
    except DatabaseError:
        return out
    age = (now - newest["ts"]).total_seconds() if newest else None
    out.update({
        "database": True,
        "state": "online" if age is not None and age <= STALE_SECONDS else "stale",
        "newest_row": newest,
        "newest_row_age_seconds": round(age) if age is not None else None,
        "collectors": [{**c, "last_seen_age_seconds": round((now - c["last_seen"]).total_seconds())} for c in collectors],
        "window": window,
    })
    return out


def annotate(rows: list[dict]) -> None:
    """Add `quality`, `period_counter` and `anomaly` to rows of the live table (in place)."""
    if not rows:
        return
    first, last = min(r["ts"] for r in rows), max(r["ts"] for r in rows)
    meta = {(m["extruder_id"], m["ts"]): m for m in LiveRowMeta.objects.filter(ts__gte=first, ts__lte=last)
            .values("extruder_id", "ts", "quality", "quality_notes", "period_counter")}
    events = {(e["extruder_id"], e["ts"]): e for e in LiveSimEvent.objects.filter(ts__gte=first, ts__lte=last)
              .values("extruder_id", "ts", "event_id", "event_type", "event_signal", "event_size")}
    for row in rows:
        key = (row["extruder_id"], row["ts"])
        m, e = meta.get(key), events.get(key)
        row["period_counter"] = m["period_counter"] if m else None
        row["quality"] = (m["quality"] + (f" ({m['quality_notes']})" if m["quality_notes"] else "")) if m else ""
        row["anomaly"] = (f"#{e['event_id']} {EVENT_TYPES.get(e['event_type'], e['event_type'])} on {e['event_signal']}"
                          f" ({e['event_size']:g} σ)") if e else ""


EXTRA_COLUMNS = [{"key": "period_counter", "type": "number"}, {"key": "quality", "type": "text"},
                 {"key": "anomaly", "type": "text"}]
