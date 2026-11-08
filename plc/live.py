"""The machine live, as the PLC sees it now (P31): a curated read of OPTA's OPC UA server for a client (BSKLAB EDGE).

Built on `service.opcua()` (one session, every variable, cached PLC_STATUS_CACHE_SECONDS), so the PLC sees one client
however many tablets poll. Read only: Start / Pause stay on the Studio's PLC page.

- identification, state and step, heartbeat and sample age;
- the active fault (code 0 = none), with its label and the PLC's severity;
- production (kg, availability, performance, quality, OEE);
- the labelled analog inputs and outputs: value, unit, the PLC's limits, and for a historian signal its Studio name and
  normal range ("outside" when the value is out of it);
- the labelled digital inputs and outputs;
- with `everything`, every variable (path, value, type), as OPTA's dashboard shows them.

Unused slots are left out.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from django.conf import settings

UNUSED = {"", "unused", "no fault"}
SLOT = re.compile(r"^(AnalogIn|AnalogOut|DigitalIn|DigitalOut)\[(\d+)\]\.(\w+)$")
PRODUCTION = ("GoodOutputKg", "ScrapOutputKg", "DowntimeMin", "AvailabilityPct", "PerformancePct", "QualityPct", "OeePct")


class LiveError(ValueError):
    """Shown to the client as the tool's error text."""


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def _age(v, now: datetime) -> float | None:
    return max(0.0, round((now - v).total_seconds(), 1)) if isinstance(v, datetime) else None


SIGNALS_KEEP_SECONDS = 300      # names and normal ranges change only when the graph is edited (a read takes ~1.4 s)
_signals_cache: dict = {}


def _signals(asset_id: str) -> dict:
    """The asset's historian signals by column: name, unit, normal range (from the graph, kept SIGNALS_KEEP_SECONDS;
    empty when the graph is down, and then asked again next time)."""
    import time
    kept = _signals_cache.get(asset_id)
    if kept and time.monotonic() - kept[0] < SIGNALS_KEEP_SECONDS:
        return kept[1]
    try:
        from chat import asset_context
        found = {s["column"]: s for s in asset_context.signal_columns(asset_id)}
    except Exception:
        return {}
    _signals_cache[asset_id] = (time.monotonic(), found)
    return found


def build(read: dict, asset_id: str, signals: dict, everything: bool = False, now: datetime | None = None) -> dict:
    """The answer from one read of `service.opcua()`."""
    now = now or datetime.now(timezone.utc)
    values = {v["path"]: v["value"] for v in read.get("variables") or []}
    slots: dict = {}
    for path, value in values.items():
        m = SLOT.match(path)
        if m:
            slots.setdefault((m.group(1), int(m.group(2))), {})[m.group(3)] = value

    analog, digital = [], []
    for (kind, index), f in sorted(slots.items()):
        label = str(f.get("Label") or "").strip()
        if label.lower() in UNUSED:
            continue
        item = {"slot": f"{kind}[{index}]", "label": label, "value": f.get("Value"),
                "direction": "in" if kind.endswith("In") else "out"}
        if kind.startswith("Analog"):
            sig = signals.get(label) or {}
            v = f.get("Value")
            low, high = sig.get("low"), sig.get("high")
            item.update({"name": sig.get("name") or label, "unit": f.get("EngineeringUnits") or sig.get("unit") or "",
                         "plc_low": f.get("LowLimit"), "plc_high": f.get("HighLimit"), "low": low, "high": high,
                         "signal_id": sig.get("signal_id"), "column": label if sig else None,
                         "outside": bool(isinstance(v, (int, float)) and ((low is not None and v < low)
                                                                          or (high is not None and v > high)))})
            analog.append(item)
        else:
            digital.append(item)
    # the historian's signals first (they have names and ranges), then raw inputs, outputs last
    analog.sort(key=lambda a: (a["direction"] != "in", a["column"] is None, int(a["slot"].split("[")[1][:-1])))

    code = values.get("Faults.ActiveCode")
    label = str(values.get("Faults.ActiveLabel") or "")
    active = bool(code) and label.lower() not in UNUSED
    out = {
        "asset_id": asset_id, "endpoint": read.get("endpoint"), "read_at": _iso(read.get("read_at")), "error": read.get("error"),
        "identification": {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith("Identification.")},
        "state": {"name": values.get("Step.StepName") or "", "step": values.get("Step.CurrentStep"),
                  "state_code": values.get("State.CurrentState"), "since": _iso(values.get("Step.StepStartUtc")),
                  "heartbeat_age_s": _age(values.get("Heartbeat.LastCommUtc"), now),
                  "sample_at": _iso(values.get("Sample.TimestampUtc")), "sample_age_s": _age(values.get("Sample.TimestampUtc"), now),
                  "period_s": values.get("Sample.PeriodSeconds")},
        "fault": {"active": active, "code": code if active else 0, "label": label if active else "",
                  "severity": values.get("Faults.Severity") if active else None,
                  "since": _iso(values.get("Faults.Timestamp")) if active else None},
        "production": {k: values.get(f"Production.{k}") for k in PRODUCTION if f"Production.{k}" in values},
        "analog": analog, "digital": digital,
    }
    if everything:
        out["variables"] = [{"path": v["path"], "value": _iso(v["value"]), "type": v.get("type"), "good": v.get("good")}
                            for v in read.get("variables") or []]
    return out


def live(asset_id: str, everything: bool = False) -> dict:
    """The machine now. Only the machine on the Studio's OPC UA connection (PLC_ASSET) can be read."""
    if asset_id.rsplit(":", 1)[-1] != settings.PLC_ASSET:
        raise LiveError(f"no OPC UA connection for {asset_id} (the Studio reads {settings.PLC_ASSET})")
    from . import service
    read = service.opcua()
    if read.get("error") and not read.get("variables"):
        return {"asset_id": asset_id, "endpoint": read.get("endpoint"), "read_at": _iso(read.get("read_at")),
                "error": f"The PLC could not be read: {read['error']}"[:300], "analog": [], "digital": [],
                "production": {}, "fault": {"active": False}, "state": {}, "identification": {}}
    return build(read, asset_id, _signals(asset_id), everything)
