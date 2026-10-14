"""The PLC page's logic (P13f): the status, cached for a moment, and Start / Pause with their safeguards.

Safeguards of a command (Angel, 2026-10-14: "safeguards ok"): PLC_CONTROL_ENABLED in .env, off by default; the
command typed again as a confirmation; one command at a time and PLC_COMMAND_SPACING_SECONDS between two; refused
when the generator is already in the asked state or the PLC cannot be read; the result checked by reading
ExtruderSim.Running again. Every attempt is recorded in PlcCommand, refusals included.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from django.conf import settings
from django.utils import timezone as dj_timezone

from . import opc, status
from .models import PlcCommand

SETTLE_SECONDS = 45          # after a Start, or after the generator is first seen running, "no new row" waits this long
VERIFY_SECONDS = 4           # how long the result of a command is looked for

_cache: dict = {}
_lock = threading.Lock()                     # one command at a time in this process
_running_since: dict = {"at": None}          # when this backend first saw the generator running


class Refused(Exception):
    def __init__(self, reason: str, code: int):
        super().__init__(reason)
        self.code = code


def _read() -> tuple[dict | None, str | None]:
    try:
        return opc.read_values(), None
    except opc.OpcError as exc:
        return None, str(exc)


def current(fresh: bool = False) -> dict:
    """The status answer: values, LEDs, facts, conditions, verdict, and what the controls allow. Read at most once
    in PLC_STATUS_CACHE_SECONDS unless `fresh`."""
    if not fresh and _cache.get("at") and time.monotonic() - _cache["at"] < settings.PLC_STATUS_CACHE_SECONDS:
        return _cache["answer"]
    values, error = _read()
    now = datetime.now(timezone.utc)
    running = bool(values and values.get("ExtruderSim.Running") is True and values.get("ExtruderSim.ClockSet") is True)
    if not running:
        _running_since["at"] = None
    elif _running_since["at"] is None:
        _running_since["at"] = now
    last_start = PlcCommand.objects.filter(command="Start", outcome="done").values_list("created_at", flat=True).first()
    settled = (_running_since["at"] is not None and (now - _running_since["at"]).total_seconds() > SETTLE_SECONDS
               and (last_start is None or (now - last_start).total_seconds() > SETTLE_SECONDS))
    answer = {
        "endpoint": settings.PLC_OPCUA_URL, "asset": settings.PLC_ASSET, "read_at": now, "error": error,
        "values": values or {}, **status.panel(values, error, now, settled),
        "control": controls(),
    }
    _cache.update(at=time.monotonic(), answer=answer)
    return answer


_opcua_cache: dict = {}


def opcua() -> dict:
    """The OPC UA server as OPTA's dashboard on port 8891 shows it: every variable, value and type. Read at most once
    in PLC_STATUS_CACHE_SECONDS."""
    if _opcua_cache.get("at") and time.monotonic() - _opcua_cache["at"] < settings.PLC_STATUS_CACHE_SECONDS:
        return _opcua_cache["answer"]
    started = time.monotonic()
    try:
        found, error = opc.read_all(), None
    except opc.OpcError as exc:
        found, error = {"browsed_now": False, "variables": []}, str(exc)
    answer = {"endpoint": settings.PLC_OPCUA_URL, "asset": settings.PLC_ASSET, "read_at": datetime.now(timezone.utc),
              "error": error, "seconds": round(time.monotonic() - started, 2), **found}
    _opcua_cache.update(at=time.monotonic(), answer=answer)
    return answer


_limits_cache: dict = {}
SLOT_LIMITS_KEEP_SECONDS = 30


def slot_limits() -> dict:
    """The PLC's LowLimit / HighLimit of each analog slot that has a label, by the label, which is the historian
    column (BSK machine template): {column: {"slot", "low", "high", "unit"}}. Kept 30 s. Raises opc.OpcError."""
    if _limits_cache.get("at") and time.monotonic() - _limits_cache["at"] < SLOT_LIMITS_KEEP_SECONDS:
        return _limits_cache["limits"]
    fields: dict = {}
    for v in opc.read_all()["variables"]:
        slot, _, name = v["path"].rpartition(".")
        if slot.startswith(("AnalogIn[", "AnalogOut[")) and name in ("Label", "LowLimit", "HighLimit", "EngineeringUnits"):
            fields.setdefault(slot, {})[name] = v["value"]
    limits = {f["Label"]: {"slot": slot, "low": f.get("LowLimit"), "high": f.get("HighLimit"), "unit": f.get("EngineeringUnits") or ""}
              for slot, f in fields.items()
              if f.get("Label") and f.get("Label") != "Unused" and isinstance(f.get("LowLimit"), (int, float))
              and isinstance(f.get("HighLimit"), (int, float)) and f["LowLimit"] < f["HighLimit"]}
    _limits_cache.update(at=time.monotonic(), limits=limits)
    return limits


def controls() -> dict:
    last = PlcCommand.objects.exclude(outcome="refused").values_list("created_at", flat=True).first()
    return {"enabled": settings.PLC_CONTROL_ENABLED, "spacing_seconds": settings.PLC_COMMAND_SPACING_SECONDS,
            "last_command_at": last, "commands": list(opc.COMMANDS)}


def command(name: str, confirm: str, by: str) -> PlcCommand:
    """Start or Pause the generator, with every safeguard. Returns the record; raises Refused (recorded too)."""
    by = (by or "").strip()[:120]

    def refuse(reason: str, code: int, **fields):
        PlcCommand.objects.create(command=str(name)[:16], requested_by=by, outcome="refused", reason=reason[:255], **fields)
        raise Refused(reason, code)

    if name not in opc.COMMANDS:
        refuse(f"not a command; the commands are {' and '.join(opc.COMMANDS)}", 400)
    if confirm != name:
        refuse(f"the confirmation must be the command itself, typed: {name}", 400)
    if not settings.PLC_CONTROL_ENABLED:
        refuse("control is off: PLC_CONTROL_ENABLED is not set in the backend's .env", 403)
    if not _lock.acquire(blocking=False):
        refuse("another command is being sent", 409)
    try:
        last = PlcCommand.objects.exclude(outcome="refused").values_list("created_at", flat=True).first()
        if last is not None and dj_timezone.now() - last < timedelta(seconds=settings.PLC_COMMAND_SPACING_SECONDS):
            wait = settings.PLC_COMMAND_SPACING_SECONDS - (dj_timezone.now() - last).total_seconds()
            refuse(f"the last command was less than {settings.PLC_COMMAND_SPACING_SECONDS} s ago; wait {wait:.0f} s", 429)
        before, error = _read()
        if before is None or before.get("Heartbeat.State") != 2:
            refuse("the PLC cannot be read: " + (error or "PLC not answering"), 503)
        running = before.get("ExtruderSim.Running") is True
        fields = {"running_before": running, "period_before": before.get("Sample.PeriodCounter")}
        if name == "Start" and running:
            refuse("the generator is already running", 409, **fields)
        if name == "Pause" and not running:
            refuse("the generator is already paused", 409, **fields)
        started = time.monotonic()
        try:
            opc.call(name)
        except opc.OpcError as exc:
            _cache.clear()
            return PlcCommand.objects.create(command=name, requested_by=by, outcome="failed", reason=f"the call failed: {exc}"[:255],
                                             seconds=round(time.monotonic() - started, 2), **fields)
        after, wanted = None, name == "Start"
        while time.monotonic() - started < VERIFY_SECONDS:
            after, _ = _read()
            if after and after.get("ExtruderSim.Running") is wanted:
                break
            time.sleep(0.5)
        _cache.clear()
        done = bool(after and after.get("ExtruderSim.Running") is wanted)
        return PlcCommand.objects.create(
            command=name, requested_by=by, outcome="done" if done else "failed",
            reason="" if done else f"the generator did not show {'running' if wanted else 'paused'} within {VERIFY_SECONDS} s",
            running_after=after.get("ExtruderSim.Running") if after else None,
            period_after=after.get("Sample.PeriodCounter") if after else None,
            seconds=round(time.monotonic() - started, 2), **fields)
    finally:
        _lock.release()


def record_json(c: PlcCommand) -> dict:
    return {"id": c.id, "created_at": c.created_at, "command": c.command, "requested_by": c.requested_by, "outcome": c.outcome,
            "reason": c.reason, "running_before": c.running_before, "running_after": c.running_after,
            "period_before": c.period_before, "period_after": c.period_after, "seconds": c.seconds}
