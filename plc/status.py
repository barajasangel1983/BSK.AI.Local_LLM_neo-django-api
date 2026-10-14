"""The PLC status panel from the values read over OPC UA (P13f): the board's LEDs, the facts and the conditions.

A copy of the rules of OPTA's panel (`plcStatus` in server/dashboard.html; specification
docs/NOTE_FOR_OPTA_plc_status.md in the collector repository), in the same order. Pure: the values and the
time are given.
"""

from __future__ import annotations

from datetime import datetime, timezone

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEVELS = ("bad", "warn", "info", "ok")
LEDS = (
    ("status", "STATUS", "blinking slowly = firmware running; fast flashing = waiting for WiFi"),
    ("user", "USER (blue)", "on = the PLC has a network address"),
    ("led1", "LED 1", "generator running"),
    ("led2", "LED 2", "machine state is RUNNING"),
    ("led3", "LED 3", "process alarm active"),
    ("led4", "LED 4", "anomaly being injected"),
)
PLC_DOWN_HELP = ("Look at the board. STATUS blinking slowly and blue USER LED on: the PLC is running and on WiFi, so its "
                 "address has probably changed (DHCP) or the WiFi path to the Mini PC is broken. STATUS flashing fast: it is "
                 "powered but cannot join the WiFi. All LEDs off: no 12-24 V supply. A red LED blinking in a repeating "
                 "long / short pattern: the firmware has crashed; press RESET once or cycle the power.")


def _time(value) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None if value <= EPOCH else value


def duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    if seconds < 86400:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    return f"{seconds // 86400} d {seconds % 86400 // 3600} h"


def _clock(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S UTC") if value else "–"


def panel(values: dict | None, error: str | None, now: datetime, settled: bool) -> dict:
    """LEDs, facts, conditions and the verdict. `settled`: the generator has run at least 45 s since this backend
    first saw it running and no Start was sent in the last 45 s (the "no new row" warning waits for it)."""
    v = values or {}
    server_up = error is None and v.get("Heartbeat.State") is not None
    linked = server_up and v.get("Heartbeat.State") == 2
    running, clock_set = v.get("ExtruderSim.Running") is True, v.get("ExtruderSim.ClockSet") is True
    period = v.get("Sample.PeriodCounter") or 0
    row_ts = _time(v.get("Sample.TimestampUtc"))
    row_age = (now - row_ts).total_seconds() - (v.get("Sample.PeriodSeconds") or 20) if period and row_ts else None
    comm = _time(v.get("Heartbeat.LastCommUtc"))
    comm_age = (now - comm).total_seconds() if comm else None
    uptime, offset = v.get("Plc.UptimeSeconds"), v.get("Plc.ClockOffsetSeconds")
    relays = [n for n in range(4) if v.get(f"DigitalOut[{n}].Value") is True]

    state = {
        "status": "blink" if linked else None, "user": True if linked else None,
        "led1": (running and clock_set) if linked else None,
        "led2": (v.get("Step.StepName") == "RUNNING" and period > 0) if linked else None,
        "led3": (v.get("DigitalIn[8].Value") is True) if linked else None,
        "led4": ((v.get("ExtruderSim.EventType") or 0) != 0) if linked else None,
    }
    leds = [{"key": key, "name": name, "meaning": meaning, "state": state[key]} for key, name, meaning in LEDS]

    facts = [
        ("PLC link", "unknown" if not server_up else "Connected" if linked else "Down"),
        ("Generator", "–" if not linked else "running" if running else "paused"),
        ("Machine state of the last row", v.get("Step.StepName") or "–" if server_up else "–"),
        ("Period counter", str(period) if server_up else "–"),
        ("Uptime", duration(uptime) if linked and uptime is not None else "–"),
        ("Started", _clock(_time(v.get("Plc.BootUtc"))) if linked and uptime else "–"),
        ("Firmware map version", f"{v.get('Plc.MapVersion')} (server expects {v.get('Plc.ExpectedMapVersion')})"
         if server_up and v.get("Plc.MapVersion") else "–"),
        ("PLC clock", "–" if not linked else "not set" if not clock_set else f"{'+' if (offset or 0) > 0 else ''}{offset} s against the Mini PC"),
        ("Last row finished", "none yet" if row_age is None else f"{duration(row_age)} ago"),
        ("Restarts seen", str(v.get("Plc.RestartCount", "–")) if server_up else "–"),
        ("Link losses seen", str(v.get("Plc.LinkLossCount", "–")) if server_up else "–"),
        ("Link up since", _clock(_time(v.get("Plc.LinkUpUtc"))) if server_up else "–"),
    ]

    c: list[tuple[str, str, str]] = []
    if not server_up:
        c.append(("bad", "OPC UA server not reachable", "The Studio cannot read the server on the Mini PC, so nothing is known about "
                  f"the PLC{f' ({error})' if error else ''}. Check the scheduled task \"BSK OPC UA Server\" and "
                  "server\\logs\\bsk_opcua_server.log on the Mini PC, and the Tailscale link."))
    elif not linked:
        since = _time(v.get("Plc.LinkDownUtc"))
        c.append(("bad", "PLC not answering" + (f" since {_clock(since)}" if since else ""),
                  "The server cannot read the PLC over Modbus, so no new rows arrive and the values are the last ones read. "
                  + PLC_DOWN_HELP))
    else:
        if v.get("Plc.MapVersion") != v.get("Plc.ExpectedMapVersion"):
            c.append(("bad", "Firmware and server do not match", f"The PLC reports map version {v.get('Plc.MapVersion')}, the "
                      f"server expects {v.get('Plc.ExpectedMapVersion')}. The values cannot be trusted. Upload the firmware that "
                      "belongs to this server, or restart the server after an upload."))
        if relays:
            c.append(("bad", f"Relay {', '.join(str(n + 1) for n in relays)} reads ON", "This firmware never switches a relay. Treat "
                      "it as a fault: check what is wired to the output and reset the PLC."))
        if comm_age is not None and comm_age > 5:
            c.append(("warn", f"PLC data is {duration(comm_age)} old", "The link is reported up but the last successful read is "
                      "old. If it keeps growing the link is failing."))
        if not clock_set:
            c.append(("warn", "PLC clock not set", "The PLC has no clock of its own and produces no rows until the server sets it, "
                      "which it does within seconds of connecting. If this stays, look at the server log."))
        elif offset is not None and abs(offset) > 2:
            c.append(("warn", f"PLC clock is {abs(offset)} s {'ahead of' if offset > 0 else 'behind'} the Mini PC", "Row timestamps "
                      "are off by that much. The server corrects the clock every 10 minutes; if the offset stays, restart the server task."))
        if clock_set and not running:
            c.append(("warn", "Generator paused", "No rows are produced until Start is pressed. The period counter continues from "
                      "where it stopped."))
        if settled and (uptime or 0) >= 60 and row_age is not None and row_age > 30:
            c.append(("warn", f"No new row for {duration(row_age)}", "A row is expected every 20 s while the generator runs. If the "
                      "heartbeat still counts, restart the PLC."))
        if uptime is not None and uptime < 600:
            c.append(("info", f"PLC started {duration(uptime)} ago", "After a start the generator begins again from its seed: the "
                      "period counter is back at 1, the day's batch numbers restart and the same values repeat with new timestamps."))
        if clock_set and running and period == 0:
            c.append(("info", "Waiting for the first row", "The first partial 20 s period after a start is not sent; the first row "
                      "appears within 40 s."))
        if v.get("DigitalIn[8].Value") is True:
            c.append(("info", f"Process alarm: {v.get('Faults.ActiveLabel') or 'code ' + str(v.get('Faults.ActiveCode'))}",
                      "This is part of the simulated extruder data, not a fault of the PLC. No action."))
        if (v.get("ExtruderSim.EventType") or 0) != 0:
            c.append(("info", f"Anomaly {v.get('ExtruderSim.EventId')} being injected on {v.get('ExtruderSim.EventSignal')}",
                      "A deliberate test disturbance in the data. No action."))
        if not any(level in ("bad", "warn") for level, _, _ in c):
            c.insert(0, ("ok", "PLC healthy", "Firmware running, link up, clock set, generator producing a row every 20 s. Machine "
                         f"state of the last row: {v.get('Step.StepName') or 'none yet'}."))
    conditions = [{"level": level, "title": title, "meaning": meaning} for level, title, meaning in c]
    # The verdict is the worst of bad and warn; "info" is news, not a fault.
    verdict = "bad" if any(x["level"] == "bad" for x in conditions) else "warn" if any(x["level"] == "warn" for x in conditions) else "ok"
    return {"leds": leds, "facts": [{"name": k, "value": val} for k, val in facts], "conditions": conditions,
            "verdict": verdict, "linked": linked, "server_up": server_up, "running": running if linked else None}
