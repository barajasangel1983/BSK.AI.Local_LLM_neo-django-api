"""The transfer check of the live historian (brief `Historian_table_brief_for_PLC.md`, section 10).

Three parts, all computed from rows already read (nothing here touches a database):

    transfer(...)   did every row arrive, once, whole and as the PLC computed it      → pass or fail
    likeness(...)   does the data look like the historian's, per machine state        → figures next to their targets
    anomalies(...)  the injected-anomaly record: events, types, lengths               → figures, and its own rules

`rows` are historian_rows as dicts, `meta` and `events` are row_meta and sim_events keyed by `ts`.
"""

import re
from collections import Counter
from datetime import timedelta
from statistics import mean, median

from .pipeline import EVENT_TYPES, PERIOD_SECONDS, is_restart

STATES = ["RUNNING", "STOPPED", "WARM UP", "CHANGEOVER", "FLUSH", "ALARMED", "COOL DOWN"]
DOWN_STATES = ("STOPPED", "ALARMED")
# Share of the rows in each state in March (brief 5.1); the PLC draws its state blocks with the same odds.
STATE_SHARE = {"RUNNING": 69.7, "STOPPED": 15.9, "WARM UP": 3.5, "CHANGEOVER": 2.8, "FLUSH": 2.7, "ALARMED": 2.7,
               "COOL DOWN": 2.6}
RECIPE_TARGET = {"RC-A": 10000.00, "RC-B": 9000.00, "RC-C": 9465.96, "RC-D": 10451.77, "RC-E": 8407.37,
                 "RC-F": 8287.95, "RC-G": 8122.10, "RC-H": 10012.23, "RC-I": 8820.56, "RC-J": 9252.52}
ALARM_CODES = {
    "OUT_OF_SPEC_DENSITY", "OUT_OF_SPEC_DIAMETER", "LO_MOISTURE", "HI_MOISTURE", "SCREEN_BLOCK", "MOTOR_OVERLOAD",
    "HIGH_SCRAP", "UNPLANNED_STOP", "DIE_PLUG", "HEAD_TEMP_DRIFT", "MOISTURE_SENSOR_FAIL", "SPEED_UNDER_TARGET",
    "FEEDER_STARVE", "HI_DIE_PRESSURE", "FEEDER_OVERLOAD", "TEMP_SENSOR_FAIL", "LOW_OEE", "LO_TEMP_ZONE_1",
    "HI_TEMP_ZONE_3", "HI_TORQUE",
}
# The normal ranges while RUNNING, taken from March's flat noise (brief section 7).
NORMAL_RANGE = {
    "throughput_actual_kg_hr": (7972, 10385), "feeder_rate_actual_kg_hr": (7959, 10391),
    "screw_speed_rpm": (405.9, 444.6), "screw_torque_pct": (88.6, 101.5), "motor_power_kw": (315.3, 385.0),
    "die_pressure_bar": (84.3, 99.2), "steam_flow_kg_hr": (96.6, 132.9), "water_injection_l_hr": (76.2, 107.6),
    "barrel_zone_1_temp_c": (81.0, 99.0), "barrel_zone_2_temp_c": (88.2, 106.8),
    "barrel_zone_3_temp_c": (95.3, 114.7), "head_temp_c": (93.9, 115.9), "product_temp_discharge_c": (93.5, 116.3),
    "product_moisture_pct": (8.15, 10.85), "bulk_density_g_l": (332.4, 377.5), "kibble_diameter_mm": (7.15, 9.85),
}
# What a day should show while RUNNING with bell-shaped noise (OPTA's simulated fortnight; VPS_HANDOFF 9.5).
# March, with flat noise: an alarm in 34 % of the rows, 13 starts an hour, quality 88 %, OEE 81 %.
RUNNING_TARGET = {"quality_pct": 95.0, "oee_pct": 88.0, "alarm_rows_pct": 10.0}
ALARM_STARTS_PER_HOUR = 6.0
ANOMALIES_PER_DAY = 6.0
MIN_GAP_BETWEEN_ANOMALIES = timedelta(minutes=30)

TEXT_COLUMNS = ("extruder_id", "batch_id", "recipe_id", "shift_code", "machine_state")
NOT_NUMBERS = TEXT_COLUMNS + ("ts", "alarm_active", "alarm_code")
ROUNDING = 0.011            # two values rounded to two decimals can be 0.01 apart from the exact relation
PERIODS_PER_HOUR = 3600 // PERIOD_SECONDS
LATE_SECONDS = 60           # a row that reaches the database later than this after its period waited in a buffer


def shift_of(ts) -> str:
    return "A" if 7 <= ts.hour < 15 else "B" if 15 <= ts.hour < 23 else "C"


def _row_faults(row: dict) -> list[str]:
    """Names of the checks one row fails: filled, two decimals, vocabulary, and the relations of brief 6.1."""
    faults = []
    numbers = {k: v for k, v in row.items() if k not in NOT_NUMBERS}
    if any(v is None for v in numbers.values()) or any(not row[c] for c in TEXT_COLUMNS) \
            or row["alarm_active"] is None or row["alarm_code"] is None:
        return ["a column is empty"]
    if any(abs(round(v, 2) - v) > 1e-9 for v in numbers.values()):
        faults.append("a number has more than two decimals")

    if row["machine_state"] not in STATES:
        faults.append("machine_state is not in the vocabulary")
    if row["shift_code"] != shift_of(row["ts"]):
        faults.append("shift_code does not follow the hour of ts")
    if row["recipe_id"] not in RECIPE_TARGET:
        faults.append("recipe_id is not in the vocabulary")
    elif abs(row["throughput_target_kg_hr"] - RECIPE_TARGET[row["recipe_id"]]) > 0.005:
        faults.append("the target is not the row's recipe's")
    if not re.fullmatch(rf"{re.escape(row['extruder_id'])}-\d{{8}}-\d{{3}}", row["batch_id"]):
        faults.append("batch_id is not EXTR01-YYYYMMDD-NNN")
    if row["alarm_code"] and row["alarm_code"] not in ALARM_CODES:
        faults.append("alarm_code is not in the vocabulary")
    if bool(row["alarm_code"]) != row["alarm_active"]:
        faults.append("alarm_code and alarm_active disagree")

    actual, target = row["throughput_actual_kg_hr"], row["throughput_target_kg_hr"]
    if abs(row["oee_pct"] - row["availability_pct"] * row["performance_pct"] * row["quality_pct"] / 10000) > 0.02:
        faults.append("oee is not availability x performance x quality")
    if target and abs(row["performance_pct"] - min(100.0, actual / target * 100)) > ROUNDING:
        faults.append("performance is not actual / target")
    total = actual * PERIOD_SECONDS / 3600
    if abs(row["good_output_kg"] + row["scrap_output_kg"] - total) > ROUNDING:
        faults.append("good + scrap is not the period's throughput")
    if abs(row["scrap_output_kg"] - total * (0.20 if row["alarm_active"] else 0.05)) > ROUNDING:
        faults.append("scrap is not 5 % (20 % with an alarm)")
    for name, low, high, upper, lower in (
            ("zone 2 - zone 1", 5, 10, "barrel_zone_2_temp_c", "barrel_zone_1_temp_c"),
            ("zone 3 - zone 2", 5, 10, "barrel_zone_3_temp_c", "barrel_zone_2_temp_c"),
            ("head - zone 3", -5, 5, "head_temp_c", "barrel_zone_3_temp_c"),
            ("product - head", -3, 3, "product_temp_discharge_c", "head_temp_c")):
        if not low - ROUNDING <= row[upper] - row[lower] <= high + ROUNDING:
            faults.append(f"{name} is outside {low} to {high} °C")
    if abs(row["downtime_min"] - (0.33 if row["machine_state"] in DOWN_STATES else 0)) > 1e-9:
        faults.append("downtime_min is not 0.33 when down, 0 otherwise")
    return faults


def transfer(rows: list[dict], meta: dict, events: dict) -> dict:
    """{"checks": [{name, ok, detail}], "info": [text]}. A row inside an injected anomaly may break a relation
    on purpose (the anomaly moves its own signal only), so those rows are counted apart and do not fail a check."""
    checks, info = [], []

    def check(name, bad, detail=""):
        checks.append({"name": name, "ok": not bad, "detail": detail})

    off_grid = [r["ts"] for r in rows if r["ts"].second % PERIOD_SECONDS or r["ts"].microsecond]
    check("every ts on the 20 s grid (second 00, 20 or 40)", off_grid, _some(off_grid))

    without_meta = [r["ts"] for r in rows if r["ts"] not in meta]
    orphans = sorted(set(meta) - {r["ts"] for r in rows})
    check("every row has its transfer record (row_meta), and the reverse", without_meta or orphans,
          _some(without_meta + orphans))

    missing, repeats, pauses, paused, restarts = 0, [], 0, timedelta(0), []
    known = [r["ts"] for r in rows if r["ts"] in meta]
    for previous, ts in zip(known, known[1:]):
        before, now = meta[previous], meta[ts]
        step, gap = now["period_counter"] - before["period_counter"], ts - previous
        if is_restart(now["period_counter"], now["plc_boot_utc"], before["period_counter"], before["plc_boot_utc"]):
            restarts.append(ts)
        elif step > 1:
            missing += step - 1
            repeats.append(f"{step - 1} before {_utc(ts)}")
        elif gap > timedelta(seconds=PERIOD_SECONDS):       # the counter held: the generator was paused
            pauses, paused = pauses + 1, paused + gap - timedelta(seconds=PERIOD_SECONDS)
    check("the period counter advances by 1 (no period lost)", missing,
          f"{missing} missing: {', '.join(repeats[:5])}" if missing else "")
    # (extruder_id, ts) is the table's key, so a period cannot be stored twice; a gap in time with an unbroken
    # counter is a pause of the generator, which produces no rows.
    info.append(f"PLC restarts: {len(restarts)}{_some(restarts, ' at ')}; "
                f"pauses of the generator: {pauses} ({_minutes(paused)} without rows)")

    faults, in_anomaly, examples = Counter(), Counter(), {}
    for row in rows:
        for fault in _row_faults(row):
            (in_anomaly if row["ts"] in events else faults)[fault] += 1
            if row["ts"] not in events:
                examples.setdefault(fault, row["ts"])
    check("every column filled, two decimals, vocabulary, and the relations of the brief (6.1)", faults,
          "; ".join(f"{name}: {n} rows, first {_utc(examples[name])}" for name, n in faults.most_common()))
    if in_anomaly:
        info.append("relations broken inside an injected anomaly (expected): "
                    + "; ".join(f"{name}: {n} rows" for name, n in in_anomaly.most_common()))

    changed = sum(1 for a, b in zip(rows, rows[1:]) if a["batch_id"] == b["batch_id"] and a["recipe_id"] != b["recipe_id"])
    check("a batch keeps one recipe", changed, f"{changed} changes inside a batch" if changed else "")

    notes = Counter(note.split(":")[0] for m in meta.values() for note in m["quality_notes"].split(",") if note)
    suspect = sum(1 for m in meta.values() if m["quality"] != "good")
    check("every row marked good by the collector", suspect, f"{suspect} suspect" if suspect else "")
    if notes:
        info.append("collector notes: " + ", ".join(f"{name} {n}" for name, n in notes.most_common()))

    # A row normally arrives within two periods of its end. Rows that arrive later waited in the collector's
    # buffer (VPS or Tailscale unreachable, or the collector started after the PLC): one line per such stretch.
    delays = {ts: (meta[ts]["received_at"] - ts).total_seconds() - PERIOD_SECONDS for ts in known}
    if delays:
        info.append(f"from the end of a period to the database: median {median(delays.values()):.0f} s")
    stretch = []
    for ts in known + [None]:
        if ts is not None and delays[ts] > LATE_SECONDS:
            stretch.append(ts)
        elif stretch:
            info.append(f"waited in the collector's buffer: {len(stretch)} rows, {_utc(stretch[0])} to "
                        f"{_utc(stretch[-1])}, delivered {_utc(meta[stretch[-1]]['received_at'])} "
                        f"(the first after {_minutes(timedelta(seconds=delays[stretch[0]]))})")
            stretch = []
    return {"checks": checks, "info": info}


def likeness(rows: list[dict], events: dict) -> dict:
    """Per-state figures, and for RUNNING the figures the brief gives targets for."""
    hours = len(rows) / PERIODS_PER_HOUR
    states = []
    for state in STATES:
        found = [r for r in rows if r["machine_state"] == state]
        blocks = sum(1 for a, b in zip([None] + rows, rows) if b["machine_state"] == state
                     and (a is None or a["machine_state"] != state))
        entry = {"state": state, "rows": len(found), "share_pct": 100 * len(found) / len(rows),
                 "target_share_pct": STATE_SHARE[state], "blocks": blocks,
                 "block_minutes": len(found) / blocks / 3 if blocks else None}
        for column in ("throughput_actual_kg_hr", "availability_pct", "performance_pct", "quality_pct", "oee_pct"):
            entry[column] = mean(r[column] for r in found) if found else None
        entry["alarm_rows_pct"] = 100 * sum(1 for r in found if r["alarm_active"]) / len(found) if found else None
        states.append(entry)

    starts = sum(1 for a, b in zip([None] + rows, rows) if b["alarm_active"]
                 and (a is None or a["alarm_code"] != b["alarm_code"]))
    running = [r for r in rows if r["machine_state"] == "RUNNING" and r["ts"] not in events]
    outside = {column: sum(1 for r in running if not low <= r[column] <= high)
               for column, (low, high) in NORMAL_RANGE.items()}
    any_outside = sum(1 for r in running
                      if any(not low <= r[column] <= high for column, (low, high) in NORMAL_RANGE.items()))
    return {
        "hours": hours, "states": states,
        "alarm_starts": starts, "alarm_starts_per_hour": starts / hours if hours else None,
        "running_rows": len(running),
        "outside_normal_pct": 100 * any_outside / len(running) if running else None,
        "outside_by_signal": {c: 100 * n / len(running) for c, n in sorted(outside.items(), key=lambda kv: -kv[1]) if n},
    }


def anomalies(rows: list[dict], events: dict) -> dict:
    """The injected anomalies of the window, one entry per event, and the rules of brief 13.2 they must keep."""
    state = {r["ts"]: r["machine_state"] for r in rows}
    found = []
    for ts in sorted(events):
        e = events[ts]
        last = found[-1] if found else None
        if last and last["event_id"] == e["event_id"] and ts - last["end"] <= timedelta(seconds=PERIOD_SECONDS):
            last["end"], last["rows"] = ts, last["rows"] + 1
        else:
            found.append({"event_id": e["event_id"], "type": EVENT_TYPES.get(e["event_type"], str(e["event_type"])),
                          "signal": e["event_signal"], "size": e["event_size"], "start": ts, "end": ts, "rows": 1,
                          "gap_before": ts - last["end"] - timedelta(seconds=PERIOD_SECONDS) if last else None})
    not_running = sum(1 for ts in events if state.get(ts, "RUNNING") != "RUNNING")
    too_close = [e["start"] for e in found if e["gap_before"] is not None and e["gap_before"] < MIN_GAP_BETWEEN_ANOMALIES]
    hours = len(rows) / PERIODS_PER_HOUR
    return {
        "events": found, "rows": len(events), "by_type": dict(Counter(e["type"] for e in found)),
        "per_day": len(found) / hours * 24 if hours else None,
        "checks": [
            {"name": "anomalies only while RUNNING", "ok": not not_running,
             "detail": f"{not_running} rows in another state" if not_running else ""},
            {"name": "at least 30 minutes between two anomalies", "ok": not too_close, "detail": _some(too_close)},
        ],
    }


def _utc(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _some(stamps: list, lead: str = "") -> str:
    if not stamps:
        return ""
    return f"{lead}{', '.join(_utc(ts) for ts in stamps[:5])}{f' and {len(stamps) - 5} more' if len(stamps) > 5 else ''}"


def _minutes(span: timedelta) -> str:
    seconds = span.total_seconds()
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min"
