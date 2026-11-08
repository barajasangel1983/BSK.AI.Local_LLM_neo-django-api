"""The digital twin (P32): each machine's manifest, generated from the graph, and its live values.

The manifest says which template draws the machine, its areas (the graph's top-level parts) with what the Studio knows
about each (approved facts and manuals from Coverage, procedures, the PLC alarms associated with it), its sensors (the
historian signals, each on the area of the part that carries it, `MONITORED_BY`), and which columns drive the motion and
heat colours. The package `@bsklab/twin` draws it; machines are data, templates are code.

Live values: the PLC over OPC UA (P31's `plc.live`), else the newest historian row, labelled as such. Never mixed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .models import TwinManifest

logger = logging.getLogger("chat")

VERSION = 1
# The graph's part types (component_type, lower case) → the template's slots.
SLOTS = {"feeder": "feeder", "conditioner": "conditioner", "screw_barrel": "barrel", "motor": "drive", "die": "die",
         "cutter": "cutter", "measurement_point": "discharge", "instrument": "scale", "controller": "panel"}
TEMPLATES = {"extruder": "single_screw_extruder"}
BINDINGS = {"single_screw_extruder": {
    "screw_rpm": "screw_speed_rpm", "throughput": "throughput_actual_kg_hr", "steam": "steam_flow_kg_hr",
    "heat": {"zone1": "barrel_zone_1_temp_c", "zone2": "barrel_zone_2_temp_c", "zone3": "barrel_zone_3_temp_c",
             "die": "head_temp_c"}}}
PROCEDURES_SHOWN = 6


class TwinError(ValueError):
    """Shown to the client as is."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def generate(asset_id: str) -> dict:
    """A manifest from the graph (areas, knowledge, alarms, sensors)."""
    from chat import asset_context
    from context_graph import coverage
    from context_graph.driver import session

    grid = coverage.grid(asset_id)
    asset_name = grid["asset"]["name"]
    with session() as s:
        info = s.run("MATCH (a:Asset {id: $id}) RETURN a.asset_type AS t", id=asset_id).single()
    template = TEMPLATES.get(str((info or {}).get("t") or "").strip().lower(), "")

    # areas: the depth-0 rows, each with the rows after it until the next area
    areas, members, current = [], {}, None
    for row in grid["rows"]:
        if row["depth"] == 0:
            slot = SLOTS.get(str(row.get("type") or "").strip().lower())
            current = {"id": row["id"], "name": row["name"], "slot": slot or f"other:{row['id']}", "facts": 0,
                       "manuals": 0}
            areas.append(current)
            members[row["id"]] = current
        if current is None:
            continue
        members[row["id"]] = current
        current["facts"] += sum(c.get("approved") or 0 for c in (row.get("cells") or {}).values())
        current["manuals"] = max(current["manuals"], row.get("manuals") or 0)
    part_ids = list(members)

    with session() as s:
        procs = s.run("""
            UNWIND $ids AS pid MATCH (c:Component {id: pid})
            OPTIONAL MATCH (c)-[]-(p1:Procedure)
            OPTIONAL MATCH (c)-[]-(al:Alarm)-[]-(p2:Procedure)
            WITH pid, collect(DISTINCT p1.name) + collect(DISTINCT p2.name) AS names
            RETURN pid, names""", ids=part_ids).data()
        alarms = s.run("""
            UNWIND $ids AS pid MATCH (c:Component {id: pid})-[:ASSOCIATED_WITH]-(al:Alarm)
            WHERE al.subtype = 'PLC alarm' RETURN pid, collect(DISTINCT coalesce(al.code, al.name)) AS codes""",
                       ids=part_ids).data()
        monitored = s.run("""
            UNWIND $ids AS pid MATCH (c:Component {id: pid})-[:MONITORED_BY]-(sig:Signal)
            RETURN sig.id AS signal, pid, c.name AS part, size([(c)-[:HAS_COMPONENT]->() | 1]) AS children""",
                          ids=part_ids).data()
    by_area: dict[str, dict] = {a["id"]: {"procedures": set(), "alarms": set()} for a in areas}
    for r in procs:
        by_area[members[r["pid"]]["id"]]["procedures"].update(n for n in r["names"] if n)
    for r in alarms:
        by_area[members[r["pid"]]["id"]]["alarms"].update(c for c in r["codes"] if c)
    where: dict[str, tuple[str, str, int]] = {}          # signal → (area id, part name); the most specific part wins
    for r in monitored:
        best = where.get(r["signal"])
        if best is None or r["children"] < best[2]:
            where[r["signal"]] = (members[r["pid"]]["id"], r["part"], r["children"])

    sensors = []
    for sig in asset_context.signal_columns(asset_id):
        found = where.get(sig["signal_id"])
        if not found or (sig.get("low") is None and sig.get("high") is None):
            continue                                  # flags and setpoints (no normal range) are not drawn as sensors
        area = next(a for a in areas if a["id"] == found[0])
        if any(x["column"] == sig["column"] for x in sensors):
            continue
        sensors.append({"column": sig["column"], "name": sig["name"], "unit": sig.get("unit") or "", "low": sig.get("low"),
                        "high": sig.get("high"), "area": area["slot"], "part": found[1]})
    columns = {s["column"] for s in sensors}
    bind = BINDINGS.get(template, {})
    bindings = {k: v for k, v in bind.items() if k != "heat" and v in columns}
    if bind.get("heat"):
        bindings["heat"] = {k: v for k, v in bind["heat"].items() if v in columns}
    return {
        "version": VERSION, "asset_id": asset_id, "asset_name": asset_name, "template": template, "enabled": True,
        "generated_at": _now().isoformat(),
        "areas": [{"id": a["id"], "name": a["name"], "slot": a["slot"],
                   "info": {"facts": a["facts"], "documents": a["manuals"],
                            "procedures": sorted(by_area[a["id"]]["procedures"])[:PROCEDURES_SHOWN]},
                   "alarms": sorted(by_area[a["id"]]["alarms"])} for a in areas],
        "sensors": sensors, "bindings": bindings,
    }


def manifest(asset_id: str, regenerate: bool = False, by: str = "") -> dict:
    """The kept manifest (generated on first use). A machine without a template still gets one: the apps then say
    "no drawing for this machine type yet"."""
    row = TwinManifest.objects.filter(asset_id=asset_id).first()
    if row is None or regenerate:
        try:
            data = generate(asset_id)
        except Exception as exc:
            if row is not None:
                logger.warning("twin: %s not regenerated (%s); the kept manifest is served", asset_id, exc)
                return {**row.manifest, "enabled": row.enabled}
            raise TwinError(f"no twin for {asset_id}: {type(exc).__name__}: {exc}"[:300])
        if row is None:
            row = TwinManifest(asset_id=asset_id)
        row.manifest, row.generated_at, row.updated_by = data, _now(), (by or row.updated_by)[:200]
        row.save()
    return {**row.manifest, "enabled": row.enabled}


def set_enabled(asset_id: str, enabled: bool, by: str = "") -> dict:
    """Show or hide the twin for this machine in every app (no deploy)."""
    manifest(asset_id)                                   # it exists from here on
    row = TwinManifest.objects.get(asset_id=asset_id)
    row.enabled, row.updated_by = bool(enabled), by[:200]
    row.save(update_fields=["enabled", "updated_by", "updated_at"])
    return {**row.manifest, "enabled": row.enabled}


# --- live ------------------------------------------------------------------------------------------------------

def _from_opc(asset_id: str) -> dict | None:
    from plc import live
    try:
        o = live.live(asset_id)
    except live.LiveError:
        return None                                   # not the machine on the Studio's OPC UA connection
    if o.get("error") or not o.get("analog"):
        return None
    values = {a["column"]: {"value": a["value"], "outside": bool(a.get("outside"))} for a in o["analog"] if a.get("column")}
    fault = o.get("fault") or {}
    return {"asset_id": asset_id, "source": "opc", "read_at": o.get("read_at"),
            "age_s": o["state"].get("sample_age_s") if o["state"].get("heartbeat_age_s") is None else o["state"]["heartbeat_age_s"],
            "state": o["state"].get("name") or "", "values": values,
            "fault": {"active": bool(fault.get("active")), "code": fault.get("code"), "label": fault.get("label") or ""},
            "error": None}


def _from_historian(asset_id: str, sensors: list[dict], opc_error: str | None) -> dict:
    from monitor import agent
    now = _now()
    try:
        h = agent.history(asset_id, now - timedelta(minutes=30), now)
        rows = agent._rows(h)
    except Exception as exc:
        return {"asset_id": asset_id, "source": "none", "read_at": None, "age_s": None, "state": "", "values": {},
                "fault": {"active": False}, "error": f"no data: {type(exc).__name__}"}
    if not rows:
        return {"asset_id": asset_id, "source": "none", "read_at": None, "age_s": None, "state": "", "values": {},
                "fault": {"active": False}, "error": "no row in the last 30 minutes"}
    r = rows[-1]
    values = {}
    for s in sensors:
        v = r.get(s["column"])
        v = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        out = v is not None and ((s.get("low") is not None and v < s["low"]) or (s.get("high") is not None and v > s["high"]))
        values[s["column"]] = {"value": v, "outside": bool(out)}
    code = str(r.get("alarm_code") or "")
    return {"asset_id": asset_id, "source": "historian", "read_at": r["ts"].isoformat(),
            "age_s": round((now - r["ts"]).total_seconds(), 1), "state": str(r.get("machine_state") or ""),
            "values": values, "fault": {"active": bool(code), "code": code, "label": code}, "error": opc_error}


def live(asset_id: str) -> dict:
    """The machine now: the PLC over OPC UA when the Studio reads this machine's PLC, else the newest historian row."""
    m = manifest(asset_id)
    try:
        found = _from_opc(asset_id)
    except Exception as exc:
        logger.warning("twin: OPC read failed for %s (%s)", asset_id, exc)
        found = None
    return found or _from_historian(asset_id, m.get("sensors") or [], None)
