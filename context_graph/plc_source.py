"""The graph against the PLC (P14a): the comparison, and the PLC as a source for GraphLab Import.

The PLC is read over OPC UA (plc.opc, BSK machine template): each analog and digital slot has a Label (the
historian column), EngineeringUnits, LowLimit and HighLimit; Faults.Labels lists the alarm codes. Nothing here
writes the graph: the comparison only reads, and the import source writes two CSV files (a tag list and an alarm
list) into GraphLab Import, where they are previewed, staged and reviewed like any uploaded file.
Plan: claude/P14_plant_model_plan.md (Studio web repository), P14a.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from django.conf import settings
from django.utils import timezone

from historian import source as historian_source

from . import structured

SLOT = re.compile(r"^((AnalogIn|AnalogOut|DigitalIn|DigitalOut)\[\d+\])\.(\w+)$")
FAULT = re.compile(r"^Faults\.Labels\[(\d+)\]$")
NO_FAULT = "No fault"
OPC_TYPES = {"Double": "float", "Float": "float", "Boolean": "bool", "String": "string",
             "Int16": "int", "Int32": "int", "Int64": "int", "UInt16": "int", "UInt32": "int", "UInt64": "int"}
TOLERANCE = 1e-6


class PlcSourceError(ValueError):
    """The asset has no PLC connection, or the PLC could not be read."""


@dataclass
class Slot:
    slot: str                  # e.g. AnalogIn[12]
    group: str                 # AnalogIn, AnalogOut, DigitalIn, DigitalOut
    label: str
    unit: str = ""
    low: float | None = None
    high: float | None = None
    opc_type: str = ""         # the OPC UA type of the slot's Value
    node: str = ""             # the Value's NodeId, with the namespace URI (stable across server restarts)


@dataclass
class Plc:
    slots: list[Slot] = field(default_factory=list)
    alarms: list[tuple[int, str]] = field(default_factory=list)    # (index in Faults.Labels, code), "No fault" left out


def _asset_key(asset_id: str) -> str:
    return asset_id.rsplit(":", 1)[-1]


def read(asset_id: str) -> Plc:
    """The PLC's slots and alarm codes, for the asset the Studio reaches over OPC UA (PLC_ASSET)."""
    if _asset_key(asset_id) != settings.PLC_ASSET:
        raise PlcSourceError(f"{asset_id} has no PLC connection (the Studio reaches {settings.PLC_ASSET} only)")
    from plc import opc
    try:
        variables = opc.read_all()["variables"]
    except opc.OpcError as exc:
        raise PlcSourceError(f"the PLC could not be read: {exc}") from exc
    fields: dict[str, dict] = {}
    alarms = []
    for v in variables:
        m = SLOT.match(v["path"])
        if m:
            fields.setdefault(m.group(1), {"group": m.group(2)})[m.group(3)] = v
            continue
        m = FAULT.match(v["path"])
        if m and isinstance(v["value"], str) and v["value"] not in ("", "Unused", NO_FAULT):
            alarms.append((int(m.group(1)), v["value"]))
    slots = []
    for slot, f in fields.items():
        label = (f.get("Label") or {}).get("value")
        if not label or label == "Unused":
            continue
        number = lambda name: f[name]["value"] if name in f and isinstance(f[name]["value"], (int, float)) else None  # noqa: E731
        slots.append(Slot(slot=slot, group=f["group"], label=label, unit=(f.get("EngineeringUnits") or {}).get("value") or "",
                          low=number("LowLimit"), high=number("HighLimit"), opc_type=(f.get("Value") or {}).get("type") or "",
                          node=f"nsu=urn:bsk:machine:v1;s=BSK.{settings.PLC_ASSET}.{slot}.Value"))
    return Plc(slots=slots, alarms=sorted(alarms))


def historian_columns() -> set[str]:
    return {f.name for f in historian_source.samples()._meta.fields} - {"pk", "ts", "extruder_id"}


SIGNALS_QUERY = """
MATCH (a:Entity {id: $asset})-[:HAS_COMPONENT*0..]->(owner:Entity)-[:MONITORED_BY]->(s:Entity)
OPTIONAL MATCH (s)-[:HAS_LIMIT]->(l:Entity)
OPTIONAL MATCH (s)-[:IMPLEMENTED_AS]->(o:Entity)
RETURN s.id AS id, s.name AS name, properties(s) AS props, owner.id AS owner, owner.name AS owner_name,
       collect(DISTINCT properties(l)) AS limits, count(DISTINCT o) AS opc_nodes
"""
ALARMS_QUERY = """
MATCH (a:Entity {id: $asset})-[:HAS_ALARM]->(al:Entity)
OPTIONAL MATCH (c:Entity)-[:ASSOCIATED_WITH]->(al)
RETURN al.id AS id, al.name AS name, properties(al) AS props, collect(c.name)[0] AS component
"""


def _graph(asset_id: str) -> tuple[dict, dict]:
    """(signals by historian column or key, alarms by code) of the asset, at any depth of its components.
    A signal: {id, name, properties, limits: [properties], opc_nodes, component (name, "" on the asset)}."""
    from .driver import session
    signals, alarms = {}, {}
    with session() as s:
        for r in s.run(SIGNALS_QUERY, asset=asset_id):
            props = dict(r["props"])
            ref = (props.get("historian_ref") or "").rsplit(".", 1)[-1]
            key = ref or r["id"].rsplit("/", 1)[-1]
            signals.setdefault(key, {"id": r["id"], "name": r["name"], "properties": props, "opc_nodes": r["opc_nodes"],
                                     "limits": [dict(x) for x in r["limits"] if x],
                                     "component": "" if r["owner"] == asset_id else r["owner_name"]})
        for r in s.run(ALARMS_QUERY, asset=asset_id):
            props = dict(r["props"])
            alarms[props.get("code") or r["name"]] = {"id": r["id"], "name": r["name"], "properties": props,
                                                      "component": r["component"] or ""}
    return signals, alarms


def _normal(signal: dict) -> dict | None:
    return next((lim for lim in signal.get("limits", []) if lim.get("kind", "normal") == "normal"), None)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOLERANCE


def compare(asset_id: str) -> dict:
    """Signal by signal, alarm by alarm: in both and the same, in both and different, only in the graph, only in
    the PLC; and the PLC's slots that are not historian columns (spare I/O). Writes nothing."""
    plc = read(asset_id)
    signals, alarms = _graph(asset_id)
    columns = historian_columns()
    rows, spare = [], []
    seen = set()
    for slot in plc.slots:
        if slot.label not in columns and slot.label not in signals:
            spare.append({"slot": slot.slot, "label": slot.label, "unit": slot.unit, "low": slot.low, "high": slot.high})
            continue
        seen.add(slot.label)
        g = signals.get(slot.label)
        plc_side = {"slot": slot.slot, "unit": slot.unit, "low": slot.low, "high": slot.high, "type": slot.opc_type}
        if g is None:
            rows.append({"kind": "signal", "key": slot.label, "status": "only_plc", "plc": plc_side, "graph": None, "differences": []})
            continue
        lim = _normal(g)
        lp = lim or {}
        graph_side = {"id": g["id"], "name": g["name"], "component": g["component"], "unit": g["properties"].get("unit") or "",
                      "low": lp.get("low"), "high": lp.get("high"), "opc_nodes": g["opc_nodes"]}
        differences = []
        if (graph_side["unit"] or "") != (slot.unit or "") and slot.group.startswith("Analog"):
            differences.append("unit")
        # An output's LowLimit / HighLimit is its range, not a normal range while running: not compared.
        if slot.group == "AnalogIn" and (slot.low is not None or slot.high is not None):
            if lim is None:
                differences.append("no normal range in the graph")
            else:
                differences += [name for name, a, b in (("low", graph_side["low"], slot.low), ("high", graph_side["high"], slot.high))
                                if not _same(a, b)]
        if not graph_side["opc_nodes"]:
            differences.append("no OPC node in the graph")
        rows.append({"kind": "signal", "key": slot.label, "status": "differs" if differences else "same",
                     "plc": plc_side, "graph": graph_side, "differences": differences})
    for key, g in sorted(signals.items()):
        if key not in seen:
            # A historian column the PLC publishes outside its I/O slots (KPIs, counts, text): not "missing".
            rows.append({"kind": "signal", "key": key, "status": "historian_only" if key in columns else "only_graph", "plc": None,
                         "graph": {"id": g["id"], "name": g["name"], "unit": g["properties"].get("unit") or ""}, "differences": []})
    plc_codes = {code: index for index, code in plc.alarms}
    for code, index in plc_codes.items():
        g = alarms.get(code)
        rows.append({"kind": "alarm", "key": code, "status": "same" if g else "only_plc", "plc": {"index": index},
                     "graph": {"id": g["id"], "description": g["properties"].get("description") or ""} if g else None,
                     "differences": []})
    for code, g in sorted(alarms.items()):
        if code not in plc_codes:
            rows.append({"kind": "alarm", "key": code, "status": "only_graph", "plc": None, "graph": {"id": g["id"]}, "differences": []})
    count = lambda kind, status: sum(1 for r in rows if r["kind"] == kind and r["status"] == status)  # noqa: E731
    summary = {kind: {status: count(kind, status) for status in ("same", "differs", "only_plc", "only_graph", "historian_only")}
               for kind in ("signal", "alarm")}
    return {"asset_id": asset_id, "read_at": timezone.now(), "endpoint": settings.PLC_OPCUA_URL, "summary": summary,
            "rows": rows, "spare_slots": spare}


# --- the PLC as an import source ---------------------------------------------------------

TAG_COLUMNS = ["Tag", "Description", "Unit", "Data Type", "Component", "Historian Ref", "Low", "High", "Limit Kind",
               "Limit Basis", "OPC Node", "OPC Slot", "OPC Type"]
ALARM_COLUMNS = ["Alarm Code", "Name", "Description", "Severity", "Component", "PLC Index"]


def _csv(columns: list[str], rows: list[dict]) -> bytes:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _fmt(value) -> str:
    return "" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def tables(asset_id: str) -> tuple[bytes, bytes, int]:
    """The tag list and the alarm list the PLC gives, as CSV, and how many spare slots were left out. Names,
    descriptions, severities and components are the graph's where it has the signal or alarm, so that staging
    them changes nothing the graph already says; what the PLC adds is the OPC node and the limit's basis."""
    plc = read(asset_id)
    signals, alarms = _graph(asset_id)
    columns = historian_columns()
    tag_rows, spare = [], 0
    for slot in plc.slots:
        if slot.label not in columns and slot.label not in signals:
            spare += 1
            continue
        g = signals.get(slot.label)
        has_limits = slot.low is not None and slot.high is not None and slot.group == "AnalogIn"   # an output's is its range
        tag_rows.append({
            "Tag": slot.label, "Description": g["name"] if g else slot.label,
            "Unit": (g["properties"].get("unit") if g else None) or slot.unit,
            "Data Type": (g["properties"].get("data_type") if g else None) or OPC_TYPES.get(slot.opc_type, "string"),
            "Component": g["component"] if g else "",
            "Historian Ref": f"historian.historian_rows.{slot.label}" if slot.label in columns else "",
            "Low": _fmt(slot.low) if has_limits else "", "High": _fmt(slot.high) if has_limits else "",
            "Limit Kind": "normal" if has_limits else "",
            "Limit Basis": f"PLC {slot.slot} LowLimit / HighLimit" if has_limits else "",
            "OPC Node": slot.node, "OPC Slot": slot.slot, "OPC Type": slot.opc_type,
        })
    alarm_rows = []
    for index, code in plc.alarms:
        g = alarms.get(code)
        p = g["properties"] if g else {}
        alarm_rows.append({"Alarm Code": code, "Name": g["name"] if g else code, "Description": p.get("description", ""),
                           "Severity": p.get("severity", ""), "Component": g["component"] if g else "", "PLC Index": str(index)})
    return _csv(TAG_COLUMNS, tag_rows), _csv(ALARM_COLUMNS, alarm_rows), spare


class _Upload:
    """What structured.add_data_file reads from an uploaded file."""

    def __init__(self, name: str, content: bytes):
        self.name, self._content, self.size = name, content, len(content)

    def read(self) -> bytes:
        return self._content


def create_data_files(asset_id: str) -> dict:
    """The PLC's tag list and alarm list as two GraphLab data files. A file the same as one already there is not
    added again (the import deduplicates by content): `created` is false and the existing file is returned."""
    tags, alarms, spare = tables(asset_id)
    key = _asset_key(asset_id)
    out = []
    for kind, content in (("tag_list", tags), ("alarm_list", alarms)):
        df, created = structured.add_data_file(_Upload(f"{key}_plc_{kind}.csv", content))
        out.append({"template": kind, "created": created, "data_file": df})
    return {"files": out, "spare_slots": spare}
