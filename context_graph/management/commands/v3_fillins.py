"""Schema v3: fill in what exists, as data files staged for review (schema review step 2, 2026-10-15).

    manage.py v3_fillins --asset bsk:asset:EXTR01 --out <folder> [--stage]

Writes four CSV files from the live graph and the library, each with its import mapping:

    <KEY>_v3_component_subtypes.csv   every part: `subtype` (schema spelling) beside its `component_type`
    <KEY>_v3_signal_subtypes.csv      every signal: `subtype` from its unit and name
    <KEY>_v3_alarm_signals.csv        every alarm: subtype "PLC alarm"; Signal HAS_ALARM Alarm where one tag is clear
    <KEY>_v3_manuals.csv              Document nodes for the library's manuals; part DOCUMENTED_BY its manual (BOM)

All additive: existing entities only gain properties they lack (an import never overwrites), relations that
exist are restated (no change), new ones are new. With --stage the files are added to GraphLab Import and
staged: pending triples, nothing in the graph until a person approves them. Every proposal is reviewed there.
"""

import csv
import json
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError

from context_graph import structured
from context_graph.driver import session

# component_type as imports wrote it -> the schema's subtype; a few parts get a v3 subtype by their key.
COMPONENT_SUBTYPE = {"heating_zone": "Heating zone", "feeder": "Feeder", "conditioner": "Conditioner", "motor": "Motor",
                     "screw_barrel": "Barrel", "die": "Die", "measurement_point": "Measurement point"}
COMPONENT_OVERRIDE = {"plc": "PLC", "plc-analog-expansion": "PLC", "plc-digital-expansion": "PLC", "mini-pc": "Computer",
                      "power-supply": "Power supply", "feeder-load-cell": "Load cell"}
# Signal subtype: by historian column first, then by unit.
SIGNAL_BY_KEY = {"alarm_active": "State", "alarm_code": "State", "machine_state": "State", "recipe_id": "State",
                 "batch_id": "State", "shift_code": "State", "oee_pct": "KPI", "availability_pct": "KPI",
                 "performance_pct": "KPI", "quality_pct": "KPI", "downtime_min": "KPI", "good_output_kg": "Counter",
                 "scrap_output_kg": "Counter", "screw_torque_pct": "Torque", "product_moisture_pct": "Moisture",
                 "steam_flow_kg_hr": "Flow", "water_injection_l_hr": "Flow"}
SIGNAL_BY_UNIT = {"°C": "Temperature", "bar": "Pressure", "kg/h": "Rate", "L/h": "Flow", "rpm": "Speed", "kW": "Power",
                  "g/L": "Density", "mm": "Dimension", "A": "Current", "V": "Voltage"}
# The PLC alarm -> its tag (ISA-18.2: an alarm belongs to the signal it watches). Proposals, reviewed in GraphLab.
ALARM_TAG = {"DIE_PLUG": "die_pressure_bar", "FEEDER_OVERLOAD": "feeder_rate_actual_kg_hr",
             "FEEDER_STARVE": "feeder_rate_actual_kg_hr", "HEAD_TEMP_DRIFT": "head_temp_c", "HIGH_SCRAP": "scrap_output_kg",
             "HI_DIE_PRESSURE": "die_pressure_bar", "HI_MOISTURE": "product_moisture_pct",
             "HI_TEMP_ZONE_3": "barrel_zone_3_temp_c", "HI_TORQUE": "screw_torque_pct", "LOW_OEE": "oee_pct",
             "LO_MOISTURE": "product_moisture_pct", "LO_TEMP_ZONE_1": "barrel_zone_1_temp_c",
             "MOISTURE_SENSOR_FAIL": "product_moisture_pct", "MOTOR_OVERLOAD": "motor_power_kw",
             "OUT_OF_SPEC_DENSITY": "bulk_density_g_l", "OUT_OF_SPEC_DIAMETER": "kibble_diameter_mm",
             "SCREEN_BLOCK": "die_pressure_bar", "SPEED_UNDER_TARGET": "screw_speed_rpm", "UNPLANNED_STOP": "machine_state"}
# Library folder -> VDI 2770 document class (the Document subtype).
DOC_CLASS = {"datasheet": "Data sheet", "manual": "Operating instructions", "drawing": "Drawing",
             "sop": "Standard operating procedure", "quality_alert": "Quality alert", "visual_aid": "Visual aid",
             "training": "Training", "report": "Report", "work_instruction": "Standard operating procedure"}

ASSET_E = {"key": "asset", "type": "Asset", "scope": True}


def mappings() -> dict:
    part = {"key": "part", "type": "Component", "id_column": "Item", "existing_only": True, "fuzzy": False,
            "properties": {"subtype": "Subtype"}}
    parent = {"key": "parent", "type": "Component", "id_column": "Parent", "existing_only": True, "fuzzy": False}
    owner = {"key": "owner", "type": "Component", "id_column": "Owner", "existing_only": True, "fuzzy": False}
    signal = {"key": "signal", "type": "Signal", "id_column": "Tag", "existing_only": True}
    return {
        "component_subtypes": {"entities": [ASSET_E, part, parent], "relationships": [
            {"from": "parent", "type": "HAS_COMPONENT", "to": "part"},
            {"from": "asset", "type": "HAS_COMPONENT", "to": "part", "unless": "parent"}]},
        "signal_subtypes": {"entities": [ASSET_E, owner, {**signal, "properties": {"subtype": "Subtype"}}], "relationships": [
            {"from": "owner", "type": "MONITORED_BY", "to": "signal"},
            {"from": "asset", "type": "MONITORED_BY", "to": "signal", "unless": "owner"}]},
        "alarm_signals": {"entities": [ASSET_E, signal, {"key": "alarm", "type": "Alarm", "id_column": "Code", "existing_only": True,
                                                          "properties": {"subtype": "Subtype"}}], "relationships": [
            {"from": "asset", "type": "HAS_ALARM", "to": "alarm"},
            {"from": "signal", "type": "HAS_ALARM", "to": "alarm"}]},
        "manuals": {"entities": [{"key": "part", "type": "Component", "id_column": "Item", "existing_only": True, "fuzzy": False},
                                 {"key": "document", "type": "Document", "id_column": "Document key", "name_column": "Title",
                                  "fuzzy": False, "properties": {"doc_key": "Document key", "subtype": "Class",
                                                                 "source_url": "Source URL", "document_id": "Library id"}}],
                    "relationships": [{"from": "part", "type": "DOCUMENTED_BY", "to": "document"}]},
    }


class Command(BaseCommand):
    help = "Schema v3: write (and optionally stage) the fill-ins of what exists, for review in GraphLab Import."

    def add_arguments(self, parser):
        parser.add_argument("--asset", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--stage", action="store_true", help="add the files to GraphLab Import and stage them (pending)")

    def handle(self, *args, asset, out, stage, **opts):
        key = asset.split(":", 2)[2]
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        with session() as s:
            parts = s.run("MATCH (a:Asset {id: $a})-[:HAS_COMPONENT*]->(c:Component) MATCH (p)-[:HAS_COMPONENT]->(c) "
                          "RETURN DISTINCT c.id AS id, c.name AS name, c.component_type AS type, c.subtype AS subtype, "
                          "c.manual_title AS manual, p.id AS parent ORDER BY id", a=asset).data()
            signals = s.run("MATCH (s:Signal) WHERE s.id STARTS WITH $p MATCH (o)-[:MONITORED_BY]->(s) "
                            "RETURN s.id AS id, s.unit AS unit, s.subtype AS subtype, collect(o.id) AS owners ORDER BY id",
                            p=f"bsk:signal:{key}/").data()
            alarms = s.run("MATCH (a:Asset {id: $a})-[:HAS_ALARM]->(al:Alarm) RETURN al.code AS code, al.id AS id ORDER BY code",
                           a=asset).data()
        if not parts:
            raise CommandError(f"{asset} has no parts in the graph")
        tail = lambda node_id: node_id.rsplit("/", 1)[-1]     # noqa: E731
        files = {}

        rows = []
        for p in parts:
            item = tail(p["id"])
            subtype = COMPONENT_OVERRIDE.get(item) or COMPONENT_SUBTYPE.get(p["type"] or "", p["type"] or "")
            parent = "" if p["parent"] == asset else tail(p["parent"])
            if subtype and not p["subtype"]:
                rows.append({"Item": item, "Parent": parent, "Name": p["name"], "Component type": p["type"] or "", "Subtype": subtype})
        files["component_subtypes"] = rows

        rows = []
        for sig in signals:
            tag = tail(sig["id"])
            subtype = SIGNAL_BY_KEY.get(tag) or SIGNAL_BY_UNIT.get(sig["unit"] or "")
            owner = next((o for o in sig["owners"] if o != asset), "")
            if subtype and not sig["subtype"]:
                rows.append({"Tag": tag, "Owner": tail(owner) if owner else "", "Unit": sig["unit"] or "", "Subtype": subtype})
        files["signal_subtypes"] = rows

        files["alarm_signals"] = [{"Code": tail(a["id"]), "Tag": ALARM_TAG.get(a["code"], ""), "Subtype": "PLC alarm"} for a in alarms]

        from ingestion.models import Document
        by_title = {d.title.strip(): d for d in Document.objects.select_related("folder") if d.title}
        rows = []
        for p in parts:
            for title in [t.strip() for t in (p["manual"] or "").split(" | ") if t.strip()]:
                d = by_title.get(title)
                if d is None:
                    continue
                kind = d.folder.kind if d.folder else ""
                rows.append({"Item": tail(p["id"]), "Document key": d.doc_key, "Title": d.title,
                             "Class": DOC_CLASS.get(kind, "Operating instructions"), "Source URL": d.source_url,
                             "Library id": str(d.id)})
        files["manuals"] = rows

        maps = mappings()
        for name, rows in files.items():
            path = out / f"{key}_v3_{name}.csv"
            if rows:
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
            self.stdout.write(f"{path.name}: {len(rows)} rows")
            if stage and rows:
                df, created = structured.add_data_file(SimpleUploadedFile(path.name, path.read_bytes()))
                result = structured.stage(df, maps[name], asset)
                self.stdout.write(f"  staged: {result['staged']} triples, {result['new_entities']} new entities, "
                                  f"{result['row_errors']} row errors{' (file already added)' if not created else ''}")
                for e in result["row_error_list"][:10]:
                    self.stdout.write(f"    row {e['row']}: {e['message']}")
        (out / f"{key}_v3_mappings.json").write_text(json.dumps(maps, indent=1))
