"""The graph against the PLC (P14a): the comparison, the PLC's tag list and alarm list, the templates that read them,
and the endpoints. The PLC and the graph are fakes: no OPC UA server and no Neo4j are reached."""

import csv
import io
import tempfile
from unittest.mock import patch

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from context_graph import plc_source, structured
from context_graph.models import DataFile

ASSET = "bsk:asset:EXTR01"


def var(path, value, type_="String"):
    return {"path": path, "value": value, "type": type_, "good": True}


def slot(name, label, unit="", low=None, high=None, value_type="Double"):
    out = [var(f"{name}.Value", 0.0, value_type), var(f"{name}.Label", label)]
    if unit or low is not None:
        out += [var(f"{name}.EngineeringUnits", unit), var(f"{name}.LowLimit", low, "Double"), var(f"{name}.HighLimit", high, "Double")]
    return out


PLC_VARIABLES = (
    slot("AnalogIn[0]", "I1 voltage", "V", 0.0, 10.0)                                     # spare
    + slot("AnalogIn[12]", "barrel_zone_1_temp_c", "°C", 81.0, 99.0)
    + slot("AnalogIn[18]", "die_pressure_bar", "bar", 84.3, 99.2)
    + slot("AnalogIn[20]", "product_moisture_pct", "%", 8.15, 10.85)
    + slot("AnalogOut[0]", "throughput_target_kg_hr", "kg/h", 0.0, 11000.0)
    + slot("DigitalIn[8]", "alarm_active", value_type="Boolean")
    + slot("DigitalOut[0]", "Relay 1", value_type="Boolean")                                # spare
    + slot("AnalogIn[23]", "Unused", "", 0.0, 0.0)                                        # not offered
    + [var("Faults.Labels[0]", "No fault"), var("Faults.Labels[1]", "DIE_PLUG"), var("Faults.Labels[2]", "NEW_FAULT"),
       var("Faults.Labels[3]", "Unused")]
)


def signal(sid, name, unit, component="", low=None, high=None, opc=0, ref=""):
    return {"id": f"bsk:signal:EXTR01/{sid}", "name": name, "component": component, "opc_nodes": opc,
            "properties": {"unit": unit, "data_type": "float", "historian_ref": ref or f"plc_1_historian.extruder_samples.{sid}"},
            "limits": [{"kind": "normal", "low": low, "high": high}] if low is not None else []}


GRAPH_SIGNALS = {
    "barrel_zone_1_temp_c": signal("barrel_zone_1_temp_c", "Barrel zone 1 temperature", "°C", "Barrel Zone 1", 81.0, 99.0),
    "die_pressure_bar": signal("die_pressure_bar", "Die pressure", "bar", "Die Head", 84.3, 99.0),          # high differs
    "throughput_target_kg_hr": signal("throughput_target_kg_hr", "Throughput (target)", "kg/h"),
    "alarm_active": signal("alarm_active", "Alarm active", "", opc=1),
    "oee_pct": signal("oee_pct", "OEE", "%"),                                                                  # a historian column
    "old_tag": signal("old_tag", "Old tag", "bar", ref="nowhere.old_tag"),                                      # gone from the PLC
}
GRAPH_ALARMS = {
    "DIE_PLUG": {"id": "bsk:alarm:EXTR01/DIE_PLUG", "name": "DIE_PLUG", "component": "Die Head",
                 "properties": {"code": "DIE_PLUG", "severity": "high", "description": "Die plugged"}},
    "OLD_ALARM": {"id": "bsk:alarm:EXTR01/OLD_ALARM", "name": "OLD_ALARM", "component": "", "properties": {"code": "OLD_ALARM"}},
}
COLUMNS = {"barrel_zone_1_temp_c", "die_pressure_bar", "product_moisture_pct", "throughput_target_kg_hr", "alarm_active", "oee_pct"}


@override_settings(PLC_ASSET="EXTR01", PLC_OPCUA_URL="opc.tcp://plc:4840/bsk/machine")
class PlcSourceTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target, kwargs in (("plc.opc.read_all", {"return_value": {"variables": PLC_VARIABLES}}),
                               ("context_graph.plc_source._graph", {"return_value": (GRAPH_SIGNALS, GRAPH_ALARMS)}),
                               ("context_graph.plc_source.historian_columns", {"return_value": COLUMNS})):
            patch(target, **kwargs).start()
        self.addCleanup(patch.stopall)

    def test_signal_by_signal_and_alarm_by_alarm(self):
        result = plc_source.compare(ASSET)
        rows = {(r["kind"], r["key"]): (r["status"], r["differences"]) for r in result["rows"]}
        self.assertEqual(rows[("signal", "barrel_zone_1_temp_c")], ("differs", ["no OPC node in the graph"]))
        self.assertEqual(rows[("signal", "die_pressure_bar")], ("differs", ["high", "no OPC node in the graph"]))
        self.assertEqual(rows[("signal", "product_moisture_pct")], ("only_plc", []))
        self.assertEqual(rows[("signal", "throughput_target_kg_hr")], ("differs", ["no OPC node in the graph"]))   # an output's range is not compared
        self.assertEqual(rows[("signal", "alarm_active")], ("same", []))
        self.assertEqual(rows[("signal", "oee_pct")], ("historian_only", []))
        self.assertEqual(rows[("signal", "old_tag")], ("only_graph", []))
        self.assertEqual(rows[("alarm", "DIE_PLUG")], ("same", []))
        self.assertEqual(rows[("alarm", "NEW_FAULT")], ("only_plc", []))
        self.assertEqual(rows[("alarm", "OLD_ALARM")], ("only_graph", []))
        self.assertEqual([s["slot"] for s in result["spare_slots"]], ["AnalogIn[0]", "DigitalOut[0]"])
        self.assertEqual(result["summary"]["signal"], {"same": 1, "differs": 3, "only_plc": 1, "only_graph": 1, "historian_only": 1})

    def test_the_plcs_tag_list_and_alarm_list_keep_what_the_graph_says_and_add_what_the_plc_has(self):
        tags, alarms, spare = plc_source.tables(ASSET)
        tag = {r["Tag"]: r for r in csv.DictReader(io.StringIO(tags.decode()))}
        self.assertEqual(spare, 2)
        self.assertEqual(set(tag), {"barrel_zone_1_temp_c", "die_pressure_bar", "product_moisture_pct", "throughput_target_kg_hr", "alarm_active"})
        zone = tag["barrel_zone_1_temp_c"]
        self.assertEqual((zone["Description"], zone["Component"], zone["Low"], zone["High"], zone["Limit Kind"], zone["Limit Basis"]),
                         ("Barrel zone 1 temperature", "Barrel Zone 1", "81", "99", "normal", "PLC AnalogIn[12] LowLimit / HighLimit"))
        self.assertEqual((zone["OPC Node"], zone["OPC Slot"], zone["OPC Type"]),
                         ("nsu=urn:bsk:machine:v1;s=BSK.EXTR01.AnalogIn[12].Value", "AnalogIn[12]", "Double"))
        self.assertEqual((tag["product_moisture_pct"]["Description"], tag["product_moisture_pct"]["Unit"]), ("product_moisture_pct", "%"))   # new: the label
        self.assertEqual((tag["throughput_target_kg_hr"]["Low"], tag["throughput_target_kg_hr"]["Limit Kind"]), ("", ""))   # an output: no normal range
        self.assertEqual(tag["alarm_active"]["Data Type"], "float")                                                        # the graph's, kept
        alarm = list(csv.DictReader(io.StringIO(alarms.decode())))
        self.assertEqual([(a["Alarm Code"], a["Name"], a["Description"], a["Component"], a["PLC Index"]) for a in alarm],
                         [("DIE_PLUG", "DIE_PLUG", "Die plugged", "Die Head", "1"), ("NEW_FAULT", "NEW_FAULT", "", "", "2")])

    def test_the_templates_read_the_opc_node_the_limit_basis_and_the_plc_index(self):
        tags, alarms, _ = plc_source.tables(ASSET)
        header = lambda content: next(csv.reader(io.StringIO(content.decode())))   # noqa: E731
        mapping = structured.suggestions(header(tags))[0]["mapping"]
        opc = next(e for e in mapping["entities"] if e["key"] == "opc")
        self.assertEqual((opc["type"], opc["id_suffix"], opc["name_column"], opc["properties"]),
                         ("OPCNode", "opc", "OPC Node", {"node_id": "OPC Node", "slot": "OPC Slot", "data_type": "OPC Type"}))
        self.assertIn({"from": "signal", "type": "IMPLEMENTED_AS", "to": "opc"}, mapping["relationships"])
        limit = next(e for e in mapping["entities"] if e["key"] == "limit")
        self.assertEqual((limit["properties"]["kind"], limit["properties"]["basis"]), ("Limit Kind", "Limit Basis"))
        signal_props = next(e for e in mapping["entities"] if e["key"] == "signal")["properties"]
        self.assertNotIn("OPC Type", signal_props.values())
        alarm_mapping = structured.suggestions(header(alarms))[0]["mapping"]
        alarm_entity = next(e for e in alarm_mapping["entities"] if e["key"] == "alarm")
        self.assertEqual((alarm_entity["name_column"], alarm_entity["properties"]["plc_index"]), ("Name", "PLC Index"))

    def test_reading_the_plc_twice_adds_the_files_once(self):
        with override_settings(GRAPH_DATA_BASE=self.tmp.name):
            client = APIClient()
            first = client.post("/api/graph/plc-source/", {"asset_id": ASSET}, format="json").json()
            again = client.post("/api/graph/plc-source/", {"asset_id": ASSET}, format="json").json()
        self.assertEqual([(f["template"], f["created"], f["data_file"]["filename"], f["data_file"]["row_count"]) for f in first["files"]],
                         [("tag_list", True, "EXTR01_plc_tag_list.csv", 5), ("alarm_list", True, "EXTR01_plc_alarm_list.csv", 2)])
        self.assertEqual(first["spare_slots"], 2)
        self.assertEqual([f["created"] for f in again["files"]], [False, False])
        self.assertEqual(DataFile.objects.count(), 2)

    def test_the_comparison_endpoint_and_what_is_refused(self):
        client = APIClient()
        body = client.get(f"/api/graph/assets/{ASSET}/plc-compare/").json()
        self.assertEqual((body["asset_id"], body["endpoint"], len(body["rows"])), (ASSET, "opc.tcp://plc:4840/bsk/machine", 10))
        other = client.get("/api/graph/assets/bsk:asset:MIXER02/plc-compare/")
        self.assertEqual((other.status_code, other.json()["error"]), (400, "bsk:asset:MIXER02 has no PLC connection (the Studio reaches EXTR01 only)"))
        from plc import opc
        with patch("plc.opc.read_all", side_effect=opc.OpcError("TimeoutError")):
            down = client.post("/api/graph/plc-source/", {"asset_id": ASSET}, format="json")
        self.assertEqual((down.status_code, down.json()["error"]), (400, "the PLC could not be read: TimeoutError"))
        self.assertEqual(client.post("/api/graph/plc-source/", {}, format="json").status_code, 400)
