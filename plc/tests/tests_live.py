"""P31: the machine live for EDGE: a curated read of the OPC UA variables (the PLC is a fake here)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import SimpleTestCase, TransactionTestCase, override_settings

from plc import live

NOW = datetime(2026, 11, 8, 20, 47, 40, tzinfo=timezone.utc)


def var(path, value, kind="Double"):
    return {"path": path, "value": value, "type": kind, "good": True}


READ = {"endpoint": "opc.tcp://x", "read_at": NOW, "error": None, "variables": [
    var("Identification.AssetNumber", "EXTR01", "String"), var("Identification.RecipeNumber", "RC-G", "String"),
    var("Heartbeat.LastCommUtc", NOW - timedelta(seconds=2), "DateTime"),
    var("Step.StepName", "RUNNING", "String"), var("Step.CurrentStep", 1, "UInt16"), var("State.CurrentState", 2, "Int32"),
    var("Step.StepStartUtc", NOW - timedelta(hours=1), "DateTime"), var("Sample.TimestampUtc", NOW - timedelta(seconds=40), "DateTime"),
    var("Sample.PeriodSeconds", 20, "UInt16"),
    var("Faults.ActiveCode", 9, "Int32"), var("Faults.ActiveLabel", "DIE_PLUG", "String"), var("Faults.Severity", 3, "Int32"),
    var("Faults.Timestamp", NOW - timedelta(minutes=3), "DateTime"),
    var("Production.OeePct", 88.6), var("Production.GoodOutputKg", 50.3), var("Production.CycleTime", 0.0),
    var("AnalogIn[0].Label", "I1 voltage", "String"), var("AnalogIn[0].Value", 0.0), var("AnalogIn[0].EngineeringUnits", "V", "String"),
    var("AnalogIn[0].LowLimit", 0.0), var("AnalogIn[0].HighLimit", 10.0),
    var("AnalogIn[9].Label", "die_pressure_bar", "String"), var("AnalogIn[9].Value", 104.2), var("AnalogIn[9].EngineeringUnits", "bar", "String"),
    var("AnalogIn[9].LowLimit", 0.0), var("AnalogIn[9].HighLimit", 200.0),
    var("AnalogIn[10].Label", "Unused", "String"), var("AnalogIn[10].Value", 0.0),
    var("AnalogOut[0].Label", "throughput_target_kg_hr", "String"), var("AnalogOut[0].Value", 10000.0),
    var("DigitalIn[8].Label", "alarm_active", "String"), var("DigitalIn[8].Value", True, "Boolean"),
    var("DigitalOut[0].Label", "Relay 1", "String"), var("DigitalOut[0].Value", False, "Boolean"),
    var("DigitalOut[5].Label", "Unused", "String"), var("DigitalOut[5].Value", False, "Boolean"),
]}
SIGNALS = {"die_pressure_bar": {"column": "die_pressure_bar", "signal_id": "bsk:signal:EXTR01/die_pressure_bar",
                                "name": "Die pressure", "unit": "bar", "low": 84.3, "high": 99.2}}


class BuildTests(SimpleTestCase):
    def test_the_curated_read(self):
        out = live.build(READ, "bsk:asset:EXTR01", SIGNALS, now=NOW)
        self.assertEqual(out["state"]["name"], "RUNNING")
        self.assertEqual((out["state"]["heartbeat_age_s"], out["state"]["sample_age_s"]), (2.0, 40.0))
        self.assertEqual(out["fault"], {"active": True, "code": 9, "label": "DIE_PLUG", "severity": 3,
                                        "since": (NOW - timedelta(minutes=3)).isoformat()})
        self.assertEqual(out["production"], {"GoodOutputKg": 50.3, "OeePct": 88.6})          # CycleTime is not kept
        die = out["analog"][0]                                                                # the historian's first
        self.assertEqual((die["name"], die["unit"], die["low"], die["high"], die["outside"]),
                         ("Die pressure", "bar", 84.3, 99.2, True))
        self.assertEqual([a["label"] for a in out["analog"]], ["die_pressure_bar", "I1 voltage", "throughput_target_kg_hr"])
        self.assertEqual(out["analog"][-1]["direction"], "out")
        self.assertEqual([d["label"] for d in out["digital"]], ["alarm_active", "Relay 1"])      # Unused left out
        self.assertNotIn("variables", out)
        everything = live.build(READ, "bsk:asset:EXTR01", SIGNALS, everything=True, now=NOW)
        self.assertEqual(len(everything["variables"]), len(READ["variables"]))
        json.dumps(everything)                                                               # plain JSON

    def test_no_fault(self):
        read = {**READ, "variables": [v for v in READ["variables"] if not v["path"].startswith("Faults.")]
                + [var("Faults.ActiveCode", 0, "Int32"), var("Faults.ActiveLabel", "No fault", "String")]}
        self.assertEqual(live.build(read, "bsk:asset:EXTR01", {}, now=NOW)["fault"],
                         {"active": False, "code": 0, "label": "", "severity": None, "since": None})


@override_settings(PLC_ASSET="EXTR01")
class ToolTests(TransactionTestCase):
    def call(self, args):
        from context_service import mcp_server
        result = asyncio.run(mcp_server.build_server().call_tool("get_opc_live", args))
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    def test_the_tool_reads_through_the_cached_reader(self):
        with patch("plc.service.opcua", return_value=READ) as read, patch("plc.live._signals", return_value=SIGNALS):
            out = self.call({"asset_id": "bsk:asset:EXTR01"})
        read.assert_called_once()
        self.assertEqual(out["fault"]["label"], "DIE_PLUG")

    def test_only_the_connected_machine_and_a_down_plc_is_an_answer(self):
        with self.assertRaisesMessage(Exception, "no OPC UA connection for bsk:asset:EXTR02"):
            self.call({"asset_id": "bsk:asset:EXTR02"})
        with patch("plc.service.opcua", return_value={"endpoint": "x", "read_at": NOW, "error": "timed out", "variables": []}):
            out = self.call({"asset_id": "bsk:asset:EXTR01"})
        self.assertIn("The PLC could not be read: timed out", out["error"])


class SignalCacheTests(SimpleTestCase):
    def test_signal_names_are_kept_and_a_failure_is_not(self):
        live._signals_cache.clear()
        with patch("chat.asset_context.signal_columns", return_value=list(SIGNALS.values())) as read:
            live._signals("bsk:asset:EXTR01")
            live._signals("bsk:asset:EXTR01")
        self.assertEqual(read.call_count, 1)
        live._signals_cache.clear()
        with patch("chat.asset_context.signal_columns", side_effect=RuntimeError("graph down")) as read:
            self.assertEqual(live._signals("bsk:asset:EXTR01"), {})
            live._signals("bsk:asset:EXTR01")
        self.assertEqual(read.call_count, 2)
        live._signals_cache.clear()
