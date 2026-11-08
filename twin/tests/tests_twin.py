"""P32: the twin's manifest from the graph (faked), the switch, the live values (OPC, else the historian), API, MCP."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from django.test import TestCase, TransactionTestCase
from rest_framework.test import APIClient

from twin import service
from twin.models import TwinManifest

ASSET = "bsk:asset:EXTR01"
GRID = {"asset": {"id": ASSET, "name": "Extruder EXTR01"}, "rows": [
    {"id": "c:die", "name": "Die Head", "depth": 0, "type": "die", "manuals": 1, "cells": {"Safety": {"approved": 10}, "Controls": {"approved": 2}}},
    {"id": "c:die-p", "name": "Die pressure transmitter", "depth": 1, "type": "sensor", "manuals": 0, "cells": {"Safety": {"approved": 3}}},
    {"id": "c:feed", "name": "Feeder", "depth": 0, "type": "feeder", "manuals": 0, "cells": {}},
    {"id": "c:odd", "name": "Something else", "depth": 0, "type": "robot", "manuals": 0, "cells": {}}]}
SIGNALS = [{"column": "die_pressure_bar", "signal_id": "s:dp", "name": "Die pressure", "unit": "bar", "low": 84.3, "high": 99.2},
           {"column": "alarm_active", "signal_id": "s:aa", "name": "Alarm active", "unit": "", "low": None, "high": None},
           {"column": "screw_speed_rpm", "signal_id": "s:ss", "name": "Screw speed", "unit": "rpm", "low": 405.9, "high": 444.6}]


class FakeSession:
    """Answers the generator's four queries."""
    def __init__(self):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **kw):
        r = MagicMock()
        if "asset_type" in query:
            r.single.return_value = {"t": "Extruder"}
        elif "Procedure" in query:
            r.data.return_value = [{"pid": "c:die-p", "names": ["Die plug clearing", None]}, {"pid": "c:die", "names": ["Die plug clearing"]}]
        elif "ASSOCIATED_WITH" in query:
            r.data.return_value = [{"pid": "c:die", "codes": ["DIE_PLUG", "HI_DIE_PRESSURE"]}]
        else:   # MONITORED_BY: the transmitter (no children) wins over the die head
            r.data.return_value = [{"signal": "s:dp", "pid": "c:die", "part": "Die Head", "children": 2},
                                   {"signal": "s:dp", "pid": "c:die-p", "part": "Die pressure transmitter", "children": 0},
                                   {"signal": "s:aa", "pid": "c:feed", "part": "Feeder", "children": 0}]
        return r


def fakes(test):
    for target, kw in (("context_graph.coverage.grid", {"return_value": GRID}),
                       ("chat.asset_context.signal_columns", {"return_value": SIGNALS}),
                       ("context_graph.driver.session", {"side_effect": lambda: FakeSession()})):
        p = patch(target, **kw)
        p.start()
        test.addCleanup(p.stop)


class ManifestTests(TestCase):
    def setUp(self):
        fakes(self)

    def test_generated_from_the_graph(self):
        m = service.manifest(ASSET)
        self.assertEqual((m["template"], m["asset_name"], m["enabled"]), ("single_screw_extruder", "Extruder EXTR01", True))
        die = m["areas"][0]
        self.assertEqual((die["slot"], die["info"]["facts"], die["info"]["documents"], die["info"]["procedures"], die["alarms"]),
                         ("die", 15, 1, ["Die plug clearing"], ["DIE_PLUG", "HI_DIE_PRESSURE"]))
        self.assertEqual(m["areas"][2]["slot"], "other:c:odd")                     # an unknown part type: no slot
        self.assertEqual(m["sensors"], [{"column": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 84.3,
                                         "high": 99.2, "area": "die", "part": "Die pressure transmitter"}])   # no flags
        self.assertEqual(m["bindings"], {"heat": {}})                              # screw speed has no part here
        self.assertEqual(TwinManifest.objects.count(), 1)

    def test_kept_switched_and_regenerated(self):
        service.manifest(ASSET)
        self.assertFalse(service.set_enabled(ASSET, False, "angel")["enabled"])
        self.assertFalse(service.manifest(ASSET)["enabled"])                       # the switch survives
        self.assertFalse(service.manifest(ASSET, regenerate=True)["enabled"])      # and a regeneration
        with patch("context_graph.coverage.grid", side_effect=RuntimeError("graph down")):
            self.assertEqual(service.manifest(ASSET, regenerate=True)["template"], "single_screw_extruder")   # kept one served

    def test_no_graph_and_nothing_kept_is_an_error(self):
        with patch("context_graph.coverage.grid", side_effect=RuntimeError("graph down")):
            with self.assertRaisesMessage(service.TwinError, "no twin for bsk:asset:EXTR01"):
                service.manifest(ASSET)


NOW = datetime(2026, 11, 8, 22, 0, tzinfo=timezone.utc)


class LiveTests(TestCase):
    def setUp(self):
        fakes(self)
        service.manifest(ASSET)

    def test_opc_first(self):
        opc = {"state": {"name": "RUNNING", "heartbeat_age_s": 1.0}, "read_at": NOW.isoformat(), "error": None,
               "fault": {"active": True, "code": 9, "label": "DIE_PLUG"},
               "analog": [{"column": "die_pressure_bar", "value": 104.0, "outside": True}, {"column": None, "value": 0}]}
        with patch("plc.live.live", return_value=opc):
            l = service.live(ASSET)
        self.assertEqual((l["source"], l["state"], l["age_s"], l["fault"]["label"]), ("opc", "RUNNING", 1.0, "DIE_PLUG"))
        self.assertEqual(l["values"], {"die_pressure_bar": {"value": 104.0, "outside": True}})

    def test_the_historian_when_the_plc_cannot_be_read(self):
        from plc import live as plc_live
        row_ts = datetime.now(timezone.utc) - timedelta(seconds=40)
        hist = {"period_seconds": 20, "signals": [], "ts": [row_ts.isoformat()],
                "series": {"die_pressure_bar": [90.0], "machine_state": ["STOPPED"], "alarm_code": ["UNPLANNED_STOP"]}}
        with patch("plc.live.live", side_effect=plc_live.LiveError("no OPC UA connection")), \
                patch("context_service.service.signal_history", return_value=hist):
            l = service.live(ASSET)
        self.assertEqual((l["source"], l["state"], l["fault"]["label"]), ("historian", "STOPPED", "UNPLANNED_STOP"))
        self.assertEqual(l["values"]["die_pressure_bar"], {"value": 90.0, "outside": False})
        self.assertGreaterEqual(l["age_s"], 40)


class ApiAndToolTests(TransactionTestCase):
    def setUp(self):
        fakes(self)

    def test_api(self):
        c = APIClient()
        self.assertEqual(c.get("/api/twin/").status_code, 400)
        self.assertEqual(c.get("/api/twin/", {"asset_id": ASSET}).data["template"], "single_screw_extruder")
        self.assertFalse(c.put("/api/twin/", {"asset_id": ASSET, "enabled": False}, format="json").data["enabled"])
        self.assertFalse(c.post("/api/twin/generate/", {"asset_id": ASSET}, format="json").data["enabled"])

    def test_tools(self):
        from context_service import mcp_server
        result = asyncio.run(mcp_server.build_server().call_tool("get_twin", {"asset_id": ASSET}))
        content = result[0] if isinstance(result, tuple) else result
        self.assertEqual(json.loads(content[0].text)["areas"][0]["name"], "Die Head")
