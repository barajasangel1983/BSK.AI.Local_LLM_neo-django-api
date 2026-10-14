"""The PLC page (P13f): the status panel's rules, and Start / Pause with every safeguard. No OPC UA server is
reached: the client is replaced by a fake PLC."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from plc import opc, service, status
from plc.models import PlcCommand

NOW = datetime(2026, 10, 14, 6, 0, 0, tzinfo=timezone.utc)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def healthy(**changes) -> dict:
    """The values of a healthy PLC at NOW: link up, clock set, generator running, a row finished 5 s ago."""
    v = {
        "Heartbeat.State": 2, "Heartbeat.LastCommUtc": NOW - timedelta(seconds=1), "Sample.PeriodCounter": 11386,
        "Sample.TimestampUtc": NOW - timedelta(seconds=25), "Sample.PeriodSeconds": 20, "Step.StepName": "RUNNING",
        "DigitalOut[0].Value": False, "DigitalOut[1].Value": False, "DigitalOut[2].Value": False, "DigitalOut[3].Value": False,
        "DigitalIn[8].Value": False, "Faults.ActiveLabel": "", "Faults.ActiveCode": 0,
        "ExtruderSim.Running": True, "ExtruderSim.ClockSet": True, "ExtruderSim.EventType": 0, "ExtruderSim.EventId": 0,
        "ExtruderSim.EventSignal": "", "Plc.UptimeSeconds": 227661, "Plc.BootUtc": datetime(2026, 10, 11, 14, 34, 21, tzinfo=timezone.utc),
        "Plc.MapVersion": 3, "Plc.ExpectedMapVersion": 3, "Plc.ClockUtc": NOW, "Plc.ClockOffsetSeconds": -1,
        "Plc.RestartCount": 0, "Plc.LinkLossCount": 0, "Plc.LinkUpUtc": NOW - timedelta(hours=5), "Plc.LinkDownUtc": EPOCH,
    }
    v.update(changes)
    return v


def titles(values, error=None, settled=True):
    return [(c["level"], c["title"]) for c in status.panel(values, error, NOW, settled)["conditions"]]


class PanelTests(SimpleTestCase):
    def test_a_healthy_plc(self):
        p = status.panel(healthy(), None, NOW, True)
        self.assertEqual((p["verdict"], [c["title"] for c in p["conditions"]]), ("ok", ["PLC healthy"]))
        self.assertEqual({led["key"]: led["state"] for led in p["leds"]},
                         {"status": "blink", "user": True, "led1": True, "led2": True, "led3": False, "led4": False})
        facts = {f["name"]: f["value"] for f in p["facts"]}
        self.assertEqual((facts["PLC link"], facts["Generator"], facts["Uptime"], facts["PLC clock"], facts["Last row finished"]),
                         ("Connected", "running", "2 d 15 h", "-1 s against the Mini PC", "5 s ago"))

    def test_the_server_or_the_plc_down_makes_the_leds_unknown(self):
        down = status.panel(None, "timeout", NOW, True)
        self.assertEqual((down["verdict"], titles(None, "timeout")[0]), ("bad", ("bad", "OPC UA server not reachable")))
        self.assertTrue(all(led["state"] is None for led in down["leds"]))
        lost = titles(healthy(**{"Heartbeat.State": 0, "Plc.LinkDownUtc": NOW - timedelta(minutes=3)}))
        self.assertEqual(lost, [("bad", "PLC not answering since 2026-10-14 05:57:00 UTC")])

    def test_every_bad_and_warn_condition_in_order(self):
        found = titles(healthy(**{"Plc.MapVersion": 2, "DigitalOut[1].Value": True, "DigitalOut[3].Value": True,
                                  "Heartbeat.LastCommUtc": NOW - timedelta(seconds=9), "Plc.ClockOffsetSeconds": 4,
                                  "Sample.TimestampUtc": NOW - timedelta(seconds=80)}))
        self.assertEqual(found, [("bad", "Firmware and server do not match"), ("bad", "Relay 2, 4 reads ON"),
                                 ("warn", "PLC data is 9 s old"), ("warn", "PLC clock is 4 s ahead of the Mini PC"),
                                 ("warn", "No new row for 1 min 0 s")])
        self.assertEqual(titles(healthy(**{"ExtruderSim.Running": False}))[0], ("warn", "Generator paused"))
        self.assertEqual(titles(healthy(**{"ExtruderSim.ClockSet": False}))[0], ("warn", "PLC clock not set"))

    def test_no_new_row_waits_until_the_generator_has_settled(self):
        old = {"Sample.TimestampUtc": NOW - timedelta(seconds=80)}
        self.assertEqual(titles(healthy(**old), settled=False), [("ok", "PLC healthy")])
        self.assertEqual(titles(healthy(**old, **{"Plc.UptimeSeconds": 50})),                       # the first 60 s of uptime
                         [("ok", "PLC healthy"), ("info", "PLC started 50 s ago")])

    def test_the_info_conditions_do_not_change_the_verdict(self):
        p = status.panel(healthy(**{"Plc.UptimeSeconds": 300, "Sample.PeriodCounter": 0, "DigitalIn[8].Value": True,
                                    "Faults.ActiveLabel": "Die pressure high", "ExtruderSim.EventType": 4,
                                    "ExtruderSim.EventId": 17, "ExtruderSim.EventSignal": "bulk_density_g_l"}), None, NOW, True)
        self.assertEqual([(c["level"], c["title"]) for c in p["conditions"]], [
            ("ok", "PLC healthy"), ("info", "PLC started 5 min 0 s ago"), ("info", "Waiting for the first row"),
            ("info", "Process alarm: Die pressure high"), ("info", "Anomaly 17 being injected on bulk_density_g_l")])
        self.assertEqual((p["verdict"], [led["state"] for led in p["leds"]][3:]), ("ok", [False, True, True]))


class FakePlc:
    """The PLC as the client sees it: Start and Pause change Running; it can be made unreachable or deaf."""

    def __init__(self, running=True):
        # The service compares with the real clock: the fake's times are moved to now.
        shift = datetime.now(timezone.utc) - NOW
        self.values = {k: (v + shift if isinstance(v, datetime) and v > EPOCH else v)
                       for k, v in healthy(**{"ExtruderSim.Running": running}).items()}
        self.calls: list[str] = []
        self.down = self.ignores = False

    def read_values(self):
        if self.down:
            raise opc.OpcError("timeout")
        return dict(self.values)

    def call(self, command):
        self.calls.append(command)
        if not self.ignores:
            self.values["ExtruderSim.Running"] = command == "Start"


@override_settings(PLC_CONTROL_ENABLED=True, PLC_COMMAND_SPACING_SECONDS=10, PLC_STATUS_CACHE_SECONDS=2)
class CommandTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.plc = FakePlc()
        patch("plc.opc.read_values", side_effect=self.plc.read_values).start()
        patch("plc.opc.call", side_effect=self.plc.call).start()
        patch("plc.service.time.sleep").start()
        patch("plc.service.VERIFY_SECONDS", 0.05).start()
        self.addCleanup(patch.stopall)
        service._cache.clear()

    def send(self, command, confirm=None, by="Admin User"):
        return self.client.post("/api/plc/command/", {"command": command, "confirm": command if confirm is None else confirm, "by": by},
                                format="json")

    def records(self):
        return [(c.command, c.outcome, c.reason[:30]) for c in PlcCommand.objects.order_by("id")]

    def test_pause_then_start_are_sent_checked_and_recorded(self):
        r = self.send("Pause")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual({k: body["command"][k] for k in ("command", "outcome", "requested_by", "running_before", "running_after", "period_before")},
                         {"command": "Pause", "outcome": "done", "requested_by": "Admin User", "running_before": True,
                          "running_after": False, "period_before": 11386})
        self.assertEqual(body["status"]["running"], False)                             # read again, not from the cache
        PlcCommand.objects.update(created_at=datetime.now(timezone.utc) - timedelta(seconds=11))
        self.assertEqual(self.send("Start").json()["command"]["outcome"], "done")
        self.assertEqual(self.plc.calls, ["Pause", "Start"])

    def test_every_safeguard_refuses_and_records_without_calling_the_plc(self):
        self.assertEqual(self.send("Reset").status_code, 400)                          # only Start and Pause
        self.assertEqual(self.send("Pause", confirm="pause").status_code, 400)        # typed exactly
        self.assertEqual(self.send("Start").status_code, 409)                          # already running
        with override_settings(PLC_CONTROL_ENABLED=False):
            off = self.send("Pause")
        self.assertEqual((off.status_code, off.json()["error"]), (403, "control is off: PLC_CONTROL_ENABLED is not set in the backend's .env"))
        self.plc.down = True
        self.assertEqual(self.send("Pause").status_code, 503)                          # the PLC cannot be read
        self.plc.down = False
        self.plc.values["Heartbeat.State"] = 0
        self.assertEqual(self.send("Pause").status_code, 503)                          # the PLC link is down
        self.assertEqual(self.plc.calls, [])
        self.assertEqual([o for _, o, _ in self.records()], ["refused"] * 6)
        self.assertEqual(self.records()[0][2], "not a command; the commands ar")

    def test_two_commands_are_at_least_the_spacing_apart(self):
        self.assertEqual(self.send("Pause").status_code, 200)
        too_soon = self.send("Start")
        self.assertEqual(too_soon.status_code, 429)
        self.assertIn("less than 10 s ago", too_soon.json()["error"])
        self.assertEqual(self.plc.calls, ["Pause"])

    def test_one_command_at_a_time(self):
        service._lock.acquire()
        try:
            self.assertEqual(self.send("Pause").status_code, 409)
        finally:
            service._lock.release()
        self.assertEqual(self.plc.calls, [])

    def test_a_command_that_does_not_take_or_cannot_be_sent_is_a_failure(self):
        self.plc.ignores = True
        r = self.send("Pause")
        self.assertEqual((r.status_code, r.json()["command"]["outcome"]), (502, "failed"))
        self.assertIn("did not show paused", r.json()["command"]["reason"])
        PlcCommand.objects.update(created_at=datetime.now(timezone.utc) - timedelta(seconds=11))
        with patch("plc.opc.call", side_effect=opc.OpcError("BadCommunicationError")):
            r = self.send("Pause")
        self.assertEqual((r.status_code, r.json()["command"]["reason"]), (502, "the call failed: BadCommunicationError"))

    def test_the_status_and_the_record_of_commands(self):
        first = self.client.get("/api/plc/status/").json()
        self.assertEqual((first["verdict"], first["asset"], first["control"]["enabled"], first["control"]["commands"]),
                         ("ok", "EXTR01", True, ["Start", "Pause"]))
        self.plc.values["Heartbeat.State"] = 0
        self.assertEqual(self.client.get("/api/plc/status/").json()["verdict"], "ok")   # cached for 2 s
        service._cache.clear()
        self.assertEqual(self.client.get("/api/plc/status/").json()["verdict"], "bad")
        self.send("Reset")
        listed = self.client.get("/api/plc/commands/").json()["commands"]
        self.assertEqual([(c["command"], c["outcome"]) for c in listed], [("Reset", "refused")])


class ClientTests(SimpleTestCase):
    def test_only_start_and_pause_can_reach_the_client(self):
        with patch("plc.opc.asyncio.run") as run, self.assertRaises(ValueError):
            opc.call("Reset")
        run.assert_not_called()


class Ref:
    def __init__(self, node_id, node_class):
        self.NodeId, self.NodeClass = node_id, node_class


class FakeNode:
    def __init__(self, tree, node_id):
        self.tree, self.nodeid = tree, node_id

    async def get_children_descriptions(self):
        return self.tree.get(self.nodeid, [])


class BrowseTests(SimpleTestCase):
    def test_the_variables_are_found_depth_first_with_one_browse_per_object(self):
        from asyncua import ua
        V, O = ua.NodeClass.Variable, ua.NodeClass.Object
        tree = {"root": [Ref("Heartbeat", O), Ref("Plc.Uptime", V)], "Heartbeat": [Ref("Heartbeat.State", V), Ref("Heartbeat.Counter", V)]}
        client = type("C", (), {"get_node": lambda self, nid: FakeNode(tree, nid)})()
        import asyncio
        found = asyncio.run(opc._variables(client, FakeNode(tree, "root")))
        self.assertEqual([n.nodeid for n in found], ["Heartbeat.State", "Heartbeat.Counter", "Plc.Uptime"])


class OpcuaPageTests(TestCase):
    def setUp(self):
        service._opcua_cache.clear()
        self.addCleanup(service._opcua_cache.clear)

    def test_every_variable_with_its_value_and_type_or_the_error(self):
        variables = [{"path": "Heartbeat.Counter", "value": 228432, "type": "UInt32", "good": True}]
        with patch("plc.opc.read_all", return_value={"browsed_now": True, "variables": variables}) as read:
            body = APIClient().get("/api/plc/opcua/").json()
            APIClient().get("/api/plc/opcua/")                                          # within 2 s: the same answer
        self.assertEqual(read.call_count, 1)
        self.assertEqual((body["error"], body["asset"], body["variables"], body["browsed_now"]), (None, "EXTR01", variables, True))
        service._opcua_cache.clear()
        with patch("plc.opc.read_all", side_effect=opc.OpcError("TimeoutError")):
            down = APIClient().get("/api/plc/opcua/").json()
        self.assertEqual((down["error"], down["variables"]), ("TimeoutError", []))
