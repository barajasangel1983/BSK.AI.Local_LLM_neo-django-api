"""P31: alarm episodes from the historian (a fake here), chattering merged, categories, the API and the MCP tools."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase
from rest_framework.test import APIClient

from alarms import service
from alarms.models import AlarmCursor, AlarmEpisode

ASSET = "bsk:asset:EXTR01"
T0 = datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc)
KNOW = {"DIE_PLUG": {"id": "bsk:alarm:EXTR01/DIE_PLUG", "code": "DIE_PLUG", "description": "Die plugged", "severity": "high"},
        "HI_MOISTURE": {"id": "bsk:alarm:EXTR01/HI_MOISTURE", "code": "HI_MOISTURE", "description": "Moisture high", "severity": "warning"}}


def code_at(t: datetime) -> str:
    """HI_MOISTURE chatters (1 min on at 10:05, 10:08, 10:20); DIE_PLUG 10:30-10:34 while ALARMED."""
    m = (t - T0).total_seconds() / 60
    if 5 <= m < 6 or 8 <= m < 9 or 20 <= m < 21:
        return "HI_MOISTURE"
    if 30 <= m < 34:
        return "DIE_PLUG"
    return ""


class Fake:
    def __init__(self, test, code=code_at):
        self.code = code
        for target, fake in (("context_service.service.signal_history", self.history),
                             ("historian.source.current", lambda: "live"), ("alarms.service.knowledge", lambda a: KNOW)):
            p = patch(target, side_effect=fake)
            p.start()
            test.addCleanup(p.stop)

    def history(self, asset_id, start, end, signals=None, bucket_minutes=None):
        start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
        first = start.replace(second=(start.second // 20) * 20, microsecond=0) + timedelta(seconds=20)
        stamps = []
        t = first
        while t <= end:
            stamps.append(t)
            t += timedelta(seconds=20)
        return {"source": "historian_live", "period_seconds": 20, "signals": [],
                "ts": [s.isoformat() for s in stamps],
                "series": {"alarm_code": [self.code(s) for s in stamps],
                           "machine_state": ["ALARMED" if self.code(s) == "DIE_PLUG" else "RUNNING" for s in stamps]}}


class EpisodeTests(TestCase):
    def test_chattering_is_one_episode_and_a_later_return_is_another(self):
        Fake(self)
        r = service.update(ASSET, until=T0 + timedelta(minutes=40), since=T0)
        self.assertEqual(r["rows"], 120)
        eps = list(AlarmEpisode.objects.order_by("started_at"))
        self.assertEqual([(e.code, e.repeats) for e in eps], [("HI_MOISTURE", 2), ("HI_MOISTURE", 1), ("DIE_PLUG", 1)])
        first = eps[0]
        self.assertEqual(round(first.alarm_seconds), 120)                     # 2 minutes on, the gap not counted
        self.assertEqual(round((first.ended_at - first.started_at).total_seconds() / 60), 4)   # 10:05 → 10:09
        self.assertEqual(eps[2].state_at_start, "ALARMED")
        self.assertEqual(AlarmCursor.objects.get().read_until, T0 + timedelta(minutes=40))

    def test_the_cursor_continues_an_episode_across_passes(self):
        Fake(self)
        service.update(ASSET, until=T0 + timedelta(minutes=32), since=T0)     # DIE_PLUG half way
        service.update(ASSET, until=T0 + timedelta(minutes=40))
        plug = AlarmEpisode.objects.get(code="DIE_PLUG")
        self.assertEqual(round(plug.alarm_seconds), 240)
        self.assertEqual(AlarmEpisode.objects.count(), 3)

    def test_categories_default_from_the_name_and_can_be_set(self):
        self.assertEqual([service.default_category(c) for c in ("HI_MOISTURE", "OUT_OF_SPEC_DENSITY", "DIE_PLUG",
                                                                 "MOISTURE_SENSOR_FAIL", "HI_DIE_PRESSURE", "LOW_OEE")],
                         ["quality", "quality", "equipment", "sensor", "process", "quality"])
        with patch("alarms.service.knowledge", return_value=KNOW):
            self.assertEqual(service.set_category(ASSET, "DIE_PLUG", "process", "angel")["category"], "process")
            with self.assertRaisesMessage(service.AlarmError, "category must be one of"):
                service.set_category(ASSET, "DIE_PLUG", "bad")


class ReadTests(TestCase):
    def setUp(self):
        Fake(self)
        service.update(ASSET, until=T0 + timedelta(minutes=34), since=T0)     # DIE_PLUG still on at the newest row

    def test_summary_active_now_and_per_code(self):
        s = service.summary(ASSET, hours=24)
        self.assertEqual([e["code"] for e in s["active"]], ["DIE_PLUG"])
        self.assertEqual(s["active"][0]["description"], "Die plugged")
        self.assertEqual([(c["code"], c["episodes"], c["repeats"]) for c in s["codes"]],
                         [("HI_MOISTURE", 2, 3), ("DIE_PLUG", 1, 1)])
        self.assertEqual(service.summary(ASSET, category="quality")["episodes"][0]["code"], "HI_MOISTURE")
        self.assertEqual(service.summary(ASSET, limit=0)["episodes"], [])
        with self.assertRaises(service.AlarmError):
            service.summary(ASSET, category="nope")

    def test_code_detail_with_knowledge_and_tickets(self):
        from problems.models import Ticket
        t = Ticket.objects.create(asset_id=ASSET, trigger="downtime", title="Down", window_start=T0 + timedelta(minutes=29),
                                  window_end=T0 + timedelta(hours=1))
        ctx = {"relationships": [{"type": "ADDRESSES", "node": {"id": "p1", "label": "Procedure", "name": "Die plug clearing",
                                                                 "properties": {"summary": "Stop feed, clear the die"}}},
                                 {"type": "ASSOCIATED_WITH", "node": {"id": "c1", "label": "Component", "name": "Die Head", "properties": {}}}]}
        with patch("context_service.service.entity_context", return_value=ctx):
            d = service.code_detail(ASSET, "DIE_PLUG")
        self.assertEqual((d["severity"], d["category"]), ("high", "equipment"))
        self.assertEqual([r["name"] for r in d["related"]], ["Die plug clearing", "Die Head"])
        self.assertEqual(d["tickets"][0]["number"], t.number)
        self.assertTrue(d["episodes"][0]["active"])

    def test_api(self):
        c = APIClient()
        self.assertEqual(c.get("/api/alarms/", {"asset_id": ASSET}).data["active"][0]["code"], "DIE_PLUG")
        self.assertEqual(c.put("/api/alarms/codes/", {"asset_id": ASSET, "code": "HI_MOISTURE", "category": "process"},
                               format="json").data["category"], "process")
        self.assertTrue(next(x for x in c.get("/api/alarms/codes/").data["codes"] if x["code"] == "HI_MOISTURE")["set"])
        self.assertEqual(c.get("/api/alarms/codes/DIE_PLUG/").status_code, 200)
        self.assertEqual(c.get("/api/alarms/", {"category": "x"}).status_code, 400)


class ToolTests(TransactionTestCase):
    def call(self, name, args):
        from context_service import mcp_server
        result = asyncio.run(mcp_server.build_server().call_tool(name, args))
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    def test_the_tools(self):
        Fake(self)
        service.update(ASSET, until=T0 + timedelta(minutes=40), since=T0)
        s = self.call("get_alarms", {"asset_id": ASSET, "hours": 9999})
        self.assertEqual(s["hours"], 744)
        self.assertEqual(len(s["episodes"]), 3)
        with patch("context_service.service.entity_context", return_value={"relationships": []}):
            d = self.call("get_alarm_code", {"asset_id": ASSET, "code": "HI_MOISTURE", "days": 500})
        self.assertEqual((d["days"], len(d["episodes"])), (90, 2))
