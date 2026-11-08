"""Context MCP server (P10b): tokens, exposure, the settings API and the tools (called in
process; the service functions are mocked)."""

import asyncio
import json
from unittest.mock import patch

from django.test import TestCase, TransactionTestCase, override_settings
from rest_framework.test import APIClient

from context_service import mcp_access, mcp_server, service
from context_service.models import McpSettings, McpToken
from usage.models import ModelCall


class TokenTests(TestCase):
    def test_token_is_shown_once_and_only_its_hash_is_stored(self):
        record, token = mcp_access.create_token("  BSKLAB   EDGE ")
        self.assertTrue(token.startswith("bsk_") and len(token) > 40)
        self.assertEqual((record.name, record.prefix), ("BSKLAB EDGE", token[:10]))
        self.assertNotIn(token, json.dumps(mcp_access.token_json(record)))
        self.assertNotEqual(record.token_hash, token)

        self.assertEqual(mcp_access.verify(token).pk, record.pk)
        self.assertIsNotNone(McpToken.objects.get(pk=record.pk).last_used_at)
        for bad in ("", "bsk_wrong", token[:-1], "Bearer " + token):
            self.assertIsNone(mcp_access.verify(bad))

        self.assertTrue(mcp_access.revoke(record.id))
        self.assertIsNone(mcp_access.verify(token))
        self.assertFalse(mcp_access.revoke(record.id))
        with self.assertRaisesMessage(ValueError, "name is required"):
            mcp_access.create_token(" ")

    @override_settings(MCP_TAILSCALE_HOST="100.92.170.72")
    def test_listen_hosts_never_all_interfaces(self):
        self.assertEqual(mcp_access.listen_hosts("off"), [])
        self.assertEqual(mcp_access.listen_hosts("local"), ["127.0.0.1"])
        self.assertEqual(mcp_access.listen_hosts("tailscale"), ["127.0.0.1", "100.92.170.72"])
        self.assertEqual(mcp_access.listen_hosts(), ["127.0.0.1"])          # the default setting
        for exposure in ("off", "local", "tailscale"):
            self.assertNotIn("0.0.0.0", mcp_access.listen_hosts(exposure))


@patch("context_service.views.mcp_answers", return_value=True)
class SettingsApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @override_settings(MCP_TAILSCALE_HOST="100.92.170.72", MCP_PORT=8002)
    def test_exposure_and_tokens(self, answers):
        d = self.client.get("/api/context/mcp/").data
        self.assertEqual((d["exposure"], d["url"], d["answering"]), ("local", "http://127.0.0.1:8002/mcp", True))
        self.assertIn("assemble_context", d["tools"])
        self.assertEqual([o["key"] for o in d["options"] if not o["available"]], ["lan", "internet"])

        d = self.client.put("/api/context/mcp/", {"exposure": "tailscale"}, format="json").data
        self.assertEqual((d["exposure"], d["url"]), ("tailscale", "http://100.92.170.72:8002/mcp"))
        self.assertEqual(McpSettings.get().exposure, "tailscale")

        d = self.client.put("/api/context/mcp/", {"exposure": "off"}, format="json").data
        self.assertEqual((d["url"], d["answering"], d["addresses"]), (None, False, []))

        for exposure in ("internet", "lan", "everywhere"):
            self.assertEqual(self.client.put("/api/context/mcp/", {"exposure": exposure}, format="json").status_code, 400)
        self.assertEqual(McpSettings.get().exposure, "off")

        created = self.client.post("/api/context/mcp/tokens/", {"name": "Claude Desktop"}, format="json")
        self.assertEqual(created.status_code, 201)
        token = created.data["token"]
        listed = self.client.get("/api/context/mcp/").data["tokens"]
        self.assertEqual([t["name"] for t in listed], ["Claude Desktop"])
        self.assertNotIn("token", listed[0])
        self.assertEqual(self.client.delete(f"/api/context/mcp/tokens/{created.data['id']}/").status_code, 204)
        self.assertIsNone(mcp_access.verify(token))
        self.assertEqual(self.client.delete(f"/api/context/mcp/tokens/{created.data['id']}/").status_code, 404)
        self.assertEqual(self.client.post("/api/context/mcp/tokens/", {}, format="json").status_code, 400)

    def test_health_entry_is_idle_when_off(self, answers):
        from chat.views import TRACKED_ENDPOINTS, _check_single_endpoint
        ep = next(e for e in TRACKED_ENDPOINTS if e["id"] == "context-mcp")
        self.assertEqual(_check_single_endpoint(ep)["status"], "online")
        McpSettings.objects.update_or_create(pk=1, defaults={"exposure": "off"})
        self.assertEqual(_check_single_endpoint(ep)["status"], "idle")
        McpSettings.objects.update_or_create(pk=1, defaults={"exposure": "local"})
        answers.return_value = False
        self.assertEqual(_check_single_endpoint(ep)["status"], "offline")


class ToolTests(TransactionTestCase):
    """The tools run service functions in worker threads (their own DB connections)."""

    def call(self, name, args):
        async def run():
            return await mcp_server.build_server().call_tool(name, args)
        return asyncio.run(run())

    def test_tools_are_registered_and_read_only(self):
        """Read-only, except the problem-solving tickets (P29): the graph and the library are never written here."""
        from context_service.views import MCP_TOOLS, TICKET_TOOLS
        tools = asyncio.run(mcp_server.build_server().list_tools())
        self.assertEqual([t.name for t in tools], MCP_TOOLS)
        self.assertTrue(all(t.description for t in tools))
        others = [t.name for t in tools if t.name not in TICKET_TOOLS]
        self.assertFalse([n for n in others if any(w in n for w in ("create", "delete", "update", "write", "approve", "save",
                                                                     "add_", "remove", "open_", "submit", "close"))])
        self.assertTrue(all("ticket" in n for n in TICKET_TOOLS))

    def test_assemble_context_passes_arguments_and_records_usage(self):
        with patch.object(service, "assemble", return_value={"packet_version": "1"}) as assemble:
            self.call("assemble_context", {"query": "die pressure", "asset_id": "bsk:asset:EXTR01", "budget_chars": 3000})
            assemble.assert_called_once_with("die pressure", "bsk:asset:EXTR01", True, 3000, None)
            self.call("assemble_context", {"query": "anything"})
            assemble.assert_called_with("anything", None, True, None, None)       # "" and 0 mean "not given"
        calls = ModelCall.objects.filter(purpose="mcp")
        self.assertEqual([(c.model_id, c.status) for c in calls], [("mcp:assemble_context", "ok")] * 2)

    def test_service_errors_reach_the_client_as_tool_errors(self):
        with patch.object(service, "entity_context", side_effect=service.ContextError("unknown entity 'x'")):
            with self.assertRaisesMessage(Exception, "unknown entity 'x'"):
                self.call("get_entity_context", {"entity_id": "x"})
        self.assertEqual(ModelCall.objects.get(purpose="mcp").status, "error")


class AuthTests(TransactionTestCase):
    def request(self, path, token=None):
        """One request through the auth middleware; returns the status the app (or the middleware) answered."""
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def run():
            headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
            await mcp_server.BearerAuth(app)({"type": "http", "path": path, "headers": headers}, None,
                                             lambda message: _collect(message))

        async def _collect(message):
            sent.append(message)
        asyncio.run(run())
        return sent[0]["status"]

    def test_token_required_except_for_health(self):
        _, token = mcp_access.create_token("EDGE")
        self.assertEqual(self.request("/mcp"), 401)
        self.assertEqual(self.request("/mcp", "bsk_wrong"), 401)
        self.assertEqual(self.request("/mcp", token), 200)
        self.assertEqual(self.request("/health"), 200)


class AnswerRouteTests(TransactionTestCase):
    """P24-3: POST /answer on the MCP server streams the engine's events, behind the token."""

    def post(self, body, token=None):
        from starlette.testclient import TestClient
        client = TestClient(mcp_server.BearerAuth(mcp_server.build_server().streamable_http_app()),
                            base_url="http://127.0.0.1:8002")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return client.post("/answer", headers=headers, content=body if isinstance(body, str) else json.dumps(body))

    def test_token_required(self):
        self.assertEqual(self.post({"question": "hi", "model": "ollama-qwen3-8b"}).status_code, 401)

    def test_events_are_streamed(self):
        _, token = mcp_access.create_token("EDGE")
        events = [{"type": "status", "text": "Reading the question…"}, {"type": "token", "text": "Hi"},
                  {"type": "done", "reply": "Hi", "sources": []}]
        with patch("chat.views.answer_events", return_value=iter(events)) as engine:
            r = self.post({"question": "hi", "model": "ollama-qwen3-8b", "asset_id": "bsk:asset:EXTR01",
                           "history": [{"role": "user", "content": "x"}], "system_prompt": "Be brief.", "picture": "A plate"},
                          token)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        got = [json.loads(chunk[6:]) for chunk in r.text.split("\n\n") if chunk.startswith("data: ")]
        self.assertEqual(got, events)
        self.assertEqual(engine.call_args.kwargs, {"question": "hi", "model_id": "ollama-qwen3-8b",
                                                   "asset_id": "bsk:asset:EXTR01", "history": [{"role": "user", "content": "x"}],
                                                   "use_rag": True, "system_prompt": "Be brief.", "picture": "A plate",
                                                   "asked_before": False})

    def test_a_bad_request_is_an_error_event(self):
        _, token = mcp_access.create_token("EDGE")
        r = self.post({"question": "hi", "model": "gpt-9"}, token)
        got = [json.loads(chunk[6:]) for chunk in r.text.split("\n\n") if chunk.startswith("data: ")]
        self.assertEqual((got[-1]["type"], got[-1]["kind"]), ("error", "bad_request"))
        self.assertEqual(self.post("not json", token).status_code, 400)


class MonitorPostsToolTests(TransactionTestCase):
    """P24-4: EDGE's Home reads the Studio's monitoring posts."""

    def test_posts_newest_first_with_paging(self):
        from datetime import datetime, timezone
        from monitor.models import MonitorPost
        for n in range(3):
            MonitorPost.objects.create(asset_id="bsk:asset:EXTR01", kind="summary", title=f"Hour {n}", text="ran",
                                       ts=datetime(2026, 11, 8, n, tzinfo=timezone.utc))
        def call(args):
            result = asyncio.run(mcp_server.build_server().call_tool("get_monitor_posts", args))
            content = result[0] if isinstance(result, tuple) else result
            return json.loads(content[0].text)
        first = call({"asset_id": "bsk:asset:EXTR01", "limit": 2})
        self.assertEqual([p["title"] for p in first["posts"]], ["Hour 2", "Hour 1"])
        self.assertTrue(first["more"])
        rest = call({"asset_id": "bsk:asset:EXTR01", "before_id": first["posts"][-1]["id"]})
        self.assertEqual(([p["title"] for p in rest["posts"]], rest["more"]), (["Hour 0"], False))


class SpcToolTests(TransactionTestCase):
    """P30: EDGE reads SPC from the Studio (read only); a chart is thinned for a tablet."""

    def call(self, name, args):
        result = asyncio.run(mcp_server.build_server().call_tool(name, args))
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    def test_thinning_keeps_shape_and_every_rule_point(self):
        from spc.service import thin_samples
        samples = [{"ts": i, "value": float(i % 50), "rules": ["beyond_limits"] if i in (7, 777, 2999) else []}
                   for i in range(3000)]
        thin = thin_samples(samples, limit=200)
        self.assertLessEqual(len(thin), 203)
        self.assertEqual([s["ts"] for s in thin], sorted(s["ts"] for s in thin))          # time order kept
        self.assertTrue({7, 777, 2999} <= {s["ts"] for s in thin})                      # rule points kept
        self.assertEqual((min(s["value"] for s in thin), max(s["value"] for s in thin)), (0.0, 49.0))
        self.assertEqual(len(thin_samples(samples[:100], limit=200)), 100)              # short: unchanged

    def test_chart_overview_and_events_are_plain_json_with_tickets(self):
        from datetime import datetime, timedelta, timezone
        from problems.models import Ticket
        from spc import service as spc
        from spc.models import SpcEvent, SpcTool
        tool = SpcTool.objects.create(asset_id="bsk:asset:EXTR01", signal_id="bsk:signal:EXTR01/scrap_output_kg",
                                      column="scrap_output_kg", name="Scrap output", unit="kg")
        t0 = datetime(2026, 11, 7, 16, 13, tzinfo=timezone.utc)
        ev = SpcEvent.objects.create(tool=tool, tool_version=1, type="control_violation", rule="beyond_limits",
                                     started_at=t0, last_at=t0 + timedelta(hours=1), value=4.6, extreme=5.2,
                                     limits={"ucl": 3.5})
        Ticket.objects.create(asset_id=tool.asset_id, trigger="spc_control", title="Scrap", window_start=t0,
                              source_key=f"spc:{ev.pk}")
        samples = [{"ts": t0 + timedelta(seconds=20 * i), "value": 2.0 + (i % 7) / 10, "range": None, "group": "",
                    "rules": ["beyond_limits"] if i == 1500 else []} for i in range(2000)]
        fake = {"tool_id": str(tool.id), "name": tool.name, "samples": samples, "window_start": t0,
                "box": {"outliers": list(range(80))}, "_active": 1}
        with patch.object(spc, "detail", return_value=fake):
            chart = self.call("get_spc_chart", {"tool_id": str(tool.id), "hours": 999})
        self.assertTrue(chart["thinned"])
        self.assertEqual(chart["sample_count_full"], 2000)
        self.assertLessEqual(len(chart["samples"]), spc.CLIENT_POINTS + 1)
        self.assertIn(["beyond_limits"], [s["rules"] for s in chart["samples"]])
        self.assertEqual(chart["window_start"], t0.isoformat())                         # ISO text, not a datetime
        self.assertEqual(len(chart["box"]["outliers"]), spc.CLIENT_OUTLIERS)
        self.assertNotIn("_active", chart)
        self.assertEqual(chart["events"][0]["ticket"]["number"], Ticket.objects.get().number)

        with self.assertRaisesMessage(Exception, "SPC tool not found"):
            self.call("get_spc_chart", {"tool_id": "not-a-tool"})

        events = self.call("list_spc_events", {"asset_id": "bsk:asset:EXTR01"})
        self.assertEqual(events["events"][0]["ticket"]["status"], "open")
        with patch.object(spc, "screening", return_value={"tools": [], "summary": {}, "newest_row": t0}) as screening:
            overview = self.call("get_spc_overview", {"asset_id": "bsk:asset:EXTR01", "hours": 500})
        screening.assert_called_once_with("bsk:asset:EXTR01", None, None, 72)            # hours capped at 72
        self.assertEqual(overview["newest_row"], t0.isoformat())
