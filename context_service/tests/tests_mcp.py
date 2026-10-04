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
        from context_service.views import MCP_TOOLS
        tools = asyncio.run(mcp_server.build_server().list_tools())
        self.assertEqual([t.name for t in tools], MCP_TOOLS)
        self.assertTrue(all(t.description for t in tools))
        self.assertFalse([t.name for t in tools if any(w in t.name for w in ("create", "delete", "update", "write", "approve"))])

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
