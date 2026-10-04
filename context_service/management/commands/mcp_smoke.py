"""Live check of the Context MCP server, as a real client would use it.

    python manage.py mcp_smoke                       # the address the Studio's setting gives clients
    python manage.py mcp_smoke --url http://127.0.0.1:8002/mcp
    python manage.py mcp_smoke --ask                 # also the `ask` tool (uses a model; slower)

Creates a temporary token, connects over Streamable HTTP with the official MCP client, calls
every tool, then deletes the token. Read-only.
"""

import asyncio
import json
import time

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from context_service import mcp_access
from context_service.models import McpSettings
from context_service.views import MCP_TOOLS


class Command(BaseCommand):
    help = "Check the Context MCP server end to end with a temporary token."

    def add_arguments(self, parser):
        parser.add_argument("--url", help="MCP address (default: the one for the current exposure setting).")
        parser.add_argument("--ask", action="store_true", help="Also call the ask tool.")
        parser.add_argument("--image", action="store_true", help="Also call describe_image (uses the BSK GPU).")

    def handle(self, *args, url=None, ask=False, image=False, **options):
        exposure = McpSettings.get().exposure
        if not url:
            if exposure == "off":
                raise CommandError("The MCP server is set to Off (Studio: Settings → MCP).")
            host = settings.MCP_TAILSCALE_HOST if exposure == "tailscale" else "127.0.0.1"
            url = f"http://{host}:{settings.MCP_PORT}/mcp"
        self.stdout.write(f"exposure: {exposure} · address: {url}")

        results: list[tuple[str, bool, str]] = []
        base = url.rsplit("/mcp", 1)[0]
        try:
            ok = requests.get(f"{base}/health", timeout=5).status_code == 200
            results.append(("health", ok, "answers" if ok else "does not answer"))
            code = requests.post(url, json={}, timeout=5).status_code
            results.append(("no token", code == 401, f"HTTP {code} (expected 401)"))
        except requests.RequestException as exc:
            raise CommandError(f"The MCP server is not reachable at {url}: {exc}")

        record, token = mcp_access.create_token("mcp_smoke (temporary)")
        try:
            results += asyncio.run(self._tools(url, token, ask, image))
        finally:
            record.delete()

        self.stdout.write("")
        for name, ok, detail in results:
            self.stdout.write((self.style.SUCCESS if ok else self.style.ERROR)(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}"))
        if not all(ok for _, ok, _ in results):
            raise CommandError("mcp_smoke failed")

    async def _tools(self, url, token, ask, image=False):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        out = []
        async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                out.append(("tools", names == MCP_TOOLS, f"{len(names)} listed"))

                async def call(name, args, describe):
                    start = time.monotonic()
                    try:
                        result = await session.call_tool(name, args)
                        text = result.content[0].text if result.content else ""
                        if result.isError:
                            return out.append((name, False, text[:200]))
                        detail = describe(json.loads(text))
                        out.append((name, True, f"{detail} ({time.monotonic() - start:.1f} s)"))
                        return json.loads(text)
                    except Exception as exc:
                        out.append((name, False, f"{type(exc).__name__}: {exc}"[:200]))

                assets = await call("list_assets", {}, lambda d: f"{len(d['assets'])} asset(s)")
                asset = (assets or {}).get("assets", [{}])[0].get("id") if assets and assets["assets"] else None
                await call("get_system_health", {}, lambda d: f"{len(d['services'])} services")
                await call("search_documents", {"query": "start up procedure", "top_k": 2},
                           lambda d: f"{len(d['results'])} excerpt(s), scope {d['scope']}")
                question = "Is anything out of its normal range?"
                await call("assemble_context", {"query": question, "asset_id": asset or "", "budget_chars": 3000},
                           lambda d: f"{len(d['graph_facts'])} facts, {len(d['document_evidence'])} excerpts, "
                                     f"{d['limits']['used_chars']} of {d['limits']['budget_chars']} characters")
                if asset:
                    await call("resolve_entity", {"text": question, "asset_id": asset}, lambda d: f"asset {d['asset']['name']}")
                    await call("get_entity_context", {"entity_id": asset}, lambda d: f"{d['name']}, {len(d['relationships'])} relationships")
                    await call("get_sources", {"entity_id": asset}, lambda d: f"origin {d.get('origin')}")
                    await call("get_operational_state", {"asset_id": asset},
                               lambda d: f"{len(d['values'])} values, state {(d['current_state'] or {}).get('machine_state')}")
                    await call("get_graph", {"asset_id": asset}, lambda d: f"{len(d['nodes'])} nodes, {len(d['links'])} links")
                docs = await call("list_documents", {}, lambda d: f"{len(d['documents'])} document(s)")
                described = next((d for d in (docs or {}).get("documents", []) if d["figures_described"]), None)
                if described:
                    detail = await call("get_document", {"document_id": described["id"]},
                                        lambda d: f"{d['filename']}: {len(d['figures'])} described figure(s)")
                    await call("get_document_page", {"document_id": described["id"], "page": 1},
                               lambda d: f"page 1 of {d['pages']}, {len(d['text'])} characters")
                    if detail and detail["figures"]:
                        start = time.monotonic()
                        result = await session.call_tool("get_figure_image", {"document_id": described["id"],
                                                                              "figure_index": detail["figures"][0]["index"]})
                        image = result.content[0] if result.content else None
                        ok = not result.isError and getattr(image, "type", "") == "image"
                        out.append(("get_figure_image", ok, f"{getattr(image, 'mimeType', '?')}, "
                                    f"{len(getattr(image, 'data', '')) * 3 // 4} bytes ({time.monotonic() - start:.1f} s)"))
                if image:
                    import base64
                    import io
                    from PIL import Image, ImageDraw, ImageFont
                    picture = Image.new("RGB", (900, 300), "white")
                    ImageDraw.Draw(picture).text((60, 100), "ALARM DIE_PLUG 104 bar", fill="black", font=ImageFont.load_default(size=56))
                    buf = io.BytesIO()
                    picture.save(buf, format="JPEG")
                    await call("describe_image", {"image_base64": base64.b64encode(buf.getvalue()).decode(), "question": "What does it say?"},
                               lambda d: ("read the text" if "DIE_PLUG" in d["description"].upper().replace(" ", "_") else "text NOT read")
                               + f": {d['description'][:80]!r}")
                if ask:
                    await call("ask", {"query": question, "asset_id": asset or ""}, lambda d: f"{len(d['answer'] or '')} characters from {d['model']}")
        return out
