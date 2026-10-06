"""The Context MCP server (P10b): the Context Service as MCP tools over Streamable HTTP.

    python manage.py mcp_server        (systemd user service: deploy/systemd/neo-context-mcp.service)

- One process, separate from the API. It listens on port MCP_PORT of the addresses the
  Studio's exposure setting allows (Off / This machine / Tailscale) and follows a change of
  that setting within a few seconds, without a restart. It never listens on all interfaces.
- Every request needs `Authorization: Bearer <token>` (tokens are created in the Studio's
  Settings → MCP). `GET /health` is open, for the Health page.
- Tools are read-only wrappers of context_service.service; results are the same JSON as the
  REST endpoints. Each call is recorded for Analytics (purpose `mcp`).

Client contract: frontend repo, claude/MCP_client_brief.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import signal
import time

import requests
import uvicorn
from asgiref.sync import sync_to_async
from django.conf import settings
from django.db import close_old_connections
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from . import mcp_access, service

logger = logging.getLogger("chat")
_client: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_client", default="-")

INSTRUCTIONS = (
    "BSKLab Context Studio: curated knowledge about the plant's machines. "
    "Use `list_assets` to see what can be asked about. For a question, call `assemble_context` "
    "(one call returns graph facts with their sources, document excerpts and current values); "
    "pass `asset_id` when the user has chosen a machine. If `resolved_scope.ambiguous` is not empty, "
    "ask the user which candidate they mean. Graph facts are authoritative; cite documents by page."
)


def _run(tool: str, fn, *args, **kwargs):
    """Call a service function from a worker thread: fresh DB connection, usage record, log."""
    from usage.models import ModelCall

    close_old_connections()
    start = time.monotonic()
    status, error = "ok", ""
    try:
        return fn(*args, **kwargs)
    except service.ContextError as exc:
        status, error = "error", str(exc)
        raise ValueError(str(exc))      # shown to the client as the tool's error text
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ms = round((time.monotonic() - start) * 1000)
        logger.info("mcp tool=%s client=%s status=%s ms=%d %s", tool, _client.get(), status, ms, error[:200])
        try:
            ModelCall.objects.create(purpose="mcp", model_id=f"mcp:{tool}", latency_ms=ms, status=status, error=error[:255])
        except Exception:
            logger.exception("could not record mcp call")
        close_old_connections()


async def _call(tool: str, fn, *args, **kwargs):
    return await sync_to_async(_run, thread_sensitive=False)(tool, fn, *args, **kwargs)


def _health() -> dict:
    data = requests.get(f"{settings.MCP_STUDIO_API}/health/status/", timeout=30).json()
    return {"services": [{k: s.get(k) for k in ("id", "name", "status", "latency")} for s in data]}


def build_server() -> FastMCP:
    """A server with the tools registered (one instance per listening address)."""
    hosts = [f"{h}:{settings.MCP_PORT}" for h in ("127.0.0.1", "localhost", settings.MCP_TAILSCALE_HOST) if h]
    mcp = FastMCP(
        "BSKLab Context Studio", instructions=INSTRUCTIONS, stateless_http=True, json_response=True,
        max_request_body_size=12 * 1024 * 1024,      # describe_image carries a base64 picture
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                                     allowed_origins=[f"http://{h}" for h in hosts]),
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok", "server": "bsk-context-mcp"})

    @mcp.tool()
    async def list_assets() -> dict:
        """The machines (assets) in the curated Context Graph that can be asked about."""
        return {"assets": await _call("list_assets", service.list_assets)}

    @mcp.tool()
    async def resolve_entity(text: str, asset_id: str = "") -> dict:
        """Find which asset and which of its components, signals, alarms or procedures a text names.
        Returns {asset, focus, ambiguous}; ambiguous candidates must be put to the user, never guessed."""
        return await _call("resolve_entity", service.resolve, text, asset_id or None)

    @mcp.tool()
    async def get_entity_context(entity_id: str) -> dict:
        """One curated entity (canonical id such as bsk:component:EXTR01/die-head) with its properties
        and direct relationships."""
        return await _call("get_entity_context", service.entity_context, entity_id)

    @mcp.tool()
    async def get_sources(entity_id: str) -> dict:
        """Where an entity came from: its origin and every evidence record (document page, import row, figure)."""
        return await _call("get_sources", service.entity_sources, entity_id)

    @mcp.tool()
    async def search_documents(query: str, asset_id: str = "", top_k: int = 5) -> dict:
        """Reranked excerpts from the document library. With asset_id, that asset's documents are searched first.
        Weak matches are not returned; when nothing passes, `closest` names the nearest documents."""
        return await _call("search_documents", service.search_documents, query, asset_id or None, top_k)

    @mcp.tool()
    async def assemble_context(query: str, asset_id: str = "", include_documents: bool = True,
                               budget_chars: int = 0, search_queries: list[str] | None = None) -> dict:
        """The Context Packet for a question: resolved asset and focus, graph facts with sources,
        document excerpts, current values against their normal range, and `context_text` (the same
        content as prompt text). budget_chars limits its size for small models (0 = default).
        search_queries: for a question with several topics, one short document search per topic
        (up to 4); `document_search` in the packet says what was searched and, when nothing was
        relevant enough, which documents came closest."""
        return await _call("assemble_context", service.assemble, query, asset_id or None, include_documents,
                           budget_chars or None, search_queries or None)

    @mcp.tool()
    async def ask(query: str, asset_id: str = "", model: str = "", include_documents: bool = True) -> dict:
        """Answer a question with one of the Studio's models, from the Context Packet. For clients
        without a model of their own. Returns {answer, model, packet}."""
        return await _call("ask", service.ask, query, asset_id or None, model or None, include_documents)

    @mcp.tool()
    async def get_operational_state(asset_id: str) -> dict:
        """Current machine state and signal values against their normal range (historian data)."""
        return await _call("get_operational_state", service.operational_state, asset_id)

    @mcp.tool()
    async def get_signal_history(asset_id: str, start: str, end: str, signals: list[str] | None = None,
                                 bucket_minutes: int = 0) -> dict:
        """Recorded values of an asset's signals between two times (ISO, start < ts <= end), oldest first,
        by column: `ts` and `series[<signal key>]`, with each signal's name, unit and normal range.
        signals: ids or keys to limit the columns (default all). bucket_minutes > 1 averages each bucket.
        At most 1,500 points and 31 days per call. Historian data (get_operational_state gives its latest time)."""
        return await _call("get_signal_history", service.signal_history, asset_id, start, end, signals or None,
                           bucket_minutes or None)

    @mcp.tool()
    async def list_documents() -> dict:
        """The document library: each document with its pages, whether it is searchable, and its figures."""
        return await _call("list_documents", service.list_documents)

    @mcp.tool()
    async def get_document(document_id: str) -> dict:
        """One document with its described figures (page, kind, caption, description)."""
        return await _call("get_document", service.get_document, document_id)

    @mcp.tool()
    async def get_document_page(document_id: str, page: int) -> dict:
        """The text of one page of a document (Markdown, with figure descriptions in place)."""
        return await _call("get_document_page", service.get_document_page, document_id, page)

    @mcp.tool()
    async def get_figure_image(document_id: str, figure_index: int) -> Image:
        """A described figure of a document as a JPEG image."""
        return Image(data=await _call("get_figure_image", service.figure_image, document_id, figure_index), format="jpeg")

    @mcp.tool()
    async def get_graph(asset_id: str, depth: int = 2) -> dict:
        """The curated graph around an asset, for a viewer: nodes (id, label, name, properties) and
        links (source, target, type). depth 1 to 3."""
        return await _call("get_graph", service.get_graph, asset_id, depth)

    @mcp.tool()
    async def describe_image(image_base64: str, question: str = "") -> dict:
        """What the vision model sees in a picture (base64 JPEG / PNG / WebP, at most 8 MB; send it
        downscaled to about 1280 px): what it shows, the legible text and tags, anything abnormal.
        The picture is not kept. Errors starting with gpu_busy / gpu_unavailable mean: try later."""
        import base64
        import binascii
        try:
            data = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_base64 is not valid base64")
        return await _call("describe_image", service.describe_image, data, question)

    @mcp.tool()
    async def get_system_health() -> dict:
        """Status of the services the Studio depends on (models, graph, document store, historian)."""
        return await _call("get_system_health", _health)

    return mcp


class BearerAuth:
    """ASGI middleware: a valid token on every request except GET /health."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/health":
            return await self.app(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        record = await sync_to_async(self._verify, thread_sensitive=False)(token)
        if record is None:
            body = json.dumps({"error": "a valid token is required (Authorization: Bearer <token>); "
                                        "create one in the Studio: Settings → MCP"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
            return await send({"type": "http.response.body", "body": body})
        _client.set(record.name)
        return await self.app(scope, receive, send)

    @staticmethod
    def _verify(token):
        close_old_connections()
        try:
            return mcp_access.verify(token)
        finally:
            close_old_connections()


def build_app():
    return BearerAuth(build_server().streamable_http_app())


class _Server(uvicorn.Server):
    """uvicorn without its own signal handling (several run in one process; `serve` handles signals)."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


def _hosts():
    close_old_connections()
    try:
        return mcp_access.listen_hosts()
    finally:
        close_old_connections()


async def serve(poll_seconds: float = 3.0, out=print) -> None:
    """Run until SIGTERM, keeping one listener per address the exposure setting allows."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    running: dict[str, tuple[_Server, asyncio.Task]] = {}

    async def close(host: str):
        server, task = running.pop(host)
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=10)
        out(f"mcp: stopped listening on {host}:{settings.MCP_PORT}")

    while not stop.is_set():
        try:
            wanted = await sync_to_async(_hosts, thread_sensitive=False)()
        except Exception as exc:        # database busy: keep the current listeners
            logger.warning("mcp: could not read the exposure setting: %s", exc)
            wanted = list(running)
        for host in [h for h in running if h not in wanted or running[h][1].done()]:
            await close(host)
        for host in [h for h in wanted if h not in running]:
            config = uvicorn.Config(build_app(), host=host, port=settings.MCP_PORT, log_level="warning", access_log=False)
            server = _Server(config)
            running[host] = (server, asyncio.create_task(server.serve()))
            out(f"mcp: listening on {host}:{settings.MCP_PORT}")     # a failed bind ends the task; retried next round
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_seconds)

    for host in list(running):
        await close(host)
