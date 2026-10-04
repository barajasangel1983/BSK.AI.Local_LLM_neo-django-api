"""REST for the Context Service (the MCP server exposes the same functions as tools).

    GET  /api/context/assets/
    POST /api/context/resolve/            {text, asset_id?}
    GET  /api/context/entities/<id>/      (ids contain ':' and '/')
    GET  /api/context/entities/<id>/sources/
    GET  /api/context/state/<asset id>/
    POST /api/context/search/             {query, asset_id?, top_k?}
    POST /api/context/assemble/           {query, asset_id?, include_documents?, budget_chars?}  → Context Packet
    POST /api/context/ask/                {query, asset_id?, model?, include_documents?}        → {answer, model, packet}

MCP server settings (the Studio's Settings → MCP):

    GET  /api/context/mcp/                exposure, addresses, whether it answers, tools, tokens
    PUT  /api/context/mcp/                {exposure: off | local | tailscale}
    POST /api/context/mcp/tokens/         {name} → the token, shown once
    DELETE /api/context/mcp/tokens/<id>/  revoke
"""

import requests
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from context_graph.driver import GraphUnavailable

from django.conf import settings

from . import mcp_access, service
from .models import McpSettings, McpToken


def _call(fn, *args, **kwargs):
    try:
        return Response(fn(*args, **kwargs))
    except service.ContextError as exc:
        text = str(exc)
        code = status.HTTP_404_NOT_FOUND if text.startswith("unknown") else status.HTTP_400_BAD_REQUEST
        return Response({"error": text}, status=code)
    except GraphUnavailable as exc:
        return Response({"error": f"Context Graph unavailable: {exc}"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except (TypeError, ValueError) as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


def _flag(value, default=True) -> bool:
    return default if value is None else str(value).lower() not in ("false", "0", "no")


@api_view(["GET"])
def assets(request):
    return _call(lambda: {"assets": service.list_assets()})


@api_view(["POST"])
def resolve(request):
    return _call(service.resolve, str(request.data.get("text") or ""), request.data.get("asset_id") or None)


@api_view(["GET"])
def entity(request, entity_id):
    return _call(service.entity_context, entity_id)


@api_view(["GET"])
def entity_sources(request, entity_id):
    return _call(service.entity_sources, entity_id)


@api_view(["GET"])
def state(request, asset_id):
    return _call(service.operational_state, asset_id)


@api_view(["POST"])
def search(request):
    if not str(request.data.get("query") or "").strip():
        return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)
    return _call(service.search_documents, str(request.data["query"]), request.data.get("asset_id") or None,
                 request.data.get("top_k"))


@api_view(["POST"])
def assemble(request):
    return _call(service.assemble, str(request.data.get("query") or ""), request.data.get("asset_id") or None,
                 _flag(request.data.get("include_documents")), request.data.get("budget_chars"))


@api_view(["POST"])
def ask(request):
    try:
        return _call(service.ask, str(request.data.get("query") or ""), request.data.get("asset_id") or None,
                     request.data.get("model") or None, _flag(request.data.get("include_documents")))
    except requests.RequestException as exc:
        return Response({"error": f"Model backend failed: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)


# --- MCP server settings ------------------------------------------------------------------

MCP_TOOLS = ["list_assets", "resolve_entity", "get_entity_context", "get_sources", "search_documents",
             "assemble_context", "ask", "get_operational_state",
             "list_documents", "get_document", "get_document_page", "get_figure_image", "get_graph",
             "describe_image",
             "get_system_health"]


def mcp_answers(host: str = "127.0.0.1") -> bool:
    try:
        return requests.get(f"http://{host}:{settings.MCP_PORT}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _mcp_json() -> dict:
    current = McpSettings.get()
    hosts = mcp_access.listen_hosts(current.exposure)
    public = settings.MCP_TAILSCALE_HOST if current.exposure == McpSettings.Exposure.TAILSCALE else "127.0.0.1"
    return {
        "exposure": current.exposure,
        "options": mcp_access.EXPOSURE_OPTIONS,
        "port": settings.MCP_PORT,
        "addresses": [f"http://{h}:{settings.MCP_PORT}/mcp" for h in hosts],
        "url": f"http://{public}:{settings.MCP_PORT}/mcp" if hosts else None,     # the one to give a client
        # The server process follows the setting within a few seconds.
        "answering": bool(hosts) and mcp_answers(),
        "tools": MCP_TOOLS,
        "tokens": [mcp_access.token_json(t) for t in McpToken.objects.all()],
    }


@api_view(["GET", "PUT"])
def mcp_settings(request):
    if request.method == "PUT":
        exposure = request.data.get("exposure")
        option = next((o for o in mcp_access.EXPOSURE_OPTIONS if o["key"] == exposure), None)
        if option is None:
            return Response({"error": "exposure must be one of off, local, tailscale"}, status=status.HTTP_400_BAD_REQUEST)
        if not option["available"]:
            return Response({"error": option["note"]}, status=status.HTTP_400_BAD_REQUEST)
        current = McpSettings.get()
        current.exposure = exposure
        current.save()
    return Response(_mcp_json())


@api_view(["POST"])
def mcp_tokens(request):
    try:
        record, token = mcp_access.create_token(request.data.get("name"))
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    # The only time the token itself is returned.
    return Response({**mcp_access.token_json(record), "token": token}, status=status.HTTP_201_CREATED)


@api_view(["DELETE"])
def mcp_token(request, token_id):
    if not mcp_access.revoke(token_id):
        return Response({"error": "token not found or already revoked"}, status=status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)
