"""REST for the Context Service (the MCP server exposes the same functions as tools).

    GET  /api/context/assets/
    POST /api/context/resolve/            {text, asset_id?}
    GET  /api/context/entities/<id>/      (ids contain ':' and '/')
    GET  /api/context/entities/<id>/sources/
    GET  /api/context/state/<asset id>/
    POST /api/context/search/             {query, asset_id?, top_k?}
    POST /api/context/assemble/           {query, asset_id?, include_documents?, budget_chars?}  → Context Packet
    POST /api/context/ask/                {query, asset_id?, model?, include_documents?}        → {answer, model, packet}
"""

import requests
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from context_graph.driver import GraphUnavailable

from . import service


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
