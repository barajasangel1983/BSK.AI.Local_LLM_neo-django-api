"""Context Graph API (`/api/graph/`).

Read endpoints (P1). Views stay thin: they parse query parameters, call
context_graph.services (the same functions MCP tools will use later) and map
errors to HTTP statuses.
"""

from functools import wraps

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import driver, services
from .repository import NodeNotFound
from .schema import SchemaError

MAX_LIMIT = 1000


def graph_errors(view):
    """Map service exceptions to HTTP responses."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except driver.GraphUnavailable as exc:
            return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except NodeNotFound as exc:
            return Response({"error": f"Unknown node: {exc}"}, status=status.HTTP_404_NOT_FOUND)
        except (SchemaError, ValueError) as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    return wrapper


def _int(request, name: str, default: int, maximum: int = MAX_LIMIT) -> int:
    try:
        value = int(request.query_params.get(name, default))
    except ValueError:
        raise ValueError(f"{name} must be an integer")
    return max(0, min(value, maximum))


@api_view(["GET"])
def graph_health(request):
    """GET /api/graph/health/ — 200 when online, 503 otherwise (body says why)."""
    result = driver.health()
    code = status.HTTP_200_OK if result["status"] == "online" else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(result, status=code)


@api_view(["GET"])
@graph_errors
def graph_schema(request):
    """GET /api/graph/schema/ — active schema (types, allowed pairs) with counts."""
    return Response(services.schema_summary())


@api_view(["GET"])
@graph_errors
def asset_list(request):
    """GET /api/graph/assets/"""
    return Response({"assets": services.list_assets()})


@api_view(["GET"])
@graph_errors
def asset_context(request, asset_id):
    """GET /api/graph/assets/<id>/context/ — hierarchy, components, signals, alarms, procedures, documents."""
    return Response(services.get_asset_context(asset_id))


@api_view(["GET"])
@graph_errors
def node_list(request):
    """GET /api/graph/nodes/?type=Signal&q=temp&limit=50&offset=0"""
    return Response(services.search_nodes(
        request.query_params.get("type") or None,
        request.query_params.get("q") or None,
        limit=max(1, _int(request, "limit", 50)),
        offset=_int(request, "offset", 0, maximum=10**6),
    ))


@api_view(["GET"])
@graph_errors
def node_detail(request, node_id):
    """GET /api/graph/nodes/<id>/ — node with its relationships."""
    return Response(services.node_detail(node_id))


@api_view(["GET"])
@graph_errors
def alarm_procedures(request, alarm_id):
    """GET /api/graph/alarms/<id>/procedures/"""
    return Response(services.alarm_procedures(alarm_id))


@api_view(["GET"])
@graph_errors
def graph_data(request):
    """GET /api/graph/data/?root=<id>&depth=2&types=Component,Signal&limit=300 — nodes + links for the visualizer."""
    types = [t for t in request.query_params.get("types", "").split(",") if t]
    return Response(services.graph_data(
        request.query_params.get("root") or None,
        depth=max(1, _int(request, "depth", 2, maximum=4)),
        labels=types or None,
        limit=max(1, _int(request, "limit", 300)),
    ))
