"""Context Graph API (`/api/graph/`).

Read endpoints (P1). Views stay thin: they parse query parameters, call
context_graph.services (the same functions MCP tools will use later) and map
errors to HTTP statuses.
"""

from functools import wraps

from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import driver, registry, services
from .repository import NodeNotFound
from .schema import SchemaError
from .services import SchemaConflict

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
        except SchemaConflict as exc:
            return Response({"error": str(exc), "conflicts": exc.conflicts}, status=status.HTTP_409_CONFLICT)
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


def _definition_from(request) -> dict:
    """A schema definition from a request body: {"definition": {...}} or {"text": "...", "format": "yaml"|"json"}."""
    if isinstance(request.data.get("definition"), dict):
        return request.data["definition"]
    text = request.data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("send a 'definition' object or schema 'text' (YAML/JSON)")
    return services.parse_schema_text(text, request.data.get("format", "yaml"))


@api_view(["GET", "POST"])
@graph_errors
def graph_schema(request):
    """GET  /api/graph/schema/ — active schema (types, allowed pairs) with counts.
    POST /api/graph/schema/ — save a new active version: {text, format, note} or {definition, note}.
    """
    if request.method == "POST":
        result = services.save_schema(_definition_from(request), note=str(request.data.get("note", ""))[:255])
        return Response({**result, "schema": services.schema_summary()}, status=status.HTTP_201_CREATED)
    return Response(services.schema_summary())


@api_view(["POST"])
@graph_errors
def schema_validate(request):
    """POST /api/graph/schema/validate/ — validate + diff against the active schema; writes nothing."""
    try:
        definition = _definition_from(request)
    except SchemaError as exc:
        return Response({"valid": False, "errors": [str(exc)], "diff": None, "conflicts": []})
    return Response(services.check_schema(definition))


@api_view(["GET"])
@graph_errors
def schema_versions(request):
    """GET /api/graph/schema/versions/"""
    return Response({"versions": registry.versions()})


@api_view(["POST"])
@graph_errors
def schema_activate(request, version):
    """POST /api/graph/schema/versions/<n>/activate/ — make an earlier version active."""
    result = services.activate_schema(version)
    return Response({**result, "schema": services.schema_summary()})


@api_view(["GET"])
@graph_errors
def schema_export(request):
    """GET /api/graph/schema/export/?fmt=yaml|json&version=<n> — download a schema definition.

    (`fmt`, not `format`: DRF reserves ?format= for its own content negotiation.)
    """
    fmt = request.query_params.get("fmt", "yaml")
    version = request.query_params.get("version")
    text = services.export_schema(int(version) if version else None, fmt)
    content_type = "application/json" if fmt == "json" else "application/yaml"
    response = HttpResponse(text, content_type=f"{content_type}; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="graph-schema{"-v" + version if version else ""}.{fmt}"'
    return response


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
    """GET /api/graph/data/?root=<id>&depth=2&types=Component,Signal&limit=300&layer=curated|lab|both&doc=<doc_key>

    Nodes + links for the visualizer. The lab layer holds free-form triples (filter by document with `doc`).
    """
    types = [t for t in request.query_params.get("types", "").split(",") if t]
    return Response(services.graph_data(
        request.query_params.get("root") or None,
        depth=max(1, _int(request, "depth", 2, maximum=4)),
        labels=types or None,
        limit=max(1, _int(request, "limit", 300)),
        layer=request.query_params.get("layer") or "curated",
        doc_key=request.query_params.get("doc") or None,
    ))
