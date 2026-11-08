"""The twin API (P32).

    GET  /api/twin/?asset_id=            the manifest (generated from the graph on first use)
    POST /api/twin/generate/  {asset_id} generate it again from the graph
    PUT  /api/twin/           {asset_id, enabled, updated_by}   show or hide the twin everywhere (no deploy)
    GET  /api/twin/live/?asset_id=       the machine now (OPC UA, else the newest historian row)
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import service


def _bad(exc):
    return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


def _asset(request):
    a = request.query_params.get("asset_id") or (request.data.get("asset_id") if isinstance(request.data, dict) else "")
    if not a:
        raise service.TwinError("asset_id is required")
    return a


@api_view(["GET", "PUT"])
def manifest(request):
    try:
        if request.method == "PUT":
            d = request.data if isinstance(request.data, dict) else {}
            return Response(service.set_enabled(_asset(request), bool(d.get("enabled")), str(d.get("updated_by") or "")))
        return Response(service.manifest(_asset(request)))
    except service.TwinError as exc:
        return _bad(exc)


@api_view(["POST"])
def generate(request):
    try:
        d = request.data if isinstance(request.data, dict) else {}
        return Response(service.manifest(_asset(request), regenerate=True, by=str(d.get("updated_by") or "")))
    except service.TwinError as exc:
        return _bad(exc)


@api_view(["GET"])
def live(request):
    try:
        return Response(service.live(_asset(request)))
    except service.TwinError as exc:
        return _bad(exc)
