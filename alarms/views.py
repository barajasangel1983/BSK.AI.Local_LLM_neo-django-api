"""The alarms API (P31).

    GET  /api/alarms/?asset_id=&hours=24&code=&category=&active=1&limit=200   episodes, active now, per code
    GET  /api/alarms/codes/?asset_id=                                        every code with its category
    PUT  /api/alarms/codes/  {asset_id, code, category}                      set a code's category
    GET  /api/alarms/codes/<code>/?asset_id=&days=30                         one code: episodes, knowledge, tickets
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import service

DEFAULT_ASSET = "bsk:asset:EXTR01"


def _bad(exc):
    return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@api_view(["GET"])
def summary(request):
    q = request.query_params
    try:
        return Response(service.summary(q.get("asset_id") or DEFAULT_ASSET, _int(q.get("hours"), 24), q.get("code") or "",
                                        q.get("category") or "", q.get("active") in ("1", "true"), _int(q.get("limit"), 200)))
    except service.AlarmError as exc:
        return _bad(exc)


@api_view(["GET", "PUT"])
def codes(request):
    if request.method == "PUT":
        d = request.data if isinstance(request.data, dict) else {}
        try:
            return Response(service.set_category(d.get("asset_id") or DEFAULT_ASSET, str(d.get("code") or ""),
                                                 str(d.get("category") or ""), str(d.get("updated_by") or "")))
        except service.AlarmError as exc:
            return _bad(exc)
    return Response({"codes": service.codes(request.query_params.get("asset_id") or DEFAULT_ASSET),
                     "categories": service.CATEGORIES})


@api_view(["GET"])
def code(request, code):
    q = request.query_params
    try:
        return Response(service.code_detail(q.get("asset_id") or DEFAULT_ASSET, code, max(1, min(_int(q.get("days"), 30), 90))))
    except service.AlarmError as exc:
        return _bad(exc)
