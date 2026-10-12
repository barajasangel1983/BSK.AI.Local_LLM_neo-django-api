"""Read-only view of the historian (the Studio's Health page → Historian Postgres).

    GET /api/historian/summary/   what the historian holds: table, rows, time range, assets, machine states
    GET /api/historian/samples/?page=&page_size=&start=&end=&state=   the rows, oldest first (start <= ts <= end)

Both take ?source=march|live (default: the source the Studio serves, settings.HISTORIAN_SOURCE).
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal

from django.db import Error as DatabaseError
from django.db import connections
from django.db.models import Count, Max, Min
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import source as historian_source

logger = logging.getLogger(__name__)

PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

_TYPES = {"DateTimeField": "time", "DecimalField": "number", "FloatField": "number", "BooleanField": "boolean"}


def _columns(model) -> list[dict]:
    return [{"key": f.name, "type": _TYPES.get(f.get_internal_type(), "text")}
            for f in model._meta.fields if f.name not in ("id", "pk")]


def _source(request) -> str:
    asked = request.query_params.get("source") or historian_source.current()
    if asked not in historian_source.SOURCES:
        raise ValueError(f"source must be one of: {', '.join(historian_source.SOURCES)}")
    return asked


def _when(value: str, name: str):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name} must be an ISO date and time, e.g. 2026-03-05T14:00:00Z")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _unavailable(exc) -> Response:
    logger.warning("historian read failed: %s", exc)
    return Response({"error": "the historian is not available"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["GET"])
def summary(request):
    try:
        source = _source(request)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    model = historian_source.samples(source)
    rows = model.objects.order_by()
    try:
        totals = rows.aggregate(rows=Count("ts"), first_ts=Min("ts"), last_ts=Max("ts"))
        assets = sorted(rows.values_list("extruder_id", flat=True).distinct())
        states = list(rows.values("machine_state").annotate(rows=Count("ts")).order_by("-rows", "machine_state"))
    except DatabaseError as exc:
        return _unavailable(exc)
    return Response({
        "source": source,
        "serving": historian_source.current(),
        "database": connections[rows.db].settings_dict["NAME"],
        "table": model._meta.db_table,
        "rows": totals["rows"],
        "first_ts": totals["first_ts"],
        "last_ts": totals["last_ts"],
        "assets": assets,
        "states": [{"state": s["machine_state"], "rows": s["rows"]} for s in states],
        "columns": _columns(model),
    })


@api_view(["GET"])
def samples(request):
    q = request.query_params
    try:
        page = max(1, int(q.get("page") or 1))
        page_size = min(MAX_PAGE_SIZE, max(1, int(q.get("page_size") or PAGE_SIZE)))
    except ValueError:
        return Response({"error": "page and page_size must be numbers"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        source = _source(request)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    model = historian_source.samples(source)
    rows = model.objects.order_by("ts", "extruder_id")
    try:
        if q.get("start"):
            rows = rows.filter(ts__gte=_when(q["start"], "start"))
        if q.get("end"):
            rows = rows.filter(ts__lte=_when(q["end"], "end"))
    except ValueError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    if q.get("state"):
        rows = rows.filter(machine_state=q["state"])

    columns = _columns(model)
    try:
        count = rows.count()
        found = list(rows.values(*[c["key"] for c in columns])[(page - 1) * page_size:page * page_size])
    except DatabaseError as exc:
        return _unavailable(exc)
    return Response({
        "source": source,
        "count": count,
        "page": page,
        "page_size": page_size,
        "pages": max(1, -(-count // page_size)),
        "columns": columns,
        "rows": [{k: float(v) if isinstance(v, Decimal) else v for k, v in row.items()} for row in found],
    })
