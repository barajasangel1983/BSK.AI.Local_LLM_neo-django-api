"""Home's API (P15a).

    GET /api/monitor/assets/                          the assets Home can show
    GET /api/monitor/overview/?asset_id=              the state now, the last 8 hours, the OEE trend
    GET /api/monitor/posts/?asset_id=&before_id=&limit=   the agent's posts, newest first
    GET /api/monitor/settings/   PUT {agent_enabled}  the agent's switch
    GET /api/monitor/calendar/?plant_id=   PUT {plant_id, time_zone, working_days, shifts, updated_by}
                                 the plant calendar (P18a0)
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from context_service.service import ContextError

from datetime import datetime, timezone

from . import agent, plant_calendar
from .models import MonitorSettings, PlantCalendar


def _asset(request):
    asset_id = request.query_params.get("asset_id") or ""
    if not asset_id:
        found = agent.assets()
        asset_id = found[0]["id"] if found else ""
    return asset_id


@api_view(["GET"])
def assets(request):
    try:
        return Response({"assets": agent.assets()})
    except Exception as exc:        # the graph is down
        return Response({"assets": [], "error": f"the assets could not be read: {exc}"[:300]},
                        status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["GET"])
def overview(request):
    asset_id = _asset(request)
    if not asset_id:
        return Response({"error": "no asset in the graph"}, status=status.HTTP_404_NOT_FOUND)
    try:
        return Response(agent.overview(asset_id))
    except ContextError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def posts(request):
    asset_id = _asset(request)
    try:
        before = int(request.query_params["before_id"]) if request.query_params.get("before_id") else None
        limit = int(request.query_params.get("limit") or 30)
    except ValueError:
        return Response({"error": "before_id and limit must be numbers"}, status=status.HTTP_400_BAD_REQUEST)
    return Response(agent.posts(asset_id, before, limit))


@api_view(["GET", "PUT"])
def settings_view(request):
    current = MonitorSettings.get()
    if request.method == "PUT":
        value = request.data.get("agent_enabled")
        if not isinstance(value, bool):
            return Response({"error": "agent_enabled must be true or false"}, status=status.HTTP_400_BAD_REQUEST)
        current.agent_enabled = value
        current.save(update_fields=["agent_enabled"])
    return Response({"agent_enabled": current.agent_enabled})


def _calendar_body(row: PlantCalendar | None) -> dict | None:
    if row is None:
        return None
    return {"time_zone": row.time_zone, "working_days": row.working_days, "shifts": row.shifts,
            "updated_by": row.updated_by, "updated_at": row.updated_at.isoformat()}


def _now(row: PlantCalendar | None) -> dict | None:
    """What the calendar says about the moment: shown on the card so a wrong time zone is easy to spot."""
    if row is None:
        return None
    cal = plant_calendar.Calendar.from_model(row)
    now = datetime.now(timezone.utc)
    shift = cal.shift_at(now)
    return {"local": cal.local(now).isoformat(timespec="minutes"), "shift": shift.as_dict() if shift else None,
            "working": cal.is_working(now)}


@api_view(["GET", "PUT"])
def calendar_view(request):
    try:
        plants = plant_calendar.plants()
    except Exception:        # the graph is down: the stored calendars can still be read and edited
        plants = [{"id": pid, "name": ""} for pid in PlantCalendar.objects.values_list("plant_id", flat=True)]
    if request.method == "PUT":
        plant_id = str(request.data.get("plant_id") or "")
        if not plant_id.startswith("bsk:plant:") or (plants and plant_id not in {p["id"] for p in plants}):
            return Response({"error": f"unknown plant {plant_id!r}", "errors": {"plant_id": "choose a plant"}},
                            status=status.HTTP_400_BAD_REQUEST)
        clean, errors = plant_calendar.validate(request.data)
        if errors:
            return Response({"error": "the calendar has errors", "errors": errors},
                            status=status.HTTP_400_BAD_REQUEST)
        PlantCalendar.objects.update_or_create(
            plant_id=plant_id, defaults={**clean, "updated_by": str(request.data.get("updated_by") or "")[:200]})
    else:
        plant_id = request.query_params.get("plant_id") or (plants[0]["id"] if plants else "")
    row = PlantCalendar.objects.filter(plant_id=plant_id).first() if plant_id else None
    return Response({"plants": plants, "plant_id": plant_id, "calendar": _calendar_body(row), "now": _now(row)})
