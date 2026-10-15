"""Home's API (P15a).

    GET /api/monitor/assets/                          the assets Home can show
    GET /api/monitor/overview/?asset_id=              the state now, the last 8 hours, the OEE trend
    GET /api/monitor/posts/?asset_id=&before_id=&limit=   the agent's posts, newest first
    GET /api/monitor/settings/   PUT {agent_enabled}  the agent's switch
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from context_service.service import ContextError

from . import agent
from .models import MonitorSettings


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
