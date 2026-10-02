"""Context Graph API (`/api/graph/`).

P0: health only. Entity/relationship endpoints arrive in later phases; they
call context_graph services so the same operations can later be exposed as
MCP tools.
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import driver


@api_view(["GET"])
def graph_health(request):
    """GET /api/graph/health/ — 200 when online, 503 otherwise (body says why)."""
    result = driver.health()
    code = status.HTTP_200_OK if result["status"] == "online" else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(result, status=code)
