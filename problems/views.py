"""The Studio's problem-solving API (P29). The Studio reads tickets and holds the settings; tickets are filled and
approved on EDGE (through the MCP tools in context_service.mcp_server).

    GET  /api/problems/tickets/?asset_id=&status=&before_id=&limit=     newest first; status "active" or one status
    GET  /api/problems/tickets/<id>/                                     one ticket, with its context and 5-Why
    GET  /api/problems/tickets/<id>/pictures/<picture id>/               a picture (JPEG)
    GET  /api/problems/tickets/<id>/report/                              the printable report (HTML, pictures inside)
    POST /api/problems/tickets/<id>/check/                               run the knowledge check now (P29c)
    POST /api/problems/tickets/<id>/recommend/                           what to check first, again (P29b)
    POST /api/problems/tickets/<id>/context/                             read the window again
    GET  /api/problems/settings/   PUT {enabled, downtime_hours, downtime_states, spc_control_hours, spc_spec_hours,
                                        check_mode, check_day, check_time, publish_folder, updated_by}
"""

from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import llm, service
from .service import TicketError


def _bad(exc: Exception, code=status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": str(exc)}, status=code)


@api_view(["GET"])
def tickets(request):
    q = request.query_params
    try:
        return Response(service.list_tickets(q.get("asset_id", ""), q.get("status", ""), int(q.get("limit") or 50),
                                             int(q.get("before_id") or 0)))
    except ValueError as exc:
        return _bad(exc)


@api_view(["GET"])
def ticket(request, ticket_id: int):
    try:
        return Response(service.ticket_json(service.get(ticket_id)))
    except TicketError as exc:
        return _bad(exc, status.HTTP_404_NOT_FOUND)


@api_view(["GET"])
def picture(request, ticket_id: int, picture_id: int):
    try:
        return HttpResponse(service.picture_bytes(ticket_id, picture_id), content_type="image/jpeg")
    except (TicketError, OSError) as exc:
        return _bad(exc, status.HTTP_404_NOT_FOUND)


@api_view(["GET"])
def report(request, ticket_id: int):
    from . import report as template
    try:
        return HttpResponse(template.html(service.get(ticket_id)), content_type="text/html; charset=utf-8")
    except TicketError as exc:
        return _bad(exc, status.HTTP_404_NOT_FOUND)


@api_view(["POST"])
def run_check(request, ticket_id: int):
    try:
        return Response(service.ticket_json(llm.request_check(service.get(ticket_id))))
    except TicketError as exc:
        return _bad(exc)


@api_view(["POST"])
def recommend(request, ticket_id: int):
    try:
        return Response(service.ticket_json(llm.recommend(service.get(ticket_id))))
    except TicketError as exc:
        return _bad(exc)
    except Exception as exc:          # the model is down
        return _bad(f"the recommendations could not be made: {exc}", status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["POST"])
def refresh(request, ticket_id: int):
    try:
        return Response(service.ticket_json(service.refresh_context(service.get(ticket_id))))
    except TicketError as exc:
        return _bad(exc)


@api_view(["GET", "PUT"])
def settings_view(request):
    if request.method == "PUT":
        try:
            return Response(service.update_settings(request.data, str(request.data.get("updated_by") or "")))
        except TicketError as exc:
            return _bad(exc)
    return Response(service.settings_json())
