"""The PLC page (P13f).

    GET  /api/plc/status/          values, LEDs, facts, conditions, verdict; what the controls allow
    POST /api/plc/command/         {"command": "Start" | "Pause", "confirm": <the same, typed>, "by": <name>}
    GET  /api/plc/commands/?limit= the record of commands, newest first (refusals included)
    GET  /api/plc/opcua/           every variable of the OPC UA server with its value and type (as the 8891 dashboard)

Only Start and Pause are ever sent to the PLC. Off until PLC_CONTROL_ENABLED=true in the backend's .env.
"""

from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import service
from .models import PlcCommand


@api_view(["GET"])
def plc_status(request):
    return Response(service.current())


@api_view(["POST"])
def plc_command(request):
    body = request.data if isinstance(request.data, dict) else {}
    try:
        record = service.command(body.get("command"), body.get("confirm"), body.get("by") or "")
    except service.Refused as exc:
        return Response({"error": str(exc), "status": service.current(fresh=True) if exc.code == 409 else None}, status=exc.code)
    code = 200 if record.outcome == "done" else 502
    return Response({"command": service.record_json(record), "status": service.current(fresh=True)}, status=code)


@api_view(["GET"])
def plc_opcua(request):
    return Response(service.opcua())


@api_view(["GET"])
def plc_commands(request):
    raw = request.query_params.get("limit") or "20"
    limit = int(raw) if raw.isdigit() else 20
    return Response({"commands": [service.record_json(c) for c in PlcCommand.objects.all()[:max(1, min(limit, 200))]]})
