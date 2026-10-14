"""SPC for the Studio's SPC page (Play Mode).

    GET    /api/spc/signals/?asset_id=            the signals a tool can be made for, with proposed values
    GET    /api/spc/tools/?asset_id=              the tools of a machine
    POST   /api/spc/tools/                        add a tool
    PUT    /api/spc/tools/order/              the order of a machine's cards: {"asset_id", "ids": [...]}
    GET | PATCH | DELETE /api/spc/tools/<id>/     one tool
    POST   /api/spc/tools/<id>/baseline/          control limits from a window of rows (preview or save)
    POST   /api/spc/baseline/preview/             the same for a block's values, before the tool is saved; stores nothing
    PUT    /api/spc/tools/<id>/limits/            control limits typed by the user
    GET    /api/spc/screening/?asset_id=&state=&tools=&window_hours=   summary and one object per tool
    GET    /api/spc/summary/?…                    the summary alone
    GET    /api/spc/tools/<id>/detail/?window_hours=   the chart, distribution and capability of one tool
    GET    /api/spc/events/?asset_id=&tool=&active=&limit=&before=    SPC events, newest first
    GET    /api/spc/tools/<id>/llm/               the LLM packet of the tool's window, the default prompt, the models
    POST   /api/spc/tools/<id>/llm/analysis/      {model, prompt, packet}: one call to the model, with the checks; nothing stored
    POST   /api/spc/events/<id>/acknowledge/      mark an event as seen

SPC events are written by the evaluator (manage.py spc_evaluate, once a minute), not by these views.

Every figure is computed by spc/engine.py from the historian's rows. Nothing here asks a model.
"""

import logging
from datetime import datetime, timezone

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import Error as DatabaseError
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from chat.asset_context import AssetScopeError
from context_graph.driver import GraphUnavailable

from . import llm, service
from .models import SpcEvent, SpcTool

logger = logging.getLogger(__name__)


def _error(text: str, code: int) -> Response:
    return Response({"error": text}, status=code)


def _guard(view):
    """Turn the service's errors into answers: 400 bad input, 409 conflict, 503 historian or graph down."""
    def wrapped(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except service.Conflict as exc:
            return _error(str(exc), status.HTTP_409_CONFLICT)
        except (service.SpcError, AssetScopeError) as exc:
            return _error(str(exc), status.HTTP_400_BAD_REQUEST)
        except GraphUnavailable:
            return _error("the Context Graph is not available", status.HTTP_503_SERVICE_UNAVAILABLE)
        except DatabaseError as exc:
            logger.warning("spc: database read failed: %s", exc)
            return _error("the historian is not available", status.HTTP_503_SERVICE_UNAVAILABLE)
    wrapped.__name__, wrapped.__doc__ = view.__name__, view.__doc__
    return wrapped


def _tool(tool_id) -> SpcTool | None:
    try:
        return SpcTool.objects.prefetch_related("limits").get(pk=tool_id)
    except (SpcTool.DoesNotExist, ValidationError, ValueError):
        return None


def _when(value, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise service.SpcError(f"{name} must be an ISO date and time, e.g. 2026-10-12T00:00:00Z")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hours(request) -> int | None:
    raw = request.query_params.get("window_hours")
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= 24:
        raise service.SpcError("window_hours must be a whole number from 1 to 24")
    return int(raw)


def _asset(request) -> str:
    asset_id = request.query_params.get("asset_id")
    if not asset_id:
        raise service.SpcError("asset_id is required")
    return asset_id


@api_view(["GET"])
@_guard
def signals(request):
    return Response(service.signals(_asset(request)))


@api_view(["GET", "POST"])
@_guard
def tools(request):
    if request.method == "POST":
        if not isinstance(request.data, dict):
            raise service.SpcError("send the tool as a JSON object")
        return Response(service.tool_json(_tool(service.create_tool(dict(request.data)).pk)), status=status.HTTP_201_CREATED)
    found = SpcTool.objects.filter(asset_id=_asset(request)).prefetch_related("limits")
    return Response({"tools": [service.tool_json(t) for t in found]})


@api_view(["PUT"])
@_guard
def order(request):
    if not isinstance(request.data, dict) or not request.data.get("asset_id"):
        raise service.SpcError("send {\"asset_id\", \"ids\"}")
    return Response({"ids": service.order_tools(request.data["asset_id"], request.data.get("ids"))})


@api_view(["GET", "PATCH", "DELETE"])
@_guard
def tool(request, tool_id):
    found = _tool(tool_id)
    if found is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        found.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    cleared = False
    if request.method == "PATCH":
        if not isinstance(request.data, dict):
            raise service.SpcError("send the changes as a JSON object")
        _, cleared = service.update_tool(found, dict(request.data))
        found = _tool(tool_id)
    return Response({**service.tool_json(found), "limits_cleared": cleared})


@api_view(["POST"])
@_guard
def baseline(request, tool_id):
    found = _tool(tool_id)
    if found is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    start, end = _when(request.data.get("start"), "start"), _when(request.data.get("end"), "end")
    return Response(service.baseline(found, start, end, preview=bool(request.data.get("preview", False))))


@api_view(["POST"])
@_guard
def baseline_preview(request):
    if not isinstance(request.data, dict):
        raise service.SpcError("send the block's values as a JSON object")
    data = dict(request.data)
    start, end = _when(data.pop("start", None), "start"), _when(data.pop("end", None), "end")
    return Response(service.preview_baseline(data, start, end))


@api_view(["PUT"])
@_guard
def limits(request, tool_id):
    found = _tool(tool_id)
    if found is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    return Response(service.set_fixed_limits(found, request.data.get("groups")))


def _screening(request) -> dict:
    raw = request.query_params.get("tools")
    ids = None if raw is None else [t for t in raw.split(",") if t]
    return service.screening(_asset(request), ids, request.query_params.get("state") or None, _hours(request))


@api_view(["GET"])
@_guard
def screening(request):
    return Response(_screening(request))


@api_view(["GET"])
@_guard
def summary(request):
    return Response({k: v for k, v in _screening(request).items() if k != "tools"})


@api_view(["GET"])
@_guard
def detail(request, tool_id):
    found = _tool(tool_id)
    if found is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    return Response(service.detail(found, _hours(request)))


@api_view(["GET"])
@_guard
def events(request):
    q = request.query_params
    limit = q.get("limit") or str(service.EVENT_PAGE)
    if not limit.isdigit():
        raise service.SpcError("limit must be a whole number")
    tool_id = q.get("tool") or None
    if tool_id and _tool(tool_id) is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    return Response(service.events(q.get("asset_id") or None, tool_id, q.get("active") in ("1", "true", "yes"), int(limit),
                                   _when(q["before"], "before") if q.get("before") else None))


@api_view(["POST"])
@_guard
def acknowledge(request, event_id):
    try:
        event = SpcEvent.objects.select_related("tool").get(pk=event_id)
    except (SpcEvent.DoesNotExist, ValidationError, ValueError):
        return _error("SPC event not found", status.HTTP_404_NOT_FOUND)
    by = request.data.get("by") if isinstance(request.data, dict) else ""
    return Response(service.event_json(service.acknowledge(event, str(by or ""))))


PACKET_MAX_CHARS = 300_000


@api_view(["GET"])
@_guard
def llm_packet(request, tool_id):
    found = _tool(tool_id)
    if found is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    data = llm.packet(found)
    text = llm.packet_text(data)
    return Response({"prompt": llm.DEFAULT_PROMPT, "packet": data, "characters": len(text),
                     # the estimate of the input: (characters of the prompt + input_extra_characters) / characters_per_token
                     "approx_tokens": llm.estimate_tokens(llm.messages_for(llm.DEFAULT_PROMPT, data)),
                     "input_extra_characters": len(llm.USER_HEAD) + len(text), "characters_per_token": llm.CHARS_PER_TOKEN,
                     "models": llm.models(), "default_model": settings.SPC_LLM_DEFAULT_MODEL,
                     "limits": {"input": settings.SPC_LLM_INPUT_LIMIT, "output": settings.SPC_LLM_OUTPUT_LIMIT}})


@api_view(["POST"])
@_guard
def llm_analysis(request, tool_id):
    """The prompt and packet the page shows, to one model, once. The answer is not stored (the usage record is)."""
    if _tool(tool_id) is None:
        return _error("SPC tool not found", status.HTTP_404_NOT_FOUND)
    body = request.data if isinstance(request.data, dict) else {}
    data, prompt, model = body.get("packet"), body.get("prompt"), body.get("model") or settings.SPC_LLM_DEFAULT_MODEL
    if not isinstance(data, dict) or not isinstance(data.get("window"), dict):
        raise service.SpcError("packet must be the JSON object of GET llm/")
    if not isinstance(prompt, str):
        raise service.SpcError("prompt must be text")
    if len(llm.packet_text(data)) > PACKET_MAX_CHARS:
        raise service.SpcError(f"the packet is longer than {PACKET_MAX_CHARS:,} characters")
    try:
        result = llm.analyze(data, prompt, model, body.get("input_limit"), body.get("output_limit"))
    except llm.ModelError as exc:
        code = status.HTTP_504_GATEWAY_TIMEOUT if "did not answer" in str(exc) else status.HTTP_502_BAD_GATEWAY
        return _error(str(exc), code)
    result["checks"] = {"numbers": llm.check_numbers(result["answer"], data), "verdict": llm.check_verdict(result["answer"], data)}
    result["prompt_changed"] = prompt.strip() != llm.DEFAULT_PROMPT.strip()
    return Response(result)

