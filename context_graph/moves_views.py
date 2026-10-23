"""P26c "Move to families" API (`/api/graph/moves/`): plans, apply and undo of moving installed parts' manual facts to
their product family or device model (context_graph/fact_moves.py).

    GET  moves/                    every family (or model without one): its parts, models and plan summary; batches
    POST moves/plan/               {target, choices?, stays?} -> the plan (nothing written)
    POST moves/apply/              {target, choices?, stays?, author?} -> the batch
    POST moves/undo/               {batch} -> the batch, undone
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import fact_moves
from .driver import session
from .models import FactMoveBatch
from .views import graph_errors


def _error(message: str, code=status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


def _graph():
    with session() as s:
        return s.execute_read(fact_moves.load_graph)


@api_view(["GET"])
@graph_errors
def moves(request):
    g = _graph()
    rows = []
    for t in fact_moves.targets(g):
        p = fact_moves.plan(g, t["id"])
        rows.append({**t, "summary": p["summary"], "counts": p["counts"],
                     "names": {i: g.props(i).get("name", i) for i in [*t["models"], *t["parts"]]}})
    return Response({"targets": rows, "batches": [fact_moves.batch_json(b) for b in FactMoveBatch.objects.all()[:50]]})


def _args(request):
    target = request.data.get("target") or ""
    choices = request.data.get("choices") or {}
    stays = request.data.get("stays") or []
    if not target:
        return None, _error("target is required")
    if not isinstance(choices, dict) or not isinstance(stays, list):
        return None, _error("choices must be an object and stays a list")
    return (target, choices, stays), None


@api_view(["POST"])
@graph_errors
def moves_plan(request):
    args, err = _args(request)
    if err:
        return err
    try:
        return Response(fact_moves.plan(_graph(), *args))
    except fact_moves.MoveError as exc:
        return _error(str(exc))


@api_view(["POST"])
@graph_errors
def moves_apply(request):
    args, err = _args(request)
    if err:
        return err
    try:
        batch = fact_moves.apply(*args, author=str(request.data.get("author") or "")[:150])
    except fact_moves.MoveError as exc:
        return _error(str(exc), status.HTTP_409_CONFLICT)
    return Response(fact_moves.batch_json(batch), status=status.HTTP_201_CREATED)


@api_view(["POST"])
@graph_errors
def moves_undo(request):
    batch = FactMoveBatch.objects.filter(pk=request.data.get("batch") or 0).first()
    if batch is None:
        return _error("unknown batch", status.HTTP_404_NOT_FOUND)
    try:
        fact_moves.undo(batch)
    except fact_moves.MoveError as exc:
        return _error(str(exc), status.HTTP_409_CONFLICT)
    batch.refresh_from_db()
    return Response(fact_moves.batch_json(batch))
