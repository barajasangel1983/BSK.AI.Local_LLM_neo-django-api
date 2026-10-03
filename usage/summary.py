"""Aggregates for GET /api/usage/summary/ (computed in Python: ranges are ≤ 90 days)."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from django.utils import timezone

from .models import ModelCall

RANGES = (7, 14, 30, 90)


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, -(-95 * len(ordered) // 100) - 1))]


def _agg(calls: list[ModelCall]) -> dict:
    latencies = [c.latency_ms for c in calls if c.status == "ok"]
    errors = sum(1 for c in calls if c.status != "ok")
    return {
        "requests": len(calls),
        "errors": errors,
        "error_rate": round(errors / len(calls), 4) if calls else 0.0,
        "prompt_tokens": sum(c.prompt_tokens or 0 for c in calls),
        "completion_tokens": sum(c.completion_tokens or 0 for c in calls),
        "estimated_calls": sum(1 for c in calls if c.tokens_estimated),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_ms": _p95(latencies),
    }


def summary(days: int, purpose: str | None = None) -> dict:
    now = timezone.now()
    since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    qs = ModelCall.objects.filter(created_at__gte=since)
    if purpose:
        qs = qs.filter(purpose=purpose)
    calls = list(qs.order_by("created_at"))
    first = ModelCall.objects.order_by("created_at").values_list("created_at", flat=True).first()

    by_model: dict[str, list] = defaultdict(list)
    by_purpose: dict[str, list] = defaultdict(list)
    by_day: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in calls:
        by_model[c.model_id].append(c)
        by_purpose[c.purpose].append(c)
        by_day[timezone.localdate(c.created_at).isoformat()][c.model_id].append(c)

    daily = []
    for i in range(days):
        day = (timezone.localdate(since) + timedelta(days=i)).isoformat()
        models = by_day.get(day, {})
        daily.append({
            "date": day,
            "models": {m: {"requests": len(cs), "prompt_tokens": sum(c.prompt_tokens or 0 for c in cs),
                           "completion_tokens": sum(c.completion_tokens or 0 for c in cs)} for m, cs in models.items()},
        })
    return {
        "days": days,
        "purpose": purpose,
        "measured_since": first.isoformat() if first else None,
        "totals": _agg(calls),
        "per_model": [{"model_id": m, **_agg(cs)} for m, cs in sorted(by_model.items(), key=lambda kv: -len(kv[1]))],
        "per_purpose": [{"purpose": p, **_agg(cs)} for p, cs in sorted(by_purpose.items(), key=lambda kv: -len(kv[1]))],
        "daily": daily,
        "purposes": sorted(ModelCall.objects.values_list("purpose", flat=True).distinct()),
    }
