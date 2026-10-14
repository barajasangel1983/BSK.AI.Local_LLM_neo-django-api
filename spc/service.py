"""SPC for the page and the evaluator: reading the historian, baselines, screening and detail.

The numbers come from spc/engine.py; this module only fetches rows and stores results.
Rows are read through chat.asset_context (the same path as the Context Service's signal
history), so SPC serves whatever historian the Studio serves.
Specification: claude/P12_SPCimplementation.md, sections 5 to 6.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import yaml
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from chat import asset_context
from historian import source as historian_source

from . import engine
from .models import SpcCursor, SpcEvent, SpcLimits, SpcTool

logger = logging.getLogger(__name__)

READ_HOURS = 4                  # one read of the historian (720 of the PLC's rows)
BASELINE_MAX_DAYS = 31
NUMERIC_FIELDS = {"FloatField", "DecimalField", "IntegerField", "BigIntegerField", "SmallIntegerField"}
TEXT_FIELDS = {"TextField", "CharField"}
DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"


class SpcError(ValueError):
    """Bad input: shown to the user as it is (HTTP 400)."""


class Conflict(SpcError):
    """The limit of tools, or a second tool for one signal (HTTP 409)."""


def asset_key(asset_id: str) -> str:
    """bsk:asset:EXTR01 → EXTR01 (what the historian calls the machine)."""
    return asset_id.rsplit(":", 1)[-1]


# --- reading the historian ----------------------------------------------------------------------

def _field_types() -> dict[str, str]:
    return {f.name: f.get_internal_type() for f in historian_source.samples()._meta.fields}


def read_rows(asset_id: str, columns: list[str], start, end) -> list[dict]:
    """Rows with start < ts <= end, oldest first, numbers as floats. Read in pieces of READ_HOURS."""
    period = historian_source.period_seconds()
    columns = sorted(set(columns) | {"machine_state"})
    rows: list[dict] = []
    while start < end:
        until = min(start + timedelta(hours=READ_HOURS), end)
        rows += asset_context.historian_rows(asset_key(asset_id), columns, start, until, READ_HOURS * 3600 // period + 10)
        start = until
    for row in rows:
        for key, value in row.items():
            if isinstance(value, Decimal):
                row[key] = float(value)
    return rows


def newest_row(asset_id: str) -> dict | None:
    """{ts, machine_state} of the asset's newest row in the served historian."""
    return (historian_source.samples().objects.filter(extruder_id=asset_key(asset_id))
            .order_by("-ts").values("ts", "machine_state").first())


def recent_states(asset_id: str) -> list[str]:
    """The machine states seen in the 7 days before the newest row."""
    newest = newest_row(asset_id)
    if not newest:
        return []
    return sorted(historian_source.samples().objects
                  .filter(extruder_id=asset_key(asset_id), ts__gt=newest["ts"] - timedelta(days=7))
                  .order_by().values_list("machine_state", flat=True).distinct())


def _samples(tool: SpcTool, rows: list[dict], states: list[str] | None = None) -> list[engine.Sample]:
    return engine.samples(rows, column=tool.column, states=tool.states if states is None else states,
                          ignore_seconds=tool.ignore_seconds, method=tool.method, subgroup_size=tool.subgroup_size,
                          group_by=tool.group_by, period=historian_source.period_seconds())


# --- 6.1 the signals a tool can be made for -----------------------------------------------------

def defaults(asset_id: str) -> dict:
    """The proposed values of an asset: {"common": {...}, "signals": {column: {...}}} (empty without a file)."""
    path = DEFAULTS_DIR / f"{asset_key(asset_id)}.yaml"
    if not path.is_file():
        return {"common": {}, "signals": {}}
    data = yaml.safe_load(path.read_text()) or {}
    return {"common": data.get("common") or {}, "signals": data.get("signals") or {}}


def signals(asset_id: str) -> dict:
    """Every numeric signal of the asset that has a historian column, with what is proposed for it."""
    types = _field_types()
    found = asset_context.signal_columns(asset_id)
    proposed = defaults(asset_id)
    taken = set(SpcTool.objects.filter(asset_id=asset_id).values_list("signal_id", flat=True))
    states = recent_states(asset_id)
    common = proposed["common"]
    out = []
    for s in found:
        if types.get(s["column"]) not in NUMERIC_FIELDS:
            continue
        own = proposed["signals"].get(s["column"])
        out.append({
            "signal_id": s["signal_id"], "column": s["column"], "name": s["name"], "unit": s["unit"],
            "normal_low": s["low"], "normal_high": s["high"], "has_tool": s["signal_id"] in taken,
            "proposed": None if own is None else {
                "lsl": own.get("lsl"), "target": own.get("target"), "usl": own.get("usl"),
                "group_by": own.get("group_by", ""), "method": own.get("method", common.get("method", engine.IMR)),
                "states": own.get("states", common.get("states", ["RUNNING"])),
                "ignore_seconds": own.get("ignore_seconds", common.get("ignore_seconds", 120)),
                "sigma_multiplier": own.get("sigma_multiplier", common.get("sigma_multiplier", 3)),
                "baseline_days": own.get("baseline_days", common.get("baseline_days", 7))},
        })
    return {
        "asset_id": asset_id, "max_tools": settings.SPC_MAX_TOOLS_PER_ASSET, "tools": len(taken),
        "group_columns": sorted(s["column"] for s in found
                                if types.get(s["column"]) in TEXT_FIELDS and s["column"] != "machine_state"),
        "states": states, "signals": sorted(out, key=lambda s: s["name"]),
    }


# --- 6.2 tools ----------------------------------------------------------------------------------

EDITABLE = ("name", "unit", "enabled", "states", "ignore_seconds", "method", "subgroup_size", "group_by", "lsl", "target",
            "usl", "limit_strategy", "sigma_multiplier", "rules", "no_variation_count", "event_actions", "window_hours")
# A change of one of these makes the stored baseline limits wrong: they are deleted.
CLEARS_LIMITS = ("method", "subgroup_size", "states", "ignore_seconds", "group_by", "sigma_multiplier", "limit_strategy")


def tool_json(tool: SpcTool) -> dict:
    out = {"id": str(tool.id), "asset_id": tool.asset_id, "signal_id": tool.signal_id, "column": tool.column,
           **{name: getattr(tool, name) for name in EDITABLE},
           "baseline_start": tool.baseline_start, "baseline_end": tool.baseline_end, "version": tool.version,
           "created_at": tool.created_at, "updated_at": tool.updated_at,
           "limits": [limits_json(lim) for lim in tool.limits.all()]}
    return out


def limits_json(lim: SpcLimits) -> dict:
    return {name: getattr(lim, name) for name in (
        "group", "source", "n_samples", "mean", "sigma_within", "sigma_overall", "range_bar", "cl", "lcl", "ucl",
        "r_cl", "r_lcl", "r_ucl", "window_start", "window_end", "computed_at", "tool_version")}


def _clean(data: dict, tool: SpcTool | None, asset_id: str) -> dict:
    """The fields of `data` that may be set, checked. Raises SpcError with a sentence for the user."""
    unknown = set(data) - set(EDITABLE) - {"asset_id", "signal_id"}
    if unknown:
        raise SpcError(f"unknown field: {', '.join(sorted(unknown))}")
    out = {k: v for k, v in data.items() if k in EDITABLE}

    def value(name):
        return out[name] if name in out else getattr(tool, name) if tool else SpcTool._meta.get_field(name).get_default()

    def number(name, low=None, high=None, whole=False, optional=False):
        if name not in out:
            return
        v = out[name]
        if v is None and optional:
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and int(v) != v):
            raise SpcError(f"{name} must be a {'whole ' if whole else ''}number")
        if (low is not None and v < low) or (high is not None and v > high):
            raise SpcError(f"{name} must be between {low} and {high}" if high is not None else f"{name} must be at least {low}")
        out[name] = int(v) if whole else float(v)

    number("ignore_seconds", 0, 86400, whole=True)
    number("subgroup_size", 2, 10, whole=True)
    number("no_variation_count", 3, 1000, whole=True)
    number("window_hours", 1, 24, whole=True)
    number("sigma_multiplier", 0.5, 6)
    for name in ("lsl", "target", "usl"):
        number(name, optional=True)
    if "method" in out and out["method"] not in (engine.IMR, engine.XBAR_R):
        raise SpcError("method must be imr or xbar_r")
    if "limit_strategy" in out and out["limit_strategy"] not in ("baseline", "fixed"):
        raise SpcError("limit_strategy must be baseline or fixed")
    if "enabled" in out and not isinstance(out["enabled"], bool):
        raise SpcError("enabled must be true or false")
    if "states" in out and (not isinstance(out["states"], list) or not out["states"]
                            or not all(isinstance(s, str) and s for s in out["states"])):
        raise SpcError("states must name at least one machine state")
    if "rules" in out:
        if not isinstance(out["rules"], list) or any(r not in engine.RULES for r in out["rules"]):
            raise SpcError(f"rules must be among: {', '.join(engine.RULES)}")
        out["rules"] = [r for r in engine.RULES if r in out["rules"]]
    if "event_actions" in out:
        actions = out["event_actions"]
        if not isinstance(actions, dict) or set(actions) - {engine.WARNING, engine.CONTROL, engine.SPEC}:
            raise SpcError("event_actions has an unknown event type")
        out["event_actions"] = {t: {"log": bool((actions.get(t) or {}).get("log", True)),
                                    "agent": bool((actions.get(t) or {}).get("agent", False))}
                                for t in (engine.WARNING, engine.CONTROL, engine.SPEC)}
    if "name" in out and (not isinstance(out["name"], str) or not out["name"].strip()):
        raise SpcError("name must not be empty")
    if "group_by" in out and out["group_by"]:
        if _field_types().get(out["group_by"]) not in TEXT_FIELDS:
            raise SpcError(f"group_by: the historian has no text column {out['group_by']!r}")
    lsl, target, usl = value("lsl"), value("target"), value("usl")
    if lsl is not None and usl is not None and not lsl < usl:
        raise SpcError("LSL must be below USL")
    if target is not None and ((lsl is not None and target < lsl) or (usl is not None and target > usl)):
        raise SpcError("the target must be between LSL and USL")
    return out


def create_tool(data: dict) -> SpcTool:
    asset_id, signal_id = data.get("asset_id"), data.get("signal_id")
    if not asset_id or not signal_id:
        raise SpcError("asset_id and signal_id are required")
    signal = next((s for s in signals(asset_id)["signals"] if s["signal_id"] == signal_id), None)
    if signal is None:
        raise SpcError("this signal has no numeric historian column, or is not a signal of the asset")
    maximum = settings.SPC_MAX_TOOLS_PER_ASSET
    if SpcTool.objects.filter(asset_id=asset_id).count() >= maximum:
        raise Conflict(f"this machine already has {maximum} SPC tools, the most allowed; delete one first")
    if SpcTool.objects.filter(asset_id=asset_id, signal_id=signal_id).exists():
        raise Conflict("this signal already has an SPC tool")
    fields = _clean(data, None, asset_id)
    fields.setdefault("name", signal["name"])
    fields.setdefault("unit", signal["unit"])
    with transaction.atomic():
        tool = SpcTool.objects.create(asset_id=asset_id, signal_id=signal_id, column=signal["column"], **fields)
        SpcCursor.objects.create(tool=tool, judged_until=timezone.now())     # the past is judged only on request
    return tool


def update_tool(tool: SpcTool, data: dict) -> tuple[SpcTool, bool]:
    """(the tool, whether its baseline limits were deleted). Every saved change raises the version."""
    if "asset_id" in data or "signal_id" in data:
        raise SpcError("a tool keeps its machine and signal; make a new tool for another signal")
    fields = _clean(data, tool, tool.asset_id)
    changed = {k: v for k, v in fields.items() if getattr(tool, k) != v}
    if not changed:
        return tool, False
    cleared = False
    with transaction.atomic():
        for name, value in changed.items():
            setattr(tool, name, value)
        tool.version += 1
        if any(name in changed for name in CLEARS_LIMITS):
            cleared = tool.limits.all().delete()[0] > 0
            tool.baseline_start = tool.baseline_end = None
        tool.save()
    return tool, cleared


# --- 6.3 limits ---------------------------------------------------------------------------------

def _store(tool: SpcTool, found: dict[str, engine.Limits], source: str, start=None, end=None) -> None:
    with transaction.atomic():
        tool.version += 1                       # new limits are a change of the tool
        tool.limits.all().delete()
        SpcLimits.objects.bulk_create([
            SpcLimits(tool=tool, source=source, window_start=start, window_end=end, tool_version=tool.version, **lim.as_dict())
            for lim in found.values()])
        tool.limit_strategy, tool.baseline_start, tool.baseline_end = source, start, end
        tool.save()


def baseline(tool: SpcTool, start, end, preview: bool) -> dict:
    """Control limits per group from the rows of (start, end]. Stored and frozen unless `preview`."""
    if not start < end:
        raise SpcError("the baseline's start must be before its end")
    if end - start > timedelta(days=BASELINE_MAX_DAYS):
        raise SpcError(f"a baseline window is at most {BASELINE_MAX_DAYS} days")
    rows = read_rows(tool.asset_id, [tool.column] + ([tool.group_by] if tool.group_by else []), start, end)
    found, skipped = engine.baseline_limits(_samples(tool, rows), method=tool.method, subgroup_size=tool.subgroup_size,
                                            k=tool.sigma_multiplier, min_samples=settings.SPC_MIN_BASELINE_SAMPLES)
    saved = False
    if not preview:
        if not found:
            raise SpcError("no group has enough samples in this window: nothing was saved")
        _store(tool, found, "baseline", start, end)
        saved = True
    return {"groups": [lim.as_dict() for lim in found.values()], "skipped": skipped, "saved": saved, "rows": len(rows)}


def set_fixed_limits(tool: SpcTool, groups: list[dict]) -> dict:
    if not isinstance(groups, list) or not groups:
        raise SpcError("give the limits of at least one group")
    found = {}
    for g in groups:
        try:
            cl, lcl, ucl = float(g["cl"]), float(g["lcl"]), float(g["ucl"])
            r_ucl = None if g.get("r_ucl") is None else float(g["r_ucl"])
        except (KeyError, TypeError, ValueError):
            raise SpcError("each group needs the numbers cl, lcl and ucl")
        group = str(g.get("group") or "")
        if bool(group) != bool(tool.group_by):
            raise SpcError("a tool with 'limits by' needs a group name for each set of limits, and one without needs none")
        try:
            found[group] = engine.fixed_limits(group, cl, lcl, ucl, r_ucl, method=tool.method,
                                               subgroup_size=tool.subgroup_size, k=tool.sigma_multiplier)
        except ValueError as exc:
            raise SpcError(str(exc))
    _store(tool, found, "fixed")
    return {"groups": [lim.as_dict() for lim in found.values()], "saved": True}


# --- 6.4 and 6.5 screening and detail -----------------------------------------------------------

def _judged_until(tool: SpcTool):
    cursor = SpcCursor.objects.filter(tool=tool).first()
    return cursor.judged_until if cursor else tool.created_at


def active_events(tool: SpcTool) -> list[dict]:
    """The tool's active SPC events as {type, rule}: those whose newest point is at most
    SPC_EVENT_ACTIVE_SECONDS older than the newest row the evaluator has judged."""
    since = _judged_until(tool) - timedelta(seconds=settings.SPC_EVENT_ACTIVE_SECONDS)
    return list(tool.events.filter(last_at__gte=since).values("type", "rule"))


def event_counts(tool: SpcTool, start, end) -> tuple[int, object]:
    """(events that started in the window, start of the newest of them)."""
    found = tool.events.filter(started_at__gt=start, started_at__lte=end)
    newest = found.order_by("-started_at").values_list("started_at", flat=True).first()
    return found.count(), newest


EVENT_PAGE = 50


def event_json(event: SpcEvent, active_since=None) -> dict:
    return {"id": str(event.id), "tool_id": str(event.tool_id), "tool_name": event.tool.name, "unit": event.tool.unit,
            "tool_version": event.tool_version, "type": event.type, "rule": event.rule, "group": event.group,
            "started_at": event.started_at, "last_at": event.last_at, "count": event.count, "value": event.value,
            "extreme": event.extreme, "machine_state": event.machine_state, "limits": event.limits,
            "active": active_since is not None and event.last_at >= active_since,
            "acknowledged_at": event.acknowledged_at, "acknowledged_by": event.acknowledged_by,
            "agent_status": event.agent_status}


def events(asset_id: str | None = None, tool_id: str | None = None, active: bool = False, limit: int = EVENT_PAGE,
           before=None) -> dict:
    """SPC events, newest first. `before` (a started_at) continues an earlier page."""
    found = SpcEvent.objects.select_related("tool")
    if tool_id:
        found = found.filter(tool_id=tool_id)
    if asset_id:
        found = found.filter(tool__asset_id=asset_id)
    if before is not None:
        found = found.filter(started_at__lt=before)
    window = timedelta(seconds=settings.SPC_EVENT_ACTIVE_SECONDS)
    since = {c.tool_id: c.judged_until - window for c in SpcCursor.objects.all()}
    limit = max(1, min(int(limit), 200))
    out = []
    for event in found.order_by("-started_at").iterator():
        item = event_json(event, since.get(event.tool_id))
        if active and not item["active"]:
            continue
        out.append(item)
        if len(out) > limit:
            break
    return {"events": out[:limit], "more": len(out) > limit}


def acknowledge(event: SpcEvent, by: str) -> SpcEvent:
    """Mark an event as seen. A second call changes nothing; the tool's status is not affected."""
    if event.acknowledged_at is None:
        event.acknowledged_at, event.acknowledged_by = timezone.now(), (by or "").strip()[:120]
        event.save(update_fields=["acknowledged_at", "acknowledged_by"])
    return event


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def _view(tool: SpcTool, rows: list[dict], end, hours: int, state: str | None) -> tuple[dict, list[engine.Sample], dict]:
    """The screening object of one tool from the asset's rows, with its samples and limits."""
    start = end - timedelta(hours=hours)
    window = [r for r in rows if start < r["ts"] <= end]
    states = tool.states if not state else [s for s in tool.states if s == state]
    found = _samples(tool, window, states)
    limits = {lim.group: lim for lim in tool.limits.all()}
    newest = window[-1] if window else None
    # The group that applies now is the newest row's: after a recipe change it is the new recipe at once,
    # also while the first rows of the new segment are still being ignored.
    group = str(newest.get(tool.group_by) or "") if newest and tool.group_by else ""
    lim = limits.get(group)
    status, reason = engine.status(enabled=tool.enabled, has_column=tool.column in _field_types(), has_limits=bool(limits),
                                   group_has_limits=lim is not None, group=group, active_events=active_events(tool))
    cap = engine.capability(found, method=tool.method, subgroup_size=tool.subgroup_size, lsl=tool.lsl, usl=tool.usl,
                            min_samples=settings.SPC_MIN_CAPABILITY_SAMPLES)
    values = [v for s in found for v in s.rows]
    violations, last = event_counts(tool, start, end)
    current = newest.get(tool.column) if newest else None
    out = {
        "tool_id": str(tool.id), "signal_id": tool.signal_id, "name": tool.name, "unit": tool.unit,
        "enabled": tool.enabled, "method": tool.method, "group_by": tool.group_by, "version": tool.version,
        "status": status, "status_reason": reason,
        "current_value": _round(current) if isinstance(current, (int, float)) else None, "current_group": group,
        "machine_state": newest.get("machine_state") if newest else None,
        "mean": _round(cap["mean"]), "sigma": _round(cap["sigma_within"]), "sample_count": cap["sample_count"],
        "cp": _round(cap["cp"], 3), "cpk": _round(cap["cpk"], 3), "pp": _round(cap["pp"], 3), "ppk": _round(cap["ppk"], 3),
        "capability_reason": cap["reason"],
        "sigma_shift": _round(engine.sigma_shift(found, limits, recent_seconds=settings.SPC_RECENT_MINUTES * 60), 2),
        "violation_count": violations, "last_violation_at": last,
        "lsl": tool.lsl, "target": tool.target, "usl": tool.usl,
        "cl": _round(lim.cl) if lim else None, "lcl": _round(lim.lcl) if lim else None, "ucl": _round(lim.ucl) if lim else None,
        "trend": engine.downsample([s.value for s in found]), "distribution": engine.histogram(values, tool.lsl, tool.usl),
    }
    return out, found, {"limits": limits, "capability": cap}


def _summary(tools: list[dict]) -> dict:
    count = lambda status: sum(1 for t in tools if t["status"] == status)   # noqa: E731
    cpks = [t["cpk"] for t in tools if t["cpk"] is not None]
    return {
        "selected_tools": len(tools), "stable": count("stable"), "warning": count("warning"), "trending": count("trending"),
        "out_of_control": count("out_of_control"), "specification_violations": count("spec_violation"),
        "paused": count("paused"), "configuration_required": count("configuration_required"),
        "average_cpk": round(sum(cpks) / len(cpks), 3) if cpks else None, "lowest_cpk": min(cpks) if cpks else None,
        "active_violations": sum(t.pop("_active", 0) for t in tools),
    }


def screening(asset_id: str, tool_ids: list[str] | None = None, state: str | None = None,
              window_hours: int | None = None) -> dict:
    """Summary and one object per tool, from one read of the historian for all tools."""
    tools = list(SpcTool.objects.filter(asset_id=asset_id).prefetch_related("limits"))
    if tool_ids is not None:
        tools = [t for t in tools if str(t.id) in tool_ids]
    newest = newest_row(asset_id)
    live = historian_source.current() == "live"
    age = (timezone.now() - newest["ts"]).total_seconds() if newest else None
    out = {"asset_id": asset_id, "source": historian_source.label(), "period_seconds": historian_source.period_seconds(),
           "window_hours": window_hours or max([t.window_hours for t in tools], default=8), "machine_state": newest["machine_state"] if newest else None,
           "newest_row": newest["ts"] if newest else None,
           "stale": bool(live and age is not None and age > historian_source.STALE_SECONDS), "tools": []}
    if newest and tools:
        hours = {t.id: window_hours or t.window_hours for t in tools}
        end = newest["ts"]
        columns = [t.column for t in tools] + [t.group_by for t in tools if t.group_by]
        rows = read_rows(asset_id, columns, end - timedelta(hours=max(hours.values())), end)
        for tool in tools:
            view, _, _ = _view(tool, rows, end, hours[tool.id], state)
            view["_active"] = len(active_events(tool))
            out["tools"].append(view)
    out["summary"] = _summary(out["tools"])
    return out


def detail(tool: SpcTool, window_hours: int | None = None) -> dict:
    """The screening object of one tool with its chart, limits per group, distribution and capability."""
    newest = newest_row(tool.asset_id)
    hours = window_hours or tool.window_hours
    if not newest:
        return {"tool_id": str(tool.id), "samples": [], "limits": [], "capability": None, "window_hours": hours}
    end = newest["ts"]
    rows = read_rows(tool.asset_id, [tool.column] + ([tool.group_by] if tool.group_by else []), end - timedelta(hours=hours), end)
    view, found, extra = _view(tool, rows, end, hours, None)
    limits = extra["limits"]
    fired = engine.judge(found, limits, rules=tool.rules, method=tool.method, subgroup_size=tool.subgroup_size,
                         lsl=tool.lsl, usl=tool.usl, no_variation_count=tool.no_variation_count)
    spans: list[dict] = []                      # where each group's limits apply, for stepped limit lines
    for s in found:
        if spans and spans[-1]["group"] == s.group:
            spans[-1]["end"] = s.ts
        else:
            spans.append({"group": s.group, "start": s.ts, "end": s.ts})
    cap = extra["capability"]
    view.update({
        "window_hours": hours, "window_start": end - timedelta(hours=hours), "window_end": end,
        "samples": [{"ts": s.ts, "value": _round(s.value), "range": _round(s.range), "group": s.group, "rules": rules}
                    for s, rules in zip(found, fired)],
        "limits": [{**limits_json(lim), "spans": [{"start": sp["start"], "end": sp["end"]} for sp in spans if sp["group"] == group]}
                   for group, lim in limits.items()],
        "groups_without_limits": sorted({s.group for s in found} - set(limits)),
        "capability": {**{k: (_round(v) if isinstance(v, float) else v) for k, v in cap.items()}, "target": tool.target},
    })
    return view
