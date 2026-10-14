"""The SPC evaluator: judges the rows that arrived since the last pass and writes SPC events.

Run by `manage.py spc_evaluate` once a minute (a systemd timer), so events exist whether or not
the SPC page is open. It needs the Studio's database and the historian only: no graph, no
document store, no model. Specification: claude/P12_SPCimplementation.md, section 7.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from django.conf import settings

from historian import source as historian_source

from . import engine, service
from .models import SpcCursor, SpcEvent, SpcTool

logger = logging.getLogger(__name__)

RUN_HISTORY = 12            # samples read before the first one judged, so the run rules have their history


def lookback(tool: SpcTool) -> timedelta:
    """How far before the first row to judge the read begins: the ignore period and RUN_HISTORY samples."""
    rows = 1 if tool.method == engine.IMR else tool.subgroup_size
    return timedelta(seconds=tool.ignore_seconds + RUN_HISTORY * rows * historian_source.period_seconds())


def evaluate_tool(tool: SpcTool, since=None, dry_run: bool = False) -> dict:
    """Judge one tool's rows after its cursor (or after `since`). Returns what was done.

    A rule that fires again within SPC_EVENT_ACTIVE_SECONDS of an event of the same rule and group
    extends that event; otherwise a new event begins. The cursor moves to the newest row read, not
    to the clock, so rows that arrive late are judged on a later pass."""
    result = {"tool": tool.name, "judged": 0, "created": 0, "extended": 0, "skipped": ""}
    if not tool.enabled:
        return {**result, "skipped": "paused"}
    limits = {lim.group: lim for lim in tool.limits.all()}
    if not limits:
        return {**result, "skipped": "no control limits"}
    cursor, _ = SpcCursor.objects.get_or_create(tool=tool, defaults={"judged_until": tool.created_at})
    start = since or cursor.judged_until
    newest = service.newest_row(tool.asset_id)
    if not newest or newest["ts"] <= start:
        return {**result, "skipped": "no new row"}

    rows = service.read_rows(tool.asset_id, [tool.column] + ([tool.group_by] if tool.group_by else []),
                             start - lookback(tool), newest["ts"])
    found = service._samples(tool, rows)
    fired = engine.judge(found, limits, rules=tool.rules, method=tool.method, subgroup_size=tool.subgroup_size,
                         lsl=tool.lsl, usl=tool.usl, no_variation_count=tool.no_variation_count)
    active = timedelta(seconds=settings.SPC_EVENT_ACTIVE_SECONDS)
    if since is not None and not dry_run:
        tool.events.filter(started_at__gt=since).delete()           # the past is judged again
    open_events: dict[tuple, SpcEvent] = {}
    for event in tool.events.filter(last_at__gte=start - active).order_by("last_at"):
        open_events[(event.rule, event.group)] = event

    changed, result["judged"], result["created"], result["extended"] = group_events(tool, found, fired, limits, start, open_events)
    if not dry_run:
        for event in changed.values():
            event.save()
        if rows:
            cursor.judged_until = max(cursor.judged_until, rows[-1]["ts"]) if since is None else rows[-1]["ts"]
            cursor.save()
    result["events"] = [{"rule": e.rule, "type": e.type, "group": e.group, "started_at": e.started_at, "last_at": e.last_at,
                         "count": e.count, "value": e.value} for e in changed.values()]
    return result


def group_events(tool: SpcTool, found, fired, limits: dict, start, open_events: dict) -> tuple[dict, int, int, int]:
    """Turn the rules that fired on the samples after `start` into events: a rule that fires again within
    SPC_EVENT_ACTIVE_SECONDS of an event of the same rule and group extends it, otherwise a new event begins.
    `open_events` ({(rule, group): event}) holds the events a new firing may extend; it is updated in place.
    Returns (the events made or changed, by id; samples judged; events started; stored events extended).
    Nothing is saved here."""
    active = timedelta(seconds=settings.SPC_EVENT_ACTIVE_SECONDS)
    changed: dict = {}
    judged = created = extended = 0
    for sample, rules in zip(found, fired):
        if sample.ts <= start:
            continue                                # lookback: history for the run rules only
        judged += 1
        lim = limits.get(sample.group)
        for rule in rules:
            kind = engine.RULES[rule]
            if not (tool.event_actions.get(kind) or {}).get("log", True):
                continue
            event = open_events.get((rule, sample.group))
            if event is not None and sample.ts - event.last_at <= active:
                event.last_at, event.count = sample.ts, event.count + 1
                if abs(sample.value - lim.cl) > abs(event.extreme - lim.cl):
                    event.extreme = sample.value
                extended += event.pk not in changed and not event._state.adding
            else:
                event = SpcEvent(tool=tool, tool_version=tool.version, type=kind, rule=rule, group=sample.group,
                                 started_at=sample.ts, last_at=sample.ts, count=1, value=sample.value, extreme=sample.value,
                                 machine_state=sample.state,
                                 limits={"cl": lim.cl, "lcl": lim.lcl, "ucl": lim.ucl, "lsl": tool.lsl, "usl": tool.usl})
                open_events[(rule, sample.group)] = event
                created += 1
            changed[event.pk] = event
    return changed, judged, created, extended


def judge_window(tool: SpcTool, limits: dict, start, end) -> list[SpcEvent]:
    """The events the tool's present configuration gives on the rows of (start, end], made in memory only.

    Reads no stored event, writes nothing and leaves the cursor alone, so the tool, its events and their
    acknowledgements are untouched. The tool and its limits need not be saved. Used by the scoring (P12d)."""
    rows = service.read_rows(tool.asset_id, [tool.column] + ([tool.group_by] if tool.group_by else []),
                             start - lookback(tool), end)
    found = service._samples(tool, rows)
    fired = engine.judge(found, limits, rules=tool.rules, method=tool.method, subgroup_size=tool.subgroup_size,
                         lsl=tool.lsl, usl=tool.usl, no_variation_count=tool.no_variation_count)
    changed, *_ = group_events(tool, found, fired, limits, start, {})
    return sorted(changed.values(), key=lambda e: e.started_at)


def evaluate(tool_id=None, since=None, dry_run: bool = False) -> list[dict]:
    """One pass over the enabled tools (or one tool). A tool that fails is logged and the others go on."""
    if historian_source.current() != "live":
        return [{"tool": "", "judged": 0, "created": 0, "extended": 0,
                 "skipped": "the Studio serves the recorded month, which is not the present: nothing is judged"}]
    tools = SpcTool.objects.prefetch_related("limits")
    tools = tools.filter(pk=tool_id) if tool_id else tools
    out = []
    for tool in tools:
        try:
            out.append(evaluate_tool(tool, since, dry_run))
        except Exception as exc:        # one tool must not stop the others
            logger.exception("spc: tool %s (%s) was not judged", tool.name, tool.id)
            out.append({"tool": tool.name, "judged": 0, "created": 0, "extended": 0, "skipped": "",
                        "error": f"{type(exc).__name__}: {exc}"[:200]})
    return out
