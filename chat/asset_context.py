"""Asset-scoped chat context: graph facts + historian values for one asset.

The curated Context Graph (context_graph.services.get_asset_context) is turned
into a compact fact sheet — location, components, signals with units and
normal ranges, alarms with components and procedures, procedures,
connections, linked documents. Each signal line carries the historian's latest
RUNNING value compared with its range (the machine state and timestamps are
stated, since operating limits apply while running). Which historian is read,
the simulated month or the PLC's rows, is decided in historian/source.py.

When the sheet doesn't fit its budget, lines matching the question are kept
first (the asset header and historian status always stay). The lab layer is
never used: it isn't trusted asset knowledge.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from django.conf import settings

from context_graph import provenance, services
from context_graph.ids import is_valid_id
from context_graph.repository import NodeNotFound

logger = logging.getLogger("chat")

# How a signal in the graph points at its historian column. The live table has the same column names,
# so the pointer is kept as a name and read from whichever historian is served.
HISTORIAN_PREFIX = "plc_1_historian.extruder_samples."
STOPWORDS = {"the", "a", "an", "of", "on", "in", "is", "are", "what", "which", "how", "do", "does", "to", "for",
             "and", "or", "it", "its", "with", "should", "i", "my", "this", "that", "be", "can", "there", "any", "now"}


class AssetScopeError(ValueError):
    """The requested asset scope is not a known asset."""


@dataclass
class Line:
    section: str
    text: str
    pinned: bool = False   # always included
    node_id: str = ""      # the entity the fact is about (for its source label, P8c)
    related: tuple[str, ...] = ()   # entities kept with it when it is in focus (an alarm's procedure; P16a)


@dataclass
class AssetContext:
    asset_id: str
    name: str
    text: str                                         # prompt block
    facts: list[str] = field(default_factory=list)    # lines actually included (for the citation)
    fact_sources: list[str] = field(default_factory=list)   # where each fact comes from (seed, document page, import row)
    total_facts: int = 0
    # {"source", "latest_ts", "state", "running_ts", "age_seconds", "stale", "no_data_reason",
    #  "collector_age_seconds", "out_of_range": [...]}
    historian: dict | None = None
    doc_keys: list[str] = field(default_factory=list)  # documents linked to the asset (DOCUMENTED_BY)
    pages: list[tuple[str, int]] = field(default_factory=list)  # (doc_key, page) of the best facts (P17a)
    # The asset's parts for the planner (P17b): {id, name, parent, model, doc_keys (its manuals, from manual_title)}
    parts: list[dict] = field(default_factory=list)
    # Structured forms of the above, for the Context Service (context_service/service.py):
    items: list[dict] = field(default_factory=list)    # {text, section, entity_id, source, focus} per included fact
    values: list[dict] = field(default_factory=list)   # {signal_id, name, value, unit, low, high, status} (last RUNNING sample)


def check_asset(asset_id: str) -> str:
    """Validate an asset scope; returns its name. Raises AssetScopeError."""
    if not is_valid_id(asset_id) or not asset_id.startswith("bsk:asset:"):
        raise AssetScopeError(f"not an asset id: {asset_id!r}")
    try:
        node = services.node_detail(asset_id)
    except NodeNotFound:
        raise AssetScopeError(f"unknown asset {asset_id!r}")
    if node.get("label") != "Asset":
        raise AssetScopeError(f"{asset_id} is a {node.get('label')}, not an asset")
    return node.get("name") or asset_id


# --- historian ---------------------------------------------------------------

def _num(value):
    return float(value) if isinstance(value, Decimal) else value


def _fmt(value) -> str:
    if value is None:
        return "?"
    value = _num(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _age(seconds: int) -> str:
    return f"{seconds} s" if seconds < 120 else f"{seconds // 60} min" if seconds < 7200 else f"{seconds // 3600} h"


def historian_snapshot(asset_key: str, columns: list[str]) -> dict | None:
    """Latest historian row for the asset, plus the last RUNNING row (limits apply while running).

    `source` is the historian's label. `age_seconds` is the age of the latest row, for the live historian
    only: its rows are the present, so an old one means nothing is arriving. `collector_age_seconds` is how
    long ago a collector last reported (None: never, or not known); it tells which part stopped."""
    try:
        from django.utils import timezone
        from historian import source
        ExtruderSample = source.samples()
        live = source.current() == "live"
    except Exception:  # historian app unavailable
        return None
    fields = {f.name for f in ExtruderSample._meta.fields}
    columns = [c for c in columns if c in fields]
    try:
        rows = ExtruderSample.objects.filter(extruder_id=asset_key)
        latest = rows.order_by("-ts").values("ts", "machine_state", *columns).first()
        if latest is None:
            return None
        running = latest if latest["machine_state"] == "RUNNING" else (
            rows.filter(machine_state="RUNNING").order_by("-ts").values("ts", *columns).first())
        since = None
        if latest["machine_state"] != "RUNNING":
            change = rows.exclude(machine_state=latest["machine_state"]).order_by("-ts").values("ts").first()
            since = change["ts"] if change else None
    except Exception as exc:  # Postgres down: answer without historian values
        logger.warning("historian snapshot failed asset=%s: %s", asset_key, exc)
        return None
    now, collector_age = timezone.now(), None
    if live:
        try:
            from django.db.models import Max
            from historian.models import LiveCollector
            seen = LiveCollector.objects.aggregate(seen=Max("last_seen"))["seen"]
            collector_age = round((now - seen).total_seconds()) if seen else None
        except Exception as exc:
            logger.warning("collector report not read: %s", exc)
    return {"latest": latest, "running": running, "state_since": since, "source": source.label(),
            "age_seconds": round((now - latest["ts"]).total_seconds()) if live else None,
            "collector_age_seconds": collector_age}


def normal_limit(limits: list[dict]) -> dict | None:
    """The normal range among a signal's limits. Since the Limits import (P14b) a signal can also have a
    specification, an alarm or a trip limit; a limit without a kind is a normal range (the seed's)."""
    return next((lim["properties"] for lim in limits if (lim["properties"].get("kind") or "normal") == "normal"), None)


def signal_columns(asset_id: str) -> list[dict]:
    """The asset's signals that have a historian column: [{column, signal_id, name, unit, low, high}].

    Raises AssetScopeError / GraphUnavailable like `build`."""
    try:
        ctx = services.get_asset_context(asset_id)
    except NodeNotFound:
        raise AssetScopeError(f"unknown asset {asset_id!r}")

    def walk(components):
        for c in components:
            yield c
            yield from walk(c["children"])

    out = []
    for s in list(ctx["signals"]) + [s for c in walk(ctx["components"]) for s in c["signals"]]:
        ref = str(s["properties"].get("historian_ref", ""))
        if not ref.startswith(HISTORIAN_PREFIX):
            continue
        limit = normal_limit(s["limits"]) or {}
        out.append({"column": ref[len(HISTORIAN_PREFIX):], "signal_id": s["id"], "name": s["name"],
                    "unit": s["properties"].get("unit", ""), "low": limit.get("low"), "high": limit.get("high")})
    return out


def historian_rows(asset_key: str, columns: list[str], start, end, limit: int) -> list[dict]:
    """Historian samples with start < ts <= end, oldest first: [{ts, <column>: value…}]. Raises when Postgres is down."""
    from historian import source
    ExtruderSample = source.samples()
    fields = {f.name for f in ExtruderSample._meta.fields}
    columns = [c for c in columns if c in fields]
    rows = ExtruderSample.objects.filter(extruder_id=asset_key, ts__gt=start, ts__lte=end).order_by("ts")
    return list(rows.values("ts", *columns)[:limit])


def _range_status(value, limit: dict | None) -> str:
    if value is None or not limit:
        return ""
    low, high = limit.get("low"), limit.get("high")
    value = _num(value)
    if low is not None and value < low:
        return "BELOW normal range"
    if high is not None and value > high:
        return "ABOVE normal range"
    return "in range"


# --- fact sheet ----------------------------------------------------------------

def _limit_text(limit: dict | None, unit: str) -> str:
    if not limit:
        return ""
    low, high = limit.get("low"), limit.get("high")
    rng = f"{_fmt(low)}–{_fmt(high)}" if low is not None and high is not None else (
        f"≥ {_fmt(low)}" if low is not None else f"≤ {_fmt(high)}")
    return f"normal {rng}{(' ' + unit) if unit else ''}"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1}


def build(asset_id: str, question: str, budget_chars: int, focus_ids: set[str] | None = None) -> AssetContext:
    """Fact sheet for `asset_id` within `budget_chars` (raises GraphUnavailable / AssetScopeError).

    `focus_ids`: entities the question is about; their facts are kept first when the sheet is cut."""
    try:
        ctx = services.get_asset_context(asset_id)
    except NodeNotFound:
        raise AssetScopeError(f"unknown asset {asset_id!r}")
    asset = ctx["asset"]
    props = asset["properties"]
    asset_key = asset_id.split(":", 2)[2]
    names = {asset_id: asset["name"]}

    def walk(components, depth=0):
        for c in components:
            names[c["id"]] = c["name"]
            yield c, depth
            yield from walk(c["children"], depth + 1)

    flat = list(walk(ctx["components"]))
    owners: dict[str, tuple[dict, list[str]]] = {}   # a signal monitored by a system and its device is listed once
    for s, owner in [(s, asset["name"]) for s in ctx["signals"]] + [(s, c["name"]) for c, _ in flat for s in c["signals"]]:
        owners.setdefault(s["id"], (s, []))[1].append(owner)
    signals = [(s, ", ".join(names)) for s, names in owners.values()]
    alarm_names = {a["id"]: a["properties"].get("code") or a["name"] for a in ctx["alarms"]}

    columns = [s["properties"].get("historian_ref", "")[len(HISTORIAN_PREFIX):] for s, _ in signals
               if str(s["properties"].get("historian_ref", "")).startswith(HISTORIAN_PREFIX)]
    snap = historian_snapshot(asset_key, columns) if columns else None

    lines: list[Line] = []
    header = f"Asset {asset_id}: {asset['name']}"
    if props.get("asset_type"):
        header += f" ({props['asset_type']})"
    if props.get("description"):
        header += f" — {props['description']}"
    lines.append(Line("ASSET", header, pinned=True))
    if ctx["hierarchy"]:
        lines.append(Line("ASSET", "Location: " + " > ".join(n["name"] for n in ctx["hierarchy"]), pinned=True,
                          node_id=asset_id))

    historian = None
    values: list[dict] = []
    if snap:
        latest, running = snap["latest"], snap["running"]
        from historian import source as historian_source
        historian_label = snap.get("source") or historian_source.label("march")
        age = snap.get("age_seconds")
        stale = age is not None and age > historian_source.STALE_SECONDS
        # The live rows are 20 s apart, so their times are given to the second.
        when = "%Y-%m-%d %H:%M:%S" if age is not None else "%Y-%m-%d %H:%M"
        kind = "rows from the PLC, one every 20 s" if age is not None else "simulated data"
        state = f"Historian ({historian_label}, {kind}): latest sample {latest['ts']:{when}} UTC, machine state {latest['machine_state']}"
        if snap["state_since"]:
            state += f" since {snap['state_since']:{when}}"
        # Which part stopped: the collector reports every period, with or without rows to deliver.
        collector_age = snap.get("collector_age_seconds")
        reason = None
        if stale:
            silent = collector_age is None or collector_age > historian_source.STALE_SECONDS
            reason = "collector_not_reporting" if silent else "no_rows_from_plc"
            state += f". No row has arrived for {_age(age)}, so this is the last known state, not the present one ("
            state += ("the collector has not reported" + (f" for {_age(collector_age)}" if collector_age is not None else "")
                      + ": it or its link to the historian is down, and it keeps the rows until it can deliver them)"
                      if silent else
                      "the collector reports but has no rows: the generator is paused, or the PLC or its link is down)")
        if running and running is not latest:
            state += f". Signal values below are from the last RUNNING sample ({running['ts']:{when}} UTC); normal ranges apply while running"
        lines.append(Line("ASSET", state + ".", pinned=True))
        historian = {"source": historian_label, "latest_ts": latest["ts"].isoformat(), "state": latest["machine_state"],
                     "running_ts": running["ts"].isoformat() if running else None,
                     "age_seconds": age, "stale": stale, "no_data_reason": reason,
                     "collector_age_seconds": collector_age, "out_of_range": []}

    for c, depth in flat:
        text = f"{'  ' * depth}- {c['name']} [{c['id']}]"
        p = c["properties"]
        kind = ", ".join(str(v) for v in (p.get("component_type"), p.get("manufacturer"), p.get("model"),
                                          p.get("part_number") and f"part {p['part_number']}") if v)
        if kind:
            text += f" ({kind})"
        if p.get("description") and str(p["description"]) != c["name"]:
            text += f": {p['description']}"
        if c["alarms"]:
            text += f"; alarms: {', '.join(alarm_names.get(a, a) for a in c['alarms'])}"
        lines.append(Line("COMPONENTS", text, node_id=c["id"]))

    for s, owner in signals:
        p = s["properties"]
        unit = p.get("unit", "")
        limit = normal_limit(s["limits"])
        column = str(p.get("historian_ref", ""))[len(HISTORIAN_PREFIX):]
        text = f"- {s['name']} [{s['id']}] on {owner}"
        detail = [d for d in (unit and f"unit {unit}", _limit_text(limit, unit)) if d]
        if snap and snap["running"] and column in snap["running"]:
            value = snap["running"][column]
            status = _range_status(value, limit)
            values.append({"signal_id": s["id"], "name": s["name"], "value": _num(value), "unit": unit,
                           "low": (limit or {}).get("low"), "high": (limit or {}).get("high"), "status": status})
            detail.append(f"last running value {_fmt(value)}{(' ' + unit) if unit else ''}{f' ({status})' if status else ''}")
            if status.endswith("normal range"):
                historian["out_of_range"].append(f"{s['name']}: {_fmt(value)} {unit} ({status})".strip())
        if detail:
            text += ": " + "; ".join(detail)
        lines.append(Line("SIGNALS", text, node_id=s["id"]))

    for a in ctx["alarms"]:
        p = a["properties"]
        text = f"- {p.get('code') or a['name']}"
        if p.get("severity"):
            text += f" (severity {p['severity']})"
        if p.get("description"):
            text += f": {p['description']}"
        if a["components"]:
            text += f"; components: {', '.join(names.get(c, c) for c in a['components'])}"
        if a["procedures"]:
            text += f"; procedures: {', '.join(pr['name'] for pr in a['procedures'])}"
        lines.append(Line("ALARMS", text, node_id=a["id"], related=tuple(pr["id"] for pr in a["procedures"])))

    for pr in ctx["procedures"]:
        p = pr["properties"]
        text = f"- {pr['name']}"
        meta = [m for m in (p.get("procedure_type"), p.get("status") and f"status {p['status']}") if m]
        if meta:
            text += f" ({', '.join(meta)})"
        if _amounts(p):
            text += f": {_amounts(p)}"
        if p.get("summary"):
            text += f": {str(p['summary']).rstrip('.')}"
        if pr.get("applies_to"):
            text += f"; applies to: {', '.join(names.get(t, t) for t in pr['applies_to'])}"
        if pr.get("addresses"):
            text += f"; addresses: {', '.join(alarm_names.get(a, a) for a in pr['addresses'])}"
        lines.append(Line("PROCEDURES", text, node_id=pr["id"], related=tuple(pr.get("addresses") or ())))

    part_lines, part_props = _part_lines(ctx, names, {a["id"] for a in ctx["alarms"]}, {pr["id"] for pr in ctx["procedures"]})
    lines += part_lines

    for conn in ctx["connections"]:
        kind = f" ({conn['kind']})" if conn.get("kind") else ""
        lines.append(Line("CONNECTIONS", f"- {names.get(conn['source'], conn['source'])} → {names.get(conn['target'], conn['target'])}{kind}",
                          node_id=f"edge:{conn['source']}>{conn['target']}"))

    for d in ctx["documents"]:
        lines.append(Line("DOCUMENTS", f"- {d['name']}" + (f" (doc_key {d['properties']['doc_key']})" if d["properties"].get("doc_key") else "")))

    focus_ids = focus_ids or set()
    selected = _fit(lines, question, budget_chars, focus_ids)
    text = _render(selected)
    cited = [ln for ln in selected if not ln.pinned or ln.text.startswith(("Location", "Historian"))]
    node_props = {asset_id: asset["properties"]}
    node_props.update({c["id"]: c["properties"] for c, _ in flat})
    node_props.update({s["id"]: s["properties"] for s, _ in signals})
    node_props.update({a["id"]: a["properties"] for a in ctx["alarms"]})
    node_props.update({pr["id"]: pr["properties"] for pr in ctx["procedures"]})
    node_props.update(part_props)
    node_props.update({f"edge:{c['source']}>{c['target']}": {"source": c.get("origin"), "evidence_ids": c.get("evidence_ids")}
                       for c in ctx["connections"]})
    manuals = {d["name"]: d["properties"]["doc_key"] for d in ctx["documents"] if d["properties"].get("doc_key")}
    labels = provenance.fact_source_labels({ln.node_id: node_props.get(ln.node_id, {}) for ln in cited if ln.node_id})
    return AssetContext(
        asset_id=asset_id, name=asset["name"], text=text,
        facts=[ln.text.strip() for ln in cited],
        fact_sources=[labels.get(ln.node_id, historian["source"] if historian and ln.text.startswith("Historian") else "")
                      for ln in cited],
        total_facts=len([ln for ln in lines if not ln.pinned]),
        historian=historian,
        doc_keys=sorted({d["properties"]["doc_key"] for d in ctx["documents"] if d["properties"].get("doc_key")}),
        items=[{"text": ln.text.strip(), "section": ln.section.lower(), "entity_id": ln.node_id,
                "source": labels.get(ln.node_id, historian["source"] if historian and ln.text.startswith("Historian") else ""),
                "focus": ln.node_id in focus_ids} for ln in cited],
        values=values,
        pages=fact_pages(lines, question, focus_ids, node_props, settings.CHAT_FACT_PAGES),
        parts=[{"id": c["id"], "name": c["name"], "parent": c.get("parent") or "",
                "model": " ".join(str(c["properties"][k]) for k in ("manufacturer", "model") if c["properties"].get(k)),
                "doc_keys": [manuals[t] for t in re.split(r"\s+\|\s+", str(c["properties"].get("manual_title") or ""))
                             if t in manuals]} for c, _ in flat],
    )


SUMMARY_CHARS = 500     # a procedure's summary on its fact line (P16a)


def _amounts(p: dict) -> str:
    """A procedure's interval and amount ("interval 5,600 h; amount 70 g/bearing")."""
    return "; ".join(f"{k} {p[k]}" for k in ("interval", "amount") if p.get(k))


def _cut(text, limit: int = SUMMARY_CHARS) -> str:
    text = str(text).strip().rstrip(".")
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _part_lines(ctx: dict, names: dict, asset_alarms: set, asset_procedures: set) -> tuple[list[Line], dict]:
    """Fact lines for the parts' alarms, procedures and failure modes (P16a, 2026-10-16), and their properties.

    A procedure reached from an alarm or a failure mode is listed once, with the part it is for. Alarms and
    procedures already listed for the asset are not repeated."""
    parts = lambda ids: ", ".join(names.get(c, c) for c in ids)
    alarm_name = {a["id"]: a["properties"].get("code") or a["name"]
                  for a in [*ctx["alarms"], *ctx.get("part_alarms", [])]}
    procedures: dict[str, dict] = {}       # id -> {node, components, addresses}

    def add_procedure(node, components=(), addresses=()):
        entry = procedures.setdefault(node["id"], {"node": node, "components": [], "addresses": []})
        entry["components"] += [c for c in components if c not in entry["components"]]
        entry["addresses"] += [a for a in addresses if a not in entry["addresses"]]

    for pr in ctx.get("part_procedures", []):
        add_procedure(pr, pr["components"], pr["addresses"])
    for a in ctx.get("part_alarms", []):
        for pr in a["procedures"]:
            add_procedure(pr, a["components"], [a["id"]])
    for f in ctx.get("failure_modes", []):
        for pr in f["fixes"]:
            add_procedure(pr, f["components"])

    lines: list[Line] = []
    props: dict = {}
    for a in ctx.get("part_alarms", []):
        if a["id"] in asset_alarms:
            continue
        p = a["properties"]
        code = str(p.get("code") or "")
        text = f"- {code} {a['name']}" if code and not a["name"].startswith(code) else f"- {a['name']}"
        meta = [m for m in (p.get("subtype"), p.get("severity") and f"severity {p['severity']}") if m]
        if meta:
            text += f" ({', '.join(meta)})"
        text += f" on {parts(a['components'])}"
        if p.get("description"):
            text += f": {_cut(p['description'])}"
        if a["procedures"]:
            text += f"; procedure: {', '.join(pr['name'] for pr in a['procedures'])}"
        lines.append(Line("PART_ALARMS", text, node_id=a["id"], related=tuple(pr["id"] for pr in a["procedures"])))
        props[a["id"]] = p

    for pid, entry in procedures.items():
        if pid in asset_procedures:
            continue
        node, p = entry["node"], entry["node"]["properties"]
        text = f"- {node['name']}"
        meta = [m for m in (p.get("subtype") or p.get("procedure_type"), p.get("status") and f"status {p['status']}") if m]
        if meta:
            text += f" ({', '.join(meta)})"
        if entry["components"]:
            text += f" for {parts(entry['components'])}"
        if _amounts(p):
            text += f": {_amounts(p)}"
        if p.get("summary"):
            text += f": {_cut(p['summary'])}"
        if entry["addresses"]:
            text += f"; addresses: {', '.join(alarm_name.get(a, a) for a in entry['addresses'])}"
        lines.append(Line("PART_PROCEDURES", text, node_id=pid, related=tuple(entry["addresses"])))
        props[pid] = p

    for f in ctx.get("failure_modes", []):
        p = f["properties"]
        text = f"- {f['name']}" + (f" ({p['subtype']})" if p.get("subtype") else "") + f" on {parts(f['components'])}"
        causes = p.get("causes")
        if causes:
            text += f": causes: {_cut('; '.join(causes) if isinstance(causes, list) else causes)}"
        if f["indicated_by"]:
            text += f"; indicated by: {', '.join(alarm_name.get(x, names.get(x, x)) for x in f['indicated_by'])}"
        if f["fixes"]:
            text += f"; fix: {', '.join(pr['name'] for pr in f['fixes'])}"
        lines.append(Line("FAILURE_MODES", text, node_id=f["id"],
                          related=tuple(pr["id"] for pr in f["fixes"]) + tuple(f["indicated_by"])))
        props[f["id"]] = p

    for sp in ctx.get("specifications", []):
        p = sp["properties"]
        text = f"- {sp['name']} of {parts(sp['components'])}"
        if _rating(p):
            text += f": {_rating(p)}"
        extra = [m for m in (p.get("condition"), p.get("subtype")) if m]
        if extra:
            text += f" ({', '.join(str(m) for m in extra)})"
        lines.append(Line("SPECIFICATIONS", text, node_id=sp["id"]))
        props[sp["id"]] = p
    return lines, props


def _rating(p: dict) -> str:
    """A Specification's value: "-200 to 600 °C", "at least -20 °C", "16.5 A", "±1.0 °C or ±1.0 %"."""
    unit = f" {p['unit']}" if p.get("unit") else ""
    low, high, value = (p.get(k) for k in ("min", "max", "value"))
    if low not in (None, "") and high not in (None, ""):
        return f"{low} to {high}{unit}"
    if low not in (None, ""):
        return f"at least {low}{unit}"
    if high not in (None, ""):
        return f"at most {high}{unit}"
    return f"{value}{unit}" if value not in (None, "") else ""


def entities(asset_id: str) -> list[dict]:
    """The asset's components, signals, alarms and procedures with the names they go by:
    [{id, label, name, keys}] — used to find what a question is about."""
    ctx = services.get_asset_context(asset_id)
    out: list[dict] = []

    def add(node, label):
        p = node.get("properties") or {}
        keys = [node["name"], *(p.get("aliases") or []), p.get("code"), p.get("tag"),
                node["id"].rsplit("/", 1)[-1].replace("_", " ").replace("-", " ")]
        out.append({"id": node["id"], "label": label, "name": node["name"], "keys": [str(k) for k in keys if k]})

    def walk(components):
        for c in components:
            add(c, "Component")
            for sig in c["signals"]:
                add(sig, "Signal")
            walk(c["children"])
    walk(ctx["components"])
    for sig in ctx["signals"]:
        add(sig, "Signal")
    for alarm in ctx["alarms"]:
        add(alarm, "Alarm")
    for proc in ctx["procedures"]:
        add(proc, "Procedure")
    # The parts' alarms, procedures and failure modes (P16a), each once.
    seen = {e["id"] for e in out}
    part_nodes = [(a, "Alarm") for a in ctx.get("part_alarms", [])]
    part_nodes += [(pr, "Procedure") for pr in ctx.get("part_procedures", [])]
    part_nodes += [(pr, "Procedure") for a in ctx.get("part_alarms", []) for pr in a["procedures"]]
    part_nodes += [(f, "FailureMode") for f in ctx.get("failure_modes", [])]
    part_nodes += [(pr, "Procedure") for f in ctx.get("failure_modes", []) for pr in f["fixes"]]
    part_nodes += [(sp, "Specification") for sp in ctx.get("specifications", [])]
    for node, label in part_nodes:
        if node["id"] not in seen:
            seen.add(node["id"])
            add(node, label)
    return out


SECTION_TITLES = {
    "ASSET": None, "COMPONENTS": "Components (tree)", "SIGNALS": "Signals",
    "ALARMS": "Alarms", "PROCEDURES": "Procedures", "CONNECTIONS": "Connections", "DOCUMENTS": "Documents",
    "PART_ALARMS": "Alarms on parts", "PART_PROCEDURES": "Procedures on parts", "FAILURE_MODES": "Failure modes",
    "SPECIFICATIONS": "Ratings of parts (datasheet values, not process limits)",
}


def _render(lines: list[Line]) -> str:
    out, current = [], None
    for ln in lines:
        if ln.section != current:
            current = ln.section
            if SECTION_TITLES.get(current):
                out.append(f"{SECTION_TITLES[current]}:")
        out.append(ln.text)
    return "\n".join(out)


def _fit(lines: list[Line], question: str, budget: int, focus_ids: set[str] | None = None) -> list[Line]:
    """All lines if they fit; otherwise pinned lines + the facts about the focus entities,
    then the best-matching ones, in original order."""
    if len(_render(lines)) <= budget:
        return lines
    scored = [i for i, _ in _ranked(lines, question, focus_ids)]
    keep = {i for i, ln in enumerate(lines) if ln.pinned}
    for i in scored:
        trial = keep | {i}
        if len(_render([lines[j] for j in sorted(trial)])) > budget:
            continue
        keep = trial
    return [lines[i] for i in sorted(keep)]


def _ranked(lines: list[Line], question: str, focus_ids: set[str] | None = None) -> list[tuple[int, bool]]:
    """Unpinned lines, best first: (index, matched), matched = in focus or sharing a question word."""
    q = _tokens(question)
    focus_ids = set(focus_ids or ())
    # A fact in focus brings its related ones: an alarm its procedure, a failure mode its fix (P16a).
    focus_ids |= {r for ln in lines if ln.node_id in focus_ids for r in ln.related}
    # A question word found in many facts counts for less ("main drive" is on each of the drive's alarms, P16a).
    words = [q & _tokens(ln.text) for ln in lines]
    df = Counter(w for ws in words for w in ws)
    weight = [sum(math.log(1 + len(lines) / df[w]) for w in ws) for ws in words]
    scored = sorted(
        (i for i, ln in enumerate(lines) if not ln.pinned),
        key=lambda i: (lines[i].node_id not in focus_ids, -weight[i], i),
    )
    return [(i, lines[i].node_id in focus_ids or weight[i] > 0) for i in scored]


def fact_pages(lines: list[Line], question: str, focus_ids: set[str] | None, node_props: dict,
               limit: int) -> list[tuple[str, int]]:
    """The document pages (doc_key, page) of the best facts for the question, best first (P17a): the facts in
    focus, then those sharing a question word. Their excerpts are offered to the reranker with the search's."""
    if limit <= 0:
        return []
    from context_graph.models import Evidence
    wanted = [eid for i, matched in _ranked(lines, question, focus_ids) if matched and lines[i].node_id
              for eid in (node_props.get(lines[i].node_id, {}).get("evidence_ids") or [])[:2]]
    found = {e.pk: e for e in Evidence.objects.filter(pk__in=wanted, document__isnull=False, page_start__isnull=False)
             .select_related("document")}
    pages: list[tuple[str, int]] = []
    for eid in wanted:
        e = found.get(eid)
        if e is not None and (e.document.doc_key, e.page_start) not in pages:
            pages.append((e.document.doc_key, e.page_start))
            if len(pages) >= limit:
                break
    return pages


# --- prompt --------------------------------------------------------------------

ASSET_PROMPT = """You are Neo, the assistant for {name} ({asset_id}) in the BSKLAB plant.

ASSET FACTS — from the curated Context Graph; treat them as authoritative:
{facts}
{historian_note}{window}
Answer about this asset. Use the asset facts first and cite them as [G]; cite document excerpts as [1], [2], … when they are provided.
If the facts and excerpts don't contain the answer, say so plainly — do not invent values, limits or procedures."""


WINDOW_NOTE = ("\n{block}\nThe historian window above is computed from the historian's rows for the time the question asks "
               "about; answer questions about that time from it, cite it as [H], and say which window it covers. "
               "Its numbers are the answer for that window: never give a live value or a latest sample from the facts "
               "above in their place (those are one moment, now). State the running minutes with an OEE, and only the "
               "downtime minutes as downtime (changeover, cool down, warm up and flush are other stops).\n")


def system_prompt(ctx: AssetContext, window: str = "") -> str:
    """`window`: the historian window block of a data question (P18a)."""
    note = ""
    if ctx.historian and ctx.historian["out_of_range"]:
        note = "\nOut of normal range in the last running sample: " + "; ".join(ctx.historian["out_of_range"]) + "\n"
    return ASSET_PROMPT.format(name=ctx.name, asset_id=ctx.asset_id, facts=ctx.text, historian_note=note,
                               window=WINDOW_NOTE.format(block=window) if window else "")


def citations(ctx: AssetContext) -> list[dict]:
    """Sources shown under the answer: the graph facts used, and the historian snapshot."""
    out = [{
        "kind": "graph", "source": "Context Graph", "asset_id": ctx.asset_id, "section_path": [ctx.name],
        "page_start": None, "page_end": None, "score": None,
        "snippet": f"{len(ctx.facts)} of {ctx.total_facts} facts about {ctx.name}",
        "facts": ctx.facts, "fact_sources": ctx.fact_sources,
    }]
    if ctx.historian:
        h = ctx.historian
        snippet = f"Latest sample {h['latest_ts'][:19 if h['age_seconds'] is not None else 16].replace('T', ' ')} UTC, state {h['state']}"
        if h["stale"]:
            snippet += f" (no row for {_age(h['age_seconds'])})"
        if h["out_of_range"]:
            snippet += f"; out of range: {', '.join(h['out_of_range'])}"
        out.append({"kind": "historian", "source": h["source"], "asset_id": ctx.asset_id, "section_path": [],
                    "page_start": None, "page_end": None, "score": None, "snippet": snippet[:400],
                    "facts": h["out_of_range"]})
    return out


def budgets(total_chars: int, facts_share: float = 0.0, excerpts_share: float = 0.0) -> tuple[int, int]:
    """(asset facts, document excerpts) character budgets for an asset-scoped prompt; a share of 0 is the
    setting's (the planner, P17b, may give others)."""
    return (int(total_chars * (facts_share or settings.ASSET_CONTEXT_SHARE)),
            int(total_chars * (excerpts_share or settings.ASSET_RAG_SHARE)))
