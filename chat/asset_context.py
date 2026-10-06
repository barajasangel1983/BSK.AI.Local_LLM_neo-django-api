"""Asset-scoped chat context: graph facts + historian values for one asset.

The curated Context Graph (context_graph.services.get_asset_context) is turned
into a compact fact sheet — location, components, signals with units and
normal ranges, alarms with components and procedures, procedures,
connections, linked documents. Each signal line carries the historian's latest
RUNNING value compared with its range (plc_1_historian; the machine state and
timestamps are stated, since operating limits apply while running).

When the sheet doesn't fit its budget, lines matching the question are kept
first (the asset header and historian status always stay). The lab layer is
never used: it isn't trusted asset knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal

from django.conf import settings

from context_graph import provenance, services
from context_graph.ids import is_valid_id
from context_graph.repository import NodeNotFound

logger = logging.getLogger("chat")

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


@dataclass
class AssetContext:
    asset_id: str
    name: str
    text: str                                         # prompt block
    facts: list[str] = field(default_factory=list)    # lines actually included (for the citation)
    fact_sources: list[str] = field(default_factory=list)   # where each fact comes from (seed, document page, import row)
    total_facts: int = 0
    historian: dict | None = None                      # {"latest_ts", "state", "running_ts", "out_of_range": [...]}
    doc_keys: list[str] = field(default_factory=list)  # documents linked to the asset (DOCUMENTED_BY)
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


def historian_snapshot(asset_key: str, columns: list[str]) -> dict | None:
    """Latest historian row for the asset, plus the last RUNNING row (limits apply while running)."""
    try:
        from historian.models import ExtruderSample
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
    return {"latest": latest, "running": running, "state_since": since}


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
        limit = s["limits"][0]["properties"] if s["limits"] else {}
        out.append({"column": ref[len(HISTORIAN_PREFIX):], "signal_id": s["id"], "name": s["name"],
                    "unit": s["properties"].get("unit", ""), "low": limit.get("low"), "high": limit.get("high")})
    return out


def historian_rows(asset_key: str, columns: list[str], start, end, limit: int) -> list[dict]:
    """Historian samples with start < ts <= end, oldest first: [{ts, <column>: value…}]. Raises when Postgres is down."""
    from historian.models import ExtruderSample
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
    signals = [(s, asset["name"]) for s in ctx["signals"]] + [(s, c["name"]) for c, _ in flat for s in c["signals"]]
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
        state = f"Historian (plc_1_historian, simulated data): latest sample {latest['ts']:%Y-%m-%d %H:%M} UTC, machine state {latest['machine_state']}"
        if snap["state_since"]:
            state += f" since {snap['state_since']:%Y-%m-%d %H:%M}"
        if running and running is not latest:
            state += f". Signal values below are from the last RUNNING sample ({running['ts']:%Y-%m-%d %H:%M} UTC); normal ranges apply while running"
        lines.append(Line("ASSET", state + ".", pinned=True))
        historian = {"latest_ts": latest["ts"].isoformat(), "state": latest["machine_state"],
                     "running_ts": running["ts"].isoformat() if running else None, "out_of_range": []}

    for c, depth in flat:
        text = f"{'  ' * depth}- {c['name']} [{c['id']}]"
        if c["properties"].get("component_type"):
            text += f" ({c['properties']['component_type']})"
        if c["properties"].get("description"):
            text += f": {c['properties']['description']}"
        if c["alarms"]:
            text += f"; alarms: {', '.join(alarm_names.get(a, a) for a in c['alarms'])}"
        lines.append(Line("COMPONENTS", text, node_id=c["id"]))

    for s, owner in signals:
        p = s["properties"]
        unit = p.get("unit", "")
        limit = s["limits"][0]["properties"] if s["limits"] else None
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
        lines.append(Line("ALARMS", text, node_id=a["id"]))

    for pr in ctx["procedures"]:
        p = pr["properties"]
        text = f"- {pr['name']}"
        meta = [m for m in (p.get("procedure_type"), p.get("status") and f"status {p['status']}") if m]
        if meta:
            text += f" ({', '.join(meta)})"
        if p.get("summary"):
            text += f": {str(p['summary']).rstrip('.')}"
        if pr.get("applies_to"):
            text += f"; applies to: {', '.join(names.get(t, t) for t in pr['applies_to'])}"
        if pr.get("addresses"):
            text += f"; addresses: {', '.join(alarm_names.get(a, a) for a in pr['addresses'])}"
        lines.append(Line("PROCEDURES", text, node_id=pr["id"]))

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
    node_props.update({f"edge:{c['source']}>{c['target']}": {"source": c.get("origin"), "evidence_ids": c.get("evidence_ids")}
                       for c in ctx["connections"]})
    labels = provenance.fact_source_labels({ln.node_id: node_props.get(ln.node_id, {}) for ln in cited if ln.node_id})
    return AssetContext(
        asset_id=asset_id, name=asset["name"], text=text,
        facts=[ln.text.strip() for ln in cited],
        fact_sources=[labels.get(ln.node_id, "plc_1_historian" if ln.text.startswith("Historian") else "")
                      for ln in cited],
        total_facts=len([ln for ln in lines if not ln.pinned]),
        historian=historian,
        doc_keys=sorted({d["properties"]["doc_key"] for d in ctx["documents"] if d["properties"].get("doc_key")}),
        items=[{"text": ln.text.strip(), "section": ln.section.lower(), "entity_id": ln.node_id,
                "source": labels.get(ln.node_id, "plc_1_historian" if ln.text.startswith("Historian") else ""),
                "focus": ln.node_id in focus_ids} for ln in cited],
        values=values,
    )


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
    return out


SECTION_TITLES = {
    "ASSET": None, "COMPONENTS": "Components (tree)", "SIGNALS": "Signals",
    "ALARMS": "Alarms", "PROCEDURES": "Procedures", "CONNECTIONS": "Connections", "DOCUMENTS": "Documents",
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
    q = _tokens(question)
    focus_ids = focus_ids or set()
    scored = sorted(
        (i for i, ln in enumerate(lines) if not ln.pinned),
        key=lambda i: (lines[i].node_id not in focus_ids, -len(q & _tokens(lines[i].text)), i),
    )
    keep = {i for i, ln in enumerate(lines) if ln.pinned}
    for i in scored:
        trial = keep | {i}
        if len(_render([lines[j] for j in sorted(trial)])) > budget:
            continue
        keep = trial
    return [lines[i] for i in sorted(keep)]


# --- prompt --------------------------------------------------------------------

ASSET_PROMPT = """You are Neo, the assistant for {name} ({asset_id}) in the BSKLAB plant.

ASSET FACTS — from the curated Context Graph; treat them as authoritative:
{facts}
{historian_note}
Answer about this asset. Use the asset facts first and cite them as [G]; cite document excerpts as [1], [2], … when they are provided.
If the facts and excerpts don't contain the answer, say so plainly — do not invent values, limits or procedures."""


def system_prompt(ctx: AssetContext) -> str:
    note = ""
    if ctx.historian and ctx.historian["out_of_range"]:
        note = "\nOut of normal range in the last running sample: " + "; ".join(ctx.historian["out_of_range"]) + "\n"
    return ASSET_PROMPT.format(name=ctx.name, asset_id=ctx.asset_id, facts=ctx.text, historian_note=note)


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
        snippet = f"Latest sample {h['latest_ts'][:16].replace('T', ' ')} UTC, state {h['state']}"
        if h["out_of_range"]:
            snippet += f"; out of range: {', '.join(h['out_of_range'])}"
        out.append({"kind": "historian", "source": "plc_1_historian", "asset_id": ctx.asset_id, "section_path": [],
                    "page_start": None, "page_end": None, "score": None, "snippet": snippet[:400],
                    "facts": h["out_of_range"]})
    return out


def budgets(total_chars: int) -> tuple[int, int]:
    """(asset facts, document excerpts) character budgets for an asset-scoped prompt."""
    return int(total_chars * settings.ASSET_CONTEXT_SHARE), int(total_chars * settings.ASSET_RAG_SHARE)
