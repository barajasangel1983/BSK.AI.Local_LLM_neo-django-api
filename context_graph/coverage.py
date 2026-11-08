"""P22f (2026-11-08): what the knowledge base knows about an asset, part by part and facet by facet.

    grid(asset_id)                      rows = the asset's parts (tree order), a cell per facet
    cell(asset_id, component_id, facet) the facts behind one cell, its pending triples and the part's manuals
    set_applicable(...)                 a person's "not applicable" / "applicable" for one part and facet

A cell counts the graph's facts of that facet the part owns: its own, and its device model's and product family's
(P26: a family's manual facts belong to every installed part of it, so both G120C drives show the G120C's hazards).
Only approved facts are in the graph; staged ones are counted apart as pending. A cell with nothing is "none", or
"n/a" when the facet doesn't apply to that kind of part (defaults below, a person's override wins), or "no manual"
when no manual describes the part.
"""

from __future__ import annotations

from collections import defaultdict

from django.db.models import Q

from . import services
from .facets import FACETS
from .models import CandidateTriple, CoverageOverride

# The kinds of fact a cell counts: get_asset_context's keys (P16a, P22a, P26e, P22g).
FACT_KEYS = {"part_alarms": "Alarm", "part_procedures": "Procedure", "failure_modes": "FailureMode",
             "specifications": "Specification", "parameters": "Parameter", "io_points": "IOPoint",
             "interfaces": "Interface", "hazards": "Hazard", "safety_functions": "SafetyFunction"}

# Facets that don't apply to a kind of part (by `component_type`, lower case), until a person says otherwise.
# Mechanical parts and assemblies have no wiring or settings; only measuring parts have a Quality facet (accuracy,
# calibration).
_MECHANICAL = {"gearbox", "heater", "die", "cutter", "screw", "screw_barrel", "feeder", "conditioner", "heating_zone",
               "measurement_point", "valve"}
_NO_QUALITY = {"drive", "motor", "contactor", "gearbox", "heater", "die", "cutter", "screw", "screw_barrel", "feeder",
               "conditioner", "heating_zone", "measurement_point", "valve", "controller"}
NOT_APPLICABLE: dict[str, set[str]] = defaultdict(set)
for kind in _MECHANICAL:
    NOT_APPLICABLE[kind] |= {"Controls", "Setup"}
for kind in ("motor", "contactor"):
    NOT_APPLICABLE[kind] |= {"Setup"}
for kind in _NO_QUALITY:
    NOT_APPLICABLE[kind] |= {"Quality"}


def default_not_applicable(component_type: str) -> set[str]:
    return set(NOT_APPLICABLE.get(str(component_type or "").strip().lower(), set()))


def _parts(ctx: dict) -> list[dict]:
    """The asset's parts in tree order: {id, name, depth, type, model, family, owners (part, model, family)}."""
    out = []

    def walk(components, depth=0):
        for c in components:
            model = c.get("device_model") or {}
            family = model.get("family") or {}
            out.append({"id": c["id"], "name": c["name"], "depth": depth,
                        "type": c["properties"].get("component_type") or "",
                        "model": model.get("name") or "", "family": family.get("name") or "",
                        "owners": [i for i in (c["id"], model.get("id"), family.get("id")) if i]})
            walk(c.get("children") or [], depth + 1)
    walk(ctx["components"])
    return out


def _facts(ctx: dict) -> list[dict]:
    """Every fact a cell can count: {id, name, label, facets, components, owners, properties}."""
    out = []
    for key, label in FACT_KEYS.items():
        for x in ctx.get(key) or []:
            out.append({"id": x["id"], "name": x["name"], "label": label, "components": x.get("components") or [],
                        "owners": x.get("owners") or [], "properties": x.get("properties") or {},
                        "facets": list((x.get("properties") or {}).get("facets") or [])})
    return out


def _manuals(ctx: dict) -> dict[str, list[dict]]:
    """part id -> the manuals that describe it (through its model or family too)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for d in ctx.get("documents") or []:
        for part in d.get("describes") or []:
            out[part].append({"name": d["name"], "doc_key": d["properties"].get("doc_key") or ""})
    return out


def _pending(parts: list[dict]) -> dict[tuple[str, str], list[CandidateTriple]]:
    """(part id, facet) -> staged triples of that facet anchored on the part, its model or its family (or, without
    an anchor, with the part as an end)."""
    owner_parts: dict[str, list[str]] = defaultdict(list)
    for p in parts:
        for o in p["owners"]:
            owner_parts[o].append(p["id"])
    ids = list(owner_parts)
    out: dict[tuple[str, str], list[CandidateTriple]] = defaultdict(list)
    rows = (CandidateTriple.objects.filter(status=CandidateTriple.Status.PENDING)
            .filter(Q(anchor_id__in=ids) | Q(anchor_id="", subject_id__in=ids) | Q(anchor_id="", object_id__in=ids))
            .select_related("document"))
    for t in rows:
        owner = t.anchor_id if t.anchor_id in owner_parts else t.subject_id if t.subject_id in owner_parts else t.object_id
        for part in owner_parts.get(owner, []):
            for f in t.facets or []:
                out[(part, f)].append(t)
    return out


def _overrides(parts: list[dict]) -> dict[tuple[str, str], bool]:
    return {(o.component_id, o.facet): o.applicable
            for o in CoverageOverride.objects.filter(component_id__in=[p["id"] for p in parts])}


def _status(approved: int, pending: int, applicable: bool, has_manual: bool) -> str:
    if approved:
        return "covered"
    if pending:
        return "pending"
    if not applicable:
        return "na"
    return "none" if has_manual else "no_manual"


def grid(asset_id: str, ctx: dict | None = None) -> dict:
    ctx = ctx or services.get_asset_context(asset_id)
    parts = _parts(ctx)
    facts = _facts(ctx)
    manuals = _manuals(ctx)
    pending = _pending(parts)
    overrides = _overrides(parts)
    approved: dict[tuple[str, str], int] = defaultdict(int)
    for x in facts:
        for part in x["components"]:
            for f in x["facets"]:
                approved[(part, f)] += 1
    rows, totals = [], {f: {"approved": 0, "pending": 0, "covered": 0} for f in FACETS}
    for p in parts:
        na = default_not_applicable(p["type"])
        cells = {}
        for f in FACETS:
            override = overrides.get((p["id"], f))
            applicable = override if override is not None else f not in na
            a, n = approved[(p["id"], f)], len(pending[(p["id"], f)])
            cells[f] = {"approved": a, "pending": n, "status": _status(a, n, applicable, bool(manuals[p["id"]])),
                        "override": override}
            totals[f]["approved"] += a
            totals[f]["pending"] += n
            totals[f]["covered"] += bool(a)
        rows.append({"id": p["id"], "name": p["name"], "depth": p["depth"], "type": p["type"], "model": p["model"],
                     "family": p["family"], "manuals": len(manuals[p["id"]]), "cells": cells})
    return {"asset": {"id": asset_id, "name": ctx["asset"]["name"]}, "facets": list(FACETS), "rows": rows,
            "totals": totals}


def cell(asset_id: str, component_id: str, facet: str, ctx: dict | None = None) -> dict:
    """The facts of one cell (where each comes from: the part, its model or its family), its pending triples by
    manual, and the part's manuals."""
    if facet not in FACETS:
        raise ValueError(f"unknown facet {facet!r}")
    ctx = ctx or services.get_asset_context(asset_id)
    parts = {p["id"]: p for p in _parts(ctx)}
    part = parts.get(component_id)
    if part is None:
        raise ValueError(f"{component_id!r} is not a part of {asset_id}")
    level = {part["owners"][0]: "part"}
    names = {component_id: part["name"]}
    for comp_ctx in _walk(ctx["components"]):
        if comp_ctx["id"] == component_id and comp_ctx.get("device_model"):
            m = comp_ctx["device_model"]
            level[m["id"]], names[m["id"]] = "model", m["name"]
            if m.get("family"):
                level[m["family"]["id"]], names[m["family"]["id"]] = "family", m["family"]["name"]
    facts = []
    for x in _facts(ctx):
        if component_id in x["components"] and facet in x["facets"]:
            owner = next((o for o in x["owners"] if o in level), "")
            facts.append({"id": x["id"], "name": x["name"], "label": x["label"], "level": level.get(owner, "part"),
                          "owner": names.get(owner, ""), "severity": x["properties"].get("severity") or "",
                          "evidence_ids": (x["properties"].get("evidence_ids") or [])[:3]})
    facts.sort(key=lambda f: (f["label"], f["name"].lower()))
    by_doc: dict[str, dict] = {}
    for t in _pending([part])[(component_id, facet)]:
        d = by_doc.setdefault(str(t.document_id or ""), {"document_id": str(t.document_id or ""),
                                                         "document": t.document.title if t.document else "",
                                                         "count": 0})
        d["count"] += 1
    applicable_default = facet not in default_not_applicable(part["type"])
    override = CoverageOverride.objects.filter(component_id=component_id, facet=facet).first()
    return {"component": {"id": component_id, "name": part["name"], "type": part["type"], "model": part["model"],
                          "family": part["family"]},
            "facet": facet, "facts": facts, "pending": list(by_doc.values()),
            "manuals": _manuals(ctx)[component_id],
            "applicable": override.applicable if override else applicable_default,
            "applicable_default": applicable_default, "override": override.applicable if override else None}


def _walk(components):
    for c in components:
        yield c
        yield from _walk(c.get("children") or [])


def set_applicable(component_id: str, facet: str, applicable: bool | None, by: str = "") -> bool | None:
    """A person's "applicable" (True) / "not applicable" (False) for one part and facet; None goes back to the
    default."""
    if facet not in FACETS:
        raise ValueError(f"unknown facet {facet!r}")
    if applicable is None:
        CoverageOverride.objects.filter(component_id=component_id, facet=facet).delete()
        return None
    CoverageOverride.objects.update_or_create(component_id=component_id, facet=facet,
                                              defaults={"applicable": applicable, "updated_by": by[:150]})
    return applicable
