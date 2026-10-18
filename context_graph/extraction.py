"""Triple extraction from library documents (GraphLab "Generate triples").

Runs as a library-worker job (kind "extract"):

    parsed document (cached Docling output)
      -> extraction windows (chunking strategy, default fixed 800/100 tokens)
      -> per window and mode: prompt (preset + active schema) -> DGX, JSON mode
      -> strict validation (+1 retry on malformed JSON)
      -> entity matching against the curated graph (existing ids or proposed ids)
      -> de-duplicated CandidateTriple rows, pending review

With `include_drawings`, the figures that Describe figures classified as drawings are also
sent to the VLM on BSK (context_graph/drawings.py) and staged the same way, with vision evidence.

Scope (step 2 of the schema review, 2026-10-15):
- `pages` ("14, 18-26"): only the windows on those pages are read.
- `follow_languages` (default on): a document that keeps some languages (Document.languages) is read on
  those pages only, as for RAG.
- `anchor_id`: the part the document describes (a generic manual, e.g. ABB's motors manual for
  `main-motor`). The prompt names it and lists the asset's existing alarms, signals and procedures by
  their exact names, so exact-name matching links to them; the product's own designations (model,
  part number) are mapped to that part instead of making a new "Motor".

Step 3 (2026-10-16):
- `tables_by_row` (default on): the document's tables are read on their own, a few rows at a time with
  their header (context_graph/tables.py), and the text windows without them.
- Properties: the model may give an entity's documented properties (schema v3 `properties`, e.g. a
  Procedure's interval and amount, a FailureMode's causes); they are staged with the triple.
- A window whose answer is invalid twice (often cut off at the output limit) is read again in two halves.
- The "Maintenance (v3)" preset: troubles as FailureModes with their causes, what to do as a Procedure.
  It asks for a short "note" before the triples (thinking is off on the DGX): without it the model skipped
  untitled lists of steps and read the wrong column of Table 7.2 (2026-10-16, ABB motors manual).

P22a (2026-10-18, schema v4): a part's ratings are Specifications (`Component -HAS_SPEC-> Specification`).
A `Component -HAS_LIMIT-> OperatingLimit` (allowed only from a Signal) is staged as one, with the limit's
values as the rating's, and the reviewer sees it was converted. The "Specifications (v4)" preset reads
technical-data pages.

The DGX's constrained decoding (guided_json / json_schema) was too slow to use
(80+ s and invalid output per window), so JSON is requested in the prompt with
`response_format: json_object` and validated here.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from usage import recorder as usage

from . import registry, repository
from .driver import GraphUnavailable, session
from .ids import clean_part, is_valid_id, make_id
from . import evidence, facets
from .identity import EntityIndex, normalize, proposed_id  # noqa: F401 (re-exported)
from .models import CandidateTriple, Evidence, PromptPreset
from .schema import Schema

logger = logging.getLogger("chat")

MODES = ("schema", "freeform", "both")
DEFAULT_CHUNKING = {"strategy": "fixed", "params": {"size": 800, "overlap": 100}}
MAX_NAME = 200
EVIDENCE_CHARS = 1500
EVIDENCE_TYPES = ("Document", "DocumentSection")
# Links the Studio suggests and a person approves, never the model (P22c, schema v6): a parameter's PLC signal, a
# terminal's wiring. They are not offered to the model, and a model's triple with one is staged with an issue.
SUGGESTED_ONLY = ("MAPS_TO", "CARRIES")

DEFAULT_PRESETS = {
    "schema": (
        "Default schema-guided",
        """You extract knowledge-graph triples about an industrial asset from technical documentation.

Entity types (use exactly these names):
{entity_types}

Relationships (predicate: allowed subject type -> object type):
{relationships}

Rules:
- Only extract facts stated in the text; do not guess.
- Use only the entity types and relationships listed above, in the allowed directions.
- Name entities as they are written in the text, keeping engineering tags (e.g. "M101", "VFD-101", "DIE_PLUG").
- When a type lists subtypes, you may add "subject_subtype" / "object_subtype" with one of them.
- Return ONLY a JSON object:
  {"triples": [{"subject": str, "subject_type": str, "predicate": str, "object": str, "object_type": str, "confidence": number 0-1}]}
- Return {"triples": []} if the text has no such facts.

Document: {document}
Section: {section}""",
    ),
    "freeform": (
        "Default free-form",
        """You extract knowledge-graph triples from technical documentation.

Rules:
- Extract the important factual relationships stated in the text as (subject, predicate, object).
- predicate: a short UPPER_SNAKE_CASE verb phrase (e.g. PART_OF, CONTROLS, REQUIRES, MEASURES).
- subject_type / object_type: a short noun for what the entity is (e.g. Component, Sensor, Setting, Material).
- Name entities as they are written in the text.
- Return ONLY a JSON object:
  {"triples": [{"subject": str, "subject_type": str, "predicate": str, "object": str, "object_type": str, "confidence": number 0-1}]}
- Return {"triples": []} if there are none.

Document: {document}
Section: {section}""",
    ),
}


# Offered beside the defaults, never the default itself (step 3, 2026-10-16).
EXTRA_PRESETS = {
    "Maintenance (v3)": (
        "schema",
        """You extract a maintenance knowledge graph about an industrial part from its manual.

Entity types (use exactly these names):
{entity_types}

Relationships (predicate: allowed subject type -> object type):
{relationships}

Properties an entity may have (only these names):
{properties}

Rules:
- Only facts stated in the text; never guess a value.
- Troubleshooting (a trouble or symptom, its causes, what to do):
  - Each trouble or symptom is ONE FailureMode named as written (e.g. "Motor overheats while running"):
    <part> HAS_FAILURE_MODE <trouble>. Its causes go in the FailureMode's "causes" property, a list of the
    causes as written. A cause is never an entity of its own.
  - What to do is ONE Procedure per trouble, named "<trouble>: what to do", subtype Corrective, with "summary"
    giving each cause and its action ("Blown fuses: replace fuses with proper type and rating; Overload trips: ...").
    <trouble> MITIGATED_BY <procedure>.
- A list of fault, warning or alarm messages (a code or name, its cause, what to do): each message is ONE Alarm,
  named as written: <part> HAS_ALARM <alarm>. What to do is ONE Procedure named "<alarm>: what to do", subtype
  Corrective, with "summary": <procedure> ADDRESSES <alarm>. The link goes from the procedure to the alarm,
  never <alarm> MITIGATED_BY <procedure>.
- Maintenance, inspection, lubrication: a Procedure named as the text names it, with a subtype and its
  "interval", "amount" and "summary" when the text gives them: <part> HAS_PROCEDURE <procedure>.
  Steps or instructions without a title of their own (a list under a heading) are the procedure named by the
  Section below, e.g. Section "Regreasing while the motor is running": <part> HAS_PROCEDURE
  "Regreasing while the motor is running", with the steps, short, in "summary".
- A table of values for many sizes or variants (e.g. lubrication intervals and grease amounts by frame size
  and speed): find the row of the part this document describes. Its frame size is usually the number in its
  model designation (e.g. 355 in "M3BP 355MLA 4"). Its speed column follows from its poles and supply frequency:
    at 50 Hz: 2-pole = 3000 r/min, 4-pole = 1500 r/min, 6-pole = 1000 r/min, 8-pole = 750 r/min;
    at 60 Hz: 2-pole = 3600 r/min, 4-pole = 1800 r/min, 6-pole = 1200 r/min, 8-pole = 900 r/min.
  From that row make <part> HAS_PROCEDURE <a Procedure named after the table and the "Rows under" heading,
  e.g. "Regreasing (roller bearings)">, with "interval" and "amount" from that row and that speed column,
  with units, and say in "summary" the table, the row and the column they come from.
  If no row applies to the part, return nothing for the table.
- INDICATED_BY only to an alarm or a signal listed as existing below, by its exact name, and only when the
  text clearly means it. Never create a new Signal or Alarm.
- A sub-part (bearing, fan, shaft seal) only when the text names it as a part of this product.
- First a key "note": one or two sentences on which procedures (with any interval or amount) and which troubles
  this text gives about the part, or that it gives none. Then the triples.
- Return ONLY compact JSON:
  {"note": str, "triples": [{"subject": str, "subject_type": str, "subject_subtype": str, "subject_properties": {},
                "predicate": str, "object": str, "object_type": str, "object_subtype": str, "object_properties": {},
                "confidence": number 0-1}]}
  Subtypes and properties are optional. Return {"triples": []} if the text has nothing of this kind.

Document: {document}
Section: {section}""",
    ),
    "Specifications (v4)": (
        "schema",
        """You extract the ratings and technical data of an industrial part from its manual or datasheet.

Entity types (use exactly these names):
{entity_types}

Relationships (predicate: allowed subject type -> object type):
{relationships}

Properties an entity may have (only these names):
{properties}

Rules:
- Only values stated in the text; never guess or convert a value.
- Each rating is ONE Specification: <part> HAS_SPEC <specification>, named after the quantity as written
  (e.g. "Measuring range", "Ambient temperature (head)", "Rated output current at 2 kHz"), with a subtype.
  - A range: "min" and "max". A single value: "value". Always "unit" when the text gives one.
  - "condition": when it applies, if the text says (a pulse frequency, a version, "with display").
  - "basis": the table, its row and its column, or the section.
- A table for many models, sizes or variants: only the row or column of the part this document describes
  (its model designation, power or order code). If none applies, return nothing for the table.
- A rating of the part is never an OperatingLimit: those are the process's limits on a signal.
- Never create a Signal, an Alarm or a Procedure.
- First a key "note": one sentence on which ratings this text gives about the part, or that it gives none.
- Return ONLY compact JSON:
  {"note": str, "triples": [{"subject": str, "subject_type": str, "predicate": str, "object": str, "object_type": str,
                "object_subtype": str, "object_properties": {}, "confidence": number 0-1}]}
  Return {"triples": []} if the text has nothing of this kind.

Document: {document}
Section: {section}""",
    ),
    "Controls & Setup (v6)": (
        "schema",
        """You extract how an industrial device is set up and connected, from its manual: its parameters, its terminals and
its communication interfaces.

Entity types (use exactly these names):
{entity_types}

Relationships (predicate: allowed subject type -> object type):
{relationships}

Properties an entity may have (only these names):
{properties}

Rules:
- Only what the text states; never guess a value, a default or a range.
- Parameters: only those the text tells you to set or check (commissioning, terminal functions, limits, ramps,
  fieldbus, motor data, protection), or that an alarm or a procedure names. Not every parameter mentioned in a
  function diagram.
  - Each is ONE Parameter: <device> HAS_PARAMETER <parameter>, named as the text names it ("Maximum speed"), with
    "code" exactly as written (p1082, r0021, 01.01) and the subtype Setting (can be changed) or Readout (display
    only, r…). "default", "min", "max", "unit", "access", "data_set" when the text gives them.
- Terminals: each is ONE IOPoint: <device> HAS_IO <terminal>, named "DI 0", "AI 0", "Relay output DO 0",
  with "connector" (-X138), "terminal" (5), "signal_type" (24 V DC, 0-10 V, 0/4-20 mA, NO contact) and "direction".
  The parameter that sets a terminal's function: <terminal> CONFIGURED_BY <parameter>.
- Interfaces: <device> COMMUNICATES_VIA <interface> (PROFINET, PROFIBUS, USS, Modbus RTU, USB, operator panel), with
  "protocol", "connector", "address_range", "telegram" when given.
- A commissioning step that sets a parameter: <procedure> SETS <parameter> (the procedure as the text names it).
- A table or list for several models or sizes: only what applies to the device this document describes.
- Never create a Signal, an Alarm, a Specification (ratings are another run) or an OperatingLimit.
- First a key "note": one sentence on what this text gives about the device's setup and connections, or that it gives
  none.
- Return ONLY compact JSON:
  {"note": str, "triples": [{"subject": str, "subject_type": str, "subject_subtype": str, "subject_properties": {},
                "predicate": str, "object": str, "object_type": str, "object_subtype": str,
                "object_properties": {}, "confidence": number 0-1}]}
  Return {"triples": []} if the text has nothing of this kind.

Document: {document}
Section: {section}""",
    ),
}
# Never from the model: identity, provenance, and `status` (draft / approved is a person's decision).
PROPERTY_SKIP = {"id", "name", "subtype", "source", "aliases", "evidence_ids", "created_by_triple", "status", "facets"}
# Multi-facet run (P22c-0): one job per document goes through these in order. Each facet run picks its own pages
# (Suggest pages with its prompt), extracts them with its prompt and stages the triples. A run is offered when its
# prompt exists and the active schema has the type it needs. New facet prompts (Controls/Setup, Safety, …) are added
# here as they are built.
# `owns`: the types a facet run is responsible for. In one job, the other facet runs leave those types to it (the
# Maintenance prompt also extracts ratings under v5: on the Opta datasheet 80 of its 82 triples duplicated the
# Ratings run's, 2026-10-18).
FACET_RUNS = (
    {"key": "maintenance", "label": "Maintenance", "preset": "Maintenance (v3)", "needs": "", "owns": ()},
    {"key": "ratings", "label": "Ratings", "preset": "Specifications (v4)", "needs": "Specification",
     "owns": ("Specification",), "keeps": ("HAS_SPEC",)},
    # Smaller windows: parameter-heavy pages give long answers; fewer facts per window keep each answer whole.
    {"key": "controls", "label": "Controls & Setup", "preset": "Controls & Setup (v6)", "needs": "Parameter",
     "owns": ("Parameter", "IOPoint", "Interface"), "chunking": {"strategy": "fixed", "params": {"size": 400, "overlap": 50}},
     # only its own links: on the G120C it also staged 30 fieldbus error codes as alarms (2026-10-18)
     "keeps": ("HAS_PARAMETER", "HAS_IO", "COMMUNICATES_VIA", "CONFIGURED_BY", "SETS")},
)
FACET_RUN_BY_KEY = {r["key"]: r for r in FACET_RUNS}


def facet_runs_available(schema: Schema | None = None) -> list[dict]:
    """The facet runs this Studio can do now: [{key, label, preset_id}] in their order."""
    schema = schema or registry.active_schema()
    ensure_default_presets()
    presets = {p.name: p for p in PromptPreset.objects.filter(mode="schema", name__in=[r["preset"] for r in FACET_RUNS])}
    return [{"key": r["key"], "label": r["label"], "preset_id": presets[r["preset"]].pk} for r in FACET_RUNS
            if r["preset"] in presets and (not r["needs"] or r["needs"] in schema.entity_types)]


MAX_PROPERTY = 1000
MAX_LIST = 40


class ExtractionError(ValueError):
    pass


def ensure_default_presets() -> None:
    """The built-in prompts, copied from code the first time; later code changes are offered, never applied (P23)."""
    wanted = [(name, mode, template, True) for mode, (name, template) in DEFAULT_PRESETS.items()]
    wanted += [(name, mode, template, False) for name, (mode, template) in EXTRA_PRESETS.items()]
    have = set(PromptPreset.objects.filter(name__in=[w[0] for w in wanted]).values_list("name", flat=True))
    for name, mode, template, is_default in wanted:
        if name not in have:
            preset = PromptPreset(name=name, mode=mode, template=template, is_default=is_default)
            preset._version_source, preset._version_note = "built-in", "the built-in text"
            preset.save()


def builtin_templates() -> dict[str, str]:
    """Name -> template of the presets the Studio ships (they can be edited and restored, not deleted)."""
    return {**{name: template for name, template in DEFAULT_PRESETS.values()},
            **{name: template for name, (_, template) in EXTRA_PRESETS.items()}}


def default_preset(mode: str) -> PromptPreset:
    ensure_default_presets()
    return PromptPreset.objects.filter(mode=mode, is_default=True).first() or PromptPreset.objects.filter(mode=mode).first()


# --- parameters --------------------------------------------------------------

def validate_params(params: dict | None) -> dict:
    """Validate/normalize extract-job parameters (called when the job is queued)."""
    from ingestion.chunking import ChunkingError, resolve_params

    params = dict(params or {})
    mode = params.get("mode", "schema")
    if mode not in MODES:
        raise ExtractionError(f"mode must be one of {', '.join(MODES)}")

    chunking = params.get("chunking") or DEFAULT_CHUNKING
    try:
        resolved = resolve_params(chunking.get("strategy", "fixed"), chunking.get("params"))
    except ChunkingError as exc:
        raise ExtractionError(str(exc))

    presets = {}
    for m in (("schema", "freeform") if mode == "both" else (mode,)):
        preset_id = (params.get("presets") or {}).get(m)
        preset = PromptPreset.objects.filter(pk=preset_id, mode=m).first() if preset_id else default_preset(m)
        if preset is None:
            raise ExtractionError(f"unknown {m} prompt preset {preset_id!r}")
        presets[m] = preset.pk

    asset_id = params.get("asset_id") or ""
    if asset_id and not is_valid_id(asset_id):
        raise ExtractionError(f"asset_id must be a canonical id (bsk:asset:...), got {asset_id!r}")
    anchor_id = params.get("anchor_id") or ""
    if anchor_id:
        if not (is_valid_id(anchor_id) and anchor_id.startswith("bsk:component:")):
            raise ExtractionError(f"anchor_id must be a component id (bsk:component:...), got {anchor_id!r}")
        if not asset_id:
            raise ExtractionError("choose the asset of the part the document describes")
    facet_runs = params.get("facet_runs") or []
    if facet_runs:
        available = [r["key"] for r in facet_runs_available()]
        wanted = available if facet_runs == "all" else facet_runs
        if not isinstance(wanted, list) or not wanted:
            raise ExtractionError("facet_runs must be \"all\" or a list of facet runs")
        unknown = [k for k in wanted if k not in available]
        if unknown:
            raise ExtractionError(f"facet run(s) not available: {', '.join(map(str, unknown))} "
                                  f"(available: {', '.join(available) or 'none'})")
        facet_runs = [k for k in available if k in wanted]       # always in the facets' order
        mode = "schema"
    return {
        "mode": mode,
        "chunking": {"strategy": chunking.get("strategy", "fixed"), "params": resolved},
        # the reviewer chose windows: a facet run keeps them (kept as given when the params are validated again)
        "chunking_custom": bool(params["chunking_custom"] if "chunking_custom" in params else params.get("chunking")),
        "presets": presets,
        "asset_id": asset_id,
        # Also read the document's described drawings with the VLM on BSK.
        "include_drawings": bool(params.get("include_drawings")),
        "pages": parse_pages(params.get("pages")),
        "follow_languages": params.get("follow_languages", True) is not False,
        "anchor_id": anchor_id,
        "tables_by_row": params.get("tables_by_row", True) is not False,
        # P22c-0: the facet runs of a multi-facet job (each picks its own pages; `pages` is ignored), and whether a
        # facet with reviewed triples in this document is run again (default: skipped).
        "facet_runs": facet_runs,
        "rerun_reviewed": bool(params.get("rerun_reviewed")),
    }


MAX_PAGE = 100_000


def parse_pages(value) -> list[list[int]]:
    """"14, 18-26" (or [[14, 14], [18, 26]]) -> [[14, 14], [18, 26]]; empty = every page."""
    if not value:
        return []
    if isinstance(value, list):
        ranges = value
    else:
        ranges = []
        for part in str(value).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            bounds = [b.strip() for b in part.replace("–", "-").split("-")]
            if len(bounds) not in (1, 2) or not all(b.isdigit() for b in bounds):
                raise ExtractionError(f"pages: {part!r} is not a page or a range like 18-26")
            ranges.append([int(bounds[0]), int(bounds[-1])])
    out = []
    for r in ranges:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(n, int) for n in r)) \
                or not 1 <= r[0] <= r[1] <= MAX_PAGE:
            raise ExtractionError(f"pages: {r!r} is not a valid range")
        out.append([r[0], r[1]])
    return sorted(out)


def pages_text(ranges: list[list[int]]) -> str:
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def select_windows(windows: list, params: dict, kept_pages: set[int] | None) -> list[tuple[int, object]]:
    """(index in the whole document, window) for the windows the run reads: on the chosen pages, and on
    pages in the kept languages. A window without page numbers is read only when no page is chosen."""
    wanted = {p for a, b in params.get("pages") or [] for p in range(a, b + 1)}
    out = []
    for i, w in enumerate(windows):
        span = set(range(w.page_start, (w.page_end or w.page_start) + 1)) if w.page_start else set()
        if wanted and not span & wanted:
            continue
        if kept_pages is not None and span and not span & kept_pages:
            continue
        out.append((i, w))
    return out


ANCHOR_TEMPLATE = """Context for this document:
- It describes the part "{name}" of the asset "{asset}"{details}. When the text refers to this product in general (its product name or model, "the motor", "the drive", "the unit", "the device"), use exactly the entity name "{name}" with type Component.
- The text may also cover other models, sizes or variants: extract only what applies to this part.
- Existing entities of this asset. When the text clearly means one of them, use its exact name; otherwise name the entity as written in the text:
{existing}"""


def anchor_context(asset_id: str, anchor_id: str) -> dict | None:
    """The prompt's context block for the part a document describes, and the names that mean that part."""
    from . import services
    ctx = services.get_asset_context(asset_id)
    components, signals = [], []

    def walk(items):
        for c in items:
            components.append(c)
            signals.extend(c["signals"])
            walk(c["children"])
    walk(ctx["components"])
    signals += ctx["signals"]
    part = next((c for c in components if c["id"] == anchor_id), None)
    if part is None:
        raise ExtractionError(f"{anchor_id!r} is not a part of {asset_id!r}")
    p = part["properties"]
    details = ", ".join(str(v) for v in (p.get("subtype") or p.get("component_type"), p.get("manufacturer"), p.get("model"),
                                         p.get("part_number") and f"part number {p['part_number']}") if v)
    listed = lambda items, n=60: "; ".join(dict.fromkeys(items))[:3000] if items else "none"   # noqa: E731
    alarms = [f"{a['properties'].get('code') or a['name']} ({a['name']})" if a['properties'].get('code') else a["name"]
              for a in ctx["alarms"]]
    procedures = [pr["name"] for pr in ctx["procedures"]]
    existing = "\n".join([f"  Alarms: {listed(alarms)}", f"  Signals: {listed([s['name'] for s in signals])}",
                          f"  Procedures: {listed(procedures)}"])
    text = ANCHOR_TEMPLATE.format(name=part["name"], asset=ctx["asset"]["name"], details=f" ({details})" if details else "",
                                  existing=existing)
    generic = {normalize(v) for v in (p.get("model"), p.get("part_number"),
                                      f"{p.get('manufacturer', '')} {p.get('model', '')}".strip()) if v}
    return {"id": anchor_id, "name": part["name"], "text": text, "generic": generic - {""}}


# --- prompt + LLM ------------------------------------------------------------

def extractable(schema: Schema) -> tuple[dict, dict]:
    """Entity types and (pairs of) relationships the LLM may produce.

    Document / DocumentSection are evidence structure, written automatically when
    a triple is approved — the model is not asked to extract them.
    """
    types = {label: desc for label, desc in schema.entity_types.items() if label not in EVIDENCE_TYPES}
    rels = {}
    for rt in schema.relationship_types.values():
        if rt.name in SUGGESTED_ONLY:
            continue
        pairs = sorted((a, b) for a, b in rt.pairs if a in types and b in types)
        if pairs:
            rels[rt.name] = pairs
    return types, rels


def render_schema(schema: Schema) -> tuple[str, str]:
    types, rels = extractable(schema)
    entity_types = "\n".join(
        (f"- {label}: {desc}" if desc else f"- {label}")
        + (f" (subtypes: {', '.join(schema.subtypes[label])})" if schema.subtypes.get(label) else "")
        for label, desc in types.items())
    relationships = "\n".join(f"- {name}: " + ", ".join(f"{a} -> {b}" for a, b in pairs) for name, pairs in rels.items())
    return entity_types, relationships


def render_properties(schema: Schema) -> str:
    """The documented properties of the types the model may produce (schema v3), for {properties}."""
    types, _ = extractable(schema)
    lines = []
    for label in types:
        props = {k: v for k, v in (schema.properties.get(label) or {}).items() if k not in PROPERTY_SKIP}
        if props:
            lines.append(f"- {label}: " + "; ".join(f"{k} ({v})" if v else k for k, v in props.items()))
    return "\n".join(lines) or "- none"


PLACEHOLDERS = ("entity_types", "relationships", "properties", "document", "section", "text")


def render_template(template: str, **values: str) -> str:
    """Replace only the known {placeholders}; other braces (e.g. a JSON example) stay as written."""
    for name in PLACEHOLDERS:
        template = template.replace("{" + name + "}", values.get(name, ""))
    return template


def build_messages(preset: PromptPreset, schema: Schema, text: str, document: str, section: str,
                   context: str = "") -> list[dict]:
    """`context`: the anchor block (the part the document describes), added after the preset's text."""
    entity_types, relationships = render_schema(schema)
    values = {"entity_types": entity_types, "relationships": relationships, "properties": render_properties(schema),
              "document": document, "section": section or "-"}
    extra = f"\n\n{context}" if context else ""
    if "{text}" in preset.template:  # the template places the text itself
        return [{"role": "system", "content": render_template(preset.template, **values, text=text) + extra}]
    return [{"role": "system", "content": render_template(preset.template, **values) + extra}, {"role": "user", "content": text}]


def call_llm(messages: list[dict]) -> str:
    with usage.track("extract", f"dgx-{settings.DGX_CHAT_MODEL}") as call:
        data = _post_llm(messages)
        content = data["choices"][0]["message"]["content"] or ""
        call.from_response(data)
        call.estimate(usage.messages_text(messages), content)
    return content


def _post_llm(messages: list[dict]) -> dict:
    resp = requests.post(
        f"{settings.DGX_API_BASE.rstrip('/')}/v1/chat/completions",
        json={
            "model": settings.DGX_CHAT_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": settings.GRAPH_EXTRACT_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=settings.GRAPH_EXTRACT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def parse_triples(content: str) -> list[dict]:
    """Strict parse of the model output into raw triple dicts; raises ExtractionError."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ExtractionError("no JSON object in the model output")
    try:
        data = json.loads(match.group(0))
    except ValueError as exc:
        raise ExtractionError(f"invalid JSON: {exc}")
    triples = data.get("triples") if isinstance(data, dict) else None
    if not isinstance(triples, list):
        raise ExtractionError("JSON has no 'triples' list")
    out = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        fields = {k: str(t.get(k) or "").strip() for k in ("subject", "subject_type", "predicate", "object", "object_type")}
        if not all(fields.values()):
            continue  # incomplete triple: skip rather than fail the window
        try:
            confidence = max(0.0, min(1.0, float(t.get("confidence"))))
        except (TypeError, ValueError):
            confidence = None
        subtypes = {k: str(t.get(k) or "").strip()[:MAX_NAME] for k in ("subject_subtype", "object_subtype") if t.get(k)}
        props = {k: clean_properties(t.get(k)) for k in ("subject_properties", "object_properties")}
        out.append({**{k: v[:MAX_NAME] for k, v in fields.items()}, **subtypes, **{k: v for k, v in props.items() if v},
                    "confidence": confidence})
    return out


def clean_properties(value) -> dict:
    """The model's properties of one entity: snake_case names; text, numbers or lists of text."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, v in value.items():
        key = str(key).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", key) or key in PROPERTY_SKIP:
            continue
        if isinstance(v, bool):
            v = str(v).lower()
        if isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, list):
            items = [str(x).strip()[:300] for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:MAX_LIST]
            if items:
                out[key] = items
        elif isinstance(v, str) and v.strip():
            out[key] = v.strip()[:MAX_PROPERTY]
    return out


def merge_properties(old: dict, new: dict) -> dict:
    """Two finds of the same triple: lists are joined, a text found again is added once; nothing is dropped."""
    out = dict(old or {})
    for key, v in (new or {}).items():
        if key not in out:
            out[key] = v
        elif isinstance(out[key], list):
            out[key] = list(dict.fromkeys(out[key] + (v if isinstance(v, list) else [str(v)])))[:MAX_LIST]
        elif isinstance(out[key], str) and isinstance(v, str) and v not in out[key]:
            out[key] = f"{out[key]}; {v}"[:MAX_PROPERTY]
    return out


def split_text(text: str) -> list[str]:
    """Two halves at a line (or sentence) boundary near the middle; [] when too short to split."""
    if len(text) < 400:
        return []
    middle = len(text) // 2
    cut = text.rfind("\n", 0, middle)
    if cut < len(text) // 4:
        cut = text.rfind(". ", 0, middle) + 1
    if cut < len(text) // 4:
        cut = middle
    halves = [text[:cut].strip(), text[cut:].strip()]
    return halves if all(halves) else []


def extract_window(messages: list[dict]) -> list[dict]:
    """LLM call + strict parse, retrying once on malformed output."""
    try:
        return parse_triples(call_llm(messages))
    except ExtractionError as first:
        logger.warning("extraction: malformed output, retrying once: %s", first)
        retry = messages + [{"role": "user", "content": "Your previous answer was not valid. Reply with ONLY the JSON object."}]
        return parse_triples(call_llm(retry))


# --- entity matching -----------------------------------------------------------

def resolve_id(index: EntityIndex, label: str, name: str, scope_key: str) -> tuple[str, bool]:
    """(id, existing) — kept for callers that don't need the match details."""
    match = index.resolve(label, name, scope_key)
    return match.id, match.existing


def triple_key(mode: str, subject_name: str, subject_type: str, predicate: str, object_name: str, object_type: str) -> tuple:
    return (mode, normalize(subject_name), subject_type, predicate, normalize(object_name), object_type)


def scope_key_for(params: dict, doc) -> str:
    """Key prefix for proposed ids: the asset scope (bsk:asset:EXTR01 -> EXTR01) or the document key."""
    return params["asset_id"].split(":", 2)[2] if (params or {}).get("asset_id") else doc.doc_key


def part_scope_for(params: dict, index: EntityIndex, scope_key: str):
    """The part a run's document describes, for part-scoped ids and matching (P21); None without one."""
    anchor = (params or {}).get("anchor_id")
    return index.part_scope(anchor, scope_key) if anchor and settings.GRAPH_PART_SCOPED_IDS else None


# --- the job -------------------------------------------------------------------

def run_extract(job) -> None:
    """Worker entry point for an extract job (ingestion.library.run_job dispatches here)."""
    from ingestion import library
    from ingestion.models import Document

    doc = Document.objects.get(pk=job.document_id)
    params = job.params
    Document.objects.filter(pk=doc.pk).update(graph_status=Document.PipelineStatus.RUNNING, graph_error="")

    if doc.parse_status != Document.ParseStatus.PARSED or not library.parsed_path(doc).exists():
        library.heartbeat(job, message="parsing first")
        library.run_parse(job)
        doc.refresh_from_db()
    if params.get("facet_runs"):
        _run_facets(job, doc, params)
    else:
        _extract(job, doc, params)


def _run_facets(job, doc, params: dict) -> None:
    """A multi-facet run (P22c-0): each facet in turn picks its pages and extracts them; one result per facet.
    A facet that fails (the DGX down, no text) doesn't stop the others; the job fails only if every facet that
    was tried failed."""
    from ingestion import library

    from . import page_suggest

    runs = params["facet_runs"]
    results: list[dict] = []
    for n, key in enumerate(runs, 1):
        run = FACET_RUN_BY_KEY[key]
        label = f"{run['label']} ({n}/{len(runs)})"
        out = {"key": key, "label": run["label"], "state": "", "pages": "", "staged": 0, "failed_calls": 0, "left": 0, "note": ""}
        results.append(out)
        preset = PromptPreset.objects.filter(mode="schema", name=run["preset"]).first()
        if preset is None:
            out.update(state="failed", note=f"prompt {run['preset']!r} not found")
            continue
        reviewed = CandidateTriple.objects.filter(document=doc, preset_name=preset.name).exclude(
            status=CandidateTriple.Status.PENDING).count()
        if reviewed and not params.get("rerun_reviewed"):
            out.update(state="skipped", note=f"{reviewed} triples already reviewed")
            continue
        library.heartbeat(job, 0, 0, f"{label}: choosing pages")
        try:
            suggestion = page_suggest.suggest(doc, preset=preset, asset_id=params["asset_id"],
                                              anchor_id=params["anchor_id"],
                                              follow_languages=params.get("follow_languages", True))
        except (ValueError, requests.RequestException) as exc:
            logger.warning("facet run %s: page choice failed doc=%s: %s", key, doc.doc_key, exc)
            out.update(state="failed", note=f"choosing pages failed: {str(exc)[:150] or exc.__class__.__name__}")
            continue
        if not suggestion["pages"]:
            out.update(state="none", note=(suggestion.get("note") or "")[:300])
            continue
        out["pages"] = suggestion["pages"]
        leave = sorted({t for other in runs if other != key for t in FACET_RUN_BY_KEY[other]["owns"]})
        sub = {**params, "mode": "schema", "presets": {"schema": preset.pk}, "pages": parse_pages(suggestion["pages"]),
               "facet_runs": [], "leave_types": leave, "keep_links": list(run.get("keeps") or ())}
        if run.get("chunking") and not params.get("chunking_custom"):
            from ingestion.chunking import resolve_params
            sub["chunking"] = {"strategy": run["chunking"]["strategy"],
                               "params": resolve_params(run["chunking"]["strategy"], run["chunking"]["params"])}
        try:
            result = _extract(job, doc, sub, label=label)
        except ExtractionError as exc:
            out.update(state="failed", note=str(exc)[:300])
            continue
        notes = [f"{result['skipped']} already reviewed" if result["skipped"] else "",
                 f"{result['left']} left to {', '.join(leave)} run" if result["left"] else ""]
        out.update(state="done", staged=result["staged"], failed_calls=result["failed"], left=result["left"],
                   note="; ".join(n for n in notes if n))

    def short(r):
        return {"done": f"{r['staged']}" + (f" ({r['failed_calls']} failed calls)" if r["failed_calls"] else ""),
                "skipped": "skipped", "none": "nothing found"}.get(r["state"], "failed")
    summary = " · ".join(f"{r['label']}: {short(r)}" for r in results)
    job.params = {**job.params, "facet_results": results}
    type(job).objects.filter(pk=job.pk).update(params=job.params)
    library.heartbeat(job, len(runs), len(runs), summary)
    tried = [r for r in results if r["state"] != "skipped"]
    if tried and all(r["state"] == "failed" for r in tried):
        raise ExtractionError("every facet failed: " + "; ".join(f"{r['label']}: {r['note']}" for r in tried))


def _extract(job, doc, params: dict, label: str = "") -> dict:
    """One prompt over the chosen pages: a single run, or one facet of a multi-facet run (`label` prefixes its
    progress messages). Returns {staged, failed, skipped}."""
    from ingestion import library
    from ingestion.chunking import chunk_parsed
    from ingestion.models import Document

    def beat(done=None, total=None, message=None):
        library.heartbeat(job, done, total, f"{label}: {message}" if label and message else message)

    parsed = library.load_parsed(doc)
    windows = chunk_parsed(parsed, params["chunking"]["strategy"], params["chunking"]["params"])
    kept = None
    if params.get("follow_languages", True) and doc.languages:
        from ingestion import languages
        kept = languages.kept_pages((parsed.get("document") or {}).get("md_content") or "", doc.languages)
    all_windows = len(windows)
    selected = select_windows(windows, params, kept)
    table_parts: list[tuple[int, object]] = []
    if params.get("tables_by_row", True):
        from . import tables
        md = (parsed.get("document") or {}).get("md_content") or ""
        wanted = {p for a, b in params.get("pages") or [] for p in range(a, b + 1)}
        for n, table in enumerate(tables.find_tables(md)):
            if (wanted and table.page not in wanted) or (kept is not None and table.page not in kept):
                continue
            table_parts += tables.windows(table, n)
        if table_parts:     # the text windows without their table lines; a window left empty is not read
            stripped = [(i, w, tables.strip_tables(w.text)) for i, w in selected]
            selected = [(i, w if text == w.text else replace(w, text=text)) for i, w, text in stripped
                        if text == w.text or len(text.split()) >= 8]
    if all_windows and not selected and not table_parts:
        raise ExtractionError("no text on the chosen pages" + (" in the kept languages" if kept is not None else ""))
    anchor = anchor_context(params["asset_id"], params["anchor_id"]) if params.get("anchor_id") else None
    drawing_figures: list[dict] = []
    if params.get("include_drawings"):
        from ingestion import figures as figure_store
        from . import drawings
        # Only the drawings on the chosen pages and in the kept languages: all 15 of the EL5000 manual's drawings
        # were sent for pages 13-16, and a wiring drawing on page 12 gave 4 wrong triples (2026-10-16).
        wanted_pages = {p for a, b in params.get("pages") or [] for p in range(a, b + 1)}
        drawing_figures = [f for f in drawings.eligible(figure_store.load(doc))
                           if (not wanted_pages or f.get("page") in wanted_pages) and (kept is None or f.get("page") in kept)]
    if not selected and not table_parts and not drawing_figures:
        raise ExtractionError("the document produced no text windows")

    schema = registry.active_schema()
    presets = {m: PromptPreset.objects.get(pk=pk) for m, pk in params["presets"].items()}
    index = EntityIndex.load()
    scope_key = scope_key_for(params, doc)
    part = part_scope_for(params, index, scope_key)
    total = (len(selected) + len(table_parts) + len(drawing_figures)) * len(presets)
    scope_note = "" if len(selected) == all_windows else f" of {all_windows}" + (
        f" (pages {pages_text(params['pages'])})" if params.get("pages") else "") + (
        f" ({', '.join(doc.languages)})" if kept is not None else "")
    beat(0, total, f"extracting from {len(selected)}{scope_note} windows"
                      + (f", {len(table_parts)} table part(s)" if table_parts else "")
                      + (f" and {len(drawing_figures)} drawing(s)" if drawing_figures else "") + f" ({', '.join(presets)})")

    staged: dict[tuple, CandidateTriple] = {}
    found: dict[int, list[Evidence]] = {}   # id(triple) -> one Evidence per window it was found in
    reviewed = {  # already approved/rejected in an earlier run: not staged again
        triple_key(t.mode, t.subject_name, t.subject_type, t.predicate, t.object_name, t.object_type)
        for t in CandidateTriple.objects.filter(document=doc).exclude(status=CandidateTriple.Status.PENDING)
    }
    done = failed = skipped = 0
    left: set[tuple] = set()      # distinct triples left to another facet run
    leave_types = set(params.get("leave_types") or ())   # owned by another facet run of the same job (P22c-0)
    keep_links = set(params.get("keep_links") or ())     # a facet run stages only its own links (empty: all)

    def stage(raw, mode, window, i, preset, make_evidence, model=None):
        """Stage the raw triples found in one place (a text window or a figure)."""
        nonlocal skipped
        for t in raw:
            if mode == "schema" and (t["subject_type"] in EVIDENCE_TYPES or t["object_type"] in EVIDENCE_TYPES):
                continue  # evidence structure is written on approval, not extracted
            if anchor:   # the product's own designations mean the anchor part
                for end in ("subject", "object"):
                    if t[f"{end}_type"] == "Component" and normalize(t[end]) in anchor["generic"]:
                        t = {**t, end: anchor["name"]}
            candidate = _candidate(t, mode, schema, index, scope_key, doc, job, window, i, preset, model=model, part=part)
            key = triple_key(mode, candidate.subject_name, candidate.subject_type, candidate.predicate,
                             candidate.object_name, candidate.object_type)
            if (candidate.subject_type in leave_types or candidate.object_type in leave_types
                    or (keep_links and candidate.predicate not in keep_links)):
                left.add(key)      # checked after staging: a part's limit becomes a Specification there
                continue
            if key in reviewed:
                skipped += 1
                continue
            if key in staged:
                if (t["confidence"] or 0) > (staged[key].confidence or 0):
                    staged[key].confidence = t["confidence"]
                staged[key].subject_props = merge_properties(staged[key].subject_props, candidate.subject_props)
                staged[key].object_props = merge_properties(staged[key].object_props, candidate.object_props)
            else:
                staged[key] = candidate
            places = found.setdefault(id(staged[key]), [])
            if all(e.chunk_index != i for e in places):   # same place twice = one place
                places.append(make_evidence(t))
                staged[key].occurrences = len(places)

    def read(i, text, section, mode, preset, halves_left=1, quiet=False) -> list[dict] | None:
        """The triples of one text; None when it failed. An invalid answer (often cut off at the output limit)
        is read again as two halves, once."""
        messages = build_messages(preset, schema, text, doc.filename, section,
                                  context=anchor["text"] if anchor and mode == "schema" else "")
        for attempt in (1, 2):
            # Heartbeat before every attempt: one attempt is at most two LLM calls
            # (malformed-output retry), which stays under the stale-job threshold.
            if not quiet:   # parallel windows: only the main thread reports (and writes the job row)
                beat(done, total, None if attempt == 1 else f"window {i + 1}: DGX slow, retrying")
            try:
                return extract_window(messages)
            except requests.RequestException as exc:  # timeout / connection: the shared DGX is busy
                if attempt == 2:
                    logger.warning("extraction window failed doc=%s window=%d mode=%s: %s", doc.doc_key, i, mode, exc)
            except Exception as exc:  # malformed output after the retry etc.: one bad window doesn't fail the run
                halves = split_text(text) if halves_left and isinstance(exc, ExtractionError) else []
                logger.warning("extraction window failed doc=%s window=%d mode=%s: %s%s", doc.doc_key, i, mode, exc,
                               "; reading it in two halves" if halves else "")
                if not halves:
                    return None
                parts = [read(i, half, section, mode, preset, halves_left - 1, quiet) for half in halves]
                return None if all(p is None for p in parts) else [t for p in parts if p for t in p]
        return None

    # GRAPH_EXTRACT_PARALLEL windows go to the DGX at once (vLLM batches them); results are staged in the document's
    # order by this thread, which alone reports progress, so a run gives the same triples whatever the setting.
    calls = [(i, window, mode, preset) for i, window in selected + table_parts for mode, preset in presets.items()]
    parallel = max(1, int(settings.GRAPH_EXTRACT_PARALLEL))

    def work(call):
        i, window, mode, preset = call
        try:
            return read(i, window.text, " > ".join(window.section_path), mode, preset, quiet=parallel > 1)
        finally:
            if parallel > 1:
                from django.db import connection
                connection.close()      # each thread's own connection (the usage records)

    executor = ThreadPoolExecutor(max_workers=parallel) if parallel > 1 else None
    try:
        for (i, window, mode, preset), raw in zip(calls, executor.map(work, calls) if executor else map(work, calls)):
            if raw is None:
                failed += 1
                raw = []
            stage(raw, mode, window, i, preset,
                  lambda t, window=window, i=i, preset=preset: _evidence(doc, window, i, preset, schema, t, params))
            done += 1
            beat(done, total)
    finally:
        if executor:   # a cancelled job stops sending the windows still queued
            executor.shutdown(wait=False, cancel_futures=True)

    drawings_note = ""
    if drawing_figures:
        from gpu import orchestrator as gpu
        preset = SimpleNamespace(name=drawings.PROMPT_NAME, version=drawings.PROMPT_VERSION)
        try:
            for figure in drawing_figures:
                place = drawings.FIGURE_CHUNK_BASE + figure["index"]
                for mode in presets:
                    beat(done, total, f"drawing: figure {figure['index'] + 1} (p.{figure['page']})")
                    raw = []
                    try:
                        raw = drawings.extract(doc, figure, mode, schema)
                    except gpu.GpuError:
                        raise
                    except Exception as exc:  # invalid answer after the retry etc.: one bad figure doesn't fail the run
                        failed += 1
                        logger.warning("drawing triples failed doc=%s figure=%d mode=%s: %s", doc.doc_key, figure["index"], mode, exc)
                    stage(raw, mode, drawings.window_for(figure), place, preset,
                          lambda t, figure=figure: drawings.evidence_for(doc, figure, schema, t), model=settings.VLM_MODEL)
                    done += 1
        except gpu.GpuError as exc:  # BSK asleep / busy: the text triples are still staged
            drawings_note = "drawings were not read: the BSK PC was not reachable"
            logger.warning("drawing triples skipped doc=%s: %s", doc.doc_key, exc)

    with transaction.atomic():
        # A new run replaces the document's pending triples of the prompts it ran (P22c-0: per prompt, not per mode:
        # a Ratings run no longer erases a Maintenance run's pending triples); reviewed ones are kept.
        ran = {p.name for p in presets.values()} | ({drawings.PROMPT_NAME} if drawing_figures else set())
        CandidateTriple.objects.filter(document=doc, status=CandidateTriple.Status.PENDING, mode__in=list(presets),
                                       preset_name__in=ran).delete()
        facets.stamp(staged.values(), registry.active_schema())
        triples = CandidateTriple.objects.bulk_create(staged.values())
        evidence.attach(triples, found)
        evidence.drop_orphans(document=doc)
    Document.objects.filter(pk=doc.pk).update(
        graph_status=Document.PipelineStatus.DONE, graph_updated_at=timezone.now(),
        graph_error="; ".join(n for n in (f"{failed} window call(s) failed" if failed else "", drawings_note) if n),
    )
    note = f", {skipped} already reviewed" if skipped else ""
    beat(total, total, f"{len(staged)} triples staged for review{note}")
    logger.info("extraction done doc=%s windows=%d/%d table_parts=%d triples=%d failed_calls=%d anchor=%s", doc.doc_key,
                len(selected), all_windows, len(table_parts), len(staged), failed, params.get("anchor_id") or "-")
    return {"staged": len(staged), "failed": failed, "skipped": skipped, "left": len(left)}


def _evidence(doc, window, i, preset, schema, t, params) -> Evidence:
    return Evidence(
        source_kind=Evidence.Kind.TEXT, document=doc,
        page_start=window.page_start, page_end=window.page_end, section_path=window.section_path,
        chunk_index=i, window=params.get("chunking") or {}, extractor="llm-text",
        model=settings.DGX_CHAT_MODEL, prompt=f"{preset.name} v{preset.version}", schema_version=schema.version,
        excerpt=window.text[:EVIDENCE_CHARS], confidence=t.get("confidence"),
    )


# A limit's properties as a rating's (P22a): the values are kept, under the Specification's names.
RATING_PROPS = {"low": "min", "high": "max", "target": "value", "unit": "unit", "basis": "basis", "group": "condition"}


def rating_props(props: dict | None) -> dict:
    """An OperatingLimit's properties as a Specification's ({"low": "-50", "high": "600"} -> min / max)."""
    out = {}
    for key, value in (props or {}).items():
        if key in RATING_PROPS and value not in (None, ""):
            out[RATING_PROPS[key]] = value
    return out


def is_part_rating(subject_type: str, predicate: str, object_type: str, schema: Schema) -> bool:
    """A part's limit the schema refuses but can hold as a rating (Component -HAS_LIMIT-> OperatingLimit, v4)."""
    return (subject_type == "Component" and predicate == "HAS_LIMIT" and object_type == "OperatingLimit"
            and "HAS_SPEC" in schema.relationship_types
            and ("Component", "Specification") in schema.relationship_types["HAS_SPEC"].pairs
            and ("Component", "OperatingLimit") not in schema.relationship_types["HAS_LIMIT"].pairs)


def _candidate(t, mode, schema, index, scope_key, doc, job, window, i, preset, model=None, part=None) -> CandidateTriple:
    issue = ""
    subject_id = object_id = ""
    subject_existing = object_existing = False
    matches: dict = {}
    props: dict = {}
    auto_fix = ""
    if mode == "schema":
        canonical_limit = schema.relationship_for(t["predicate"], "Signal", "OperatingLimit")
        if canonical_limit == "HAS_LIMIT" and is_part_rating(t["subject_type"], "HAS_LIMIT", t["object_type"], schema):
            # A part's rating read as a limit (the overnight batch's 29, 2026-10-17): staged as a Specification.
            auto_fix = "a part's rating: was Component -HAS_LIMIT-> OperatingLimit"
            t = {**t, "predicate": "HAS_SPEC", "object_type": "Specification",
                 "object_properties": rating_props(t.get("object_properties"))}
        # A synonym of an allowed relationship ("drives", "is monitored by") becomes the schema type.
        canonical = schema.relationship_for(t["predicate"], t["subject_type"], t["object_type"])
        if canonical:
            t = {**t, "predicate": canonical}
        else:
            # A link the schema allows only the other way round is flipped (the ACS880 fault list came as
            # "Alarm -MITIGATED_BY-> Procedure", 2026-10-16); the reviewer sees it was.
            flipped = schema.reversed_relationship(t["subject_type"], t["predicate"], t["object_type"])
            if flipped:
                auto_fix = f"flipped from {t['subject_type']} -{t['predicate']}-> {t['object_type']}"[:255]
                t = {**_swap_ends(t), "predicate": flipped}
        for end in ("subject", "object"):
            subtype = schema.subtype_for(t[f"{end}_type"], t.get(f"{end}_subtype", ""))
            if subtype:
                props.setdefault(end, {})["subtype"] = subtype
            # Documented properties only (schema v3); a schema without them takes none.
            known = schema.properties.get(t[f"{end}_type"]) or {}
            given = {k: v for k, v in (t.get(f"{end}_properties") or {}).items() if k in known}
            if given:
                props.setdefault(end, {}).update(given)
        for end in ("subject", "object"):      # P22c: one form of a parameter code ("P310", "p0310", "p1070[0]")
            p = props.get(end) or {}
            if t[f"{end}_type"] == "Parameter" and p.get("code"):
                code, position = normalize_code(str(p["code"]))
                p["code"] = code
                if position and not p.get("data_set"):
                    p["data_set"] = position
        if t["subject_type"] not in schema.entity_types or t["object_type"] not in schema.entity_types:
            issue = f"type not in schema ({t['subject_type']} / {t['object_type']})"
        else:
            try:
                schema.check_relationship(t["subject_type"], t["predicate"], t["object_type"])
            except Exception as exc:
                issue = str(exc)[:255]
            if t["predicate"] in SUGGESTED_ONLY:
                issue = f"{t['predicate']} is suggested by the Studio and approved, not extracted"
            try:
                sm = index.resolve(t["subject_type"], t["subject"], scope_key, part=part, **_keyed(t, "subject", props, scope_key, part))
                om = index.resolve(t["object_type"], t["object"], scope_key, part=part, **_keyed(t, "object", props, scope_key, part))
                subject_id, subject_existing, object_id, object_existing = sm.id, sm.existing, om.id, om.existing
                matches = {"subject_match": sm.kind, "subject_candidates": sm.candidates,
                           "object_match": om.kind, "object_candidates": om.candidates}
            except ValueError as exc:  # a name with no usable id characters
                issue = str(exc)[:255]
    return CandidateTriple(
        document=doc, job=job, mode=mode,
        subject_name=t["subject"], subject_type=t["subject_type"], subject_id=subject_id, subject_existing=subject_existing,
        predicate=t["predicate"],
        object_name=t["object"], object_type=t["object_type"], object_id=object_id, object_existing=object_existing,
        confidence=t["confidence"], issue=issue, auto_fix=auto_fix,
        chunk_index=i, page_start=window.page_start, page_end=window.page_end, section_path=window.section_path,
        evidence_text=window.text[:EVIDENCE_CHARS],
        model=model or settings.DGX_CHAT_MODEL, preset_name=preset.name, preset_version=preset.version,
        schema_version=schema.version, **matches,
        subject_props=props.get("subject", {}), object_props=props.get("object", {}),
    )


_SIEMENS_CODE = re.compile(r"^([pr])\s*0*(\d{1,5})\s*(?:\[(.*?)\]|[.-](\w+))?$", re.I)


def normalize_code(code: str) -> tuple[str, str]:
    """A parameter code in one form, and its index: "P310" -> ("p0310", ""), "p1070[0]" -> ("p1070", "0"),
    "r0021" stays. Other makers' codes ("01.01", "99.06") are kept as written."""
    code = code.strip()
    m = _SIEMENS_CODE.match(code)
    if not m:
        return code, ""
    letter, number, index, suffix = m.groups()
    return f"{letter.lower()}{int(number):04d}", (index or suffix or "").strip()


def _keyed(t: dict, end: str, props: dict, scope_key: str, part) -> dict:
    """A parameter's id comes from its code, a terminal's from its connector and terminal (P22c): the same p1082 on
    twenty pages is one node, whatever name each page gives it. Other types: by name, as before."""
    from .identity import PART_LABELS
    label, p = t[f"{end}_type"], props.get(end) or {}
    if label == "Parameter" and p.get("code"):
        key, tag = str(p["code"]).strip(), str(p["code"]).strip()
    elif label == "IOPoint" and p.get("terminal"):
        key, tag = f"{p.get('connector', '')} {p['terminal']}".strip(), None
    else:
        return {}
    where = part.key if part is not None and label in PART_LABELS else scope_key
    try:
        return {"proposed": proposed_id(label, key, where), "tag": tag}
    except ValueError:
        return {}


def _swap_ends(t: dict) -> dict:
    """`t` with the subject's name, type, subtype and properties as the object's, and the other way round."""
    ends = {f"{end}{suffix}" for end in ("subject", "object") for suffix in ("", "_type", "_subtype", "_properties")}
    out = {k: v for k, v in t.items() if k not in ends}
    for a, b in (("subject", "object"), ("object", "subject")):
        for suffix in ("", "_type", "_subtype", "_properties"):
            if b + suffix in t:
                out[a + suffix] = t[b + suffix]
    return out
