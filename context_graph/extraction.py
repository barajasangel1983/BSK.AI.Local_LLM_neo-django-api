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

from . import registry, repository, review_checks
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
# Product families and device models (schema v7, P26a) come from the BOM's model list, not from a manual's text: they
# are not offered to the model, and a model's triple with one is staged with an issue, except the family or model the
# run is anchored on (P26b; the model names it as a Component and it is mapped to the anchor when staged).
NOT_EXTRACTED = ("ProductFamily", "DeviceModel")
ANCHOR_LEVELS = ("part", "model", "family", "auto")
# P26 D2: with "auto", a part's ratings are read for its model (the row of its size), everything else for its family.
AUTO_LEVEL = {"ratings": "model"}
# "Barrel zone 1 power controller HAS_COMPONENT Barrel zone 1 power controller" (2898, approved 2026-10-17): a part
# that is its own sub-part left the component tree. Staged with this issue; never approved.
SELF_LINK = "the link would point to itself (subject and object are the same entity)"

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
  - A Procedure is what someone does on the product where it is installed. Not one: what the manufacturer does
    at the factory, and text that explains what a test or a feature is for. One task is one Procedure, even when
    the text describes it twice.
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
- A sub-part (bearing, fan, shaft seal, indicator LED) only when the text names it as a part built into this
  product. Not sub-parts: accessories, mounting adapters, and protection the text recommends installing outside
  it (fuses, circuit breakers), even when the text gives their makers or order numbers.
- Name each item as the text names it, never after a figure's callout number ("LED 1", "item 3").
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
  - Name it after its own row or line, with the row's label as the text writes it, never after a table's title.
    Never "Row 3" or "Column 2" in a name.
- A Specification needs a value, or a min/max, that the text states for it. An empty cell ("not required", "–",
  "n/a", a symbol) or a condition alone (a voltage, a temperature) is not a value: skip the row. A yes or no and
  a value in words are values: keep them ("yes", "No, external fuse required", "DIN rail", a list of approvals).
- One Specification holds the values of one row or one statement. Never combine the values of different rows or
  of different features into one. When one row gives a value for each of several conditions (input voltages,
  temperatures, loads, columns such as "120 V AC" and "230 V AC"), each value is its own Specification with its
  condition in "condition": keep them all.
- A test condition (a frequency band, a test voltage) and a pass/fail criterion are the "condition" of the result
  they belong to, never a Specification of their own. Each test level with its own result is its own
  Specification, with its level in "condition".
- A value or range the user can set (a setting, a configuration or signal option, a threshold, a factory setting)
  is a parameter, not a rating: return nothing for it. What the device can do is still a rating, even when the
  user picks within it (a selectable resolution of 12-16 bits, a supported range).
- Variants: when the text gives values for several variants of the product (sizes, frame sizes, versions; not
  operating conditions such as a voltage or a temperature), use only the variant
  the context names; if the context names none, give each value with its variant in "condition".
- One value per property: never list alternatives in one property. Values for different conditions are not
  alternatives: each is its own Specification (above).
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
- A parameter is a fact here only when the text does one of these with it:
  - tells the reader to set, enter, select, change or check it ("Set p1082 to …", "Set parameter 99.04", "Enter the
    measuring range in the setup menu");
  - lists it as a step or a row of commissioning (quick commissioning, basic settings);
  - names it as the function of a terminal or an interface (an input's function, the fieldbus address);
  - names it in an alarm, a fault or a procedure.
  A parameter number that only appears as a label in a figure, a function diagram, a block diagram, or a list or
  index of parameters is not a fact here: leave it out. A readout only when the text says to check it.
  - A parameter the text tells you to set but names without its number ("set the controller parameters KP, TI and
    Td", "select the application class") is still a fact: keep it, named as the text names it, with "code" empty
    unless its number is written next to it in the same text. A number shown on an operator panel's display is its
    code as written.
  - Each is ONE Parameter: <device> HAS_PARAMETER <parameter>, named exactly as the text names it, in at most six
    words ("Maximum speed", "Damping"), with "code" exactly as the manual writes it (p1082, 01.01, 2.03, a menu
    path) and the subtype Setting (can be changed) or Readout (display only). Only the properties the text states: "default", "min", "max", "unit",
    "access", "data_set". No descriptions.
  - A Parameter is one named setting the text gives, never a category or a general phrase ("the signaling
    parameters", "the settings").
- Terminals: each is ONE IOPoint per signal channel of the device: a digital or analog input or output, a relay
  output, a sensor input, a supply output or a supply input. <device> HAS_IO <terminal>, named as the manual names
  that channel (for example "DI 0", "DI1", "Current output 1", "Relay 2"); when the text gives a short code and a
  name together ("Rem, remote input"), use the name and put the code in "terminal". Its properties, one value each:
  - "terminal": all its terminal numbers, with each contact's role in brackets when the text gives it
    ("18 (NC), 19 (NO), 20 (COM)", "3 (+), 4 (−)");
  - "connector": the terminal strip or connector the channel is on;
  - "signal_type" (24 V DC, 0-10 V, 4-20 mA, relay contact) and "direction" when given.
  Not an IOPoint, even when the text lists it with the terminals:
  - one contact or pin of a channel (+ or −, NO, NC, COM): it goes in that channel's "terminal";
  - a terminal strip or a connector as a whole (a name such as "terminal strip …", "connector …" or only a strip
    designation): never return it as an IOPoint or a Component; put it in the "connector" of each channel on it;
  - a bit of a control word or a status word, and a signal inside the device;
  - an LED, a display or a switch on the device;
  - a terminal named only by its kind ("a digital input", "an input for an external fault"), and a figure's
    callout number.
- <terminal> CONFIGURED_BY <parameter> only when the text says this parameter sets this terminal's function or
  type. A parameter in a neighbouring row or line of a table is not that.
- Variants: when the text shows terminals, connectors or values for several variants (sizes, frame sizes,
  versions), use only those of the variant the context names. If the context names no variant, keep only what all
  the variants share.
- One value per property: when the text gives several, take the one for this device; never list alternatives in
  one property.
- Interfaces: <device> COMMUNICATES_VIA <interface>, ONE per communication protocol or port, named by the protocol as
  the manual writes it (for example PROFINET, Modbus RTU, HART, EtherNet/IP, IO-Link, a service or operator panel
  port), with "protocol", "connector", "address_range", "telegram" when given; data formats (telegrams, profiles,
  register maps) are its properties. Configuration software and apps are not interfaces.
- A commissioning step that sets a parameter: <procedure> SETS <parameter> (the procedure as the text names it).
- Numbered presets of what the inputs and outputs do (macros, application presets, default settings): only the
  parameter that selects them as ONE Parameter, with the presets in "default" or "min"/"max" as the text gives
  them. Not each preset's terminal assignments: only one preset is active on a device.
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
PROPERTY_SKIP = {"id", "name", "subtype", "source", "aliases", "evidence_ids", "created_by_triple", "status", "facets",
                 "configuration", "configuration_basis"}   # an installed part's options come from the BOM, not a manual
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
    anchor_level = params.get("anchor_level") or "part"
    if anchor_level not in ANCHOR_LEVELS:
        raise ExtractionError(f"anchor_level must be one of {', '.join(ANCHOR_LEVELS)}, got {anchor_level!r}")
    if anchor_id:
        from .device_models import kind_of
        if not (is_valid_id(anchor_id) and kind_of(anchor_id)):
            raise ExtractionError("anchor_id must be an installed part, a device model or a product family "
                                  f"(bsk:component: / bsk:device-model: / bsk:product-family:), got {anchor_id!r}")
        if kind_of(anchor_id) == "part" and not asset_id:
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
    anchor_part, facet_anchors = "", {}
    if anchor_id and anchor_level != "part":
        anchor_part = params.get("anchor_part") or (anchor_id if anchor_id.startswith("bsk:component:") else "")
        anchor_id, facet_anchors = _anchor_for_level(anchor_part or anchor_id, anchor_level, facet_runs, presets)
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
        # P26b: the level the run was asked for, the installed part it was chosen from, and with "auto" and facet runs
        # the anchor of each facet run (Ratings on the model, the rest on the family).
        "anchor_level": anchor_level,
        "anchor_part": anchor_part,
        "facet_anchors": facet_anchors,
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


def _anchor_for_level(start: str, level: str, facet_runs: list, presets: dict) -> tuple[str, dict]:
    """The anchor of a run asked for at `level` from `start` (an installed part, a model or a family), and with
    "auto" and facet runs each facet run's anchor."""
    from .device_models import levels, pick
    try:
        found = levels(start)
    except (ValueError, GraphUnavailable) as exc:
        raise ExtractionError(f"can't find the model or family of {start}: {exc}") from exc
    try:
        if level == "auto" and facet_runs:
            anchors = {key: pick(found, AUTO_LEVEL.get(key, "family"))["id"] for key in facet_runs}
            return start, anchors
        if level == "auto":     # one prompt: the Ratings prompt reads the model, any other the family
            ratings = FACET_RUN_BY_KEY["ratings"]["preset"]
            level = "model" if PromptPreset.objects.filter(pk=presets.get("schema"), name=ratings).exists() else "family"
        return pick(found, level)["id"], {}
    except ValueError as exc:
        raise ExtractionError(str(exc)) from exc


def anchors_of(params: dict) -> list[str]:
    """Every anchor a job's runs use (one, or one per facet run with "auto")."""
    return list(dict.fromkeys([*(params.get("facet_anchors") or {}).values()] or
                              ([params["anchor_id"]] if params.get("anchor_id") else [])))


def reviewed_on_parts(doc, anchor_id: str) -> str:
    """P26b D1: why a family or model run on `doc` is refused, or "". A manual already reviewed on the family's
    installed parts would give every fact a second copy; those facts move to the family in P26c first."""
    from django.db.models import Q

    from .device_models import kind_of, levels
    if not anchor_id or kind_of(anchor_id) == "part":
        return ""
    parts = {p["id"]: p["name"] for p in levels(anchor_id)["parts"]}
    if not parts:
        return ""
    done = (CandidateTriple.objects.filter(document=doc).exclude(status=CandidateTriple.Status.PENDING)
            .filter(Q(anchor_id__in=list(parts)) | Q(anchor_id="", subject_id__in=list(parts))
                    | Q(anchor_id="", object_id__in=list(parts))))
    counts: dict[str, int] = {}
    for anchor, subject, obj in done.values_list("anchor_id", "subject_id", "object_id"):
        part = anchor if anchor in parts else subject if subject in parts else obj
        counts[part] = counts.get(part, 0) + 1
    if not counts:
        return ""
    on = ", ".join(f"{parts[p]} ({n})" for p, n in sorted(counts.items(), key=lambda pn: parts[pn[0]]))
    return (f"{doc.filename} was already reviewed on {on}: those facts move to the family in P26c; "
            "until then run this manual on the part")


ANCHOR_TEMPLATE = """Context for this document:
- It describes the part "{name}" of the asset "{asset}"{details}. When the text refers to this product in general (its product name or model, "the motor", "the drive", "the unit", "the device"), use exactly the entity name "{name}" with type Component.
- The text may also cover other models, sizes or variants: extract only what applies to this part.{configuration}
- Existing entities of this asset. When the text clearly means one of them, use its exact name; otherwise name the entity as written in the text:
{existing}"""


PRODUCT_TEMPLATE = """Context for this document:
- It describes the {what} "{name}"{details}. When the text refers to this product in general (its series, model or product name, "the drive", "the unit", "the device"), use exactly the entity name "{name}" with type Component.
- The text may also cover other {others}: extract only what applies to this {what}.{configuration}
- Existing entities of this {what}. When the text clearly means one of them, use its exact name; otherwise name the entity as written in the text:
{existing}"""


def _listed(items) -> str:
    return "; ".join(dict.fromkeys(items))[:3000] if items else "none"


def _variant_line(kind: str, anchor_props: dict, models: list[dict]) -> str:
    """P26d (D3 of P26b-2): the variant of the product in use (frame size, size class), from the device models'
    `variant`, so the prompts' variant rule can keep that variant's values and leave the others'."""
    if kind == "model":
        variant = str(anchor_props.get("variant") or "").strip()
        return f"\n- Its variant: {variant}" if variant else ""
    variants = {m["name"]: str(m["properties"].get("variant") or "").strip() for m in models}
    if not any(variants.values()):
        return ""
    if len(set(variants.values())) == 1:
        return f"\n- The variant of its models in use: {next(iter(variants.values()))}"
    return "\n- The variants of its models in use: " + "; ".join(f"{n}: {v or 'not given'}" for n, v in variants.items())


def product_context(anchor_id: str) -> dict:
    """The prompt's context block for a run anchored on a product family or a device model (P26b): what it is, its
    models or family, its existing facts by their exact names, and the names that mean it (its own, its models',
    designations and order codes). Mentions of it are mapped to the anchor itself (type ProductFamily/DeviceModel)."""
    from .device_models import kind_of, levels
    from .identity import OWNING
    found = levels(anchor_id)
    kind = kind_of(anchor_id)
    anchor = found[kind]
    p = anchor["properties"]
    maker = p.get("manufacturer") or ""
    if kind == "family":
        models = [m["name"] for m in found["models"]]
        details = ", ".join(x for x in (maker, models and f"models: {'; '.join(models)}") if x)
        named = [anchor, *found["models"]]
    else:
        details = ", ".join(x for x in (maker, p.get("designation") and f"designation {p['designation']}",
                                        p.get("order_code") and f"order code {p['order_code']}",
                                        found["family"] and f"of the family {found['family']['name']}") if x)
        named = [anchor]
    configurations = {(c["properties"].get("configuration") or "") for c in found["parts"]}
    configuration = (f"\n- Its configuration on the installed parts: {configurations.pop()}"
                     if kind == "model" and len(configurations) == 1 and "" not in configurations else "")
    configuration += _variant_line(kind, p, found["models"])
    owners = [anchor_id, *([found["family"]["id"]] if kind == "model" and found["family"] else []),
              *([m["id"] for m in found["models"]] if kind == "family" else [])]
    with session() as s:
        facts = s.run("MATCH (o:Entity)-[r]->(x:Entity) WHERE o.id IN $owners AND type(r) IN $owning "
                      "RETURN DISTINCT labels(x) AS labels, x.name AS name, x.code AS code ORDER BY name",
                      owners=owners, owning=list(OWNING)).data()
    by = lambda label: [f"{f['code']} ({f['name']})" if f["code"] else f["name"]          # noqa: E731
                        for f in facts if repository.node_label(f["labels"]) == label]
    existing = "\n".join([f"  Alarms: {_listed(by('Alarm'))}", f"  Parameters: {_listed(by('Parameter'))}",
                          f"  Procedures: {_listed(by('Procedure'))}"])
    what = "product family" if kind == "family" else "device model"
    text = PRODUCT_TEMPLATE.format(what=what, name=anchor["name"], details=f" ({details})" if details else "",
                                   others="products" if kind == "family" else "models, sizes or variants",
                                   configuration=configuration, existing=existing)
    generic = set()
    for n in named:
        q = n["properties"]
        generic |= {normalize(v) for v in (n["name"], q.get("designation"), q.get("order_code"),
                                           f"{maker} {n['name']}", *(q.get("aliases") or [])) if v}
    return {"id": anchor_id, "name": anchor["name"], "label": anchor["label"], "text": text, "generic": generic - {""}}


def anchor_context(asset_id: str, anchor_id: str) -> dict | None:
    """The prompt's context block for the part a document describes, and the names that mean that part. A product
    family or a device model has its own block (P26b)."""
    from . import services
    if not anchor_id.startswith("bsk:component:"):
        return product_context(anchor_id)
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
    configuration = f"\n- This part's configuration: {p['configuration']}" if p.get("configuration") else ""
    text = ANCHOR_TEMPLATE.format(name=part["name"], asset=ctx["asset"]["name"], details=f" ({details})" if details else "",
                                  configuration=configuration, existing=existing)
    generic = {normalize(v) for v in (p.get("model"), p.get("part_number"),
                                      f"{p.get('manufacturer', '')} {p.get('model', '')}".strip()) if v}
    return {"id": anchor_id, "name": part["name"], "label": "Component", "text": text, "generic": generic - {""}}


def map_anchor(t: dict, anchor: dict | None) -> dict:
    """The product's own designations mean the anchor: a part, or the family or model itself (P26b)."""
    if anchor:
        for end in ("subject", "object"):
            if (t[f"{end}_type"] in ("Component", anchor.get("label", "Component"))
                    and normalize(t[end]) in anchor["generic"] | {normalize(anchor["name"])}):
                t = {**t, end: anchor["name"], f"{end}_type": anchor.get("label", "Component"), f"{end}_anchor": True}
    return t


# --- prompt + LLM ------------------------------------------------------------

def extractable(schema: Schema) -> tuple[dict, dict]:
    """Entity types and (pairs of) relationships the LLM may produce.

    Document / DocumentSection are evidence structure, written automatically when
    a triple is approved — the model is not asked to extract them.
    """
    types = {label: desc for label, desc in schema.entity_types.items()
             if label not in EVIDENCE_TYPES and label not in NOT_EXTRACTED}
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


# Free text, where each find can add its part (a procedure's steps over two windows, the tables a rating was read
# from): joined, not voted on. Every other property (terminal, code, default, unit, subtype …) keeps one value.
JOINED_PROPS = {"basis", "summary", "description", "cause", "causes", "consequence", "corrective_action", "effect", "notes"}


def merge_properties(old: dict, new: dict) -> dict:
    """Two finds of the same triple: lists are joined, a property missing is added. A text property keeps its value
    (P26b-2: joining gave "5; DI 0; 5; DI 0; F-DI 0"; `Votes` picks the most frequent at the end of a run), except
    the JOINED_PROPS, where a text found again is added once. With GRAPH_SINGLE_VALUES off, every text is joined."""
    out = dict(old or {})
    for key, v in (new or {}).items():
        if key not in out:
            out[key] = v
        elif isinstance(out[key], list):
            out[key] = list(dict.fromkeys(out[key] + (v if isinstance(v, list) else [str(v)])))[:MAX_LIST]
        elif (key in JOINED_PROPS or not settings.GRAPH_SINGLE_VALUES) and isinstance(out[key], str) \
                and isinstance(v, str) and v not in out[key]:
            out[key] = f"{out[key]}; {v}"[:MAX_PROPERTY]
    return out


def _same(v) -> str:
    return " ".join(str(v).split()).casefold()


def differing(old: dict, new: dict) -> list[str]:
    """The single-valued properties two finds give different values for (P26b-2), as review notes."""
    if not settings.GRAPH_SINGLE_VALUES:
        return []
    return [disagreement(k, [(str(old[k]), 1), (str(v), 1)]) for k, v in (new or {}).items()
            if k in (old or {}) and k not in JOINED_PROPS and not isinstance(v, list) and not isinstance(old[k], list)
            and _same(old[k]) != _same(v)]


def disagreement(prop: str, counts: list[tuple[str, int]]) -> str:
    """The review's note for a property found with different values: "pages disagree on terminal: 5 (3×) | F-DI 0"."""
    return f"pages disagree on {prop}: " + " | ".join(f"{v} ({n}×)" if n > 1 else v for v, n in counts)


class Votes:
    """P26b-2: the values each staged triple's single-valued properties were found with, so the run keeps the most
    frequent (the first on a tie) and tells the reviewer about the others."""

    def __init__(self):
        self.counts: dict[tuple[int, str, str], dict[str, list]] = {}

    def add(self, triple, end: str, props: dict | None) -> None:
        for key, v in (props or {}).items():
            if isinstance(v, list) or key in JOINED_PROPS:
                continue
            seen = self.counts.setdefault((id(triple), end, key), {})
            entry = seen.setdefault(_same(v), [v, 0])
            entry[1] += 1

    def settle(self, triples) -> int:
        """Each property to its most frequent value; the others in the triple's issue (one already there is kept)."""
        changed = 0
        for t in triples:
            notes = []
            for end in ("subject", "object"):
                props = getattr(t, f"{end}_props") or {}
                for key in list(props):
                    seen = self.counts.get((id(t), end, key)) or {}
                    if len(seen) < 2:
                        continue
                    ranked = sorted(seen.values(), key=lambda e: -e[1])        # stable: the first found wins a tie
                    props[key] = ranked[0][0]
                    notes.append(disagreement(key, [(str(v), n) for v, n in ranked]))
                setattr(t, f"{end}_props", props)
            if notes:
                changed += 1
                if not t.issue:
                    t.issue = "; ".join(notes)[:255]
        return changed


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
    """Key prefix for proposed ids: the asset scope (bsk:asset:EXTR01 -> EXTR01) or the document key. A run on a
    product family or a device model: its own key (global ids, P26b)."""
    anchor = (params or {}).get("anchor_id") or ""
    if anchor and not anchor.startswith("bsk:component:"):
        return anchor.split(":", 2)[2]
    return params["asset_id"].split(":", 2)[2] if (params or {}).get("asset_id") else doc.doc_key


def part_scope_for(params: dict, index: EntityIndex, scope_key: str):
    """The part a run's document describes, for part-scoped ids and matching (P21); None without one."""
    anchor = (params or {}).get("anchor_id")
    if not anchor:
        return None
    # a family's or model's facts always get its ids; a part's only with part-scoped ids on
    return index.part_scope(anchor, scope_key) if settings.GRAPH_PART_SCOPED_IDS or not anchor.startswith("bsk:component:") else None


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
    try:      # P26b-1: across the job's facet runs (the same name under two types, likely duplicates); never fails the job
        flagged = review_checks.flag_job(job)
        if flagged:
            logger.info("review checks flagged %d triples doc=%s", flagged, doc.doc_key)
    except Exception:
        logger.exception("review checks failed doc=%s", doc.doc_key)


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
        # Reviewed in this document *for this part* (P22c-3a): one manual can describe two parts (the G120C: the feed
        # screw drive and the cutter drive); the second part's run is not skipped because of the first's review.
        anchor = (params.get("facet_anchors") or {}).get(key) or params.get("anchor_id") or ""
        reviewed = CandidateTriple.objects.filter(document=doc, preset_name=preset.name).exclude(
            status=CandidateTriple.Status.PENDING)
        if anchor:
            reviewed = reviewed.filter(anchor_id=anchor)
        reviewed = reviewed.count()
        if reviewed and not params.get("rerun_reviewed"):
            out.update(state="skipped", note=f"{reviewed} triples already reviewed")
            continue
        library.heartbeat(job, 0, 0, f"{label}: choosing pages")
        try:
            suggestion = page_suggest.suggest(doc, preset=preset, asset_id=params["asset_id"],
                                              anchor_id=anchor,
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
        sub = {**params, "anchor_id": anchor, "mode": "schema", "presets": {"schema": preset.pk},
               "pages": parse_pages(suggestion["pages"]),
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
                 (f"{result['left']} left to the {', '.join(leave)} run" if leave else
                  f"{result['left']} outside this facet") if result["left"] else ""]
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

    dry = bool(params.get("dry_run"))      # P26b-2: the real path, nothing staged (manage.py extract_dry_run)

    def beat(done=None, total=None, message=None):
        if job is not None:
            library.heartbeat(job, done, total, f"{label}: {message}" if label and message else message)

    refused = "" if dry else reviewed_on_parts(doc, params.get("anchor_id") or "")
    if refused:
        raise ExtractionError(refused)
    parsed = library.load_parsed(doc)
    windows = chunk_parsed(parsed, params["chunking"]["strategy"], params["chunking"]["params"])
    kept = None
    if params.get("follow_languages", True) and doc.languages:
        from ingestion import languages
        kept = languages.kept_pages((parsed.get("document") or {}).get("md_content") or "", doc.languages)
    all_windows = len(windows)
    selected = select_windows(windows, params, kept)
    snap = None if dry else library.text_snapshot(doc, parsed)     # P25b: spans of the windows in the text
    originals = {i: w.text for i, w in selected}                    # a window's text before its tables are read apart
    table_parts: list[tuple[int, object]] = []
    if params.get("tables_by_row", True):
        from . import tables
        md = (parsed.get("document") or {}).get("md_content") or ""
        wanted = {p for a, b in params.get("pages") or [] for p in range(a, b + 1)}
        captions = set()
        for n, table in enumerate(tables.find_tables(md)):
            if (wanted and table.page not in wanted) or (kept is not None and table.page not in kept):
                continue
            table_parts += tables.windows(table, n)
            captions.add(table.caption)
        if table_parts:     # the text windows without their table lines; a window left empty is not read
            stripped = [(i, w, tables.strip_tables(w.text, captions - {""})) for i, w in selected]
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
    presets = params.get("preset_objects") or {m: PromptPreset.objects.get(pk=pk) for m, pk in params["presets"].items()}
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
    votes, channels = Votes(), Channels()   # P26b-2: one value per property, one id per channel
    reviewed = set() if dry else {  # already approved/rejected in an earlier run: not staged again
        triple_key(t.mode, t.subject_name, t.subject_type, t.predicate, t.object_name, t.object_type)
        for t in CandidateTriple.objects.filter(document=doc).exclude(status=CandidateTriple.Status.PENDING)
    }
    done = failed = skipped = 0
    left: set[tuple] = set()      # distinct triples left to another facet run
    strips: set[str] = set()      # strips and connectors returned as IOPoints, dropped (P26b-2)
    leave_types = set(params.get("leave_types") or ())   # owned by another facet run of the same job (P22c-0)
    keep_links = set(params.get("keep_links") or ())     # a facet run stages only its own links (empty: all)

    def stage(raw, mode, window, i, preset, make_evidence, model=None):
        """Stage the raw triples found in one place (a text window or a figure)."""
        nonlocal skipped
        for t in raw:
            if mode == "schema" and (t["subject_type"] in EVIDENCE_TYPES or t["object_type"] in EVIDENCE_TYPES):
                continue  # evidence structure is written on approval, not extracted
            t = map_anchor(t, anchor)
            candidate = _candidate(t, mode, schema, index, scope_key, doc, job, window, i, preset, model=model, part=part)
            candidate.anchor_id = params.get("anchor_id") or ""
            if settings.GRAPH_SINGLE_VALUES and strip_as_channel(candidate):
                strips.add(candidate.object_name if candidate.object_type == "IOPoint" else candidate.subject_name)
                continue
            channels.unify(candidate)
            key = triple_key(mode, candidate.subject_name, candidate.subject_type, candidate.predicate,
                             candidate.object_name, candidate.object_type)
            if (candidate.subject_type in leave_types or candidate.object_type in leave_types
                    or (keep_links and candidate.predicate not in keep_links)):
                left.add(key)      # checked after staging: a part's limit becomes a Specification there
                continue
            if key in reviewed:
                skipped += 1
                continue
            if settings.GRAPH_SINGLE_VALUES:
                votes.add(staged.get(key, candidate), "subject", candidate.subject_props)
                votes.add(staged.get(key, candidate), "object", candidate.object_props)
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
    located: dict[int, dict] = {}

    def span_of(i, window) -> dict:
        """The window's place in the snapshot, found once per window (P25b): a table part has its rows already."""
        if snap is None:
            return {}
        if i not in located:
            if getattr(window, "row_start", None):
                exact = window.char_start is not None
                located[i] = {"text_hash": snap.hash, "char_start": window.char_start, "char_end": window.char_end,
                              "span_status": "exact" if exact else "not_found",
                              "row_start": window.row_start, "row_end": window.row_end}
            else:
                located[i] = snap.locate(originals.get(i, window.text), window.page_start, window.page_end).fields(snap.hash)
        return located[i]
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
                  lambda t, window=window, i=i, preset=preset: _evidence(doc, window, i, preset, schema, t, params,
                                                                         span_of(i, window)))
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
                          lambda t, figure=figure: drawings.evidence_for(doc, figure, schema, t, snap), model=settings.VLM_MODEL)
                    done += 1
        except gpu.GpuError as exc:  # BSK asleep / busy: the text triples are still staged
            drawings_note = "drawings were not read: the BSK PC was not reachable"
            logger.warning("drawing triples skipped doc=%s: %s", doc.doc_key, exc)

    disagreed = votes.settle(staged.values())
    if dry:
        return {"staged": len(staged), "failed": failed, "skipped": skipped, "left": len(left), "disagreed": disagreed,
                "strips_dropped": sorted(strips), "triples": list(staged.values()), "evidence": {k: len(v) for k, v in found.items()}}
    with transaction.atomic():
        # A new run replaces the document's pending triples of the prompts it ran (P22c-0: per prompt, not per mode:
        # a Ratings run no longer erases a Maintenance run's pending triples); reviewed ones are kept. With a part, only
        # that part's runs are replaced, as the facet skip above (the G120C's feed drive run erased the cutter drive's
        # pending ratings, 2026-10-21).
        ran = {p.name for p in presets.values()} | ({drawings.PROMPT_NAME} if drawing_figures else set())
        replaced = CandidateTriple.objects.filter(document=doc, status=CandidateTriple.Status.PENDING,
                                                  mode__in=list(presets), preset_name__in=ran)
        if params.get("anchor_id"):
            replaced = replaced.filter(anchor_id=params["anchor_id"])
        replaced.delete()
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
    logger.info("extraction done doc=%s windows=%d/%d table_parts=%d triples=%d failed_calls=%d strips_dropped=%d anchor=%s",
                doc.doc_key, len(selected), all_windows, len(table_parts), len(staged), failed, len(strips),
                params.get("anchor_id") or "-")
    return {"staged": len(staged), "failed": failed, "skipped": skipped, "left": len(left)}


def _evidence(doc, window, i, preset, schema, t, params, span: dict | None = None) -> Evidence:
    return Evidence(
        source_kind=Evidence.Kind.TEXT, document=doc,
        page_start=window.page_start, page_end=window.page_end, section_path=window.section_path,
        chunk_index=i, window=params.get("chunking") or {}, extractor="llm-text",
        model=settings.DGX_CHAT_MODEL, prompt=f"{preset.name} v{preset.version}", schema_version=schema.version,
        excerpt=window.text[:EVIDENCE_CHARS], confidence=t.get("confidence"), **(span or {}),
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


# A value that is only a number or a range ("6", "88.8 x 56.8 x 45", "12...16", "≥ 0.001")
_PLAIN_VALUE = re.compile(r"^[\s\d.,…≥≤<>~±+\-–x×/]*\d[\s\d.,…≥≤<>~±+\-–x×/]*$")


def without_repeated_unit(p: dict) -> dict:
    """A rating whose value repeats its unit ("6 ms" with unit "ms" showed as "6 ms ms", P25 D6): the unit leaves the
    value, only when what is left is a plain number or range ("1 MB of RAM" with unit "RAM" stays as it is)."""
    out = {}
    unit = str(p.get("unit") or "").strip()
    for key in ("value", "min", "max"):
        v = p.get(key)
        if unit and isinstance(v, str) and v.rstrip().endswith(" " + unit):
            rest = v.rstrip()[: -len(unit)].rstrip()
            if _PLAIN_VALUE.match(rest):
                out[key] = rest
    return out


def is_part_rating(subject_type: str, predicate: str, object_type: str, schema: Schema) -> bool:
    """A part's limit the schema refuses but can hold as a rating (Component -HAS_LIMIT-> OperatingLimit, v4); a
    family's or model's too (P26b)."""
    return (subject_type in ("Component", *NOT_EXTRACTED) and predicate == "HAS_LIMIT" and object_type == "OperatingLimit"
            and "HAS_SPEC" in schema.relationship_types
            and (subject_type, "Specification") in schema.relationship_types["HAS_SPEC"].pairs
            and (subject_type, "OperatingLimit") not in schema.relationship_types["HAS_LIMIT"].pairs)


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
            if t[f"{end}_type"] == "Specification":
                p.update(without_repeated_unit(p))
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
            not_extracted = [end for end in ("subject", "object") if t[f"{end}_type"] in NOT_EXTRACTED]
            try:
                sm = index.resolve(t["subject_type"], t["subject"], scope_key, part=part, **_keyed(t, "subject", props, scope_key, part))
                om = index.resolve(t["object_type"], t["object"], scope_key, part=part, **_keyed(t, "object", props, scope_key, part))
                subject_id, subject_existing, object_id, object_existing = sm.id, sm.existing, om.id, om.existing
                matches = {"subject_match": sm.kind, "subject_candidates": sm.candidates,
                           "object_match": om.kind, "object_candidates": om.candidates}
                if subject_id == object_id and not issue:
                    issue = SELF_LINK
                ids = {"subject": subject_id, "object": object_id}
                if any(ids[end] != (part.anchor_id if part is not None else None) for end in not_extracted):
                    issue = "product families and device models come from the BOM's model list, not extracted"
            except ValueError as exc:  # a name with no usable id characters
                issue = str(exc)[:255]
        if not issue:     # P26b-1: what the QUINT4-PS review rejected by hand (a rating with no value, a "Row 3" name)
            issue = review_checks.triple_issue(t["object_type"], t["object"], props.get("object") or {})
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


_GROUP_INDEX = re.compile(r"^(\d{1,2})\.(\d{1,2})$")


def normalize_code(code: str) -> tuple[str, str]:
    """A parameter code in one form, and its index: "P310" -> ("p0310", ""), "p1070[0]" -> ("p1070", "0"),
    "r0021" stays; a group.index code has two digits each ("1.1" -> "01.01"). Anything else is kept as written."""
    code = code.strip()
    group = _GROUP_INDEX.match(code)
    if group:   # ABB style "group.index", two digits each; a parsed table drops zeros ("1.1" is 01.01, "46.3" is 46.03)
        return f"{int(group.group(1)):02d}.{int(group.group(2)):02d}", ""
    m = _SIEMENS_CODE.match(code)
    if not m:
        return code, ""
    letter, number, index, suffix = m.groups()
    return f"{letter.lower()}{int(number):04d}", (index or suffix or "").strip()


_FIRST_TERMINAL = re.compile(r"^\s*(\d{1,4})(?!\s*[.,]\d)\b")


def first_terminal(terminal) -> str:
    """The first terminal number of a channel ("18, 19, 20" -> "18", "3 (AI 0+), 4 (AI 0-)" -> "3"); "" for none
    ("DI 0", "L1")."""
    first = re.split(r"[,;]", str(terminal or ""), maxsplit=1)[0]
    m = _FIRST_TERMINAL.match(first)
    return m.group(1) if m else ""


# P26b-2: a strip or connector as a whole is not a channel. The G120C runs returned "Terminal strip -X136" as an
# IOPoint even with a prompt rule against it; dropped at staging, counted in the result. Only a name that is the strip
# word with at most a designation ("Terminal strip -X136", "Connector X1", "-X138"), never a channel named after it.
_STRIP_NAME = re.compile(r"^\s*(?:(?:terminal\s+(?:strip|block)|connector|plug)s?\s*[:.]?\s*(?:-?[a-z]{0,3}\d+[a-z]?)?"
                         r"|-?x\d+[a-z]?)\s*$", re.I)


def strip_as_channel(c) -> bool:
    return any(getattr(c, f"{end}_type") == "IOPoint" and _STRIP_NAME.match(getattr(c, f"{end}_name") or "")
               for end in ("subject", "object"))


class Channels:
    """P26b-2: within one run, one channel is one IOPoint id. A channel staged first by its name (DI-0) and then with
    its strip and terminal (X138-5), or the other way round, was two nodes on the G120C. Later finds take the id of
    the first one with the same name, or with the same strip and first terminal number (the strip must agree: many
    devices number each strip from 1)."""

    def __init__(self):
        self.by_name: dict[str, str] = {}
        self.by_terminal: dict[tuple[str, str], str] = {}

    def unify(self, c) -> None:
        if not settings.GRAPH_SINGLE_VALUES:
            return
        for end in ("subject", "object"):
            if getattr(c, f"{end}_type") != "IOPoint" or getattr(c, f"{end}_existing"):
                continue
            p = getattr(c, f"{end}_props") or {}
            name = normalize(getattr(c, f"{end}_name"))
            term = (_same(p.get("connector") or ""), first_terminal(p.get("terminal")))
            known = self.by_name.get(name) or (self.by_terminal.get(term) if term[1] else None)
            node_id = known or getattr(c, f"{end}_id")
            if known and known != getattr(c, f"{end}_id"):
                setattr(c, f"{end}_id", known)
            self.by_name.setdefault(name, node_id)
            if term[1]:
                self.by_terminal.setdefault(term, node_id)


def _keyed(t: dict, end: str, props: dict, scope_key: str, part) -> dict:
    """A parameter's id comes from its code, a terminal's from its connector and terminal (P22c): the same p1082 on
    twenty pages is one node, whatever name each page gives it. Other types: by name, as before."""
    from .identity import PART_LABELS
    label, p = t[f"{end}_type"], props.get(end) or {}
    if t.get(f"{end}_anchor") and part is not None and label != "Component":
        return {"proposed": part.anchor_id}      # the family or model the run is on (P26b)
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
