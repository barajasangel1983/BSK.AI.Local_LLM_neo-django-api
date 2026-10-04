"""Triples from drawings (P9a.3): the VLM on BSK reads a figure and proposes relationships.

Used by "Generate triples" when `include_drawings` is set. Only figures that Describe
figures classified as a drawing, schematic or diagram are sent. The results are staged
like text triples (same identity resolution, schema checks and review) with vision
evidence: page, figure number and box. Nothing reaches the graph without approval.

The request is kept small on purpose (the VLM has an 8192-token context and a
1000-token answer): relationships only, compact JSON, a cap on their number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.conf import settings

from gpu import vlm
from ingestion import figures

from .models import Evidence

logger = logging.getLogger("chat")

PROMPT_NAME = "Drawing"
PROMPT_VERSION = 1
MAX_TRIPLES = 25
FIGURE_CHUNK_BASE = 100_000        # chunk_index of figure evidence: 100000 + figure number (never a text window)

_FORMAT = (
    'Return ONLY compact JSON on one line, no explanations:\n'
    '{"triples":[{"subject":str,"subject_type":str,"predicate":str,"object":str,"object_type":str}]}\n'
    f'At most {MAX_TRIPLES} triples, the most important first. Return {{"triples":[]}} if the drawing shows none.'
)
PROMPTS = {
    "schema": (
        "You read an engineering drawing and extract knowledge-graph triples about an industrial asset.\n\n"
        "Entity types (use exactly these names):\n{entity_types}\n\n"
        "Relationships (predicate: allowed subject type -> object type):\n{relationships}\n\n"
        "Rules:\n"
        "- Only relationships the drawing shows: lines, arrows and connections between labelled items.\n"
        "- Name each item by the tag or label written next to it (e.g. \"M101\", \"VFD-101\"). Skip items without a legible label.\n"
        "- Use only the entity types and relationships listed above, in the allowed directions.\n"
        "- Do not guess.\n" + _FORMAT
    ),
    "freeform": (
        "You read an engineering drawing or diagram and extract knowledge-graph triples.\n\n"
        "Rules:\n"
        "- Only relationships the drawing shows: lines, arrows and connections between labelled items.\n"
        "- Name each item by the tag or label written next to it. Skip items without a legible label.\n"
        "- predicate: a short UPPER_SNAKE_CASE verb phrase (e.g. FEEDS, DRIVES, CONNECTED_TO, PART_OF).\n"
        "- subject_type / object_type: a short noun for what the item is (e.g. Motor, Valve, Sensor, Block).\n"
        "- Do not guess.\n" + _FORMAT
    ),
}
RETRY = ('Your previous answer was not valid JSON (it may have been cut off). Reply with ONLY the compact JSON object, '
         'with at most 12 triples.')


@dataclass
class FigureWindow:
    """What `_candidate` needs from a text window, for a figure."""
    page_start: int
    page_end: int
    section_path: list[str] = field(default_factory=list)
    text: str = ""


def eligible(data: dict | None) -> list[dict]:
    """Described figures that are drawings (not photos, charts, tables, …)."""
    return [f for f in (data or {}).get("figures", [])
            if f.get("status") == "done" and f.get("kind") in figures.DRAWING_KINDS]


def window_for(figure: dict) -> FigureWindow:
    title = f"Figure {figure['index'] + 1}" + (f": {figure['caption']}" if figure.get("caption") else "")
    return FigureWindow(page_start=figure["page"], page_end=figure["page"], section_path=[title[:200]],
                        text=figure.get("description") or "")


def prompt_for(mode: str, schema, figure: dict) -> str:
    from .extraction import render_schema

    entity_types, relationships = render_schema(schema)
    prompt = PROMPTS[mode].replace("{entity_types}", entity_types).replace("{relationships}", relationships)
    if figure.get("caption"):
        prompt += f"\nCaption in the document: {figure['caption']}"
    return prompt


def extract(doc, figure: dict, mode: str, schema) -> list[dict]:
    """Raw triples from one figure (parsed like text output), retrying once on malformed JSON.

    Raises gpu.GpuError when BSK can't be used, ExtractionError when the answer stays invalid.
    """
    from .extraction import ExtractionError, parse_triples

    path = figures.image_path(doc, figure["index"])
    jpeg = path.read_bytes() if path.exists() else figures.crop(doc, figure)[0]
    prompt = prompt_for(mode, schema, figure)
    kwargs = {"max_tokens": settings.DRAWING_MAX_TOKENS, "json_mode": True, "purpose": "vlm-drawing"}
    try:
        return parse_triples(vlm.complete(jpeg, prompt, **kwargs))[:MAX_TRIPLES]
    except ExtractionError as first:
        logger.warning("drawing triples: malformed output, retrying once: %s", first)
        return parse_triples(vlm.complete(jpeg, f"{prompt}\n\n{RETRY}", **kwargs))[:MAX_TRIPLES]


def evidence_for(doc, figure: dict, schema, triple: dict) -> Evidence:
    window = window_for(figure)
    return Evidence(
        source_kind=Evidence.Kind.VISION, document=doc,
        page_start=figure["page"], page_end=figure["page"], section_path=window.section_path,
        region={"page": figure["page"], **(figure.get("bbox") or {})}, figure_id=str(figure["index"]),
        chunk_index=FIGURE_CHUNK_BASE + figure["index"], extractor="vlm",
        model=settings.VLM_MODEL, prompt=f"{PROMPT_NAME} v{PROMPT_VERSION}", schema_version=schema.version,
        excerpt=window.text[:1500], confidence=triple.get("confidence"),
    )
