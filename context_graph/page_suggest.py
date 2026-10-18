"""Suggest pages for Generate triples (2026-10-16).

Instead of the reviewer typing the pages to read ("14, 18-26"), the model picks them from the document's
outline: one line per page with its headings, table captions and first words (not the full text). One call
(or a few for a long document) to the DGX, seconds instead of a run over every page. The reviewer sees the
sections with a reason each and can edit the pages before running.

A document in several languages is outlined in one of them (the first kept language, else English), so the
same chapter is not suggested twice. Tried on the ABB motors manual (272 pages, 9 languages): 18-21 and
24-26 (Maintenance, Troubleshooting) against the hand-picked 14, 18-26.
"""

from __future__ import annotations

import json
import re

from ingestion.chunker import PAGE_BREAK

from . import extraction, registry
from .models import PromptPreset

HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")
CAPTION_RE = re.compile(r"^\s*((?:Table|Tabla|Tabelle|Tableau)\s+[\w.\-]+.*)$", re.IGNORECASE)
FIRST_WORDS = 160           # characters of a page's text after its headings
HEADINGS = 300              # characters of a page's headings
OUTLINE_BUDGET = 24_000     # characters of outline per call; a longer document is suggested in parts

SYSTEM = """You choose the pages of a document to read for a knowledge graph.
Subject: {subject}
What will be extracted: {goal}
Entity types: {types}

The user message is the document's outline: one line per page with its headings and table captions, then its first words; "Pages 24-26" is a page with the pages after it that have no heading (they continue it).
Pick the pages that hold facts about the subject for that goal. Such facts are usually in: {where}.
Manuals are usually written for a whole product family or series: what applies to the family applies to the subject (a series' fault list is the subject's fault list). Leave out only what is about other models, sizes or options.
Leave out pages that do not hold such facts: covers, contents, declarations, general safety notes, transport and storage, pages only about other models or options, long reference lists of settings or parameters (a list of fault, warning or alarm messages is not one: keep it), figures without text. Keep a table's continuation pages with it.
Answer in JSON: {{"note": "one sentence", "sections": [{{"pages": "18-21", "title": "7 Maintenance", "why": "short reason"}}]}}
If no page applies, answer {{"note": "why", "sections": []}}."""


# Where a preset's facts usually are (P22a): the maintenance chapters, or the technical data.
WHERE = ("troubleshooting; fault, warning and alarm messages with their causes and remedies; maintenance, inspection "
         "and lubrication (intervals, amounts, steps); procedures with steps; technical data, ratings and operating limits; "
         "protections and monitoring")
WHERE_BY_PRESET = {
    "Specifications (v4)": ("technical data and specifications; ratings (electrical data, rated current, power, pulse "
                            "frequency); measuring range, accuracy, repeatability, response time; ambient, storage and operating "
                            "conditions; dimensions and weights; tables of ratings for several models or sizes (keep them: "
                            "the subject's row is chosen later)"),
    "Controls & Setup (v6)": ("commissioning and quick commissioning; parameter lists and the parameters to set (motor "
                              "data, limits, ramps, control mode); terminal strips, inputs and outputs and their wiring "
                              "examples; the default settings or macros of the interfaces; fieldbus and communication "
                              "(PROFINET, PROFIBUS, USS, Modbus, addresses, telegrams). Not function diagrams of internal "
                              "control, nor troubleshooting"),
}


class SuggestError(ValueError):
    pass


def outline_language(page_langs: list[str], kept: list[str]) -> str:
    """The one language the outline is made in ("" = every page)."""
    present = [lang for lang in page_langs if lang]
    if len(set(present)) <= 1:
        return ""
    for lang in kept:
        if lang in present:
            return lang
    return "en" if "en" in present else max(set(present), key=present.count)


def outline(md: str, language: str = "", page_langs: list[str] | None = None) -> list[tuple[int, int, str]]:
    """(first page, last page, outline line) for the pages in `language` (or every page).

    A page with headings starts an entry; the pages after it without a heading continue it ("Pages 579-628:
    Warning, fault and pure event messages"). The model sees chapters, not hundreds of lines, and choosing an
    entry takes its continuation pages: one line per page cut the ACS880 fault list at a call boundary and
    the model kept only its first page (2026-10-16).
    """
    pages = md.split(PAGE_BREAK)
    page_langs = page_langs or [""] * len(pages)
    entries: list[list] = []        # [first, last, headings, first words]
    for n, (page, lang) in enumerate(zip(pages, page_langs), start=1):
        if language and lang and lang != language:
            continue
        lines = page.splitlines()
        heads = [m.group(1) for m in map(HEADING_RE.match, lines) if m]
        heads += [m.group(1).strip() for m in map(CAPTION_RE.match, lines) if m]
        if not heads and entries and entries[-1][1] == n - 1:
            entries[-1][1] = n
            continue
        body = " ".join(line for line in lines if not HEADING_RE.match(line))
        first = re.sub(r"\s+", " ", re.sub(r"[|#*_]|-{2,}", " ", body)).strip()[:FIRST_WORDS]
        entries.append([n, n, " | ".join(heads)[:HEADINGS] or "(no heading)", first])
    return [(a, b, f"Page {a}: {h} — {w}" if a == b else f"Pages {a}-{b}: {h} — {w}") for a, b, h, w in entries]


def goal_of(preset: PromptPreset | None) -> str:
    """The preset's first paragraph: what it extracts ("You extract a maintenance knowledge graph …")."""
    if preset is None:
        return "facts about the subject for an industrial knowledge graph"
    first = preset.template.strip().split("\n\n")[0].strip()
    return re.sub(r"\s+", " ", first)[:400]


def _ranges(pages: set[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for p in sorted(pages):
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return out


def suggest(doc, *, preset: PromptPreset | None = None, asset_id: str = "", anchor_id: str = "",
            follow_languages: bool = True) -> dict:
    """The pages of `doc` to read, with the sections the model chose and why."""
    from ingestion import languages, library

    parsed = library.load_parsed(doc)
    md = (parsed.get("document") or {}).get("md_content") or ""
    if not md.strip():
        raise SuggestError("the document has no parsed text")
    page_langs = languages.page_languages(md)
    language = outline_language(page_langs, doc.languages if follow_languages else [])
    lines = outline(md, language, page_langs)

    if anchor_id:
        subject = extraction.anchor_context(asset_id, anchor_id)["text"].splitlines()[1].strip("- ")
    elif asset_id:
        from . import services
        # A generic manual (a drive's firmware manual) is about one of the asset's parts: without this the model
        # found nothing "about Extruder EXTR01" in the ACS880 manual (2026-10-16).
        subject = (f'the asset "{services.get_asset_context(asset_id)["asset"]["name"]}" and its parts. The document may be '
                   "a generic manual for one of its parts (a motor, a drive, a sensor): its content about that product applies")
    else:
        subject = f'the product that the document "{doc.filename}" describes'
    entity_types, _ = extraction.extractable(registry.active_schema())
    where = WHERE_BY_PRESET.get(preset.name, WHERE) if preset else WHERE
    system = SYSTEM.format(subject=subject, goal=goal_of(preset), types=", ".join(entity_types), where=where)

    entry_of = {p: (a, b) for a, b, _ in lines for p in range(a, b + 1)}
    chosen: set[int] = set()
    sections: list[dict] = []
    notes: list[str] = []
    parts, part, size = [], [], 0
    for _, _, line in lines:        # a long outline in parts of OUTLINE_BUDGET characters
        if part and size + len(line) > OUTLINE_BUDGET:
            parts.append(part)
            part, size = [], 0
        part.append(line)
        size += len(line) + 1
    if part:
        parts.append(part)
    for part in parts:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(part)}]
        try:
            answer = json.loads(extraction.call_llm(messages) or "{}")
        except (ValueError, TypeError) as exc:
            raise SuggestError(f"the model's answer was not valid JSON ({exc})")
        if answer.get("note"):
            notes.append(str(answer["note"]))
        for s in answer.get("sections") or []:
            if not isinstance(s, dict):
                continue
            try:
                ranges = extraction.parse_pages(str(s.get("pages") or ""))
            except extraction.ExtractionError:
                continue
            # Only pages of the outline, each with its whole entry (a table or a list with its continuation pages).
            pages = {q for a, b in ranges for p in range(a, b + 1) if p in entry_of
                     for q in range(entry_of[p][0], entry_of[p][1] + 1)}
            if not pages:
                continue
            chosen |= pages
            sections.append({"pages": extraction.pages_text(_ranges(pages)), "title": str(s.get("title") or "")[:200],
                             "why": str(s.get("why") or "")[:300]})
    return {
        "pages": extraction.pages_text(_ranges(chosen)),
        "sections": sections,
        "note": " ".join(notes)[:600],
        "language": language,
        "outline_pages": len(entry_of),
        "total_pages": len(md.split(PAGE_BREAK)),
    }
