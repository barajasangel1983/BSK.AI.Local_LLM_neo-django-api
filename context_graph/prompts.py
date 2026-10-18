"""Prompt management (P23): versions kept like schemas, validation, export / import, and a test on a page.

`PromptPreset` is the active copy that extraction reads; `PromptVersion` keeps every text. Saving makes a new
version (the highest number + 1) and makes it active; activating copies an older version back (no new number).
"""

from __future__ import annotations

import re
import time
from types import SimpleNamespace

import yaml
from django.db import transaction
from django.db.models import Count, Max

from . import extraction, registry
from .models import CandidateTriple, GraphSchemaVersion, PromptPreset, PromptVersion

SOURCES = ("built-in", "edit", "import", "adopt", "migrated")
MAX_TEST_PAGES = 3
MAX_TEST_WINDOWS = 8
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_RELATION = re.compile(r"\b[A-Z]{2,}(?:_[A-Z]+)+\b")       # HAS_SPEC, MITIGATED_BY: link names as prompts write them
_WORD = re.compile(r"\b[A-Z][a-zA-Z]+\b")


class PromptError(ValueError):
    pass


def facet_of(preset: PromptPreset) -> dict:
    """The facet run a prompt serves ({key, label}), or Other."""
    run = next((r for r in extraction.FACET_RUNS if r["preset"] == preset.name), None)
    return {"key": run["key"], "label": run["label"]} if run else {"key": "other", "label": "Other"}


def builtin_update(preset: PromptPreset) -> str | None:
    """The code's built-in text when it differs from the active version (P23: offered, never applied)."""
    text = extraction.builtin_templates().get(preset.name)
    return text if text is not None and text != preset.template else None


def check(template: str, mode: str, schema=None) -> dict:
    """Problems that stop a save (`errors`) and things worth a look (`warnings`)."""
    errors, warnings = [], []
    if not template.strip():
        errors.append("the prompt is empty")
    if mode not in PromptPreset.Mode.values:
        errors.append(f"mode must be one of {', '.join(PromptPreset.Mode.values)}")
    unknown = sorted(set(_PLACEHOLDER.findall(template)) - set(extraction.PLACEHOLDERS))
    if unknown:
        warnings.append(f"not placeholders (left as written): {', '.join('{' + u + '}' for u in unknown)}; "
                        f"placeholders: {', '.join('{' + p + '}' for p in extraction.PLACEHOLDERS)}")
    if mode == "schema":
        schema = schema or registry.active_schema()
        if "{entity_types}" not in template or "{relationships}" not in template:
            warnings.append("a schema prompt usually lists {entity_types} and {relationships}")
        links = set(schema.relationship_types)
        named = sorted({r for r in _RELATION.findall(template) if r not in links})
        if named:
            warnings.append(f"links not in the active schema (v{schema.version}): {', '.join(named)}")
        every_type = {t for v in GraphSchemaVersion.objects.all() for t in (v.definition.get("entity_types") or {})}
        missing = sorted({w for w in _WORD.findall(template) if w in every_type and w not in schema.entity_types})
        if missing:
            warnings.append(f"types not in the active schema (v{schema.version}): {', '.join(missing)}")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


@transaction.atomic
def save_version(preset: PromptPreset, template: str, note: str = "", source: str = "edit", author: str = "") -> PromptVersion:
    """A new version (the highest number + 1), active at once (Angel's decision 1)."""
    result = check(template, preset.mode)
    if not result["valid"]:
        raise PromptError("; ".join(result["errors"]))
    if template == preset.template:
        raise PromptError("no change: the text is the active version's")
    top = preset.versions.aggregate(top=Max("version"))["top"] or preset.version
    preset.template, preset.version = template, max(top, preset.version) + 1
    preset._version_note, preset._version_source, preset._version_author = note, source, author
    preset.save()
    return preset.versions.get(version=preset.version)


def activate(preset: PromptPreset, version: int) -> PromptPreset:
    """An earlier (or later) version becomes the active copy again; its number is kept."""
    v = preset.versions.filter(version=version).first()
    if v is None:
        raise PromptError(f"{preset.name} has no version {version}")
    preset.template, preset.version = v.template, v.version
    preset.save()
    return preset


def versions_json(preset: PromptPreset) -> list[dict]:
    made = dict(CandidateTriple.objects.filter(preset_name=preset.name).values_list("preset_version")
                .annotate(n=Count("id")).values_list("preset_version", "n"))
    return [{"version": v.version, "template": v.template, "note": v.note, "schema_version": v.schema_version,
             "source": v.source, "author": v.author, "created_at": v.created_at.isoformat(),
             "active": v.version == preset.version, "triples": made.get(v.version, 0)}
            for v in preset.versions.all()]


# --- export / import ---------------------------------------------------------------

def export_one(preset: PromptPreset, version: int | None = None) -> dict:
    v = preset.versions.get(version=version or preset.version)
    return {"name": preset.name, "mode": preset.mode, "facet": facet_of(preset)["label"], "version": v.version,
            "schema_version": v.schema_version, "note": v.note, "template": v.template}


def dump(data) -> str:
    class Literal(str):
        pass
    yaml.add_representer(Literal, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"),
                         Dumper=yaml.SafeDumper)

    def lit(x):
        return {k: (Literal(v) if k == "template" else v) for k, v in x.items()}
    body = {"prompts": [lit(p) for p in data["prompts"]]} if "prompts" in data else lit(data)
    return yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=120)


def import_text(text: str, author: str = "") -> list[dict]:
    """One prompt or a bundle ({prompts: [...]}): each becomes a new version of the prompt of that name (or a new
    custom prompt); identical text is skipped. Returns one result per prompt."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PromptError(f"not YAML: {exc}")
    items = data.get("prompts") if isinstance(data, dict) and "prompts" in data else [data]
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise PromptError("expected a prompt (name, mode, template) or {prompts: [...]}")
    results = []
    with transaction.atomic():
        for item in items:
            name, mode, template = str(item.get("name") or "").strip(), item.get("mode"), str(item.get("template") or "")
            if not name or not template.strip():
                raise PromptError("each prompt needs a name and a template")
            note = str(item.get("note") or f"imported (was version {item.get('version', '?')})")[:255]
            preset = PromptPreset.objects.filter(name=name).first()
            if preset is None:
                result = check(template, mode)
                if not result["valid"]:
                    raise PromptError(f"{name}: " + "; ".join(result["errors"]))
                preset = PromptPreset(name=name, mode=mode, template=template)
                preset._version_note, preset._version_source, preset._version_author = note, "import", author
                preset.save()
                results.append({"name": name, "result": "new prompt", "version": preset.version})
            elif template == preset.template:
                results.append({"name": name, "result": "unchanged", "version": preset.version})
            else:
                if mode and mode != preset.mode:
                    raise PromptError(f"{name}: the file says mode {mode!r}, the prompt is {preset.mode!r}")
                v = save_version(preset, template, note=note, source="import", author=author)
                results.append({"name": name, "result": "new version", "version": v.version})
    return results


# --- test on a page ----------------------------------------------------------------

def test_on_pages(doc, template: str, mode: str, pages: str, asset_id: str = "", anchor_id: str = "",
                  name: str = "(draft)", version: int = 0) -> dict:
    """Run a prompt (saved or not) on up to MAX_TEST_PAGES pages; the triples as they would be staged. Stores nothing."""
    from ingestion import library
    from ingestion.chunking import chunk_parsed
    from ingestion.models import Document

    from .identity import EntityIndex

    if mode != "schema":
        raise PromptError("Test on a page is for schema prompts")
    if doc.parse_status != Document.ParseStatus.PARSED or not library.parsed_path(doc).exists():
        raise PromptError("parse the document first (Library → Parse)")
    ranges = extraction.parse_pages(pages)
    wanted = {p for a, b in ranges for p in range(a, b + 1)}
    if not wanted or len(wanted) > MAX_TEST_PAGES:
        raise PromptError(f"choose 1 to {MAX_TEST_PAGES} pages")
    params = extraction.validate_params({"asset_id": asset_id, "anchor_id": anchor_id, "pages": pages})
    parsed = library.load_parsed(doc)
    windows = chunk_parsed(parsed, extraction.DEFAULT_CHUNKING["strategy"], extraction.DEFAULT_CHUNKING["params"])
    selected = extraction.select_windows(windows, params, None)[:MAX_TEST_WINDOWS]
    if not selected:
        raise PromptError("no text on those pages")
    schema = registry.active_schema()
    preset = SimpleNamespace(name=name, version=version, template=template, mode=mode)
    anchor = extraction.anchor_context(params["asset_id"], params["anchor_id"]) if params["anchor_id"] else None
    index = EntityIndex.load()
    scope_key = extraction.scope_key_for(params, doc)
    part = extraction.part_scope_for(params, index, scope_key)
    started, out, failed = time.time(), [], 0
    for i, window in selected:
        messages = extraction.build_messages(preset, schema, window.text, doc.filename, " > ".join(window.section_path),
                                             context=anchor["text"] if anchor else "")
        try:
            raw = extraction.extract_window(messages)
        except Exception:      # a bad answer or the DGX: reported, the other windows still run
            failed += 1
            continue
        for t in raw:
            if t["subject_type"] in extraction.EVIDENCE_TYPES or t["object_type"] in extraction.EVIDENCE_TYPES:
                continue
            c = extraction._candidate(t, "schema", schema, index, scope_key, doc, None, window, i, preset, part=part)
            from . import facets
            facets.stamp([c], schema)
            out.append({"subject": {"name": c.subject_name, "type": c.subject_type, "id": c.subject_id,
                                    "properties": c.subject_props},
                        "predicate": c.predicate,
                        "object": {"name": c.object_name, "type": c.object_type, "id": c.object_id,
                                   "properties": c.object_props},
                        "issue": c.issue, "facets": c.facets, "confidence": c.confidence,
                        "page": window.page_start, "auto_fix": c.auto_fix})
    return {"triples": out, "windows": len(selected), "failed_windows": failed,
            "seconds": round(time.time() - started, 1), "schema_version": schema.version}
