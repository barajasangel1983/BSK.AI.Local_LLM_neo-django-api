"""Triple extraction from library documents (GraphLab "Generate triples").

Runs as a library-worker job (kind "extract"):

    parsed document (cached Docling output)
      -> extraction windows (chunking strategy, default fixed 800/100 tokens)
      -> per window and mode: prompt (preset + active schema) -> DGX, JSON mode
      -> strict validation (+1 retry on malformed JSON)
      -> entity matching against the curated graph (existing ids or proposed ids)
      -> de-duplicated CandidateTriple rows, pending review

The DGX's constrained decoding (guided_json / json_schema) was too slow to use
(80+ s and invalid output per window), so JSON is requested in the prompt with
`response_format: json_object` and validated here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from . import registry, repository
from .driver import GraphUnavailable, session
from .ids import clean_part, is_valid_id, make_id
from .models import CandidateTriple, PromptPreset
from .schema import Schema

logger = logging.getLogger("chat")

MODES = ("schema", "freeform", "both")
DEFAULT_CHUNKING = {"strategy": "fixed", "params": {"size": 800, "overlap": 100}}
MAX_NAME = 200
EVIDENCE_CHARS = 1500
EVIDENCE_TYPES = ("Document", "DocumentSection")

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
- Name entities as they are written in the text (e.g. "die head", "DIE_PLUG").
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


class ExtractionError(ValueError):
    pass


def ensure_default_presets() -> None:
    for mode, (name, template) in DEFAULT_PRESETS.items():
        PromptPreset.objects.get_or_create(name=name, defaults={"mode": mode, "template": template, "is_default": True})


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
    return {
        "mode": mode,
        "chunking": {"strategy": chunking.get("strategy", "fixed"), "params": resolved},
        "presets": presets,
        "asset_id": asset_id,
    }


# --- prompt + LLM ------------------------------------------------------------

def extractable(schema: Schema) -> tuple[dict, dict]:
    """Entity types and (pairs of) relationships the LLM may produce.

    Document / DocumentSection are evidence structure, written automatically when
    a triple is approved — the model is not asked to extract them.
    """
    types = {label: desc for label, desc in schema.entity_types.items() if label not in EVIDENCE_TYPES}
    rels = {}
    for rt in schema.relationship_types.values():
        pairs = sorted((a, b) for a, b in rt.pairs if a in types and b in types)
        if pairs:
            rels[rt.name] = pairs
    return types, rels


def render_schema(schema: Schema) -> tuple[str, str]:
    types, rels = extractable(schema)
    entity_types = "\n".join(f"- {label}: {desc}" if desc else f"- {label}" for label, desc in types.items())
    relationships = "\n".join(f"- {name}: " + ", ".join(f"{a} -> {b}" for a, b in pairs) for name, pairs in rels.items())
    return entity_types, relationships


PLACEHOLDERS = ("entity_types", "relationships", "document", "section", "text")


def render_template(template: str, **values: str) -> str:
    """Replace only the known {placeholders}; other braces (e.g. a JSON example) stay as written."""
    for name in PLACEHOLDERS:
        template = template.replace("{" + name + "}", values.get(name, ""))
    return template


def build_messages(preset: PromptPreset, schema: Schema, text: str, document: str, section: str) -> list[dict]:
    entity_types, relationships = render_schema(schema)
    values = {"entity_types": entity_types, "relationships": relationships, "document": document, "section": section or "-"}
    if "{text}" in preset.template:  # the template places the text itself
        return [{"role": "system", "content": render_template(preset.template, **values, text=text)}]
    return [{"role": "system", "content": render_template(preset.template, **values)}, {"role": "user", "content": text}]


def call_llm(messages: list[dict]) -> str:
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
    return resp.json()["choices"][0]["message"]["content"] or ""


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
        out.append({**{k: v[:MAX_NAME] for k, v in fields.items()}, "confidence": confidence})
    return out


def extract_window(messages: list[dict]) -> list[dict]:
    """LLM call + strict parse, retrying once on malformed output."""
    try:
        return parse_triples(call_llm(messages))
    except ExtractionError as first:
        logger.warning("extraction: malformed output, retrying once: %s", first)
        retry = messages + [{"role": "user", "content": "Your previous answer was not valid. Reply with ONLY the JSON object."}]
        return parse_triples(call_llm(retry))


# --- entity matching -----------------------------------------------------------

def normalize(name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    return re.sub(r"^(the|a|an) ", "", name)


@dataclass
class EntityIndex:
    """Curated nodes by label for matching extracted names to existing ids."""

    by_label: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)  # label -> [(id, name, norm)]

    @classmethod
    def load(cls) -> "EntityIndex":
        index = cls()
        try:
            with session() as s:
                rows = s.run("MATCH (n:Entity) RETURN n.id AS id, n.name AS name, labels(n) AS labels").data()
        except GraphUnavailable:
            return index
        for row in rows:
            label = repository.node_label(row["labels"])
            index.by_label.setdefault(label, []).append((row["id"], row["name"] or "", normalize(row["name"] or "")))
        return index

    def has(self, node_id: str) -> bool:
        return any(node_id == i for rows in self.by_label.values() for i, _, _ in rows)

    def match_exact(self, label: str, name: str) -> str | None:
        """Exact (normalized) name, or the id's last key part (tags like EXTR01, DIE_PLUG)."""
        target = normalize(name)
        if not target:
            return None
        for node_id, _, norm in self.by_label.get(label, []):
            if norm == target or normalize(node_id.rsplit("/", 1)[-1].split(":")[-1]) == target:
                return node_id
        return None

    def match(self, label: str, name: str) -> str | None:
        target = normalize(name)
        if not target:
            return None
        exact = self.match_exact(label, name)
        if exact:
            return exact
        candidates = self.by_label.get(label, [])
        tokens = set(target.split())
        best, best_score = None, 0.0
        for node_id, _, norm in candidates:
            other = set(norm.split())
            if not other:
                continue
            score = len(tokens & other) / len(tokens | other)
            if len(target) >= 4 and (target in norm or norm in target):
                score = max(score, 0.75)
            if score > best_score:
                best, best_score = node_id, score
        return best if best_score >= 0.6 else None


def proposed_id(label: str, name: str, scope_key: str) -> str:
    part = clean_part(name)
    return make_id(label, part) if label == "Asset" else make_id(label, scope_key, part)


def resolve_id(index: EntityIndex, label: str, name: str, scope_key: str) -> tuple[str, bool]:
    """(id, existing): an existing entity's id when the name matches one, else a proposed id."""
    match = index.match(label, name)
    return (match, True) if match else (proposed_id(label, name, scope_key), False)


def triple_key(mode: str, subject_name: str, subject_type: str, predicate: str, object_name: str, object_type: str) -> tuple:
    return (mode, normalize(subject_name), subject_type, predicate, normalize(object_name), object_type)


def scope_key_for(params: dict, doc) -> str:
    """Key prefix for proposed ids: the asset scope (bsk:asset:EXTR01 -> EXTR01) or the document key."""
    return params["asset_id"].split(":", 2)[2] if (params or {}).get("asset_id") else doc.doc_key


# --- the job -------------------------------------------------------------------

def run_extract(job) -> None:
    """Worker entry point for an extract job (ingestion.library.run_job dispatches here)."""
    from ingestion import library
    from ingestion.chunking import chunk_parsed
    from ingestion.models import Document

    doc = Document.objects.get(pk=job.document_id)
    params = job.params
    Document.objects.filter(pk=doc.pk).update(graph_status=Document.PipelineStatus.RUNNING, graph_error="")

    if doc.parse_status != Document.ParseStatus.PARSED or not library.parsed_path(doc).exists():
        library.heartbeat(job, message="parsing first")
        library.run_parse(job)
        doc.refresh_from_db()

    windows = chunk_parsed(library.load_parsed(doc), params["chunking"]["strategy"], params["chunking"]["params"])
    if not windows:
        raise ExtractionError("the document produced no text windows")

    schema = registry.active_schema()
    presets = {m: PromptPreset.objects.get(pk=pk) for m, pk in params["presets"].items()}
    index = EntityIndex.load()
    scope_key = scope_key_for(params, doc)
    total = len(windows) * len(presets)
    library.heartbeat(job, 0, total, f"extracting from {len(windows)} windows ({', '.join(presets)})")

    staged: dict[tuple, CandidateTriple] = {}
    reviewed = {  # already approved/rejected in an earlier run: not staged again
        triple_key(t.mode, t.subject_name, t.subject_type, t.predicate, t.object_name, t.object_type)
        for t in CandidateTriple.objects.filter(document=doc).exclude(status=CandidateTriple.Status.PENDING)
    }
    done = failed = skipped = 0
    for i, window in enumerate(windows):
        section = " > ".join(window.section_path)
        for mode, preset in presets.items():
            raw = []
            messages = build_messages(preset, schema, window.text, doc.filename, section)
            for attempt in (1, 2):
                # Heartbeat before every attempt: one attempt is at most two LLM calls
                # (malformed-output retry), which stays under the stale-job threshold.
                library.heartbeat(job, done, total, None if attempt == 1 else f"window {i + 1}: DGX slow, retrying")
                try:
                    raw = extract_window(messages)
                    break
                except requests.RequestException as exc:  # timeout / connection: the shared DGX is busy
                    if attempt == 2:
                        failed += 1
                        logger.warning("extraction window failed doc=%s window=%d mode=%s: %s", doc.doc_key, i, mode, exc)
                except Exception as exc:  # malformed output after the retry etc.: one bad window doesn't fail the run
                    failed += 1
                    logger.warning("extraction window failed doc=%s window=%d mode=%s: %s", doc.doc_key, i, mode, exc)
                    break
            for t in raw:
                if mode == "schema" and (t["subject_type"] in EVIDENCE_TYPES or t["object_type"] in EVIDENCE_TYPES):
                    continue  # evidence structure is written on approval, not extracted
                candidate = _candidate(t, mode, schema, index, scope_key, doc, job, window, i, preset)
                key = triple_key(mode, candidate.subject_name, candidate.subject_type, candidate.predicate,
                                 candidate.object_name, candidate.object_type)
                if key in reviewed:
                    skipped += 1
                elif key in staged:
                    staged[key].occurrences += 1
                    if (t["confidence"] or 0) > (staged[key].confidence or 0):
                        staged[key].confidence = t["confidence"]
                else:
                    staged[key] = candidate
            done += 1
            library.heartbeat(job, done, total)

    with transaction.atomic():
        # A new run replaces the document's pending triples of the modes it ran; reviewed ones are kept.
        CandidateTriple.objects.filter(document=doc, status=CandidateTriple.Status.PENDING, mode__in=list(presets)).delete()
        CandidateTriple.objects.bulk_create(staged.values())
    Document.objects.filter(pk=doc.pk).update(
        graph_status=Document.PipelineStatus.DONE, graph_updated_at=timezone.now(),
        graph_error=f"{failed} window call(s) failed" if failed else "",
    )
    note = f", {skipped} already reviewed" if skipped else ""
    library.heartbeat(job, total, total, f"{len(staged)} triples staged for review{note}")
    logger.info("extraction done doc=%s windows=%d triples=%d failed_calls=%d", doc.doc_key, len(windows), len(staged), failed)


def _candidate(t, mode, schema, index, scope_key, doc, job, window, i, preset) -> CandidateTriple:
    issue = ""
    subject_id = object_id = ""
    subject_existing = object_existing = False
    if mode == "schema":
        if t["subject_type"] not in schema.entity_types or t["object_type"] not in schema.entity_types:
            issue = f"type not in schema ({t['subject_type']} / {t['object_type']})"
        else:
            try:
                schema.check_relationship(t["subject_type"], t["predicate"], t["object_type"])
            except Exception as exc:
                issue = str(exc)[:255]
            try:
                subject_id, subject_existing = resolve_id(index, t["subject_type"], t["subject"], scope_key)
                object_id, object_existing = resolve_id(index, t["object_type"], t["object"], scope_key)
            except ValueError as exc:  # a name with no usable id characters
                issue = str(exc)[:255]
    return CandidateTriple(
        document=doc, job=job, mode=mode,
        subject_name=t["subject"], subject_type=t["subject_type"], subject_id=subject_id, subject_existing=subject_existing,
        predicate=t["predicate"],
        object_name=t["object"], object_type=t["object_type"], object_id=object_id, object_existing=object_existing,
        confidence=t["confidence"], issue=issue,
        chunk_index=i, page_start=window.page_start, page_end=window.page_end, section_path=window.section_path,
        evidence_text=window.text[:EVIDENCE_CHARS],
        model=settings.DGX_CHAT_MODEL, preset_name=preset.name, preset_version=preset.version,
        schema_version=schema.version,
    )
