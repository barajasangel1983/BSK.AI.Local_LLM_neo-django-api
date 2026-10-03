"""Triple extraction API (`/api/graph/`): prompt presets, extract jobs, staged-triple review.

    GET    presets/                       list prompt presets (defaults created on first use)
    POST   presets/                       create {name, mode, template}
    PATCH  presets/<id>/                  update {name?, template?, is_default?} — a template change bumps the version
    DELETE presets/<id>/                  delete (not the built-in defaults)
    GET    extract/config/                model, modes, default windows, window strategies, schema (read-only)
    POST   extract/                       Generate triples {document_ids, mode, chunking, presets, asset_id}
    GET    triples/?document=&data_file=&status=&mode=&issues=1&new=1&q=&offset=&limit=   (&fields=ids: matching ids only)
    PATCH  triples/<id>/                  edit a pending/rejected triple
    POST   triples/approve/ | reject/ | delete/   {ids: [...]}
    POST   triples/<id>/promote/          lab triple -> curated {subject_type, predicate, object_type, subject_id?, object_id?}
"""

from django.db import transaction
from django.db.models import Count, Q
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ingestion import library
from ingestion.library_views import job_json
from ingestion.models import Document, Job

from . import extraction, registry, triples
from .evidence import evidence_json
from .models import CandidateTriple, Evidence, PromptPreset
from .views import _int, graph_errors

MAX_TRIPLES = 500
MAX_IDS = 10000


def _error(message: str, code=status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


# --- presets -----------------------------------------------------------------

def preset_json(p: PromptPreset) -> dict:
    builtin = {name: template for name, template in extraction.DEFAULT_PRESETS.values()}
    return {
        "id": p.pk, "name": p.name, "mode": p.mode, "template": p.template, "version": p.version,
        "is_default": p.is_default, "builtin": p.name in builtin,
        "builtin_template": builtin.get(p.name),  # lets the UI restore a built-in preset
        "placeholders": list(extraction.PLACEHOLDERS),
        "created_at": p.created_at.isoformat(), "updated_at": p.updated_at.isoformat(),
    }


def _set_default(preset: PromptPreset) -> None:
    PromptPreset.objects.filter(mode=preset.mode).exclude(pk=preset.pk).update(is_default=False)


@api_view(["GET", "POST"])
def presets(request):
    extraction.ensure_default_presets()
    if request.method == "GET":
        return Response({"presets": [preset_json(p) for p in PromptPreset.objects.all()]})
    name = str(request.data.get("name") or "").strip()
    mode = request.data.get("mode")
    template = str(request.data.get("template") or "")
    if not name or len(name) > 100:
        return _error("name is required (max 100 characters)")
    if mode not in PromptPreset.Mode.values:
        return _error(f"mode must be one of {', '.join(PromptPreset.Mode.values)}")
    if not template.strip():
        return _error("template is required")
    if PromptPreset.objects.filter(name=name).exists():
        return _error(f"a preset named {name!r} already exists", status.HTTP_409_CONFLICT)
    with transaction.atomic():
        preset = PromptPreset.objects.create(name=name, mode=mode, template=template,
                                             is_default=bool(request.data.get("is_default")))
        if preset.is_default:
            _set_default(preset)
    return Response(preset_json(preset), status=status.HTTP_201_CREATED)


@api_view(["PATCH", "DELETE"])
def preset_detail(request, preset_id):
    preset = PromptPreset.objects.filter(pk=preset_id).first()
    if preset is None:
        return _error("preset not found", status.HTTP_404_NOT_FOUND)
    builtin_names = {name for name, _ in extraction.DEFAULT_PRESETS.values()}
    if request.method == "DELETE":
        if preset.name in builtin_names:
            return _error("built-in presets can't be deleted (edit or restore them instead)", status.HTTP_409_CONFLICT)
        was_default, mode = preset.is_default, preset.mode
        preset.delete()
        if was_default:  # fall back to the built-in default for that mode
            PromptPreset.objects.filter(mode=mode, name__in=builtin_names).update(is_default=True)
        return Response(status=status.HTTP_204_NO_CONTENT)

    data = request.data
    with transaction.atomic():
        if "name" in data:
            name = str(data["name"] or "").strip()
            if preset.name in builtin_names and name != preset.name:
                return _error("built-in presets can't be renamed")
            if not name or len(name) > 100:
                return _error("name is required (max 100 characters)")
            if PromptPreset.objects.filter(name=name).exclude(pk=preset.pk).exists():
                return _error(f"a preset named {name!r} already exists", status.HTTP_409_CONFLICT)
            preset.name = name
        if "template" in data:
            template = str(data["template"] or "")
            if not template.strip():
                return _error("template must not be empty")
            if template != preset.template:
                preset.template = template
                preset.version += 1   # triples record name + version of the prompt that produced them
        if data.get("is_default") is True:
            preset.is_default = True
            _set_default(preset)
        preset.save()
    return Response(preset_json(preset))


# --- extract -------------------------------------------------------------------

@api_view(["GET"])
def extract_config(request):
    """Server-side extraction settings for the GraphLab Extract UI (read-only)."""
    from django.conf import settings

    from ingestion import chunking

    schema = registry.active_schema()
    types, rels = extraction.extractable(schema)
    return Response({
        "model": settings.DGX_CHAT_MODEL,
        "provider": "DGX",
        "modes": list(extraction.MODES),
        "default_chunking": extraction.DEFAULT_CHUNKING,
        "chunking_strategies": chunking.STRATEGIES,
        "timeout_seconds": settings.GRAPH_EXTRACT_TIMEOUT,
        "schema": {  # what schema-mode triples may use (evidence types excluded)
            "version": schema.version,
            "entity_types": list(types),
            "relationships": [{"name": name, "pairs": pairs, "aliases": list(schema.relationship_types[name].aliases)}
                              for name, pairs in rels.items()],
            "subtypes": {label: list(schema.subtypes.get(label, ())) for label in types if schema.subtypes.get(label)},
        },
    })


@api_view(["POST"])
def extract(request):
    ids = request.data.get("document_ids") or []
    if not isinstance(ids, list) or not ids:
        return _error("document_ids must be a non-empty list")
    docs = {str(d.id): d for d in Document.objects.filter(pk__in=ids)}
    missing = [i for i in ids if str(i) not in docs]
    if missing:
        return _error(f"unknown document(s): {', '.join(map(str, missing))}", status.HTTP_404_NOT_FOUND)
    params = {k: request.data.get(k) for k in ("mode", "chunking", "presets", "asset_id") if request.data.get(k)}
    try:
        params = extraction.validate_params(params)   # validate once, before anything is queued
        jobs = [library.enqueue(docs[str(i)], Job.Kind.EXTRACT, params) for i in ids]
    except ValueError as exc:
        return _error(str(exc))
    return Response({"jobs": [job_json(j) for j in jobs]}, status=status.HTTP_202_ACCEPTED)


# --- triples -------------------------------------------------------------------

def triple_json(t: CandidateTriple) -> dict:
    return {
        "id": t.pk, "document_id": str(t.document_id) if t.document_id else None,
        "data_file_id": str(t.data_file_id) if t.data_file_id else None, "source": t.source,
        "job_id": str(t.job_id) if t.job_id else None,
        "mode": t.mode, "status": t.status, "layer": t.layer or None,
        "subject": {"name": t.subject_name, "type": t.subject_type, "id": t.subject_id or None, "existing": t.subject_existing,
                    "match": t.subject_match or None, "candidates": t.subject_candidates or [],
                    "properties": t.subject_props or {}},
        "predicate": t.predicate,
        "object": {"name": t.object_name, "type": t.object_type, "id": t.object_id or None, "existing": t.object_existing,
                   "match": t.object_match or None, "candidates": t.object_candidates or [],
                   "properties": t.object_props or {}},
        "confidence": t.confidence, "issue": t.issue, "occurrences": t.occurrences, "edited": t.edited,
        # Every place the triple was found (P8a); `evidence` below is the first one (kept for older clients).
        "sources": [evidence_json(e) for e in t.evidence.all()],
        "evidence": {"chunk_index": t.chunk_index, "page_start": t.page_start, "page_end": t.page_end,
                     "section_path": t.section_path, "text": t.evidence_text, "row": t.row_number},
        "provenance": {"model": t.model, "preset": t.preset_name, "preset_version": t.preset_version,
                       "schema_version": t.schema_version},
        "created_at": t.created_at.isoformat(), "updated_at": t.updated_at.isoformat(),
        "committed_at": t.committed_at.isoformat() if t.committed_at else None,
    }


@api_view(["GET"])
def triple_list(request):
    qp = request.query_params
    qs = CandidateTriple.objects.select_related("document").prefetch_related("evidence")
    if qp.get("document"):
        qs = qs.filter(document_id=qp["document"])
    if qp.get("data_file"):
        qs = qs.filter(data_file_id=qp["data_file"])
    if qp.get("status"):
        qs = qs.filter(status__in=qp["status"].split(","))
    if qp.get("mode"):
        qs = qs.filter(mode=qp["mode"])
    if qp.get("issues") in ("1", "true"):
        qs = qs.exclude(issue="")
    if qp.get("new") in ("1", "true"):   # triples that would add an entity to the graph
        qs = qs.filter(Q(subject_existing=False) | Q(object_existing=False))
    if qp.get("q"):
        q = qp["q"]
        qs = qs.filter(Q(subject_name__icontains=q) | Q(object_name__icontains=q) | Q(predicate__icontains=q))
    counts = {s: 0 for s in CandidateTriple.Status.values}
    for row in qs.order_by().values("status").annotate(n=Count("id")):
        counts[row["status"]] = row["n"]
    if qp.get("fields") == "ids":   # "select all matching" in the review table
        return Response({"total": qs.count(), "ids": list(qs.values_list("id", flat=True)[:MAX_IDS])})
    try:
        offset = _int(request, "offset", 0, maximum=10**9)
        limit = max(1, _int(request, "limit", 200, maximum=MAX_TRIPLES))
    except ValueError as exc:
        return _error(str(exc))
    return Response({
        "total": qs.count(), "counts": counts, "offset": offset, "limit": limit,
        "triples": [triple_json(t) for t in qs[offset:offset + limit]],
    })


@api_view(["GET"])
def evidence_detail(request, evidence_id):
    """GET /api/graph/evidence/<id>/ — one evidence record and the triples staged from it."""
    e = Evidence.objects.filter(pk=evidence_id).first()
    if e is None:
        return _error("evidence not found", status.HTTP_404_NOT_FOUND)
    return Response({**evidence_json(e), "triple_ids": list(e.triples.values_list("id", flat=True))})


@api_view(["PATCH"])
@graph_errors
def triple_detail(request, triple_id):
    triple = CandidateTriple.objects.select_related("document", "data_file", "job").filter(pk=triple_id).first()
    if triple is None:
        return _error("triple not found", status.HTTP_404_NOT_FOUND)
    try:
        triples.edit(triple, dict(request.data))
    except triples.TripleError as exc:
        return _error(str(exc))
    return Response(triple_json(triple))


def _selected(request) -> list[CandidateTriple] | Response:
    ids = request.data.get("ids") or []
    if not isinstance(ids, list) or not ids:
        return _error("ids must be a non-empty list")
    if len(ids) > MAX_TRIPLES:
        return _error(f"at most {MAX_TRIPLES} triples per request")
    found = list(CandidateTriple.objects.select_related("document", "data_file", "job").filter(pk__in=ids))
    if len(found) != len(set(map(str, ids))):
        known = {str(t.pk) for t in found}
        return _error(f"unknown triple(s): {', '.join(str(i) for i in ids if str(i) not in known)}",
                      status.HTTP_404_NOT_FOUND)
    return found


def _settle_graph_status(doc_ids) -> None:
    """A document with approved triples counts as in the graph."""
    for doc in Document.objects.filter(pk__in=set(doc_ids)):
        if doc.graph_status in (Document.PipelineStatus.QUEUED, Document.PipelineStatus.RUNNING):
            continue
        has_any = doc.triples.exists()
        doc.graph_status = Document.PipelineStatus.DONE if has_any else Document.PipelineStatus.NONE
        doc.save(update_fields=["graph_status"])


@api_view(["POST"])
@graph_errors
def triples_approve(request):
    selected = _selected(request)
    if isinstance(selected, Response):
        return selected
    return Response(triples.approve(selected))


@api_view(["POST"])
def triples_reject(request):
    selected = _selected(request)
    if isinstance(selected, Response):
        return selected
    return Response({"rejected": triples.reject(selected)})


@api_view(["POST"])
@graph_errors
def triples_delete(request):
    selected = _selected(request)
    if isinstance(selected, Response):
        return selected
    doc_ids = [t.document_id for t in selected if t.document_id]
    deleted = triples.delete(selected)
    _settle_graph_status(doc_ids)
    return Response({"deleted": deleted})


@api_view(["POST"])
@graph_errors
def triple_promote(request, triple_id):
    triple = CandidateTriple.objects.select_related("document", "data_file", "job").filter(pk=triple_id).first()
    if triple is None:
        return _error("triple not found", status.HTTP_404_NOT_FOUND)
    missing = [k for k in ("subject_type", "predicate", "object_type") if not request.data.get(k)]
    if missing:
        return _error(f"required: {', '.join(missing)}")
    try:
        triples.promote(triple, dict(request.data))
    except triples.TripleError as exc:
        return _error(str(exc))
    triple.refresh_from_db()
    return Response(triple_json(triple))
