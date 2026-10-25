"""Structured data import API (`/api/graph/`): data files, column mappings, staging.

    GET    datafiles/                     list data files (with triple counts)
    POST   datafiles/                     upload CSV / XLSX (multipart "files")
    GET    datafiles/<id>/                detail: columns, sample rows, template suggestions
    PATCH  datafiles/<id>/                {sheet?, header_row?} — re-reads the columns
    DELETE datafiles/<id>/                delete (approved triples leave the graph)
    POST   datafiles/<id>/preview/        {mapping, asset_id} -> errors, stats, first triples (nothing is saved)
    POST   datafiles/<id>/stage/          {mapping, asset_id} -> staged triples for review
    POST   plc-source/                    {asset_id} -> the PLC's tag list and alarm list as two data files (P14a)
    GET    assets/<id>/plc-compare/       the graph against the PLC, signal by signal and alarm by alarm (P14a)
    GET    mappings/                      saved column mappings
    POST   mappings/                      {name, mapping} (same name: replaced)
    DELETE mappings/<id>/

Staged triples are reviewed through triples/?data_file=<id> (see extraction_views).
"""

from django.db.models import Count
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from . import plc_source, structured
from .models import CandidateTriple, DataFile, ImportMapping
from .views import graph_errors

PREVIEW_TRIPLES = 50
PREVIEW_ERRORS = 100


def _error(message: str, code=status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


def datafile_json(df: DataFile, counts: dict | None = None) -> dict:
    if counts is None:
        counts = {s: 0 for s in CandidateTriple.Status.values}
        for row in df.triples.order_by().values("status").annotate(n=Count("id")):
            counts[row["status"]] = row["n"]
    return {
        "id": str(df.id), "filename": df.filename, "key": df.key, "kind": df.kind, "size": df.size,
        "sheets": df.sheets, "sheet": df.sheet, "header_row": df.header_row,
        "columns": df.columns, "row_count": df.row_count,
        "asset_id": df.asset_id or None, "mapping": df.mapping or None,
        "uploaded_at": df.uploaded_at.isoformat(), "staged_at": df.staged_at.isoformat() if df.staged_at else None,
        "triples": counts,
        "superseded_by": str(df.superseded_by_id) if df.superseded_by_id else None,       # P26f
        "superseded_by_uploaded_at": df.superseded_by.uploaded_at.isoformat() if df.superseded_by_id else None,
        "superseded_at": df.superseded_at.isoformat() if df.superseded_at else None,
    }


def _preview_triple(t: CandidateTriple) -> dict:
    return {
        "row": t.row_number, "occurrences": t.occurrences,
        "subject": {"name": t.subject_name, "type": t.subject_type, "id": t.subject_id, "existing": t.subject_existing,
                    "match": t.subject_match or None, "candidates": t.subject_candidates, "properties": t.subject_props,
                    "changes": (t.replaces or {}).get(t.subject_id) or {}},
        "predicate": t.predicate,
        "object": {"name": t.object_name, "type": t.object_type, "id": t.object_id, "existing": t.object_existing,
                   "match": t.object_match or None, "candidates": t.object_candidates, "properties": t.object_props,
                   "changes": (t.replaces or {}).get(t.object_id) or {}},
    }


@api_view(["GET", "POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def datafiles(request):
    if request.method == "GET":
        counts: dict = {}
        for row in (CandidateTriple.objects.filter(data_file__isnull=False).order_by()
                    .values("data_file_id", "status").annotate(n=Count("id"))):
            counts.setdefault(row["data_file_id"], {s: 0 for s in CandidateTriple.Status.values})[row["status"]] = row["n"]
        empty = {s: 0 for s in CandidateTriple.Status.values}
        files = list(DataFile.objects.select_related("superseded_by"))
        # P26f: older versions of a file are hidden unless ?all=1; the list says how many, and how many still have
        # pending triples, so nothing waiting for review is out of sight without a word.
        older = [df for df in files if df.superseded_by_id]
        shown = files if request.query_params.get("all") in ("1", "true") else [df for df in files if not df.superseded_by_id]
        return Response({"data_files": [datafile_json(df, counts.get(df.id, empty)) for df in shown],
                         "older_versions": {"count": len(older),
                                            "with_pending": sum(1 for df in older if counts.get(df.id, empty)["pending"])}})
    files = request.FILES.getlist("files")
    if not files:
        return _error("no files uploaded (multipart field 'files')")
    results = []
    for f in files:
        try:
            df, created = structured.add_data_file(f)
            results.append({"filename": f.name, "created": created, "data_file": datafile_json(df)})
        except structured.ImportError_ as exc:
            results.append({"filename": f.name, "created": False, "error": str(exc)})
    return Response({"results": results}, status=status.HTTP_201_CREATED)


@api_view(["GET", "PATCH", "DELETE"])
@graph_errors
def datafile_detail(request, file_id):
    df = DataFile.objects.filter(pk=file_id).first()
    if df is None:
        return _error("data file not found", status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        return Response({"data_file_id": str(file_id), **structured.delete_data_file(df)})
    if request.method == "PATCH":
        sheet = request.data.get("sheet", df.sheet)
        if df.kind == "xlsx" and sheet not in df.sheets:
            return _error(f"unknown sheet {sheet!r}")
        try:
            header_row = int(request.data.get("header_row", df.header_row))
        except (TypeError, ValueError):
            return _error("header_row must be an integer")
        if header_row < 1:
            return _error("header_row starts at 1")
        df.sheet, df.header_row = sheet, header_row
        try:
            structured.refresh_columns(df)
        except structured.ImportError_ as exc:
            return _error(str(exc))
        df.save()
    return Response({**datafile_json(df), "sample": structured.sample(df),
                     "suggestions": structured.suggestions(df.columns)})


def _mapping_request(request, df: DataFile):
    mapping = request.data.get("mapping")
    asset_id = str(request.data.get("asset_id") or "")
    return mapping, asset_id, structured.check(df, mapping, asset_id)


@api_view(["POST"])
def datafile_preview(request, file_id):
    df = DataFile.objects.filter(pk=file_id).first()
    if df is None:
        return _error("data file not found", status.HTTP_404_NOT_FOUND)
    mapping, asset_id, errors = _mapping_request(request, df)
    if errors:
        return Response({"valid": False, "errors": errors, "stats": None, "triples": [], "row_errors": []})
    try:
        result = structured.build(df, mapping, asset_id)
    except structured.ImportError_ as exc:
        return Response({"valid": False, "errors": [str(exc)], "stats": None, "triples": [], "row_errors": []})
    return Response({
        "valid": True, "errors": [], "stats": result.stats(),
        "triples": [_preview_triple(t) for t in result.triples[:PREVIEW_TRIPLES]],
        "row_errors": result.row_errors[:PREVIEW_ERRORS],
    })


@api_view(["POST"])
def datafile_stage(request, file_id):
    df = DataFile.objects.filter(pk=file_id).first()
    if df is None:
        return _error("data file not found", status.HTTP_404_NOT_FOUND)
    try:
        result = structured.stage(df, request.data.get("mapping"), str(request.data.get("asset_id") or ""))
    except structured.ImportError_ as exc:
        return _error(str(exc))
    return Response({**result, "data_file": datafile_json(df)})


@api_view(["GET", "POST"])
def mappings(request):
    if request.method == "GET":
        return Response({"mappings": [{"id": m.pk, "name": m.name, "mapping": m.mapping,
                                       "updated_at": m.updated_at.isoformat()} for m in ImportMapping.objects.all()]})
    name = str(request.data.get("name") or "").strip()
    mapping = request.data.get("mapping")
    if not name or len(name) > 100:
        return _error("name is required (max 100 characters)")
    if not isinstance(mapping, dict) or not mapping.get("entities") or not mapping.get("relationships"):
        return _error("mapping must have entities and relationships")
    saved, created = ImportMapping.objects.update_or_create(name=name, defaults={"mapping": mapping})
    return Response({"id": saved.pk, "name": saved.name, "mapping": saved.mapping, "updated_at": saved.updated_at.isoformat()},
                    status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


@api_view(["DELETE"])
def mapping_detail(request, mapping_id):
    deleted, _ = ImportMapping.objects.filter(pk=mapping_id).delete()
    if not deleted:
        return _error("mapping not found", status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@graph_errors
def plc_import(request):
    """The PLC (OPC UA) as an import source: two data files, staged and reviewed like uploaded ones."""
    asset_id = (request.data or {}).get("asset_id") if isinstance(request.data, dict) else None
    if not asset_id:
        return _error("asset_id is required")
    try:
        made = plc_source.create_data_files(asset_id)
    except plc_source.PlcSourceError as exc:
        return _error(str(exc))
    except structured.ImportError_ as exc:
        return _error(str(exc))
    return Response({"spare_slots": made["spare_slots"],
                     "files": [{"template": f["template"], "created": f["created"], "data_file": datafile_json(f["data_file"])}
                               for f in made["files"]]})


@api_view(["GET"])
@graph_errors
def plc_compare(request, asset_id):
    try:
        return Response(plc_source.compare(asset_id))
    except plc_source.PlcSourceError as exc:
        return _error(str(exc))

