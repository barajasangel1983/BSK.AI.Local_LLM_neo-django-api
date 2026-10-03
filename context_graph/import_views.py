"""Structured data import API (`/api/graph/`): data files, column mappings, staging.

    GET    datafiles/                     list data files (with triple counts)
    POST   datafiles/                     upload CSV / XLSX (multipart "files")
    GET    datafiles/<id>/                detail: columns, sample rows, template suggestions
    PATCH  datafiles/<id>/                {sheet?, header_row?} — re-reads the columns
    DELETE datafiles/<id>/                delete (approved triples leave the graph)
    POST   datafiles/<id>/preview/        {mapping, asset_id} -> errors, stats, first triples (nothing is saved)
    POST   datafiles/<id>/stage/          {mapping, asset_id} -> staged triples for review
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

from . import structured
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
    }


def _preview_triple(t: CandidateTriple) -> dict:
    return {
        "row": t.row_number, "occurrences": t.occurrences,
        "subject": {"name": t.subject_name, "type": t.subject_type, "id": t.subject_id, "existing": t.subject_existing,
                    "match": t.subject_match or None, "candidates": t.subject_candidates, "properties": t.subject_props},
        "predicate": t.predicate,
        "object": {"name": t.object_name, "type": t.object_type, "id": t.object_id, "existing": t.object_existing,
                   "match": t.object_match or None, "candidates": t.object_candidates, "properties": t.object_props},
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
        return Response({"data_files": [datafile_json(df, counts.get(df.id, empty)) for df in DataFile.objects.all()]})
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
