"""/api/graph/prompts/ — the Prompts tab (P23): prompts by facet, every version kept, validate, export / import,
built-in updates and a test on a page."""

import re

import requests
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ingestion.models import Document

from . import extraction, prompts
from .models import PromptPreset
from .views import graph_errors


def _error(message: str, code=status.HTTP_400_BAD_REQUEST) -> Response:
    return Response({"error": message}, status=code)


def prompt_json(p: PromptPreset, versions: bool = False) -> dict:
    active = p.versions.filter(version=p.version).first()
    out = {"id": p.pk, "name": p.name, "mode": p.mode, "template": p.template, "version": p.version,
           "is_default": p.is_default, "builtin": p.name in extraction.builtin_templates(), "facet": prompts.facet_of(p),
           "schema_version": active.schema_version if active else None, "builtin_update": prompts.builtin_update(p),
           "placeholders": list(extraction.PLACEHOLDERS), "updated_at": p.updated_at.isoformat()}
    if versions:
        out["versions"] = prompts.versions_json(p)
    return out


def _preset(prompt_id):
    return PromptPreset.objects.filter(pk=prompt_id).first()


@api_view(["GET"])
def prompt_list(request):
    extraction.ensure_default_presets()
    order = {r["key"]: n for n, r in enumerate(extraction.FACET_RUNS)}
    items = sorted(PromptPreset.objects.all(), key=lambda p: (order.get(prompts.facet_of(p)["key"], 99), p.mode, p.name))
    return Response({"prompts": [prompt_json(p) for p in items],
                     "facets": [{"key": r["key"], "label": r["label"]} for r in extraction.FACET_RUNS]})


@api_view(["GET"])
def prompt_detail(request, prompt_id):
    p = _preset(prompt_id)
    return Response(prompt_json(p, versions=True)) if p else _error("prompt not found", status.HTTP_404_NOT_FOUND)


@api_view(["POST"])
def prompt_save(request, prompt_id):
    """A new version, active at once."""
    p = _preset(prompt_id)
    if p is None:
        return _error("prompt not found", status.HTTP_404_NOT_FOUND)
    try:
        prompts.save_version(p, str(request.data.get("template") or ""), note=str(request.data.get("note") or ""),
                             author=str(request.data.get("author") or ""))
    except prompts.PromptError as exc:
        return _error(str(exc))
    return Response(prompt_json(p, versions=True), status=status.HTTP_201_CREATED)


@api_view(["POST"])
def prompt_activate(request, prompt_id, version):
    p = _preset(prompt_id)
    if p is None:
        return _error("prompt not found", status.HTTP_404_NOT_FOUND)
    try:
        prompts.activate(p, version)
    except prompts.PromptError as exc:
        return _error(str(exc), status.HTTP_404_NOT_FOUND)
    return Response(prompt_json(p, versions=True))


@api_view(["POST"])
def prompt_adopt_builtin(request, prompt_id):
    """The code's newer built-in text as a new version."""
    p = _preset(prompt_id)
    if p is None:
        return _error("prompt not found", status.HTTP_404_NOT_FOUND)
    text = prompts.builtin_update(p)
    if text is None:
        return _error("the active version is the built-in text")
    try:
        prompts.save_version(p, text, note="adopted the updated built-in text", source="adopt",
                             author=str(request.data.get("author") or ""))
    except prompts.PromptError as exc:
        return _error(str(exc))
    return Response(prompt_json(p, versions=True))


@api_view(["POST"])
@graph_errors
def prompt_validate(request):
    return Response(prompts.check(str(request.data.get("template") or ""), request.data.get("mode") or "schema"))


def _yaml_response(text: str, filename: str) -> HttpResponse:
    response = HttpResponse(text, content_type="application/yaml; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@api_view(["GET"])
def prompt_export(request, prompt_id):
    p = _preset(prompt_id)
    if p is None:
        return _error("prompt not found", status.HTTP_404_NOT_FOUND)
    version = request.query_params.get("version")
    if version and not p.versions.filter(version=int(version) if version.isdigit() else -1).exists():
        return _error(f"{p.name} has no version {version}", status.HTTP_404_NOT_FOUND)
    data = prompts.export_one(p, int(version) if version else None)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", p.name).strip("-").lower()
    return _yaml_response(prompts.dump(data), f"prompt-{slug}-v{data['version']}.yaml")


@api_view(["GET"])
def prompt_export_all(request):
    extraction.ensure_default_presets()
    return _yaml_response(prompts.dump({"prompts": [prompts.export_one(p) for p in PromptPreset.objects.all()]}),
                          "prompts.yaml")


@api_view(["POST"])
def prompt_import(request):
    try:
        results = prompts.import_text(str(request.data.get("text") or ""), author=str(request.data.get("author") or ""))
    except prompts.PromptError as exc:
        return _error(str(exc))
    return Response({"results": results})


@api_view(["POST"])
@graph_errors
def prompt_test(request):
    """Run a prompt on up to 3 pages; nothing is stored."""
    data = request.data
    doc = Document.objects.filter(pk=data.get("document_id") or None).first()
    if doc is None:
        return _error("document not found", status.HTTP_404_NOT_FOUND)
    p = _preset(data.get("prompt_id")) if data.get("prompt_id") else None
    template = str(data.get("template") or (p.template if p else ""))
    mode = p.mode if p else str(data.get("mode") or "schema")
    try:
        result = prompts.test_on_pages(doc, template, mode, str(data.get("pages") or ""),
                                       asset_id=str(data.get("asset_id") or ""), anchor_id=str(data.get("anchor_id") or ""),
                                       name=p.name if p else "(draft)",
                                       version=p.version if p and template == p.template else 0)
    except (prompts.PromptError, extraction.ExtractionError) as exc:
        return _error(str(exc))
    except requests.RequestException as exc:
        return _error(f"the DGX did not answer ({exc.__class__.__name__})", status.HTTP_502_BAD_GATEWAY)
    return Response(result)
