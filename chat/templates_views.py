"""Prompt templates, shared by everyone (P15b).

    GET    /api/chat/templates/            the list
    POST   /api/chat/templates/            {name, content, category?}
    DELETE /api/chat/templates/<id>/
    POST   /api/chat/templates/import/     {templates: [{name, content, category}]}: a browser's own templates,
                                           once; one with the same name and content is not added twice
"""

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import PromptTemplate


def _json(t: PromptTemplate) -> dict:
    return {"id": str(t.id), "name": t.name, "content": t.content, "category": t.category,
            "created_at": t.created_at.isoformat()}


def _clean(data) -> tuple[dict | None, str]:
    name = str((data or {}).get("name") or "").strip()[:120]
    content = str((data or {}).get("content") or "").strip()
    category = str((data or {}).get("category") or "General").strip()[:60] or "General"
    if not name or not content:
        return None, "a template needs a name and a text"
    return {"name": name, "content": content, "category": category}, ""


@api_view(["GET", "POST"])
def templates(request):
    if request.method == "POST":
        fields, error = _clean(request.data)
        if error:
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)
        return Response(_json(PromptTemplate.objects.create(**fields)), status=status.HTTP_201_CREATED)
    return Response({"templates": [_json(t) for t in PromptTemplate.objects.all()]})


@api_view(["DELETE"])
def template_detail(request, pk):
    deleted, _ = PromptTemplate.objects.filter(pk=pk).delete()
    if not deleted:
        return Response({"error": "template not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
def templates_import(request):
    items = request.data.get("templates")
    if not isinstance(items, list):
        return Response({"error": "templates must be a list"}, status=status.HTTP_400_BAD_REQUEST)
    added = 0
    for item in items[:200]:
        fields, error = _clean(item if isinstance(item, dict) else {})
        if error or PromptTemplate.objects.filter(name=fields["name"], content=fields["content"]).exists():
            continue
        PromptTemplate.objects.create(**fields)
        added += 1
    return Response({"added": added, "templates": [_json(t) for t in PromptTemplate.objects.all()]})
