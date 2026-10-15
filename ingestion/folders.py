"""Library folders: create, rename, move, delete; move documents between folders.

A folder is a kind of document (its root folder's `kind`), so it travels with the
document's chunks as metadata (`doc_type`, `folder_path`): a search can be limited
to SOPs or Quality Alerts without new embeddings. Deleting a folder never deletes a
document: its documents go to another folder or to Unfiled.
"""

from __future__ import annotations

import re

from django.db import transaction
from django.db.models import Count

from .models import Document, Folder
from .vector_store import VectorStore

MAX_DEPTH = 6
UNFILED = "unfiled"


class FolderError(ValueError):
    """Invalid folder name, move or delete."""


def kind_for(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:64] or "folder"


def _clean(name) -> str:
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    if not name:
        raise FolderError("a folder needs a name")
    if len(name) > 120 or "/" in name:
        raise FolderError("a folder name is at most 120 characters, without '/'")
    return name


def _check_free(name: str, parent: Folder | None, exclude: Folder | None = None) -> None:
    clash = Folder.objects.filter(parent=parent, name__iexact=name)
    if exclude:
        clash = clash.exclude(pk=exclude.pk)
    if clash.exists():
        raise FolderError(f"there is already a folder {name!r} here")


def ancestors(folder: Folder | None) -> list[Folder]:
    chain = []
    while folder is not None:
        chain.append(folder)
        folder = folder.parent
    return chain[::-1]


def path(folder: Folder | None) -> str:
    return "/".join(f.name for f in ancestors(folder))


def subtree(folder: Folder) -> list[Folder]:
    out, todo = [], [folder]
    while todo:
        f = todo.pop()
        out.append(f)
        todo += list(f.children.all())
    return out


def tree() -> dict:
    """Every folder (flat, with parent and counts) and the Unfiled count."""
    counts = dict(Document.objects.values_list("folder").annotate(n=Count("id")))
    folders = list(Folder.objects.all())
    by_id = {f.id: f for f in folders}

    def full_path(f):
        names = []
        while f is not None:
            names.append(f.name)
            f = by_id.get(f.parent_id)
        return "/".join(reversed(names))

    return {"folders": [{"id": str(f.id), "name": f.name, "parent": str(f.parent_id) if f.parent_id else None,
                         "kind": f.kind, "path": full_path(f), "position": f.position, "standard": f.standard,
                         "document_count": counts.get(f.id, 0)} for f in folders],
            "unfiled_count": counts.get(None, 0)}


def get(folder_id) -> Folder | None:
    if folder_id in (None, "", UNFILED):
        return None
    try:
        folder = Folder.objects.filter(pk=folder_id).first()
    except Exception:   # not a UUID
        folder = None
    if folder is None:
        raise FolderError(f"unknown folder {folder_id!r}")
    return folder


def get_by_path(folder_path: str, create: bool = False) -> Folder | None:
    """`Manuals/ABB` -> the folder (made on the way when `create`)."""
    parent = None
    for name in [p for p in folder_path.split("/") if p.strip()]:
        name = _clean(name)
        folder = Folder.objects.filter(parent=parent, name__iexact=name).first()
        if folder is None:
            if not create:
                raise FolderError(f"no folder {folder_path!r}")
            folder = create_folder(name, parent)
        parent = folder
    return parent


@transaction.atomic
def create_folder(name, parent: Folder | None = None) -> Folder:
    name = _clean(name)
    if parent is not None and len(ancestors(parent)) >= MAX_DEPTH:
        raise FolderError(f"folders nest at most {MAX_DEPTH} levels")
    _check_free(name, parent)
    position = (Folder.objects.filter(parent=parent).order_by("-position").values_list("position", flat=True).first() or 0) + 1
    return Folder.objects.create(name=name, parent=parent, kind=parent.kind if parent else kind_for(name), position=position)


@transaction.atomic
def update_folder(folder: Folder, name=None, parent=..., ) -> Folder:
    """Rename and/or move (`parent`: a folder, or None for the top level)."""
    moved = parent is not ... and parent != folder.parent
    new_parent = parent if moved else folder.parent
    if moved and new_parent is not None:
        if any(f.pk == folder.pk for f in ancestors(new_parent)):
            raise FolderError("a folder cannot go inside itself")
        depth = max(len(ancestors(f)) - len(ancestors(folder)) for f in subtree(folder)) + 1
        if len(ancestors(new_parent)) + depth > MAX_DEPTH:
            raise FolderError(f"folders nest at most {MAX_DEPTH} levels")
    new_name = _clean(name) if name is not None else folder.name
    _check_free(new_name, new_parent, exclude=folder)
    folder.name, folder.parent = new_name, new_parent
    if new_parent is None and moved:
        folder.kind = kind_for(new_name) if not folder.standard else folder.kind
    folder.save()
    kind = folder.parent.kind if folder.parent else folder.kind
    for f in subtree(folder):
        if f.kind != kind:
            Folder.objects.filter(pk=f.pk).update(kind=kind)
    sync_chunks(Document.objects.filter(folder__in=subtree(folder)))
    return folder


@transaction.atomic
def delete_folder(folder: Folder, move_to=...) -> dict:
    """Delete a folder and its subfolders. Their documents must go somewhere (`move_to`:
    a folder outside it, or None for Unfiled); without `move_to` only an empty folder goes."""
    folders = subtree(folder)
    docs = Document.objects.filter(folder__in=folders)
    count = docs.count()
    if count and move_to is ...:
        raise FolderError(f"{folder.name!r} holds {count} document(s): choose where they go")
    if move_to is not ... and move_to is not None and any(f.pk == move_to.pk for f in folders):
        raise FolderError("the documents cannot go to a folder that is being deleted")
    target = None if move_to is ... else move_to
    moved = list(docs)
    docs.update(folder=target)
    for f in sorted(folders, key=lambda f: -len(ancestors(f))):   # children first (PROTECT)
        f.delete()
    sync_chunks(Document.objects.filter(pk__in=[d.pk for d in moved]))
    return {"folders_deleted": len(folders), "documents_moved": count}


def move_documents(ids: list, folder: Folder | None) -> int:
    docs = Document.objects.filter(pk__in=ids)
    if docs.count() != len(set(map(str, ids))):
        raise FolderError("unknown document(s)")
    docs.update(folder=folder)
    sync_chunks(Document.objects.filter(pk__in=ids))
    return len(ids)


def chunk_metadata(doc: Document) -> dict:
    """What a chunk carries about its document's place (written on embed, updated on a move)."""
    folder = doc.folder
    return {"doc_type": folder.kind if folder else UNFILED, "folder_path": path(folder) if folder else "",
            "plant_asset": doc.plant_asset or "", "title": doc.title or doc.filename}


def sync_chunks(docs) -> None:
    """Update the chunks of embedded documents (metadata only; no new embeddings)."""
    docs = [d for d in docs.select_related("folder") if d.rag_chunk_count]
    if not docs:
        return
    store = VectorStore()
    for doc in docs:
        store.update_document_metadata(doc.doc_key, chunk_metadata(doc))
