"""Versioned global schema registry (SQLite) — the active version validates graph writes."""

from __future__ import annotations

from django.db import transaction

from .models import GraphSchemaVersion
from .schema import Schema, SchemaError, validate_definition


def active_schema() -> Schema:
    """The active schema; falls back to the bundled default if none is stored yet."""
    row = GraphSchemaVersion.objects.filter(is_active=True).first()
    if row is None:
        return Schema.from_yaml()
    return Schema.from_definition(row.definition, version=row.version)


@transaction.atomic
def save_version(definition: dict, note: str = "", activate: bool = True) -> GraphSchemaVersion:
    """Store a new schema version (validated); optionally make it the active one."""
    errors = validate_definition(definition)
    if errors:
        raise SchemaError("; ".join(errors))
    latest = GraphSchemaVersion.objects.order_by("-version").first()
    if activate:
        GraphSchemaVersion.objects.filter(is_active=True).update(is_active=False)
    return GraphSchemaVersion.objects.create(
        version=(latest.version + 1) if latest else 1,
        name=definition.get("name", "Schema"),
        definition=definition,
        note=note,
        is_active=activate,
    )


@transaction.atomic
def activate(version: int) -> GraphSchemaVersion:
    row = GraphSchemaVersion.objects.filter(version=version).first()
    if row is None:
        raise SchemaError(f"unknown schema version {version}")
    GraphSchemaVersion.objects.filter(is_active=True).exclude(pk=row.pk).update(is_active=False)
    row.is_active = True
    row.save(update_fields=["is_active"])
    return row


def versions() -> list[dict]:
    return [
        {"version": v.version, "name": v.name, "note": v.note, "is_active": v.is_active,
         "created_at": v.created_at.isoformat()}
        for v in GraphSchemaVersion.objects.all()
    ]


def definition_for(version: int | None) -> dict:
    """Stored definition of `version` (active one if None; bundled default if nothing stored)."""
    if version is None:
        return active_schema().definition
    row = GraphSchemaVersion.objects.filter(version=version).first()
    if row is None:
        raise SchemaError(f"unknown schema version {version}")
    return row.definition
