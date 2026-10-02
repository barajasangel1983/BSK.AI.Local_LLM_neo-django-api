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
