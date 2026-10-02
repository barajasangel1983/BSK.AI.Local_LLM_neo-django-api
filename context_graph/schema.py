"""Context Graph schema: entity types, relationship types and allowed pairs.

The schema is data (a dict / YAML document), stored as versions in
GraphSchemaVersion; the active version validates every write. Labels and
relationship types are interpolated into Cypher (Cypher can't parameterize
them), so only names that pass IDENTIFIER rules and exist in the schema are
ever used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .ids import type_slug

LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")      # PascalCase node labels
REL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")          # UPPER_SNAKE relationship types
RESERVED_LABELS = {"Entity", "Lab"}                 # base label / free-form lab layer
DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schemas" / "industrial_v1.yaml"


class SchemaError(ValueError):
    """Invalid schema definition, or a write the schema doesn't allow."""


@dataclass(frozen=True)
class RelationshipType:
    name: str
    description: str
    pairs: frozenset[tuple[str, str]]


@dataclass
class Schema:
    name: str
    description: str
    entity_types: dict[str, str]                       # label -> description
    relationship_types: dict[str, RelationshipType]
    version: int | None = None
    definition: dict = field(default_factory=dict, repr=False)

    # --- construction -----------------------------------------------------

    @classmethod
    def from_definition(cls, definition: dict, version: int | None = None) -> "Schema":
        errors = validate_definition(definition)
        if errors:
            raise SchemaError("; ".join(errors))
        entity_types = {
            label: (spec or {}).get("description", "")
            for label, spec in definition["entity_types"].items()
        }
        relationship_types = {
            name: RelationshipType(
                name=name,
                description=(spec or {}).get("description", ""),
                pairs=frozenset(tuple(p) for p in spec["pairs"]),
            )
            for name, spec in definition["relationship_types"].items()
        }
        return cls(
            name=definition.get("name", "Schema"),
            description=definition.get("description", ""),
            entity_types=entity_types,
            relationship_types=relationship_types,
            version=version,
            definition=definition,
        )

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_SCHEMA_PATH, version: int | None = None) -> "Schema":
        return cls.from_definition(yaml.safe_load(Path(path).read_text()), version=version)

    # --- checks used before any Cypher is built ---------------------------

    def check_label(self, label: str) -> str:
        if label not in self.entity_types:
            raise SchemaError(f"unknown entity type {label!r}")
        return label

    def check_relationship(self, from_label: str, rel: str, to_label: str) -> str:
        spec = self.relationship_types.get(rel)
        if spec is None:
            raise SchemaError(f"unknown relationship type {rel!r}")
        if (from_label, to_label) not in spec.pairs:
            raise SchemaError(f"{from_label} -{rel}-> {to_label} is not allowed by the schema")
        return rel

    def label_for_slug(self, slug: str) -> str | None:
        """Entity type whose ID slug is `slug` (bsk:<slug>:...)."""
        return next((label for label in self.entity_types if type_slug(label) == slug), None)

    def constraint_statements(self) -> list[str]:
        """Idempotent Cypher for constraints/indexes (labels already validated)."""
        statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.description]",
        ]
        statements += [
            f"CREATE INDEX {type_slug(label).replace('-', '_')}_name IF NOT EXISTS FOR (n:{label}) ON (n.name)"
            for label in self.entity_types
        ]
        return statements


def validate_definition(definition: dict) -> list[str]:
    """Return a list of problems (empty if the definition is valid)."""
    errors: list[str] = []
    if not isinstance(definition, dict):
        return ["schema must be a mapping"]
    entity_types = definition.get("entity_types")
    rel_types = definition.get("relationship_types")
    if not isinstance(entity_types, dict) or not entity_types:
        errors.append("entity_types must be a non-empty mapping")
        entity_types = {}
    if not isinstance(rel_types, dict):
        errors.append("relationship_types must be a mapping")
        rel_types = {}

    slugs: dict[str, str] = {}
    for label in entity_types:
        if not LABEL_RE.match(str(label)):
            errors.append(f"entity type {label!r} must be PascalCase letters/digits")
        elif label in RESERVED_LABELS:
            errors.append(f"entity type {label!r} is reserved")
        slug = type_slug(str(label))
        if slug in slugs:
            errors.append(f"entity types {slugs[slug]!r} and {label!r} produce the same ID prefix")
        slugs[slug] = label

    for name, spec in rel_types.items():
        if not REL_RE.match(str(name)):
            errors.append(f"relationship type {name!r} must be UPPER_SNAKE_CASE")
        pairs = (spec or {}).get("pairs") if isinstance(spec, dict) else None
        if not pairs:
            errors.append(f"relationship type {name!r} needs at least one [from, to] pair")
            continue
        for pair in pairs:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                errors.append(f"{name}: pair {pair!r} must be [from, to]")
                continue
            for end in pair:
                if end not in entity_types:
                    errors.append(f"{name}: {end!r} is not an entity type")
    return errors


def diff(old: Schema, new: Schema) -> dict:
    """What changes from `old` to `new` (entity types, relationship types, allowed pairs)."""
    old_rels, new_rels = old.relationship_types, new.relationship_types
    changed_pairs = {}
    for name in sorted(set(old_rels) & set(new_rels)):
        added = sorted(new_rels[name].pairs - old_rels[name].pairs)
        removed = sorted(old_rels[name].pairs - new_rels[name].pairs)
        if added or removed:
            changed_pairs[name] = {"added": [list(p) for p in added], "removed": [list(p) for p in removed]}
    return {
        "added_entity_types": sorted(set(new.entity_types) - set(old.entity_types)),
        "removed_entity_types": sorted(set(old.entity_types) - set(new.entity_types)),
        "added_relationship_types": sorted(set(new_rels) - set(old_rels)),
        "removed_relationship_types": sorted(set(old_rels) - set(new_rels)),
        "changed_pairs": changed_pairs,
    }


def is_empty_diff(d: dict) -> bool:
    return not any(d[k] for k in ("added_entity_types", "removed_entity_types", "added_relationship_types",
                                  "removed_relationship_types", "changed_pairs"))
