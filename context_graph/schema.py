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
    aliases: tuple[str, ...] = ()       # phrases that mean this relationship ("drives", "runs", …) — P8b


@dataclass
class Schema:
    name: str
    description: str
    entity_types: dict[str, str]                       # label -> description
    relationship_types: dict[str, RelationshipType]
    subtypes: dict[str, tuple[str, ...]] = field(default_factory=dict)   # label -> allowed subtypes (P8b)
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
                aliases=tuple(str(a) for a in (spec or {}).get("aliases") or ()),
            )
            for name, spec in definition["relationship_types"].items()
        }
        subtypes = {
            label: tuple(str(t) for t in (spec or {}).get("subtypes") or ())
            for label, spec in definition["entity_types"].items() if (spec or {}).get("subtypes")
        }
        return cls(
            name=definition.get("name", "Schema"),
            description=definition.get("description", ""),
            entity_types=entity_types,
            relationship_types=relationship_types,
            subtypes=subtypes,
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

    def subtype_for(self, label: str, value: str) -> str | None:
        """The schema's spelling of a subtype of `label` (case / spacing insensitive), or None."""
        key = _vocab_key(value)
        return next((t for t in self.subtypes.get(label, ()) if _vocab_key(t) == key), None) if key else None

    def relationship_for(self, phrase: str, from_label: str | None = None, to_label: str | None = None) -> str | None:
        """Relationship type for a name or alias ("drives", "is monitored by" …); with labels, only an allowed pair."""
        key = _vocab_key(phrase)
        if not key:
            return None
        for rt in self.relationship_types.values():
            if key == _vocab_key(rt.name) or key in {_vocab_key(a) for a in rt.aliases}:
                if from_label is None or (from_label, to_label) in rt.pairs:
                    return rt.name
        return None

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


def _vocab_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


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
        sub = (entity_types[label] or {}).get("subtypes") if isinstance(entity_types[label], dict) else None
        if sub is not None and not (isinstance(sub, list) and all(isinstance(t, str) and t.strip() for t in sub)):
            errors.append(f"entity type {label!r}: subtypes must be a list of names")

    seen_aliases: dict[str, str] = {}
    for name, spec in rel_types.items():
        if not REL_RE.match(str(name)):
            errors.append(f"relationship type {name!r} must be UPPER_SNAKE_CASE")
        pairs = (spec or {}).get("pairs") if isinstance(spec, dict) else None
        aliases = (spec or {}).get("aliases") if isinstance(spec, dict) else None
        if aliases is not None:
            if not (isinstance(aliases, list) and all(isinstance(a, str) and a.strip() for a in aliases)):
                errors.append(f"relationship type {name!r}: aliases must be a list of phrases")
            else:
                for a in aliases:
                    key = _vocab_key(a)
                    if key in seen_aliases and seen_aliases[key] != name:
                        errors.append(f"alias {a!r} is used by both {seen_aliases[key]} and {name}")
                    seen_aliases[key] = name
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
        # Vocabulary (P8b): subtypes of entity types and aliases of relationship types.
        "changed_vocabulary": _vocabulary_diff(old, new),
    }


def _vocabulary_diff(old: "Schema", new: "Schema") -> dict:
    out = {}
    for label in sorted(set(old.subtypes) | set(new.subtypes)):
        a, b = set(old.subtypes.get(label, ())), set(new.subtypes.get(label, ()))
        if a != b:
            out[f"{label} subtypes"] = {"added": sorted(b - a), "removed": sorted(a - b)}
    for name in sorted(set(old.relationship_types) & set(new.relationship_types)):
        a, b = set(old.relationship_types[name].aliases), set(new.relationship_types[name].aliases)
        if a != b:
            out[f"{name} aliases"] = {"added": sorted(b - a), "removed": sorted(a - b)}
    return out


def is_empty_diff(d: dict) -> bool:
    return not any(d.get(k) for k in ("added_entity_types", "removed_entity_types", "added_relationship_types",
                                      "removed_relationship_types", "changed_pairs", "changed_vocabulary"))
