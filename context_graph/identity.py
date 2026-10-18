"""Identity resolution (P8b): which curated entity does an extracted name refer to?

One order for every source (document extraction, imports, edits, promote, later the VLM):

  1. explicit canonical id                        -> match "id"
  2. same asset + exact engineering tag            -> "tag"
  3. exact tag + same type, any scope              -> "tag"
  4. exact (normalized) name or alias              -> "name" / "alias"
  5. fuzzy name similarity                         -> "possible" + candidates (never auto-assigned)
  6. nothing                                       -> "new" (a proposed id)

Tags are explicit engineering identifiers found in names ("M101", "VFD-101", "TT_101",
"DIE_PLUG", historian columns like "die_pressure_bar"). Fuzzy similarity only ever
*suggests*: a "possible" end must be resolved in review (pick a candidate or "new")
before the triple can be approved — "Barrel Zone 1" is never merged with "Barrel Zone 2".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import repository
from .driver import GraphUnavailable, session
from .ids import clean_part, is_valid_id, make_id

MATCH_KINDS = ("id", "tag", "alias", "name", "possible", "new")
EXISTING = ("id", "tag", "alias", "name")
POSSIBLE_THRESHOLD = 0.5
MAX_CANDIDATES = 3

# Letters followed by digits (M101, VFD-101, TT_101A, EXTR01), UPPER_SNAKE codes (DIE_PLUG),
# snake_case keys with ≥ 3 parts (die_pressure_bar).
_TAG_PATTERNS = (
    re.compile(r"\b[A-Za-z]{1,6}[-_ ]?\d{1,5}[A-Za-z]?\b"),
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b"),
)


def normalize(name) -> str:
    # str(): imported properties can be numbers (an all-digit part number from a CSV)
    return re.sub(r"[^a-z0-9]+", " ", str(name if name is not None else "").lower()).strip()


def tag_key(tag: str) -> str:
    """Comparable form of a tag: VFD-101, vfd_101 and VFD 101 are the same tag."""
    return re.sub(r"[^A-Z0-9]", "", (tag or "").upper())


def tags_in(name: str) -> list[str]:
    """Engineering tags found in a name, most specific first (≥ 3 chars, at least one digit or an underscore code)."""
    found: list[str] = []
    for pattern in _TAG_PATTERNS:
        for m in pattern.finditer(name or ""):
            tag = m.group(0)
            if len(tag_key(tag)) >= 3 and tag_key(tag) not in {tag_key(t) for t in found}:
                found.append(tag)
    return found


@dataclass
class Node:
    id: str
    label: str
    name: str
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()          # tag keys: the `tag` / `code` property and the id's last key part

    @property
    def names(self) -> set[str]:
        return {normalize(self.name)} | {normalize(a) for a in self.aliases}


@dataclass
class Match:
    id: str
    kind: str                            # one of MATCH_KINDS
    candidates: list[dict] = field(default_factory=list)   # possible matches: [{id, name, score}]

    @property
    def existing(self) -> bool:
        return self.kind in EXISTING


# Facts that belong to a part (P21): their new ids carry the part, and they are matched within its family only.
PART_LABELS = ("Alarm", "Procedure", "FailureMode", "Component", "Specification", "Parameter", "IOPoint", "Interface")
# Component -> fact links that make a part own a fact; a procedure is also owned through what it serves.
OWNING = ("HAS_ALARM", "HAS_PROCEDURE", "HAS_FAILURE_MODE", "HAS_SPEC", "HAS_PARAMETER", "HAS_IO", "COMMUNICATES_VIA")


@dataclass(frozen=True)
class PartScope:
    """The part a manual describes (extraction's anchor), for its new ids and its matching (P21)."""
    anchor_id: str
    key: str                     # id key of its new nodes: "EXTR01/cutter-motor"
    family: frozenset[str]       # the part, its parent parts and its sub-parts


@dataclass
class EntityIndex:
    """Curated entities by label, for resolving names to ids."""

    by_label: dict[str, list[Node]] = field(default_factory=dict)
    # P21: which parts own a fact, the part tree, and the parts list's components (matched by name asset-wide).
    owners: dict[str, set[str]] = field(default_factory=dict)
    parents: dict[str, str] = field(default_factory=dict)
    bom: set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "EntityIndex":
        index = cls()
        try:
            with session() as s:
                rows = s.run(
                    "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
                    "coalesce(n.aliases, []) AS aliases, n.tag AS tag, n.code AS code").data()
                links = s.run(
                    "MATCH (c:Component)-[r]->(n:Entity) WHERE type(r) IN $owning RETURN c.id AS part, n.id AS id",
                    owning=list(OWNING)).data()
                serves = s.run(
                    "MATCH (p:Procedure)-[:ADDRESSES]->(x:Entity) RETURN p.id AS id, x.id AS served "
                    "UNION MATCH (x:FailureMode)-[:MITIGATED_BY]->(p:Procedure) RETURN p.id AS id, x.id AS served").data()
                tree = s.run("MATCH (p:Component)-[:HAS_COMPONENT]->(c:Component) RETURN p.id AS parent, c.id AS id").data()
                bom = s.run("MATCH (c:Component) WHERE c.source = 'structured' RETURN c.id AS id").data()
        except GraphUnavailable:
            return index
        for link in links:
            index.owners.setdefault(link["id"], set()).add(link["part"])
        for row in serves:                              # a procedure belongs to the parts of what it serves
            if row["served"] in index.owners:
                index.owners.setdefault(row["id"], set()).update(index.owners[row["served"]])
        index.parents = {row["id"]: row["parent"] for row in tree}
        index.bom = {row["id"] for row in bom}
        for row in rows:
            index.add(row["id"], repository.node_label(row["labels"]), row["name"] or "", row["aliases"],
                      [t for t in (row["tag"], row["code"]) if t])
        return index

    def add(self, node_id: str, label: str, name: str, aliases=(), tags=()) -> None:
        last_key = node_id.rsplit("/", 1)[-1].split(":")[-1]
        keys = {tag_key(t) for t in [*tags, last_key] if len(tag_key(t)) >= 3}
        self.by_label.setdefault(label, []).append(Node(node_id, label, name, tuple(aliases), tuple(sorted(keys))))

    def has(self, node_id: str) -> bool:
        return any(n.id == node_id for nodes in self.by_label.values() for n in nodes)

    def get(self, node_id: str) -> Node | None:
        return next((n for nodes in self.by_label.values() for n in nodes if n.id == node_id), None)

    def part_scope(self, anchor_id: str, scope_key: str) -> PartScope:
        """The scope of a run whose document describes `anchor_id` (a component id of the asset `scope_key`)."""
        _, _, key = anchor_id.split(":", 2)
        part_key = key.split("/", 1)[1] if key.startswith(f"{scope_key}/") else key
        family = {anchor_id}
        up = self.parents.get(anchor_id)
        while up and up not in family:
            family.add(up)
            up = self.parents.get(up)
        children: dict[str, list[str]] = {}
        for child, parent in self.parents.items():
            children.setdefault(parent, []).append(child)
        todo = [anchor_id]
        while todo:
            for child in children.get(todo.pop(), []):
                if child not in family:
                    family.add(child)
                    todo.append(child)
        return PartScope(anchor_id, f"{scope_key}/{part_key}", frozenset(family))

    def _visible(self, label: str, nodes: list[Node], part: PartScope | None) -> list[Node]:
        """The nodes a name of this run may match: with a part, its family's and the unowned ones (P21)."""
        if part is None or label not in PART_LABELS:
            return nodes
        if label == "Component":
            return [n for n in nodes if n.id in part.family or n.id in self.bom]
        return [n for n in nodes if not self.owners.get(n.id) or self.owners[n.id] & part.family]

    # --- resolution -------------------------------------------------------------

    def resolve(self, label: str, name: str, scope_key: str, proposed: str | None = None,
                fuzzy: bool = True, tag: str | None = None, part: PartScope | None = None) -> Match:
        """Resolve `name` (an entity of type `label`) to an existing id, a possible match or a new id.

        `part` (P21): the part the document describes. A new alarm, procedure, failure mode or sub-part gets
        its id under the part, and names are matched only within the part's family (and unowned nodes)."""
        everything = self.by_label.get(label, [])
        nodes = self._visible(label, everything, part)
        scoped = part is not None and label in PART_LABELS
        new_id = proposed or proposed_id(label, name, part.key if scoped else scope_key)

        # 1. explicit canonical id (the name *is* an id, or the proposed id already exists); any part
        for candidate in (name.strip(), new_id):
            if is_valid_id(candidate) and any(n.id == candidate for n in everything):
                return Match(candidate, "id")

        # 2–3. engineering tags: same asset scope first, then any scope of the same type
        tags = [tag_key(t) for t in ([tag] if tag else []) + tags_in(name)]
        tags = [t for t in dict.fromkeys(tags) if len(t) >= 3]
        if tags:
            for in_scope in (True, False):
                hits = [n for n in nodes if set(tags) & set(n.tags)
                        and (not in_scope or f":{scope_key}/" in n.id or n.id.endswith(f":{scope_key}"))]
                if len(hits) == 1:
                    return Match(hits[0].id, "tag")
                if len(hits) > 1:   # ambiguous tag: let a person choose
                    return Match(new_id, "possible", [{"id": n.id, "name": n.name, "score": 1.0} for n in hits[:MAX_CANDIDATES]])

        # 4. exact name or alias
        target = normalize(name)
        if target:
            for n in nodes:
                if target == normalize(n.name):
                    return Match(n.id, "name")
            for n in nodes:
                if target in {normalize(a) for a in n.aliases}:
                    return Match(n.id, "alias")

        # 5. fuzzy: suggestions only
        if fuzzy and target:
            scored = sorted(((score(target, n), n) for n in nodes), key=lambda sn: -sn[0])
            candidates = [{"id": n.id, "name": n.name, "score": round(sc, 2)} for sc, n in scored
                          if sc >= POSSIBLE_THRESHOLD][:MAX_CANDIDATES]
            if candidates:
                return Match(new_id, "possible", candidates)

        # 6. new entity
        return Match(new_id, "new")


def score(target: str, node: Node) -> float:
    """Best token similarity between a normalized name and a node's name / aliases (0–1)."""
    tokens = set(target.split())
    best = 0.0
    for other in node.names:
        words = set(other.split())
        if not words:
            continue
        sim = len(tokens & words) / len(tokens | words)
        if len(target) >= 4 and (target in other or other in target):
            sim = max(sim, 0.75)
        # Differing numbers or codes ("zone 1" vs "zone 2", "F01000" vs "A798", "F30001" vs "0001") mean
        # different things: never suggested (P21; they were capped at 0.5, which still suggested them).
        if {w for w in tokens if any(c.isdigit() for c in w)} != {w for w in words if any(c.isdigit() for c in w)}:
            continue
        best = max(best, sim)
    return best


def proposed_id(label: str, name: str, scope_key: str) -> str:
    """`scope_key` may have several parts ("EXTR01/cutter-motor", P21): each stays an id key part."""
    part = clean_part(name)
    return make_id(label, part) if label == "Asset" else make_id(label, *scope_key.split("/"), part)
