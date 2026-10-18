"""Knowledge facets (P22b): tags that say which kind of knowledge a node or a staged triple is.

A facet is a tag, not a separate graph: every facet hangs off the same device. The tags are *derived* from the
active schema's `facets:` section (by type, by subtype, by link), never typed by hand, so a change of the mapping
is one recompute (`manage.py graph_facets`), not a migration.

    facets:
      types:    {Alarm: [Troubleshooting], …}                   # every node of the type
      subtypes: {Procedure: {Calibration: [Setup, Quality], …}}  # by subtype ("A; B" counts as both)
      defaults: {Procedure: [Maintenance]}                      # a node of the type with no subtype
      links:                                                    # the node at `end` of such a link
        - {rel: ADDRESSES, end: from, facets: [Troubleshooting]}

Types with no rule (Plant … Asset, Component, Document, DocumentSection) are the core: no facet.
"""

from __future__ import annotations

FACETS = ("Maintenance", "Safety", "Installation", "Controls", "Operation", "Setup", "Troubleshooting", "Quality")
ENDS = ("from", "to")


def _names(value) -> list[str] | None:
    return list(value) if isinstance(value, list) and all(isinstance(v, str) for v in value) else None


def validate(section, entity_types: dict, rel_types: dict) -> list[str]:
    """Problems in a schema's `facets:` section (empty if valid, or if there is no section)."""
    if section is None:
        return []
    if not isinstance(section, dict):
        return ["facets must be a mapping"]
    errors: list[str] = []

    def check(where: str, value) -> None:
        names = _names(value)
        if names is None:
            errors.append(f"facets: {where} must be a list of facet names")
            return
        for n in names:
            if n not in FACETS:
                errors.append(f"facets: {where}: {n!r} is not a facet ({', '.join(FACETS)})")

    unknown = set(section) - {"types", "subtypes", "defaults", "links"}
    if unknown:
        errors.append(f"facets: unknown key(s) {', '.join(sorted(unknown))}")
    for key in ("types", "defaults"):
        for label, value in (section.get(key) or {}).items():
            if label not in entity_types:
                errors.append(f"facets: {key}: {label!r} is not an entity type")
            check(f"{key}.{label}", value)
    for label, by_subtype in (section.get("subtypes") or {}).items():
        if label not in entity_types:
            errors.append(f"facets: subtypes: {label!r} is not an entity type")
            continue
        allowed = set((entity_types[label] or {}).get("subtypes") or ()) if isinstance(entity_types[label], dict) else set()
        if not isinstance(by_subtype, dict):
            errors.append(f"facets: subtypes.{label} must map a subtype to facets")
            continue
        for sub, value in by_subtype.items():
            if sub not in allowed:
                errors.append(f"facets: subtypes.{label}: {sub!r} is not a subtype of {label}")
            check(f"subtypes.{label}.{sub}", value)
    links = section.get("links") or []
    if not isinstance(links, list):
        errors.append("facets: links must be a list")
        links = []
    for i, rule in enumerate(links):
        if not isinstance(rule, dict) or rule.get("rel") not in rel_types or rule.get("end") not in ENDS:
            errors.append(f"facets: links[{i}] needs a relationship type `rel` and `end` (from or to)")
            continue
        check(f"links[{i}]", rule.get("facets"))
    return errors


def mapping_of(schema) -> dict | None:
    """The active mapping, or None when the schema has no `facets:` section (v4 and earlier)."""
    section = (schema.definition or {}).get("facets")
    return section if isinstance(section, dict) else None


def subtypes_of(value) -> list[str]:
    """A node's subtype values: "Operation; Commissioning" is both."""
    return [s.strip() for s in str(value or "").split(";") if s.strip()]


def compute(mapping: dict | None, label: str, subtype=None, links=()) -> list[str]:
    """The facets of one node: its type, its subtype(s) (or the type's default without one), and the links
    it has. `links` holds (relationship type, end) pairs, end "from" when the node is the link's start."""
    if not mapping or not label:
        return []
    found = set(mapping.get("types", {}).get(label) or ())
    subs = subtypes_of(subtype)
    if subs:
        by_subtype = (mapping.get("subtypes") or {}).get(label) or {}
        for s in subs:
            found |= set(by_subtype.get(s) or ())
    else:
        found |= set((mapping.get("defaults") or {}).get(label) or ())
    have = set(links)
    for rule in mapping.get("links") or ():
        if (rule.get("rel"), rule.get("end")) in have:
            found |= set(rule.get("facets") or ())
    return [f for f in FACETS if f in found]


def for_triple(mapping: dict | None, t) -> list[str]:
    """A staged triple's facets: those of its two ends as this triple would leave them (the core ends add none)."""
    if not mapping or t.mode != "schema":
        return []
    found = set(compute(mapping, t.subject_type, (t.subject_props or {}).get("subtype"), [(t.predicate, "from")]))
    found |= set(compute(mapping, t.object_type, (t.object_props or {}).get("subtype"), [(t.predicate, "to")]))
    return [f for f in FACETS if f in found]


def stamp(triples, schema) -> None:
    """Set `facets` on unsaved or about-to-be-saved triples (staging, edit, make_rating)."""
    mapping = mapping_of(schema)
    for t in triples:
        t.facets = for_triple(mapping, t)


# --- the graph ----------------------------------------------------------------------

_NODES = ("MATCH (n:Entity) {where} "
          "OPTIONAL MATCH (n)-[r]-(:Entity) "
          "WITH n, collect(DISTINCT [type(r), CASE WHEN startNode(r) = n THEN 'from' ELSE 'to' END]) AS links "
          "RETURN n.id AS id, [x IN labels(n) WHERE x <> 'Entity'][0] AS label, n.subtype AS subtype, "
          "n.name AS name, coalesce(n.facets, []) AS current, links")


def plan(tx, mapping: dict | None, ids: list[str] | None = None) -> list[dict]:
    """Every node (or the given ones) with its current and computed facets. Reads only."""
    where = "WHERE n.id IN $ids" if ids is not None else ""
    rows = tx.run(_NODES.format(where=where), ids=list(ids or [])).data()
    out = []
    for r in rows:
        links = [tuple(x) for x in r["links"] if x and x[0]]
        out.append({"id": r["id"], "label": r["label"], "name": r["name"], "subtype": r["subtype"],
                    "current": list(r["current"]), "facets": compute(mapping, r["label"], r["subtype"], links)})
    return out


def write(tx, rows: list[dict]) -> int:
    """Set the computed facets where they differ from the stored ones (an empty list removes the property).
    Writes only `facets`. Returns the number of nodes changed."""
    changed = [{"id": r["id"], "facets": r["facets"]} for r in rows if r["facets"] != r["current"]]
    if changed:
        tx.run("UNWIND $rows AS row MATCH (n:Entity {id: row.id}) "
               "SET n.facets = CASE WHEN size(row.facets) = 0 THEN null ELSE row.facets END", rows=changed)
    return len(changed)


def refresh(tx, schema, ids) -> int:
    """Recompute the facets of these nodes in the same transaction as the write that touched them (approve,
    uncommit). No-op when the active schema has no facet mapping."""
    mapping = mapping_of(schema)
    ids = [i for i in dict.fromkeys(ids) if i]
    if mapping is None or not ids:
        return 0
    return write(tx, plan(tx, mapping, ids))
