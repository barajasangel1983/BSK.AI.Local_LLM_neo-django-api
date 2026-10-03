"""Cypher access to the curated Context Graph.

Every curated node has the base label :Entity plus exactly one schema label
(e.g. :Asset) and a unique canonical `id`. Labels and relationship types are
validated against the schema (and identifier rules) before being placed in a
query; all values are passed as parameters.

Functions take a neo4j transaction/session (`tx`) so callers control
transactions; they return plain dicts.
"""

from __future__ import annotations

from typing import Any

from .ids import is_valid_id, parse_id
from .schema import LABEL_RE, REL_RE, Schema, SchemaError

MAX_DEPTH = 4


class NodeNotFound(LookupError):
    pass


# --- serialization -----------------------------------------------------------

def _value(v: Any) -> Any:
    if hasattr(v, "iso_format"):  # neo4j.time.DateTime / Date
        return v.iso_format()
    if isinstance(v, list):
        return [_value(x) for x in v]
    return v


def node_label(labels) -> str:
    return next((label for label in labels if label != "Entity"), "Entity")


def serialize_node(node) -> dict:
    props = {k: _value(v) for k, v in dict(node).items()}
    return {"id": props.pop("id"), "label": node_label(node.labels), "name": props.get("name", ""), "properties": props}


def serialize_rel(rel, source: str, target: str) -> dict:
    # Endpoint ids are passed in: a relationship returned on its own doesn't
    # carry its nodes' properties.
    return {
        "source": source,
        "target": target,
        "type": rel.type,
        "properties": {k: _value(v) for k, v in dict(rel).items()},
    }


def _clean_props(props: dict) -> dict:
    """Neo4j properties must be primitives or lists of primitives."""
    out = {}
    for key, value in (props or {}).items():
        if key == "id":
            continue
        if isinstance(value, dict):
            raise SchemaError(f"property {key!r} must not be a mapping")
        out[key] = value
    return out


# --- schema maintenance ----------------------------------------------------

def apply_constraints(tx, schema: Schema) -> list[str]:
    statements = schema.constraint_statements()
    for statement in statements:
        tx.run(statement)
    return statements


# --- writes ----------------------------------------------------------------

def upsert_node(tx, schema: Schema, label: str, node_id: str, props: dict) -> dict:
    schema.check_label(label)
    if not LABEL_RE.match(label):
        raise SchemaError(f"invalid label {label!r}")
    kind, _ = parse_id(node_id)
    if schema.label_for_slug(kind) != label:
        raise SchemaError(f"ID {node_id!r} doesn't match entity type {label}")

    existing = tx.run("MATCH (n:Entity {id: $id}) RETURN labels(n) AS labels", id=node_id).single()
    if existing and node_label(existing["labels"]) not in (label, "Entity"):
        raise SchemaError(f"{node_id} already exists as {node_label(existing['labels'])}, not {label}")

    record = tx.run(
        f"MERGE (n:Entity {{id: $id}}) "
        f"ON CREATE SET n.created_at = datetime() "
        f"SET n:{label}, n += $props, n.updated_at = datetime() "
        f"RETURN n",
        id=node_id,
        props=_clean_props(props),
    ).single()
    return serialize_node(record["n"])


def upsert_relationship(tx, schema: Schema, from_id: str, rel: str, to_id: str, props: dict | None = None) -> dict:
    ends = tx.run(
        "OPTIONAL MATCH (a:Entity {id: $f}) OPTIONAL MATCH (b:Entity {id: $t}) "
        "RETURN labels(a) AS fa, labels(b) AS fb",
        f=from_id, t=to_id,
    ).single()
    if not ends or ends["fa"] is None:
        raise NodeNotFound(from_id)
    if ends["fb"] is None:
        raise NodeNotFound(to_id)
    schema.check_relationship(node_label(ends["fa"]), rel, node_label(ends["fb"]))
    if not REL_RE.match(rel):
        raise SchemaError(f"invalid relationship type {rel!r}")

    record = tx.run(
        f"MATCH (a:Entity {{id: $f}}), (b:Entity {{id: $t}}) "
        f"MERGE (a)-[r:{rel}]->(b) "
        f"ON CREATE SET r.created_at = datetime() "
        f"SET r += $props "
        f"RETURN r",
        f=from_id, t=to_id, props=_clean_props(props or {}),
    ).single()
    return serialize_rel(record["r"], from_id, to_id)


def delete_all_entities(tx) -> int:
    """Remove every curated node (and its relationships). Dev/reset only."""
    return tx.run("MATCH (n:Entity) DETACH DELETE n RETURN count(n) AS n").single()["n"]


# --- reads -----------------------------------------------------------------

def get_node(tx, node_id: str) -> dict:
    record = tx.run("MATCH (n:Entity {id: $id}) RETURN n", id=node_id).single()
    if record is None:
        raise NodeNotFound(node_id)
    return serialize_node(record["n"])


def get_neighbors(tx, node_id: str, limit: int = 200) -> list[dict]:
    rows = tx.run(
        "MATCH (n:Entity {id: $id})-[r]-(m:Entity) "
        "RETURN r, m, startNode(r) = n AS outgoing "
        "ORDER BY type(r), m.name LIMIT $limit",
        id=node_id, limit=limit,
    )
    return [
        {"direction": "out" if row["outgoing"] else "in", "type": row["r"].type,
         "properties": {k: _value(v) for k, v in dict(row["r"]).items()}, "node": serialize_node(row["m"])}
        for row in rows
    ]


def list_nodes(tx, label: str | None, query: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
    where, params = [], {"limit": limit, "offset": offset}
    if label:
        where.append("$label IN labels(n)")
        params["label"] = label
    if query:
        where.append("(toLower(n.name) CONTAINS toLower($q) OR toLower(n.id) CONTAINS toLower($q))")
        params["q"] = query
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = tx.run(f"MATCH (n:Entity) {clause} RETURN count(n) AS n", **params).single()["n"]
    rows = tx.run(f"MATCH (n:Entity) {clause} RETURN n ORDER BY n.name, n.id SKIP $offset LIMIT $limit", **params)
    return [serialize_node(row["n"]) for row in rows], total


def counts(tx) -> dict:
    labels = {
        node_label(row["labels"]): row["n"]
        for row in tx.run("MATCH (n:Entity) RETURN labels(n) AS labels, count(*) AS n")
    }
    rels = {row["type"]: row["n"] for row in tx.run(
        "MATCH (:Entity)-[r]->(:Entity) RETURN type(r) AS type, count(*) AS n")}
    return {"nodes": labels, "relationships": rels}


def subgraph(tx, root_id: str | None, depth: int, labels: list[str] | None, limit: int) -> dict:
    """Nodes + links for the visualizer: around `root_id` (up to `depth` hops) or the whole graph."""
    depth = max(1, min(int(depth), MAX_DEPTH))
    params: dict = {"limit": limit, "labels": labels or []}
    label_filter = "(size($labels) = 0 OR any(l IN labels(m) WHERE l IN $labels))"
    if root_id:
        if not is_valid_id(root_id):
            raise NodeNotFound(root_id)
        params["root"] = root_id
        node_query = (
            f"MATCH (r:Entity {{id: $root}}) "
            f"OPTIONAL MATCH (r)-[*1..{depth}]-(m:Entity) WHERE {label_filter} "
            f"WITH r, collect(DISTINCT m)[..$limit] AS ms "
            f"RETURN [r] + [x IN ms WHERE x <> r] AS nodes"
        )
        record = tx.run(node_query, **params).single()
        if record is None:
            raise NodeNotFound(root_id)
        nodes = record["nodes"]
    else:
        nodes = [row["m"] for row in tx.run(
            f"MATCH (m:Entity) WHERE {label_filter} RETURN m LIMIT $limit", **params)]

    ids = [n["id"] for n in nodes]
    rels = tx.run(
        "MATCH (a:Entity)-[r]->(b:Entity) WHERE a.id IN $ids AND b.id IN $ids RETURN r, a.id AS s, b.id AS t",
        ids=ids,
    )
    return {
        "nodes": [serialize_node(n) for n in nodes],
        "links": [serialize_rel(row["r"], row["s"], row["t"]) for row in rels],
    }


# --- lab layer (free-form triples, kept apart from the curated graph) ---------

def lab_summary(tx) -> dict:
    """Size of the lab layer, overall and per source document."""
    docs = [{"doc_key": row["doc"], "nodes": row["n"]} for row in tx.run(
        "MATCH (n:Lab) RETURN n.doc_key AS doc, count(n) AS n ORDER BY n DESC")]
    rels = tx.run("MATCH (:Lab)-[r:LAB_RELATION]->(:Lab) RETURN count(r) AS n").single()["n"]
    return {"nodes": sum(d["nodes"] for d in docs), "relationships": rels, "documents": docs}


def lab_subgraph(tx, doc_key: str | None, limit: int) -> dict:
    """Lab nodes + LAB_RELATION links; a link's `type` is its free-form predicate."""
    nodes = [row["n"] for row in tx.run(
        "MATCH (n:Lab) WHERE $doc IS NULL OR n.doc_key = $doc RETURN n LIMIT $limit", doc=doc_key, limit=limit)]
    ids = [n["id"] for n in nodes]
    rels = tx.run(
        "MATCH (a:Lab)-[r:LAB_RELATION]->(b:Lab) WHERE a.id IN $ids AND b.id IN $ids RETURN r, a.id AS s, b.id AS t",
        ids=ids,
    )
    links = []
    for row in rels:
        link = serialize_rel(row["r"], row["s"], row["t"])
        link["type"] = link["properties"].get("predicate", "RELATED_TO")
        link["layer"] = "lab"
        links.append(link)
    out_nodes = []
    for n in nodes:
        node = serialize_node(n)
        node["layer"] = "lab"
        out_nodes.append(node)
    return {"nodes": out_nodes, "links": links}
