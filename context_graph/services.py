"""Context Graph domain operations.

Plain Python functions (no Django request objects) used by the API views,
management commands and, later, MCP tools. They return JSON-serializable
dicts and raise:
  driver.GraphUnavailable   graph disabled / not configured / unreachable
  schema.SchemaError        write or input rejected by the schema
  repository.NodeNotFound   unknown canonical ID
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from . import registry, repository
from .driver import session
from .ids import is_valid_id, make_id
from .repository import NodeNotFound
from .schema import Schema, SchemaError

_SHORT_REF = re.compile(r"^([A-Z][A-Za-z0-9]*):(.+)$")


# --- schema ---------------------------------------------------------------

def init_graph() -> dict:
    """Store the default schema as v1 if the registry is empty; apply Neo4j constraints/indexes."""
    from .models import GraphSchemaVersion

    created = None
    if not GraphSchemaVersion.objects.exists():
        created = registry.save_version(Schema.from_yaml().definition, note="Default Industrial schema").version
    schema = registry.active_schema()
    with session() as s:
        statements = s.execute_write(repository.apply_constraints, schema)
    return {"schema_version": schema.version, "created_version": created, "statements": len(statements)}


def schema_summary() -> dict:
    schema = registry.active_schema()
    with session() as s:
        stats = s.execute_read(repository.counts)
    return {
        "name": schema.name,
        "version": schema.version,
        "description": schema.description,
        "entity_types": [
            {"name": label, "description": desc, "count": stats["nodes"].get(label, 0)}
            for label, desc in schema.entity_types.items()
        ],
        "relationship_types": [
            {"name": rt.name, "description": rt.description, "pairs": sorted(list(p) for p in rt.pairs),
             "count": stats["relationships"].get(rt.name, 0)}
            for rt in schema.relationship_types.values()
        ],
    }


# --- seed / bulk load -------------------------------------------------------

def resolve_ref(schema: Schema, ref: str) -> tuple[str, str]:
    """'Asset:EXTR01' or 'bsk:asset:EXTR01' -> (label, canonical id)."""
    if is_valid_id(ref):
        label = schema.label_for_slug(ref.split(":", 2)[1])
        if label is None:
            raise SchemaError(f"{ref!r} has no matching entity type")
        return label, ref
    match = _SHORT_REF.match(ref or "")
    if not match:
        raise SchemaError(f"invalid node reference {ref!r}")
    label, key = match.groups()
    schema.check_label(label)
    return label, make_id(label, *key.split("/"))


def plan_seed(document: dict, schema: Schema) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, str, dict]]]:
    """Validate a seed document fully before anything is written.

    Returns (nodes, relationships) as (label, id, props) / (from_id, type, to_id, props).
    """
    source = document.get("source", "seed")
    nodes, labels_by_id, errors = [], {}, []
    for entry in document.get("nodes") or []:
        try:
            label, node_id = resolve_ref(schema, entry.get("ref") or entry.get("id", ""))
        except SchemaError as exc:
            errors.append(str(exc))
            continue
        props = {k: v for k, v in entry.items() if k not in ("ref", "id")}
        props.setdefault("source", source)
        if node_id in labels_by_id:
            errors.append(f"duplicate node {node_id}")
        labels_by_id[node_id] = label
        nodes.append((label, node_id, props))

    rels = []
    for entry in document.get("relationships") or []:
        try:
            from_label, from_id = resolve_ref(schema, entry["from"])
            to_label, to_id = resolve_ref(schema, entry["to"])
            for node_id in (from_id, to_id):
                if node_id not in labels_by_id:
                    raise SchemaError(f"relationship refers to {node_id}, which the seed doesn't define")
            schema.check_relationship(from_label, entry["type"], to_label)
        except (SchemaError, KeyError) as exc:
            errors.append(f"{entry}: {exc}")
            continue
        props = {k: v for k, v in entry.items() if k not in ("from", "type", "to")}
        props.setdefault("source", source)
        rels.append((from_id, entry["type"], to_id, props))

    if errors:
        raise SchemaError(f"{len(errors)} problem(s): " + "; ".join(errors[:10]))
    return nodes, rels


def load_seed(path: str | Path, dry_run: bool = False) -> dict:
    """Idempotently MERGE a seed file into the graph (one transaction)."""
    document = yaml.safe_load(Path(path).read_text())
    schema = registry.active_schema()
    nodes, rels = plan_seed(document, schema)
    result = {"name": document.get("name"), "nodes": len(nodes), "relationships": len(rels),
              "schema_version": schema.version, "dry_run": dry_run}
    if dry_run:
        return result

    def write(tx):
        for label, node_id, props in nodes:
            repository.upsert_node(tx, schema, label, node_id, props)
        for from_id, rel, to_id, props in rels:
            repository.upsert_relationship(tx, schema, from_id, rel, to_id, props)

    with session() as s:
        s.execute_write(write)
    return result


def reset_graph() -> int:
    with session() as s:
        return s.execute_write(repository.delete_all_entities)


# --- reads ------------------------------------------------------------------

def list_assets() -> list[dict]:
    with session() as s:
        items, _ = s.execute_read(repository.list_nodes, "Asset", None, 500, 0)
    return items


def node_detail(node_id: str) -> dict:
    with session() as s:
        node = s.execute_read(repository.get_node, node_id)
        node["relationships"] = s.execute_read(repository.get_neighbors, node_id)
    return node


def search_nodes(label: str | None, query: str | None, limit: int = 50, offset: int = 0) -> dict:
    if label:
        registry.active_schema().check_label(label)
    with session() as s:
        items, total = s.execute_read(repository.list_nodes, label, query, limit, offset)
    return {"total": total, "offset": offset, "limit": limit, "items": items}


def graph_data(root_id: str | None, depth: int = 2, labels: list[str] | None = None, limit: int = 300) -> dict:
    schema = registry.active_schema()
    for label in labels or []:
        schema.check_label(label)
    with session() as s:
        return s.execute_read(repository.subgraph, root_id, depth, labels, limit)


def alarm_procedures(alarm_id: str) -> dict:
    def read(tx):
        alarm = repository.get_node(tx, alarm_id)
        if alarm["label"] != "Alarm":
            raise NodeNotFound(alarm_id)
        rows = tx.run(
            "MATCH (p:Procedure)-[:ADDRESSES]->(a:Entity {id: $id}) "
            "OPTIONAL MATCH (p)-[:APPLIES_TO]->(t:Entity) "
            "RETURN p, collect(DISTINCT t) AS targets ORDER BY p.name",
            id=alarm_id,
        )
        procedures = [
            {**repository.serialize_node(row["p"]),
             "applies_to": [repository.serialize_node(t) for t in row["targets"]]}
            for row in rows
        ]
        return {"alarm": alarm, "procedures": procedures}

    with session() as s:
        return s.execute_read(read)


def get_asset_context(asset_id: str) -> dict:
    """Everything the graph knows about one asset, structured for UI/agents.

    Answers: what asset is this, where it sits, its components (tree) and how
    they connect, which signals monitor them (with limits / OPC nodes), which
    alarms apply and which procedures address them, and which documents
    describe it.
    """

    def read(tx):
        asset = repository.get_node(tx, asset_id)
        if asset["label"] != "Asset":
            raise NodeNotFound(asset_id)

        hierarchy = _hierarchy(tx, asset_id)

        components = {
            row["c"]["id"]: {**repository.serialize_node(row["c"]), "parent": row["parent"],
                             "signals": [], "alarms": [], "children": []}
            for row in tx.run(
                "MATCH (a:Entity {id: $id})-[:HAS_COMPONENT*]->(c:Component) "
                "MATCH (p:Entity)-[:HAS_COMPONENT]->(c) "
                "RETURN DISTINCT c, p.id AS parent",
                id=asset_id,
            )
        }
        owner_ids = [asset_id, *components]

        signals_by_owner: dict[str, list] = {}
        for row in tx.run(
            "MATCH (o:Entity)-[:MONITORED_BY]->(s:Signal) WHERE o.id IN $owners "
            "OPTIONAL MATCH (s)-[:HAS_LIMIT]->(l:OperatingLimit) "
            "OPTIONAL MATCH (s)-[:IMPLEMENTED_AS]->(n:OPCNode) "
            "RETURN o.id AS owner, s, collect(DISTINCT l) AS limits, collect(DISTINCT n) AS opc "
            "ORDER BY s.name",
            owners=owner_ids,
        ):
            signals_by_owner.setdefault(row["owner"], []).append({
                **repository.serialize_node(row["s"]),
                "limits": [repository.serialize_node(l) for l in row["limits"]],
                "opc_nodes": [repository.serialize_node(n) for n in row["opc"]],
            })

        for row in tx.run(
            "MATCH (c:Component)-[:ASSOCIATED_WITH]->(al:Alarm) WHERE c.id IN $ids "
            "RETURN c.id AS comp, al.id AS alarm",
            ids=list(components),
        ):
            components[row["comp"]]["alarms"].append(row["alarm"])

        for comp_id, comp in components.items():
            comp["signals"] = signals_by_owner.get(comp_id, [])
        roots = []
        for comp in sorted(components.values(), key=lambda c: (c["name"], c["id"])):
            parent = components.get(comp["parent"])
            (parent["children"] if parent else roots).append(comp)

        connections = [
            {"source": row["s"], "target": row["t"], "kind": row["kind"]}
            for row in tx.run(
                "MATCH (a:Component)-[r:CONNECTED_TO]->(b:Component) WHERE a.id IN $ids "
                "RETURN a.id AS s, b.id AS t, r.kind AS kind",
                ids=list(components),
            )
        ]

        alarms = [
            {**repository.serialize_node(row["al"]),
             "components": row["comps"],
             "procedures": [repository.serialize_node(p) for p in row["procs"]]}
            for row in tx.run(
                "MATCH (a:Entity {id: $id})-[:HAS_ALARM]->(al:Alarm) "
                "OPTIONAL MATCH (c:Component)-[:ASSOCIATED_WITH]->(al) "
                "OPTIONAL MATCH (p:Procedure)-[:ADDRESSES]->(al) "
                "RETURN al, collect(DISTINCT c.id) AS comps, collect(DISTINCT p) AS procs ORDER BY al.code",
                id=asset_id,
            )
        ]

        procedures = [
            {**repository.serialize_node(row["p"]), "applies_to": row["targets"], "addresses": row["alarms"]}
            for row in tx.run(
                "MATCH (a:Entity {id: $id})-[:HAS_PROCEDURE]->(p:Procedure) "
                "OPTIONAL MATCH (p)-[:APPLIES_TO]->(t:Entity) "
                "OPTIONAL MATCH (p)-[:ADDRESSES]->(al:Alarm) "
                "RETURN p, collect(DISTINCT t.id) AS targets, collect(DISTINCT al.id) AS alarms ORDER BY p.name",
                id=asset_id,
            )
        ]

        documents = [
            {**repository.serialize_node(row["d"]), "describes": row["about"],
             "sections": [repository.serialize_node(sec) for sec in row["sections"]]}
            for row in tx.run(
                "MATCH (o:Entity)-[:DOCUMENTED_BY]->(d:Document) WHERE o.id IN $owners "
                "OPTIONAL MATCH (d)-[:HAS_SECTION]->(sec:DocumentSection) "
                "RETURN d, collect(DISTINCT o.id) AS about, collect(DISTINCT sec) AS sections ORDER BY d.name",
                owners=owner_ids,
            )
        ]

        return {
            "asset": asset,
            "hierarchy": hierarchy,
            "components": roots,
            "connections": connections,
            "signals": signals_by_owner.get(asset_id, []),
            "alarms": alarms,
            "procedures": procedures,
            "documents": documents,
            "counts": {
                "components": len(components),
                "signals": sum(len(v) for v in signals_by_owner.values()),
                "alarms": len(alarms),
                "procedures": len(procedures),
                "documents": len(documents),
            },
        }

    with session() as s:
        return s.execute_read(read)


def _hierarchy(tx, asset_id: str) -> list[dict]:
    """Plant → Area → Line chain above the asset (top first)."""
    record = tx.run(
        "MATCH path = (top:Entity)-[:CONTAINS*]->(a:Entity {id: $id}) "
        "WHERE NOT ()-[:CONTAINS]->(top) "
        "RETURN nodes(path) AS chain ORDER BY length(path) DESC LIMIT 1",
        id=asset_id,
    ).single()
    if record is None:
        return []
    return [repository.serialize_node(n) for n in record["chain"][:-1]]
