"""P26c: moving today's manual facts off the installed parts to their product family or device model.

Every fact read from a manual before P26b belongs to one installed part (`bsk:parameter:EXTR01/cutter-drive/p1082`).
A "target" is a product family (or a device model without one); its installed parts are the Components that are
INSTANCE_OF its models. For each target:

    plan(graph, target)   what would move, merge and conflict (pure: works on a loaded `Graph`, no writes)
    apply(target, ...)    one write transaction; a `FactMoveBatch` keeps every touched node and link as it was
    undo(batch)           puts the batch's nodes and links back (the latest applied batch only)

A fact moves **in place**: the node keeps its properties, evidence and links, its id becomes the one under the new
owner's key (`bsk:parameter:siemens/sinamics-g120c/p1082`) and the old id goes into `merged_ids`, which identity,
`repository.get_node` and approval resolve (P26c D1). Ratings go to the part's model, everything else to the family
(P26 D2); a fact's own sub-facts (a sub-part's ratings, an alarm's procedures) follow it to the family. The BOM's
links (`source` structured / seed) are machine facts and stay.

Copies of one fact from two installed parts (the two G120C drives read from one manual) become one node: same new id,
the same parameter code, or the same connector + terminal. Ratings identical on every model of the family go to the
family. A group whose members set a property to two different values is a **conflict**: by default each member is
kept on its own model (D3); the reviewer can keep one member's values (its id) or all of them ("both", joined with "; ").
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from django.db import transaction
from django.utils import timezone

from .driver import session
from .identity import OWNING, PART_LABELS, normalize
from .ids import type_slug
from .models import CandidateTriple, FactMove, FactMoveBatch

CARRY = (*OWNING, "HAS_COMPONENT")            # links that make a node a part's own
MACHINE = ("MAPS_TO", "CARRIES")              # a family fact's link to this machine: needs `instance` (P26 D3)
BOOKKEEPING = {"id", "name", "aliases", "evidence_ids", "created_at", "updated_at", "created_by_triple", "source",
               "doc_key", "facets", "merged_ids", "subtype", "document_id", "data_file"}
MODEL_LEVEL = ("Specification",)              # ratings: the model's row of the manual
JOINED = {"basis"}                            # where a value was read: the members' are joined, never a conflict
CHOICES = ("models", "both")                  # or a member's id: keep its values


class MoveError(ValueError):
    pass


# --- the graph, loaded once ---------------------------------------------------------------------------------------

@dataclass
class Graph:
    nodes: dict[str, dict] = field(default_factory=dict)       # id -> {label, props}
    rels: list[dict] = field(default_factory=list)             # {eid, type, a, b, props}

    def out(self, node_id: str):
        return self._out.get(node_id, [])

    def into(self, node_id: str):
        return self._in.get(node_id, [])

    def index(self) -> "Graph":
        self._out, self._in = defaultdict(list), defaultdict(list)
        for r in self.rels:
            self._out[r["a"]].append(r)
            self._in[r["b"]].append(r)
        return self

    def label(self, node_id: str) -> str:
        return (self.nodes.get(node_id) or {}).get("label", "")

    def props(self, node_id: str) -> dict:
        return (self.nodes.get(node_id) or {}).get("props", {})


def load_graph(tx) -> Graph:
    g = Graph()
    for row in tx.run("MATCH (n:Entity) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props"):
        label = next((x for x in row["labels"] if x != "Entity"), "Entity")
        g.nodes[row["id"]] = {"label": label, "labels": list(row["labels"]), "props": dict(row["props"])}
    for row in tx.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN elementId(r) AS eid, type(r) AS type, a.id AS a, "
                      "b.id AS b, properties(r) AS props"):
        g.rels.append({"eid": row["eid"], "type": row["type"], "a": row["a"], "b": row["b"], "props": dict(row["props"])})
    return g.index()


def _key(node_id: str) -> str:
    return node_id.split(":", 2)[2] if node_id.count(":") >= 2 else ""


def _slug(node_id: str) -> str:
    return node_id.split(":", 2)[1] if node_id.count(":") >= 2 else ""


# --- targets ------------------------------------------------------------------------------------------------------

def targets(g: Graph) -> list[dict]:
    """Every family, and every model without one, with its models and installed parts."""
    model_family = {r["a"]: r["b"] for r in g.rels if r["type"] == "VARIANT_OF"}
    part_model = {r["a"]: r["b"] for r in g.rels if r["type"] == "INSTANCE_OF"}
    out: dict[str, dict] = {}
    for model in {*model_family, *part_model.values()}:
        target = model_family.get(model, model)
        t = out.setdefault(target, {"id": target, "name": g.props(target).get("name", target),
                                    "level": "family" if target in model_family.values() else "model",
                                    "models": [], "parts": []})
        if model not in t["models"]:
            t["models"].append(model)
    for part, model in part_model.items():
        out[model_family.get(model, model)]["parts"].append(part)
    for t in out.values():
        t["models"].sort()
        t["parts"].sort()
    return sorted(out.values(), key=lambda t: t["name"].lower())


def _bom(g: Graph) -> set[str]:
    """The machine's own components: the BOM's and seed's, and every installed part of a model."""
    installed = {r["a"] for r in g.rels if r["type"] == "INSTANCE_OF"}
    return installed | {i for i, n in g.nodes.items() if n["label"] in ("Component", "Asset", "Line", "Area", "Plant")
                        and (n["props"].get("source") in ("structured",) or str(n["props"].get("source", "")).startswith("seed"))}


# --- plan ---------------------------------------------------------------------------------------------------------

def _owned(g: Graph, part: str, bom: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """The part's manual facts: {node: how it was reached ("direct" | parent id)} and {node: owning link type} for the
    direct ones. Direct: an owning link from the part made from a manual (source text), or an id under the part's key.
    Then a sub-part's own facts, an alarm's / failure mode's procedures."""
    key = _key(part)
    found: dict[str, str] = {}
    link: dict[str, str] = {}
    for r in g.out(part):
        if r["type"] in CARRY and r["props"].get("source") == "text" and r["b"] not in bom:
            found.setdefault(r["b"], "direct")
            link.setdefault(r["b"], r["type"])
    for node_id, n in g.nodes.items():
        if n["label"] in PART_LABELS and node_id not in bom and _key(node_id).startswith(key + "/"):
            found.setdefault(node_id, "direct")
    todo = list(found)
    while todo:
        x = todo.pop()
        nxt = [r["b"] for r in g.out(x) if r["type"] in CARRY or r["type"] == "MITIGATED_BY"]
        nxt += [r["a"] for r in g.into(x) if r["type"] == "ADDRESSES"]
        for y in nxt:
            if y not in found and y not in bom and g.label(y) in PART_LABELS:
                found[y] = x
                todo.append(y)
    return found, link


def _rest(old_id: str, part: str) -> str:
    """The id's key below its old owner: below the part's key, or below the asset's ("EXTR01/Fan-unit" -> "Fan-unit")."""
    key, part_key = _key(old_id), _key(part)
    if key.startswith(part_key + "/"):
        return key[len(part_key) + 1:]
    asset = part_key.split("/", 1)[0]
    return key[len(asset) + 1:] if key.startswith(asset + "/") else key


def _items(v) -> list[str]:
    """A value's items: a run joins the values it read twice with "; " ("2; 7; DI 2")."""
    return [x.strip() for x in str(v).split(";") if x.strip()]


def _joined(values: list) -> str | None:
    """The members' values as one, when they agree: every value's items are within the longest one's (one run read
    "2", the other "2; 7; DI 2"), or the property is a "where it was read". None when two values disagree."""
    set_ = [v for v in values if v not in (None, "", [])]
    if not set_:
        return None
    items = [_items(v) for v in set_]
    widest = max(items, key=lambda i: (len(i), sum(len(x) for x in i)))
    if all(all(any(_within(x, y) for y in widest) for x in i) for i in items):
        return "; ".join(widest) if len(widest) > 1 else str(set_[items.index(widest)])
    return None


def _within(x: str, y: str) -> bool:
    """Item `x` says no more than `y`: the same numbers ("150" / "150 rpm", "0" / "0.0") and no word `y` lacks."""
    nums = lambda s: tuple(float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", s))
    words = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    return nums(x) == nums(y) and words(x) <= words(y)


def _conflicts(g: Graph, members: list[str]) -> dict[str, list]:
    """Properties two members set to values that don't agree: {prop: [value per member]}."""
    keys = {k for m in members for k in g.props(m) if k not in BOOKKEEPING and k not in JOINED}
    out = {}
    for k in sorted(keys):
        values = [g.props(m).get(k) for m in members]
        set_ = [v for v in values if v not in (None, "", [])]
        if len({str(v).strip() for v in set_}) > 1 and _joined(values) is None:
            out[k] = values
    return out


def _agreed(g: Graph, members: list[str]) -> dict:
    """The joined value of every property the members agree on but wrote differently (and of `basis`)."""
    keys = {k for m in members for k in g.props(m) if k not in BOOKKEEPING}
    out = {}
    for k in keys:
        values = [g.props(m).get(k) for m in members]
        if len({str(v).strip() for v in values if v not in (None, "", [])}) > 1:
            if k in JOINED:
                out[k] = "; ".join(dict.fromkeys(x for v in values if v not in (None, "", []) for x in _items(v)))
            elif _joined(values) is not None:
                out[k] = _joined(values)
    return out


def _rating_signature(g: Graph, node_id: str) -> tuple:
    p = g.props(node_id)
    return (normalize(p.get("name")), *(str(p.get(k) or "").strip() for k in ("value", "min", "max", "unit", "condition")))


def plan(g: Graph, target_id: str, choices: dict | None = None, stays: list[str] | None = None) -> dict:
    """What moving `target_id`'s installed parts' manual facts would do. `choices` {group id: models|a|b|both} and
    `stays` (old ids left on their part) are the reviewer's; without them, the defaults (D3: conflicts per model)."""
    choices, stays = choices or {}, set(stays or ())
    target = next((t for t in targets(g) if t["id"] == target_id), None)
    if target is None:
        raise MoveError(f"{target_id} is not a product family or a device model of an installed part")
    bom = _bom(g)
    family_key = _key(target_id)
    part_model = {r["a"]: r["b"] for r in g.rels if r["type"] == "INSTANCE_OF" and r["a"] in target["parts"]}
    inside = set(target["parts"]) | {target_id, *target["models"]}

    reached: dict[str, dict] = {}      # node -> {part, how, link}
    for part in target["parts"]:
        found, link = _owned(g, part, bom)
        for node_id, how in found.items():
            reached.setdefault(node_id, {"part": part, "how": how, "link": link.get(node_id, ""), "parts": set()})
            reached[node_id]["parts"].add(part)

    # A fact also owned from outside this family (another family's part, the asset) is shared: it stays, reported.
    shared, skipped_stay = [], []
    for node_id in list(reached):
        outside = {r["a"] for r in g.into(node_id) if r["type"] in CARRY and r["a"] not in inside and r["a"] not in reached}
        if outside:
            shared.append({"id": node_id, "label": g.label(node_id), "name": g.props(node_id).get("name", ""),
                           "owners": sorted(outside)})
            del reached[node_id]
    # Stays: the node and what was reached only through it.
    if stays:
        changed = True
        while changed:
            changed = False
            for node_id, info in list(reached.items()):
                if node_id in stays or info["how"] in stays:
                    if node_id not in stays:
                        stays.add(node_id)
                        changed = True
        for node_id in [n for n in reached if n in stays]:
            info = reached.pop(node_id)
            skipped_stay.append({"id": node_id, "label": g.label(node_id), "name": g.props(node_id).get("name", ""),
                                 "part": info["part"]})

    def dest_of(node_id: str, info: dict) -> str:
        if info["how"] == "direct" and g.label(node_id) in MODEL_LEVEL and len(info["parts"]) == 1:
            return part_model.get(info["part"], target_id)
        return target_id

    # Ratings identical on every model of the family (with ratings): one, on the family.
    rated_models = {dest_of(n, i) for n, i in reached.items() if i["how"] == "direct" and g.label(n) in MODEL_LEVEL}
    to_family: set[str] = set()
    if target["level"] == "family" and len(rated_models) > 1:
        by_sig = defaultdict(set)
        for n, i in reached.items():
            if i["how"] == "direct" and g.label(n) in MODEL_LEVEL:
                by_sig[_rating_signature(g, n)].add(n)
        for sig, members in by_sig.items():
            if {dest_of(m, reached[m]) for m in members} == rated_models:
                to_family |= members

    moves = {}
    for node_id, info in reached.items():
        dest = target_id if node_id in to_family else dest_of(node_id, info)
        rest = _rest(node_id, info["part"])
        moves[node_id] = {"old_id": node_id, "label": g.label(node_id), "name": g.props(node_id).get("name", ""),
                          "part": info["part"], "parts": sorted(info["parts"]), "dest": dest,
                          "new_id": f"bsk:{_slug(node_id)}:{_key(dest)}/{rest}", "link": info["link"],
                          "via": "" if info["how"] == "direct" else info["how"]}

    # Groups: one new id; or one parameter code / one terminal in one destination.
    groups: dict[str, list[str]] = defaultdict(list)
    alias_of: dict[tuple, str] = {}
    for node_id in sorted(moves):
        m, p = moves[node_id], g.props(node_id)
        same = None
        if m["label"] == "Parameter" and p.get("code"):
            same = (m["dest"], "code", normalize(p["code"]))
        elif m["label"] == "IOPoint" and p.get("terminal"):
            same = (m["dest"], "terminal", normalize(p.get("connector")), normalize(p["terminal"]))
        new_id = alias_of.get(same, m["new_id"]) if same else m["new_id"]
        if same:
            alias_of.setdefault(same, new_id)
        m["new_id"] = new_id
        groups[new_id].append(node_id)

    merges, conflicts = [], []
    for new_id, members in sorted(groups.items()):
        existing = new_id in g.nodes and new_id not in moves      # already on the family / model (a P26b run)
        everyone = ([new_id] if existing else []) + members
        if len(everyone) < 2:
            continue
        differ = _conflicts(g, everyone)
        if differ:
            choice = choices.get(new_id, "models")
            if choice not in CHOICES and choice not in everyone:
                raise MoveError(f"choice for {new_id} must be models, both or one of its members' ids")
            conflicts.append({"id": new_id, "label": moves[members[0]]["label"], "choice": choice,
                              "properties": sorted(differ),
                              "members": [{"id": m, "name": g.props(m).get("name", ""), "part": moves[m]["part"] if m in moves else "",
                                           "model": part_model.get(moves[m]["part"], "") if m in moves else "",
                                           "values": {k: g.props(m).get(k) for k in differ},
                                           "pages": _pages(g, m)} for m in everyone]})
            if choice == "models":
                for m in members:      # each on its own model: no merge
                    model = part_model.get(moves[m]["part"], target_id)
                    moves[m]["dest"] = model
                    moves[m]["new_id"] = f"bsk:{_slug(m)}:{_key(model)}/{_rest(m, moves[m]['part'])}"
                    moves[m]["conflict"] = new_id
                continue
        survivor = new_id if existing else min(members, key=lambda m: (str(g.props(m).get("created_at", "")), m))
        merges.append({"id": new_id, "label": moves[members[0]]["label"], "survivor": survivor,
                       "members": [{"id": m, "name": g.props(m).get("name", ""),
                                    "part": moves[m]["part"] if m in moves else ""} for m in everyone],
                       "choice": choices.get(new_id) if differ else None, "agreed": _agreed(g, everyone)})
    collisions = defaultdict(list)
    for m in moves.values():
        collisions[m["new_id"]].append(m["old_id"])
    merged_ids = {x["id"] for x in merges}
    clash = [k for k, v in collisions.items() if (len(v) > 1 or (k in g.nodes and k not in moves)) and k not in merged_ids]
    if clash:
        raise MoveError(f"new ids taken by other nodes: {', '.join(clash[:5])}")

    machine = [{"type": r["type"], "from": r["a"], "to": r["b"], "instance": moves[r["a"]]["part"]}
               for r in g.rels if r["a"] in moves and (r["type"] in MACHINE or (r["type"] == "INDICATED_BY" and g.label(r["b"]) == "Signal"))
               and not r["props"].get("instance")]
    docs = sorted({(moves[r["b"]]["dest"], r["props"].get("doc_key")) for part in target["parts"] for r in g.out(part)
                   if r["b"] in moves and r["props"].get("doc_key")})
    have_doc = {(r["a"], g.props(r["b"]).get("doc_key")) for r in g.rels if r["type"] == "DOCUMENTED_BY"}
    documents = [{"owner": o, "doc_key": d, "document": f"bsk:document:{d}"} for o, d in docs
                 if (o, d) not in have_doc and f"bsk:document:{d}" in g.nodes]
    counts = defaultdict(lambda: defaultdict(int))
    for m in moves.values():
        counts[m["label"]]["family" if m["dest"] == target_id else "model"] += 1
    return {"target": target, "moves": sorted(moves.values(), key=lambda m: (m["label"], m["name"].lower(), m["old_id"])),
            "merges": merges, "conflicts": conflicts, "shared": shared, "stays": skipped_stay,
            "machine_links": machine, "documents": documents,
            "counts": {k: dict(v) for k, v in sorted(counts.items())},
            "summary": {"moved": len(moves), "merged_away": sum(len(x["members"]) - 1 for x in merges),
                        "merges": len(merges), "conflicts": len(conflicts), "shared": len(shared),
                        "stays": len(skipped_stay), "machine_links": len(machine), "documents": len(documents)}}


def _pages(g: Graph, node_id: str) -> list:
    return sorted({(r["props"].get("doc_key"), r["props"].get("page_start")) for r in g.into(node_id)
                   if r["props"].get("page_start") is not None}, key=str)[:3]


# --- apply / undo -------------------------------------------------------------------------------------------------

def _plain(v):
    """A Neo4j value as JSON: temporal values tagged so undo writes them back as temporal values."""
    if hasattr(v, "iso_format"):
        return {"$dt": v.iso_format()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _props_out(props: dict) -> dict:
    return {k: _plain(v) for k, v in props.items()}


def _set_props(var: str, props: dict) -> tuple[str, dict]:
    """A SET clause writing `props` back (temporal values through datetime())."""
    plain = {k: v for k, v in props.items() if not (isinstance(v, dict) and "$dt" in v)}
    temporal = {k: v["$dt"] for k, v in props.items() if isinstance(v, dict) and "$dt" in v}
    parts = [f"{var} = $plain"] + [f"{var}.`{k}` = datetime($t{i})" for i, k in enumerate(temporal)]
    return "SET " + ", ".join(parts), {"plain": plain, **{f"t{i}": v for i, v in enumerate(temporal.values())}}


def _snapshot(g: Graph, ids: set[str]) -> dict:
    nodes = [{"id": i, "labels": g.nodes[i]["labels"], "props": _props_out(g.nodes[i]["props"])} for i in sorted(ids)]
    seen, rels = set(), []
    for i in ids:
        for r in [*g.out(i), *g.into(i)]:
            if r["eid"] not in seen:
                seen.add(r["eid"])
                rels.append({"type": r["type"], "a": r["a"], "b": r["b"], "props": _props_out(r["props"])})
    return {"nodes": nodes, "rels": rels}


_UNION = ("triple_ids", "evidence_ids")


def _repoint(tx, rel_type: str, from_id: str, to_id: str, other: str, outgoing: bool) -> None:
    """Move one link of `from_id` (to / from `other`) onto `to_id`, keeping its properties; ids and evidence of a link
    that already exists are joined."""
    a, b = ("$to", "$other") if outgoing else ("$other", "$to")
    old = "(f)-[r:`%s`]->(o)" % rel_type if outgoing else "(o)-[r:`%s`]->(f)" % rel_type
    union = ", ".join(f"r2.{k} = [x IN coalesce(r2.{k}, []) WHERE NOT x IN coalesce(r.{k}, [])] + coalesce(r.{k}, [])"
                      for k in _UNION)
    tx.run(f"MATCH (f:Entity {{id: $from}}), (o:Entity {{id: $other}}), (t:Entity {{id: $to}}) "
           f"MATCH {old} "
           f"MERGE ({'t' if outgoing else 'o'})-[r2:`{rel_type}`]->({'o' if outgoing else 't'}) "
           f"ON CREATE SET r2 = properties(r) ON MATCH SET {union} DELETE r",
           **{"from": from_id, "to": to_id, "other": other})


def apply(target_id: str, choices: dict | None = None, stays: list[str] | None = None, author: str = "") -> FactMoveBatch:
    """Move the target's facts (one Neo4j write transaction; the batch row in the same SQLite transaction)."""
    _no_jobs()
    with session() as s:
        g = s.execute_read(load_graph)
        p = plan(g, target_id, choices, stays)
        if not p["moves"]:
            raise MoveError("nothing to move")
        touched = {m["old_id"] for m in p["moves"]} | {x["id"] for x in p["merges"] if x["id"] in g.nodes}
        before = _snapshot(g, touched)
        triples = _triples_to_anchor(p)
        before["triples"] = [{"id": t.pk, "anchor_id": t.anchor_id, "moved_from": t.moved_from} for t in triples]
        # SQLite first (the batch, its moves, the triples' anchors), the graph last: a failed graph write rolls the
        # SQLite side back; once the graph is written, only `created_links` is left to save.
        with transaction.atomic():
            batch = FactMoveBatch.objects.create(target_id=target_id, choices={"choices": choices or {}, "stays": sorted(stays or ())},
                                                 summary=p["summary"], before=before, author=author)
            _record(batch, p)
            for t in triples:
                t.moved_from, t.anchor_id = t.anchor_id, _triple_dest(p, t)
                t.save(update_fields=["anchor_id", "moved_from", "updated_at"])
            created = s.execute_write(_write, g, p)
        batch.created_links = created
        batch.save(update_fields=["created_links"])
    return batch


def _no_jobs() -> None:
    from ingestion.backup import active_jobs

    if active_jobs():
        raise MoveError("a library job is running: try again when it is done")


def batch_json(b: FactMoveBatch) -> dict:
    return {"id": b.pk, "target": b.target_id, "status": b.status, "summary": b.summary, "author": b.author,
            "created_at": b.created_at.isoformat(), "undone_at": b.undone_at.isoformat() if b.undone_at else None,
            "choices": b.choices}


def _write(tx, g: Graph, p: dict) -> list[dict]:
    moves = {m["old_id"]: m for m in p["moves"]}
    parts = p["target"]["parts"]
    # 1. merges: every other member's links onto the survivor, its names and evidence onto it, then it goes
    for group in p["merges"]:
        survivor = group["survivor"]
        others = [m["id"] for m in group["members"] if m["id"] != survivor]
        for m in others:      # the links as they are now (an earlier group may have added some)
            for r in tx.run("MATCH (m:Entity {id: $id})-[r]->(o:Entity) RETURN type(r) AS type, o.id AS other", id=m).data():
                if r["other"] not in (survivor, m):
                    _repoint(tx, r["type"], m, survivor, r["other"], True)
            for r in tx.run("MATCH (o:Entity)-[r]->(m:Entity {id: $id}) RETURN type(r) AS type, o.id AS other", id=m).data():
                if r["other"] not in (survivor, m):
                    _repoint(tx, r["type"], m, survivor, r["other"], False)
        sp = g.props(survivor)
        aliases = list(dict.fromkeys([*(sp.get("aliases") or []),
                                      *[x for m in others for x in [g.props(m).get("name"), *(g.props(m).get("aliases") or [])]
                                        if x and normalize(x) != normalize(sp.get("name"))]]))
        evidence = list(dict.fromkeys([*(sp.get("evidence_ids") or []), *[e for m in others for e in g.props(m).get("evidence_ids") or []]]))
        merged = list(dict.fromkeys([*(sp.get("merged_ids") or []), *others,
                                     *[x for m in others for x in g.props(m).get("merged_ids") or []]]))
        fill = {}
        for m in others:
            for k, v in g.props(m).items():
                if k not in BOOKKEEPING and v not in (None, "", []) and sp.get(k) in (None, "", []) and k not in fill:
                    fill[k] = v
        fill.update(group.get("agreed") or {})
        conflict = next((c for c in p["conflicts"] if c["id"] == group["id"]), None)
        if conflict:
            values = {k: [mm["values"].get(k) for mm in conflict["members"]] for k in conflict["properties"]}
            for k, vals in values.items():
                if conflict["choice"] == "both":
                    fill[k] = "; ".join(dict.fromkeys(str(v) for v in vals if v not in (None, "", [])))
                else:
                    pick = next(mm for mm in conflict["members"] if mm["id"] == conflict["choice"])["values"].get(k)
                    if pick not in (None, "", []):
                        fill[k] = pick
        tx.run("MATCH (n:Entity {id: $id}) SET n += $fill, n.aliases = $aliases, n.evidence_ids = $ev, "
               "n.merged_ids = $merged, n.updated_at = datetime()",
               id=survivor, fill={k: _back(v) for k, v in fill.items()}, aliases=aliases, ev=evidence, merged=merged)
        tx.run("MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n", ids=others)
    # 2. renames: each remaining node to its new id; the old one into merged_ids
    gone = {m["id"] for x in p["merges"] for m in x["members"] if m["id"] != x["survivor"]}
    for old, m in moves.items():
        if old in gone:
            continue
        if old != m["new_id"]:
            tx.run("MATCH (n:Entity {id: $old}) SET n.id = $new, n.updated_at = datetime(), "
                   "n.merged_ids = [x IN coalesce(n.merged_ids, []) WHERE x <> $old] + $old", old=old, new=m["new_id"])
    # 3. owning links from the installed parts to the family / model
    final = {}
    for old, m in moves.items():
        final.setdefault(m["new_id"], m["dest"])
    for new_id, dest in final.items():
        for part in parts:
            for rel_type in CARRY:
                if tx.run(f"MATCH (:Entity {{id: $p}})-[r:`{rel_type}`]->(:Entity {{id: $x}}) RETURN count(r) AS n",
                          p=part, x=new_id).single()["n"]:
                    _repoint(tx, rel_type, part, dest, new_id, True)
    # 4. a family fact's machine links name their installed part
    for link in p["machine_links"]:
        new_from = moves[link["from"]]["new_id"]
        tx.run(f"MATCH (:Entity {{id: $a}})-[r:`{link['type']}`]->(:Entity {{id: $b}}) WHERE r.instance IS NULL "
               "SET r.instance = $inst", a=new_from, b=link["to"], inst=link["instance"])
    # 5. the family / model is documented by the manuals its facts came from
    created = []
    for d in p["documents"]:
        row = tx.run("MATCH (o:Entity {id: $o}), (d:Entity {id: $d}) "
                     "MERGE (o)-[r:DOCUMENTED_BY]->(d) ON CREATE SET r.source = 'p26c', r.created_at = datetime(), "
                     "r.new = true WITH r, r.new AS fresh REMOVE r.new RETURN fresh", o=d["owner"], d=d["document"]).single()
        if row and row["fresh"]:
            created.append({"type": "DOCUMENTED_BY", "a": d["owner"], "b": d["document"]})
    return created


def _back(v):
    return v["$dt"] if isinstance(v, dict) and "$dt" in v else v


def _triples_to_anchor(p: dict) -> list[CandidateTriple]:
    """The approved manual triples of the target's installed parts: their anchor becomes the family / model."""
    return list(CandidateTriple.objects.filter(anchor_id__in=p["target"]["parts"], document__isnull=False,
                                               status=CandidateTriple.Status.APPROVED))


def _triple_dest(p: dict, t: CandidateTriple) -> str:
    moves = {m["old_id"]: m for m in p["moves"]}
    for end in (t.object_id, t.subject_id):
        if end in moves:
            return moves[end]["dest"]
    return p["target"]["id"]


def _record(batch: FactMoveBatch, p: dict) -> None:
    survivor_of = {m["id"]: x["survivor"] for x in p["merges"] for m in x["members"]}
    final = {m["old_id"]: m["new_id"] for m in p["moves"]}
    rows = []
    for m in p["moves"]:
        merged = survivor_of.get(m["old_id"]) not in (None, m["old_id"])
        kind = FactMove.Kind.MERGE if merged else (FactMove.Kind.MODEL if m.get("conflict") else FactMove.Kind.MOVE)
        rows.append(FactMove(batch=batch, kind=kind, label=m["label"], old_id=m["old_id"],
                             new_id=m["new_id"] if not merged else final.get(survivor_of[m["old_id"]], m["new_id"]),
                             owner_before=m["part"], owner_after=m["dest"]))
    rows += [FactMove(batch=batch, kind=FactMove.Kind.STAY, label=s["label"], old_id=s["id"], new_id=s["id"],
                      owner_before=s["part"], owner_after=s["part"]) for s in p["stays"]]
    FactMove.objects.bulk_create(rows)


def undo(batch: FactMoveBatch) -> dict:
    """Put a batch back: its nodes and links as they were, the triples' anchors. Only the latest applied batch."""
    _no_jobs()
    if batch.status != FactMoveBatch.Status.APPLIED:
        raise MoveError("this batch is not applied")
    later = FactMoveBatch.objects.filter(status=FactMoveBatch.Status.APPLIED, created_at__gt=batch.created_at)
    if later.exists():
        raise MoveError("undo the later batches first: " + ", ".join(str(b.pk) for b in later))
    final = sorted({m.new_id for m in batch.moves.all() if m.kind != FactMove.Kind.STAY}
                   | {n["id"] for n in batch.before.get("nodes", [])})
    with session() as s:
        s.execute_write(_restore, batch, final)
    with transaction.atomic():
        for t in batch.before.get("triples", []):
            CandidateTriple.objects.filter(pk=t["id"]).update(anchor_id=t["anchor_id"], moved_from=t["moved_from"])
        batch.status, batch.undone_at = FactMoveBatch.Status.UNDONE, timezone.now()
        batch.save(update_fields=["status", "undone_at"])
    return {"nodes": len(batch.before.get("nodes", [])), "links": len(batch.before.get("rels", []))}


def _restore(tx, batch: FactMoveBatch, final: list[str]) -> None:
    for link in batch.created_links or []:
        tx.run(f"MATCH (:Entity {{id: $a}})-[r:`{link['type']}`]->(:Entity {{id: $b}}) WHERE r.source = 'p26c' DELETE r",
               a=link["a"], b=link["b"])
    tx.run("MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n", ids=final)
    for n in batch.before["nodes"]:
        labels = ":".join(f"`{x}`" for x in n["labels"] if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", x))
        clause, params = _set_props("n", n["props"])
        tx.run(f"CREATE (n:{labels}) {clause}", **params)
    for r in batch.before["rels"]:
        clause, params = _set_props("r", r["props"])
        found = tx.run(f"MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}}) CREATE (a)-[r:`{r['type']}`]->(b) {clause} "
                       "RETURN count(r) AS n", a=r["a"], b=r["b"], **params).single()["n"]
        if not found:
            raise MoveError(f"can't restore {r['a']} -{r['type']}-> {r['b']}: an end is missing")


# --- old ids ------------------------------------------------------------------------------------------------------

def current_id(tx, node_id: str) -> str:
    """`node_id`, or the node that holds it in `merged_ids` (a fact moved or merged in P26c)."""
    if not node_id or tx.run("MATCH (n:Entity {id: $id}) RETURN n.id", id=node_id).single():
        return node_id
    row = tx.run("MATCH (n:Entity) WHERE $id IN n.merged_ids RETURN n.id AS id LIMIT 1", id=node_id).single()
    return row["id"] if row else node_id
