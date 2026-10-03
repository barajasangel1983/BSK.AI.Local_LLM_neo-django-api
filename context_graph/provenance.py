"""Traceability (P8c): where did a graph entity or relationship come from?

Answers "why is this in the graph?" for Explore and the asset chat:
- the origin written on the node / edge (`source`: a seed such as `seed:extr01-v1`,
  `text` = document extraction, `structured` = import, `library` = document evidence),
- every Evidence record behind it (document + page / section + excerpt, or data
  file + row; extractor, model, prompt), with the approved triples that used it.
"""

from __future__ import annotations

from collections import OrderedDict

from .driver import session
from .evidence import evidence_json
from .models import CandidateTriple, Evidence
from .repository import NodeNotFound, _value

MAX_SOURCES = 200


def origin_label(source: str | None) -> str:
    source = source or ""
    if source.startswith("seed:"):
        return f"Seed ({source.split(':', 1)[1]})"
    return {"text": "Document extraction", "structured": "Structured import", "library": "Document library",
            "seed": "Seed"}.get(source, source or "Unknown")


def evidence_items(evidence_ids) -> list[dict]:
    """Evidence records with their file names and the approved triples that used them."""
    rows = (Evidence.objects.filter(pk__in=list(evidence_ids)[:MAX_SOURCES])
            .select_related("document", "data_file").prefetch_related("triples").order_by("id"))
    items = []
    for e in rows:
        item = evidence_json(e)
        item["document"] = {"id": str(e.document_id), "filename": e.document.filename} if e.document_id else None
        item["data_file"] = {"id": str(e.data_file_id), "filename": e.data_file.filename} if e.data_file_id else None
        item["label"] = short_label(e)
        item["triples"] = [
            {"id": t.pk, "subject": t.subject_name, "predicate": t.predicate, "object": t.object_name,
             "status": t.status, "approved_at": t.committed_at.isoformat() if t.committed_at else None}
            for t in e.triples.all() if t.status == CandidateTriple.Status.APPROVED
        ]
        items.append(item)
    return items


def short_label(e: Evidence) -> str:
    """e.g. "Extruder manual.pdf p.4", "tags.csv row 12"."""
    if e.data_file_id:
        return f"{e.data_file.filename} row {e.row_number}" if e.row_number else e.data_file.filename
    if e.document_id:
        page = ""
        if e.page_start:
            page = f" p.{e.page_start}" + (f"–{e.page_end}" if e.page_end and e.page_end != e.page_start else "")
        return f"{e.document.filename}{page}"
    return e.source_kind


def _approved_evidence(node_id: str) -> set[int]:
    """Evidence of approved triples naming this node (in case a node predates P8a links)."""
    ids = (CandidateTriple.objects.filter(status=CandidateTriple.Status.APPROVED)
           .filter(subject_id=node_id) | CandidateTriple.objects.filter(status=CandidateTriple.Status.APPROVED,
                                                                          object_id=node_id))
    return set(Evidence.objects.filter(triples__in=ids).values_list("id", flat=True))


def node_sources(node_id: str) -> dict:
    if node_id.startswith("lab:"):
        return _lab_node_sources(node_id)
    with session() as s:
        row = s.run("MATCH (n:Entity {id: $id}) RETURN properties(n) AS p, [l IN labels(n) WHERE l <> 'Entity'][0] AS label",
                    id=node_id).single()
    if row is None:
        raise NodeNotFound(node_id)
    props = {k: _value(v) for k, v in row["p"].items()}
    ids = set(props.get("evidence_ids") or []) | _approved_evidence(node_id)
    return {
        "id": node_id, "label": row["label"], "name": props.get("name", ""),
        "aliases": props.get("aliases") or [], "subtype": props.get("subtype"), "tag": props.get("tag"),
        "origin": origin_label(props.get("source")), "source": props.get("source"),
        "created_by_triple": props.get("created_by_triple"),
        "sources": evidence_items(ids),
    }


def _lab_node_sources(node_id: str) -> dict:
    with session() as s:
        row = s.run("MATCH (n:Lab {id: $id}) OPTIONAL MATCH (n)-[r:LAB_RELATION]-() "
                    "RETURN n.name AS name, n.type AS type, collect(DISTINCT r.triple_id) AS tids", id=node_id).single()
    if row is None or row["name"] is None:
        raise NodeNotFound(node_id)
    ids = Evidence.objects.filter(triples__in=[t for t in row["tids"] if t is not None]).values_list("id", flat=True)
    return {"id": node_id, "label": "Lab", "name": row["name"], "aliases": [], "subtype": row["type"], "tag": None,
            "origin": "Free-form extraction (lab layer)", "source": "text", "created_by_triple": None,
            "sources": evidence_items(set(ids))}


def edge_sources(from_id: str, rel: str, to_id: str) -> dict:
    """Sources of one relationship (curated: by its ends and type)."""
    with session() as s:
        row = s.run("MATCH (:Entity {id: $f})-[r]->(:Entity {id: $t}) WHERE type(r) = $rel RETURN properties(r) AS p",
                    f=from_id, t=to_id, rel=rel).single()
    if row is None:
        raise NodeNotFound(f"{from_id} -{rel}-> {to_id}")
    props = {k: _value(v) for k, v in row["p"].items()}
    ids = set(props.get("evidence_ids") or [])
    ids |= set(Evidence.objects.filter(triples__in=props.get("triple_ids") or []).values_list("id", flat=True))
    return {"from": from_id, "type": rel, "to": to_id, "origin": origin_label(props.get("source")),
            "source": props.get("source"), "kind": props.get("kind"), "sources": evidence_items(ids)}


def triple_sources(triple_id: int) -> dict:
    """Sources of a lab-layer relationship (its link carries the triple id)."""
    t = CandidateTriple.objects.filter(pk=triple_id).first()
    if t is None:
        raise NodeNotFound(f"triple {triple_id}")
    return {"from": t.subject_name, "type": t.predicate, "to": t.object_name, "origin": origin_label(t.source),
            "source": t.source, "kind": None, "sources": evidence_items(t.evidence.values_list("id", flat=True))}


def fact_source_labels(node_props: dict[str, dict]) -> dict[str, str]:
    """Short source label per node (for asset-chat graph facts): seed, or the first document page / import row."""
    first_evidence = {}
    wanted = {nid: (p.get("evidence_ids") or [None])[0] for nid, p in node_props.items() if p.get("evidence_ids")}
    for e in Evidence.objects.filter(pk__in=[i for i in wanted.values() if i]).select_related("document", "data_file"):
        first_evidence[e.pk] = short_label(e)
    labels = OrderedDict()
    for nid, p in node_props.items():
        if wanted.get(nid) in first_evidence:
            labels[nid] = first_evidence[wanted[nid]]
        else:
            labels[nid] = origin_label(p.get("source"))
    return labels
