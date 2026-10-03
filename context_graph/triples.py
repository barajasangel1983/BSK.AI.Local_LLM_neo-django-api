"""Review of extracted triples: edit, approve (commit to Neo4j), reject, delete, promote.

Curated layer (schema triples): the normal :Entity graph, validated by the
active schema. Every written relationship carries provenance (`source: "text"`,
`triple_id`, document, page, model, preset); nodes created by a triple record
`created_by_triple`. The document and its section are written as Document /
DocumentSection nodes (evidence pointers — the text stays in the library) and
linked to what they describe where the schema allows it.

Structured imports (triples built from a data file's rows) use the same curated
commit with `source: "structured"`: they carry node properties (written when the
entity is new; only missing ones are added to an existing entity, and removed
again if the triple is deleted) and have no document evidence nodes.

Lab layer (free-form triples): `(:Lab {id, name, type})-[:LAB_RELATION
{predicate, triple_id}]->(:Lab)`, kept apart from the curated graph; a lab
triple can be promoted to the curated layer once mapped onto the schema.
"""

from __future__ import annotations

import hashlib
import re

from django.utils import timezone

from . import registry, repository
from .driver import session
from . import evidence
from .extraction import EntityIndex, normalize, resolve_id, scope_key_for
from .ids import is_valid_id, make_id
from .models import CandidateTriple
from .schema import SchemaError

EDITABLE = ("subject_name", "subject_type", "subject_id", "predicate", "object_name", "object_type", "object_id")
EVIDENCE_RELS = ("DESCRIBES", "HAS_SECTION", "DOCUMENTED_BY")
STAGED_SOURCES = ["text", "structured"]   # `source` of nodes/edges created by approved triples


class TripleError(ValueError):
    pass


# --- editing ---------------------------------------------------------------------

def edit(triple: CandidateTriple, changes: dict) -> CandidateTriple:
    if triple.status == CandidateTriple.Status.APPROVED:
        raise TripleError("approved triples can't be edited — delete it, or promote a lab triple")
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise TripleError(f"not editable: {', '.join(sorted(unknown))}")
    for key, value in changes.items():
        setattr(triple, key, str(value or "").strip()[:512])
    if triple.mode == "schema":
        _reresolve(triple, changes)
        triple.issue = _schema_issue(triple)
    triple.edited = True
    triple.save()
    return triple


def _reresolve(t: CandidateTriple, changes: dict, index=None) -> None:
    """Re-match/propose ids for ends whose name or type changed (unless an id was given)."""
    index = index or EntityIndex.load()
    if t.data_file_id:
        scope = t.data_file.asset_id.split(":", 2)[2] if t.data_file.asset_id else t.data_file.key
    else:
        scope = scope_key_for(t.job.params if t.job else {}, t.document)
    for end in ("subject", "object"):
        given = changes.get(f"{end}_id")
        if given:
            setattr(t, f"{end}_existing", bool(index.has(given)))
        elif f"{end}_name" in changes or f"{end}_type" in changes or not getattr(t, f"{end}_id"):
            try:
                node_id, existing = resolve_id(index, getattr(t, f"{end}_type"), getattr(t, f"{end}_name"), scope)
            except ValueError:
                node_id, existing = "", False
            setattr(t, f"{end}_id", node_id)
            setattr(t, f"{end}_existing", existing)


def _schema_issue(t: CandidateTriple) -> str:
    schema = registry.active_schema()
    try:
        schema.check_label(t.subject_type)
        schema.check_label(t.object_type)
        schema.check_relationship(t.subject_type, t.predicate, t.object_type)
        for node_id, label in ((t.subject_id, t.subject_type), (t.object_id, t.object_type)):
            if not is_valid_id(node_id) or schema.label_for_slug(node_id.split(":", 2)[1]) != label:
                return f"{node_id or '(empty id)'} is not a valid {label} id"
    except SchemaError as exc:
        return str(exc)[:255]
    return ""


# --- commit ------------------------------------------------------------------------

def _provenance(t: CandidateTriple) -> dict:
    if t.data_file_id:
        return {"source": "structured", "triple_id": t.pk, "data_file_id": str(t.data_file_id),
                "data_file": t.data_file.filename, "row": t.row_number, "imported_at": t.created_at.isoformat()}
    return {
        "source": "text", "triple_id": t.pk, "document_id": str(t.document_id), "doc_key": t.document.doc_key,
        "page_start": t.page_start, "page_end": t.page_end, "confidence": t.confidence, "model": t.model,
        "preset": f"{t.preset_name} v{t.preset_version}", "extracted_at": t.created_at.isoformat(),
    }


def _section_id(t: CandidateTriple) -> str:
    digest = hashlib.sha1(" > ".join(t.section_path or []).encode()).hexdigest()[:8]
    return make_id("DocumentSection", t.document.doc_key, f"p{t.page_start or 0}-{digest}")


def _supports(tx, schema, t: CandidateTriple, from_id: str, rel: str, to_id: str, props: dict) -> None:
    """Create (or reuse) a relationship and record that triple `t` supports it.

    Relationships are MERGEd per (from, type, to), so several triples — or the
    historian seed — can back the same edge. Each supporting triple is listed in
    `triple_ids`; provenance properties are only written on edges that text
    created, never on seeded ones.
    """
    repository.upsert_relationship(tx, schema, from_id, rel, to_id, {})   # validates + MERGE
    tx.run(
        f"MATCH (:Entity {{id: $f}})-[r:{rel}]->(:Entity {{id: $t}}) "   # rel validated above
        f"SET r.source = coalesce(r.source, $src), "
        f"    r.triple_ids = [x IN coalesce(r.triple_ids, []) WHERE x <> $tid] + $tid, "
        f"    r.evidence_ids = [x IN coalesce(r.evidence_ids, []) WHERE NOT x IN $ev] + $ev "
        f"WITH r WHERE r.source = $src SET r += $props",
        f=from_id, t=to_id, tid=t.pk, props=props, src=t.source, ev=_evidence_ids(t),
    )


def _commit_curated(tx, t: CandidateTriple, schema, asset_scope: str | None) -> None:
    issue = _schema_issue(t)
    if issue:
        raise TripleError(issue)
    doc = t.document
    origin = {"doc_key": doc.doc_key} if doc else {"data_file": t.data_file.filename}
    applied, aliased = {}, {}
    ev = _evidence_ids(t)
    for node_id, label, name, props in ((t.subject_id, t.subject_type, t.subject_name, t.subject_props),
                                        (t.object_id, t.object_type, t.object_name, t.object_props)):
        existing = tx.run("MATCH (n:Entity {id: $id}) RETURN keys(n) AS keys, n.name AS name, "
                          "coalesce(n.aliases, []) AS aliases", id=node_id).single()
        if not existing:
            repository.upsert_node(tx, schema, label, node_id, {**(props or {}), "name": name, "source": t.source,
                                                                "created_by_triple": t.pk, **origin})
        else:
            # Existing entities keep their name and properties; only missing properties are added,
            # and a different name becomes an alias (never a rename).
            missing = {k: v for k, v in (props or {}).items() if k not in existing["keys"]}
            if missing:
                tx.run("MATCH (n:Entity {id: $id}) SET n += $props", id=node_id, props=missing)
                applied[node_id] = sorted(missing)
            claimed = apply_alias(tx, node_id, name, existing["name"], existing["aliases"])
            if claimed:
                aliased.setdefault(node_id, []).append(claimed)
        tx.run("MATCH (n:Entity {id: $id}) "
               "SET n.evidence_ids = [x IN coalesce(n.evidence_ids, []) WHERE NOT x IN $ev] + $ev", id=node_id, ev=ev)
    t.applied_props, t.applied_aliases = applied, aliased
    _supports(tx, schema, t, t.subject_id, t.predicate, t.object_id, _provenance(t))
    if doc is None:
        return   # structured import: no document evidence nodes

    # Evidence pointers: document + section, linked to what they describe (where the schema allows).
    doc_id = make_id("Document", doc.doc_key)
    repository.upsert_node(tx, schema, "Document", doc_id, {"name": doc.filename, "doc_key": doc.doc_key,
                                                            "document_id": str(doc.id), "source": "library"})
    section_id = _section_id(t)
    repository.upsert_node(tx, schema, "DocumentSection", section_id, {
        "name": (t.section_path or [doc.filename])[-1], "section_path": t.section_path or [],
        "page_start": t.page_start, "page_end": t.page_end, "doc_key": doc.doc_key,
        "document_id": str(doc.id), "source": "library"})
    repository.upsert_relationship(tx, schema, doc_id, "HAS_SECTION", section_id, {"source": "library"})
    for node_id, label in {(t.subject_id, t.subject_type), (t.object_id, t.object_type)}:
        if _allowed(schema, "DocumentSection", "DESCRIBES", label):
            _supports(tx, schema, t, section_id, "DESCRIBES", node_id, {"doc_key": doc.doc_key})
    if asset_scope and _allowed(schema, "Asset", "DOCUMENTED_BY", "Document"):
        if tx.run("MATCH (a:Asset {id: $id}) RETURN a.id", id=asset_scope).single():
            repository.upsert_relationship(tx, schema, asset_scope, "DOCUMENTED_BY", doc_id, {"source": "library"})


def _evidence_ids(t: CandidateTriple) -> list[int]:
    if not hasattr(t, "_evidence_ids"):
        t._evidence_ids = evidence.ids_for(t) if t.pk else []
    return t._evidence_ids


def apply_alias(tx, node_id: str, name: str, node_name: str, aliases: list[str]) -> str | None:
    """Record `name` as an alias of an existing node; returns the alias this triple now co-owns, if any.

    A name equal to the node's own name adds nothing. A new name is added. A name that
    matches an alias another approved triple added is shared, so the alias goes only when
    its last user goes; aliases from the seed or a person are never claimed.
    """
    if not name or normalize(name) == normalize(node_name or ""):
        return None
    same = [a for a in aliases if normalize(a) == normalize(name)]
    if not same:
        tx.run("MATCH (n:Entity {id: $id}) SET n.aliases = coalesce(n.aliases, []) + $name", id=node_id, name=name)
        return name
    return same[0] if _alias_claimed(node_id, same[0]) else None


def _alias_claimed(node_id: str, alias: str) -> bool:
    """The alias was added by an approved triple (not by the seed or a person)."""
    return any(alias in (claims or {}).get(node_id, []) for claims in
               CandidateTriple.objects.filter(status=CandidateTriple.Status.APPROVED,
                                              applied_aliases__has_key=node_id).values_list("applied_aliases", flat=True))


def _alias_still_used(t: CandidateTriple, node_id: str, name: str) -> bool:
    """Another approved triple still names `node_id` this way (so the alias stays)."""
    from django.db.models import Q
    others = (CandidateTriple.objects.filter(status=CandidateTriple.Status.APPROVED).exclude(pk=t.pk)
              .filter(Q(subject_id=node_id) | Q(object_id=node_id)).values_list("subject_id", "subject_name", "object_name"))
    target = normalize(name)
    return any(normalize(sn if sid == node_id else on) == target for sid, sn, on in others)


def _allowed(schema, from_label: str, rel: str, to_label: str) -> bool:
    spec = schema.relationship_types.get(rel)
    return bool(spec) and (from_label, to_label) in spec.pairs


def lab_id(doc_key: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "entity"
    return f"lab:{doc_key}:{slug}"


def _commit_lab(tx, t: CandidateTriple) -> None:
    doc_key = t.document.doc_key
    for name, kind in ((t.subject_name, t.subject_type), (t.object_name, t.object_type)):
        tx.run("MERGE (n:Lab {id: $id}) ON CREATE SET n.created_at = datetime() "
               "SET n.name = $name, n.type = $type, n.doc_key = $doc_key",
               id=lab_id(doc_key, name), name=name, type=kind, doc_key=doc_key)
    tx.run("MATCH (a:Lab {id: $s}), (b:Lab {id: $o}) "
           "MERGE (a)-[r:LAB_RELATION {triple_id: $tid}]->(b) SET r += $props",
           s=lab_id(doc_key, t.subject_name), o=lab_id(doc_key, t.object_name), tid=t.pk,
           props={**_provenance(t), "predicate": t.predicate, "evidence_ids": _evidence_ids(t)})


def approve(triples: list[CandidateTriple]) -> dict:
    """Commit pending/rejected triples. Returns {approved: [ids], errors: {id: message}, layers: {curated, lab}}."""
    schema = registry.active_schema()
    approved, errors, layers = [], {}, {"curated": 0, "lab": 0}
    with session() as s:
        for t in triples:
            if t.status == CandidateTriple.Status.APPROVED:
                continue
            asset_scope = (t.job.params.get("asset_id") if t.job else "") or None
            try:
                if t.mode == "schema":
                    s.execute_write(_commit_curated, t, schema, asset_scope)
                    t.layer = CandidateTriple.Layer.CURATED
                else:
                    s.execute_write(_commit_lab, t)
                    t.layer = CandidateTriple.Layer.LAB
            except (TripleError, SchemaError, repository.NodeNotFound) as exc:
                errors[t.pk] = str(exc)[:255]
                continue
            t.status, t.committed_at = CandidateTriple.Status.APPROVED, timezone.now()
            t.save(update_fields=["status", "layer", "committed_at", "applied_props", "applied_aliases", "updated_at"])
            approved.append(t.pk)
            layers[t.layer] += 1
    return {"approved": approved, "errors": errors, "layers": layers}


def reject(triples: list[CandidateTriple]) -> int:
    ids = [t.pk for t in triples if t.status == CandidateTriple.Status.PENDING]
    return CandidateTriple.objects.filter(pk__in=ids).update(status=CandidateTriple.Status.REJECTED, updated_at=timezone.now())


def _uncommit(tx, t: CandidateTriple) -> None:
    """Remove what an approved triple wrote; clean up what it leaves orphaned."""
    if t.layer == CandidateTriple.Layer.LAB:
        tx.run("MATCH ()-[r:LAB_RELATION {triple_id: $tid}]->() DELETE r", tid=t.pk)
        tx.run("MATCH (n:Lab) WHERE n.id IN $ids AND NOT (n)--() DELETE n",
               ids=[lab_id(t.document.doc_key, t.subject_name), lab_id(t.document.doc_key, t.object_name)])
        return
    ev = _evidence_ids(t)
    for node_id in {t.subject_id, t.object_id}:
        tx.run("MATCH (n:Entity {id: $id}) SET n.evidence_ids = [x IN coalesce(n.evidence_ids, []) WHERE NOT x IN $ev]",
               id=node_id, ev=ev)
    for node_id, names in (t.applied_aliases or {}).items():
        keep = [n for n in names if _alias_still_used(t, node_id, n)]
        drop = [n for n in names if n not in keep]
        if drop:
            tx.run("MATCH (n:Entity {id: $id}) SET n.aliases = [a IN coalesce(n.aliases, []) WHERE NOT a IN $drop]",
                   id=node_id, drop=drop)
    # Drop this triple's support; delete text-created edges nothing supports any more.
    tx.run(
        "MATCH (:Entity)-[r]->(:Entity) WHERE $tid IN coalesce(r.triple_ids, []) "
        "SET r.triple_ids = [x IN r.triple_ids WHERE x <> $tid], "
        "    r.evidence_ids = [x IN coalesce(r.evidence_ids, []) WHERE NOT x IN $ev] "
        "WITH r WHERE size(r.triple_ids) = 0 AND r.source IN $sources DELETE r",
        tid=t.pk, sources=STAGED_SOURCES, ev=ev,
    )
    # Properties this triple added to entities that already existed.
    for node_id, names in (t.applied_props or {}).items():
        tx.run("MATCH (n:Entity {id: $id}) SET n += $unset", id=node_id, unset={name: None for name in names})
    # Text-created entities with no remaining (non-evidence) relationships.
    tx.run(
        "MATCH (n:Entity) WHERE n.id IN $ids AND n.source IN $sources "
        "AND NOT EXISTS { MATCH (n)-[r]-() WHERE NOT type(r) IN $evidence } DETACH DELETE n",
        ids=[t.subject_id, t.object_id], evidence=list(EVIDENCE_RELS), sources=STAGED_SOURCES,
    )
    if t.document is None:
        return
    # Sections that no longer describe anything, then documents with no sections left.
    tx.run(
        "MATCH (s:DocumentSection {doc_key: $doc, source: 'library'}) "
        "WHERE NOT (s)-[:DESCRIBES]->() DETACH DELETE s",
        doc=t.document.doc_key,
    )
    tx.run(
        "MATCH (d:Document {doc_key: $doc, source: 'library'}) "
        "WHERE NOT (d)-[:HAS_SECTION]->() DETACH DELETE d",
        doc=t.document.doc_key,
    )


def delete(triples: list[CandidateTriple]) -> int:
    committed = [t for t in triples if t.status == CandidateTriple.Status.APPROVED]
    if committed:
        with session() as s:
            for t in committed:
                s.execute_write(_uncommit, t)
    count = len(triples)
    evidence_ids = list(CandidateTriple.evidence.through.objects.filter(candidatetriple_id__in=[t.pk for t in triples])
                        .values_list("evidence_id", flat=True))
    CandidateTriple.objects.filter(pk__in=[t.pk for t in triples]).delete()
    evidence.drop_orphans(pk__in=evidence_ids)
    return count


def promote(triple: CandidateTriple, mapping: dict) -> dict:
    """Map a free-form triple onto the schema and commit it to the curated layer.

    `mapping` gives schema types/predicate (and optionally ids/names); missing ids
    are proposed, or matched to existing entities by the caller beforehand.
    """
    if triple.mode != "freeform":
        raise TripleError("only free-form (lab) triples can be promoted")
    candidate = CandidateTriple(**{f.attname: getattr(triple, f.attname) for f in triple._meta.concrete_fields})
    for key in ("subject_type", "predicate", "object_type", "subject_id", "object_id", "subject_name", "object_name"):
        if key in mapping:
            setattr(candidate, key, str(mapping[key] or "").strip())
    # Free-form triples have no ids yet: match existing entities or propose ids.
    _reresolve(candidate, {k: v for k, v in mapping.items() if k in ("subject_id", "object_id")} | {
        "subject_type": candidate.subject_type, "object_type": candidate.object_type})
    issue = _schema_issue(candidate)
    if issue:
        raise TripleError(issue)

    if triple.status == CandidateTriple.Status.APPROVED:
        with session() as s:
            s.execute_write(_uncommit, triple)       # leave the lab layer
    candidate.mode, candidate.status, candidate.layer, candidate.issue, candidate.edited = (
        "schema", CandidateTriple.Status.PENDING, "", "", True)
    candidate.committed_at = None
    candidate.save()
    candidate.evidence.set(triple.evidence.all())
    result = approve([candidate])
    if result["errors"]:
        CandidateTriple.objects.filter(pk=candidate.pk).update(issue=result["errors"][candidate.pk])
        raise TripleError(result["errors"][candidate.pk])
    return result
