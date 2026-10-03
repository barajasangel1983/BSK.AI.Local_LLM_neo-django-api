"""Evidence records (P8a): where each staged / approved fact was found."""

from __future__ import annotations

from .models import CandidateTriple, Evidence

EXCERPT_CHARS = 1500


def evidence_json(e: Evidence) -> dict:
    return {
        "id": e.pk, "source_kind": e.source_kind,
        "document_id": str(e.document_id) if e.document_id else None,
        "data_file_id": str(e.data_file_id) if e.data_file_id else None,
        "page_start": e.page_start, "page_end": e.page_end, "section_path": e.section_path,
        "row": e.row_number, "region": e.region or None, "figure_id": e.figure_id or None,
        "chunk_index": e.chunk_index, "extractor": e.extractor, "model": e.model, "prompt": e.prompt,
        "schema_version": e.schema_version, "excerpt": e.excerpt, "confidence": e.confidence,
        "created_at": e.created_at.isoformat(),
    }


def attach(triples: list[CandidateTriple], pending: dict[int, list[Evidence]]) -> int:
    """Save the evidence collected for freshly bulk-created triples and link it.

    `pending` maps id(triple object) -> unsaved Evidence rows (bulk_create sets the pks).
    """
    rows, owners = [], []
    for t in triples:
        for e in pending.get(id(t), []):
            rows.append(e)
            owners.append(t.pk)
    Evidence.objects.bulk_create(rows)
    Link = CandidateTriple.evidence.through
    Link.objects.bulk_create([Link(candidatetriple_id=tid, evidence_id=e.pk) for tid, e in zip(owners, rows)])
    return len(rows)


def drop_orphans(**scope) -> int:
    """Delete evidence no triple refers to any more (after pending triples were replaced or deleted)."""
    deleted, _ = Evidence.objects.filter(triples__isnull=True, **scope).delete()
    return deleted


def ids_for(triple: CandidateTriple) -> list[int]:
    return sorted(triple.evidence.values_list("id", flat=True))
