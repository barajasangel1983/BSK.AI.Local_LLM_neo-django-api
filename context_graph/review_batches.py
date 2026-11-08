"""P27c (2026-11-07): apply the automated review, and undo it.

`plan()` reads a `review_replay` report (and optionally a verdict sheet) and decides, without writing, what happens to
each row of one document; `apply()` carries the plan out as one `ReviewBatch`; `undo()` takes a batch back, graph
included. No judging here: the report's tiers are reused.

    report tier   auto -> approve, reject -> reject, reviewer -> doubtful (pending), leave -> leave (pending)
    sheet verdict approve | reject | leave | ask (doubtful) | fix (an edit, then approve), wins over the tier

A row changed since the report (no longer pending, edited, another fact or quote, deleted) is skipped and counted:
a stale report is never applied (D3). Hazards are approved before their measures; a measure whose hazard didn't
commit stays pending ("leave").
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from django.utils import timezone

from . import review_assist as ra, triples as triple_ops
from .driver import session
from .models import CandidateTriple, ReviewBatch

TOOL_VERSION = "p27-v1"
AUTHOR = f"auto:{TOOL_VERSION}"
TIER_OF = {"auto": "approve", "reject": "reject", "reviewer": "doubtful", "leave": "leave"}
SHEET_OF = {"approve": "approve", "reject": "reject", "leave": "leave", "ask": "doubtful", "fix": "fix"}
# What an edit of the batch changes on a triple, kept before and after so an undo can put it back.
SNAPSHOT = ("subject_name", "subject_type", "subject_id", "subject_existing", "subject_match", "subject_candidates",
            "object_name", "object_type", "object_id", "object_existing", "object_match", "object_candidates",
            "issue", "facets", "edited")
T = CandidateTriple


class ReviewError(ValueError):
    pass


@dataclass
class Step:
    triple: CandidateTriple
    action: str              # approve | reject | leave | doubtful | fix
    reason: str
    subject: str = ""        # fix: the hazard a measure goes under (by name)
    hazard: str = ""         # fix: the new name of a hazard and its measures
    rename_to: str = ""      # the report's kept hazard (normalized name) for a measure of a repeated hazard


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    skipped: dict[int, str] = field(default_factory=dict)    # {triple id: why it changed since the report}

    def counts(self) -> dict:
        c = Counter(s.action for s in self.steps)
        return {"approve": c["approve"], "fix": c["fix"], "reject": c["reject"], "leave": c["leave"],
                "doubtful": c["doubtful"], "skipped": len(self.skipped)}


def _fact(t: CandidateTriple) -> str:
    return f"{t.subject_type}:{t.subject_name} -{t.predicate}-> {t.object_type}:{t.object_name}"


def _matches_doc(t: CandidateTriple, doc: str) -> bool:
    d = t.document
    return bool(d) and (str(d.id) == doc or (d.title or "").startswith(doc) or (d.filename or "").startswith(doc))


def plan(report: dict, sheet: dict | None = None, doc: str = "") -> Plan:
    """What applying `report` (+ `sheet`, {id: {verdict, reason, subject?, hazard?}}) would do to one document's rows
    (`doc`: its id, or the start of its title or file name; every document when empty). Writes nothing."""
    sheet = {int(k): v for k, v in (sheet or {}).items()}
    rows = {r["id"]: r for r in report["rows"]}
    # A sheet verdict on a row outside the report (an override of a fact the tool decided) needs the report's row.
    found = {t.id: t for t in T.objects.select_related("document", "job").prefetch_related("evidence")
             .filter(id__in=list(rows))}
    out = Plan()
    for i, r in rows.items():
        t = found.get(i)
        if t is None:
            out.skipped[i] = "no longer staged"
            continue
        if doc and not _matches_doc(t, doc):
            continue
        why = _changed(t, r)
        if why:
            out.skipped[i] = why
            continue
        v = sheet.get(i)
        if v:
            action = SHEET_OF.get(v["verdict"])
            if not action:
                raise ReviewError(f"{i}: unknown sheet verdict {v['verdict']!r}")
            out.steps.append(Step(t, action, f"sheet: {v.get('reason', '')}", subject=v.get("subject") or "",
                                  hazard=v.get("hazard") or "", rename_to=r.get("rename_subject_to") or ""))
        else:
            action = TIER_OF.get(r["tier"])
            if not action:
                raise ReviewError(f"{i}: unknown tier {r['tier']!r}")
            out.steps.append(Step(t, action, r.get("reason") or "", rename_to=r.get("rename_subject_to") or ""))
    return out


def _changed(t: CandidateTriple, r: dict) -> str:
    if t.status != T.Status.PENDING:
        return f"now {t.status}"
    if t.edited and not r.get("edited"):
        return "edited since the report"
    if _fact(t) != r["fact"]:
        return "another fact than in the report"
    if r.get("quote") is not None and ra.source_of(t)[0][:300] != r["quote"]:
        return "another quote than in the report"
    return ""


# --- apply -----------------------------------------------------------------------------------------------------------

def _is_measure(t: CandidateTriple) -> bool:
    return t.predicate == "PROTECTED_BY" and t.subject_type == "Hazard"


def _snap(t: CandidateTriple) -> dict:
    return {k: getattr(t, k) for k in SNAPSHOT}


def _hazard_of(t: CandidateTriple, name: str = "") -> CandidateTriple | None:
    """The approved HAS_HAZARD triple a measure hangs on: by id, or by `name` (default: the measure's subject name)
    on the same document and anchor."""
    base = T.objects.filter(status=T.Status.APPROVED, predicate="HAS_HAZARD", document_id=t.document_id,
                            anchor_id=t.anchor_id)
    if not name:
        by_id = base.filter(object_id=t.subject_id).first()
        if by_id:
            return by_id
        name = t.subject_name
    target = ra._norm(name)
    return next((h for h in base.order_by("id") if ra._norm(h.object_name) == target), None)


def apply(p: Plan, scope: dict, started_by: str = "") -> ReviewBatch:
    """Carry out a plan as one batch: hazards and other facts first, then measures, then rejects. Each triple gets its
    tier, reason and batch; a fact that can't be approved stays pending as doubtful, with the error."""
    batch = ReviewBatch.objects.create(tool_version=TOOL_VERSION, scope=scope, author=AUTHOR, started_by=started_by)
    before, after, result = {}, {}, Counter()
    renamed = {}   # (document, anchor, normalized old hazard name) -> new name, for the measures that follow

    def mark(t: CandidateTriple, tier: str, reason: str) -> None:
        T.objects.filter(pk=t.pk).update(review_tier=tier, review_reason=reason[:255], review_batch=batch)
        result[tier] += 1

    def edit(t: CandidateTriple, changes: dict) -> bool:
        snap = _snap(t)
        try:
            triple_ops.edit(t, changes)
        except triple_ops.TripleError as exc:
            mark(t, T.ReviewTier.DOUBTFUL, f"edit failed: {exc}")
            result["failed"] += 1
            return False
        before.setdefault(str(t.pk), snap)
        after[str(t.pk)] = _snap(t)
        return True

    def approve(t: CandidateTriple, reason: str) -> bool:
        res = triple_ops.approve([t], automatic=True)
        if t.pk in res["approved"]:
            mark(t, T.ReviewTier.AUTO, reason)
            return True
        mark(t, T.ReviewTier.DOUBTFUL, f"approval failed: {res['errors'].get(t.pk, '?')}")
        result["failed"] += 1
        return False

    to_approve = [s for s in p.steps if s.action in ("approve", "fix")]
    # 1. Hazards and every fact that isn't a measure; a hazard renamed by the sheet first takes its new name.
    for s in sorted((s for s in to_approve if not _is_measure(s.triple)), key=lambda s: s.triple.pk):
        t = s.triple
        if s.action == "fix":
            if t.predicate != "HAS_HAZARD" or not s.hazard:
                mark(t, T.ReviewTier.DOUBTFUL, f"sheet fix not understood: {s.reason}")
                continue
            old = (t.document_id, t.anchor_id, ra._norm(t.object_name))
            if not edit(t, {"object_name": s.hazard}):
                continue
            renamed[old] = s.hazard
        approve(t, s.reason)
    # 2. Measures: under the sheet's hazard, the renamed hazard, or the kept one of a repeat; only if it committed.
    for s in sorted((s for s in to_approve if _is_measure(s.triple)), key=lambda s: s.triple.pk):
        t = s.triple
        if s.action == "fix" and not (s.subject or s.hazard):
            mark(t, T.ReviewTier.DOUBTFUL, f"sheet fix not understood: {s.reason}")
            continue
        target = (s.subject or s.hazard or renamed.get((t.document_id, t.anchor_id, ra._norm(t.subject_name)))
                  or s.rename_to or "")
        hazard = _hazard_of(t, target)
        if hazard is None:
            mark(t, T.ReviewTier.LEAVE, f"its hazard didn't commit ({target or t.subject_name})"[:255])
            continue
        if hazard.object_id != t.subject_id and not edit(
                t, {"subject_name": hazard.object_name, "subject_id": hazard.object_id}):
            continue
        approve(t, s.reason)
    # 3. Rejects; 4. the rest stays pending with its tier and reason.
    rejects = [s for s in p.steps if s.action == "reject"]
    triple_ops.reject([s.triple for s in rejects])
    for s in rejects:
        mark(s.triple, T.ReviewTier.REJECT, s.reason)
    for s in p.steps:
        if s.action in ("leave", "doubtful"):
            mark(s.triple, s.action, s.reason)
    batch.before, batch.after = before, after
    batch.summary = {"plan": p.counts(), "approved": result[T.ReviewTier.AUTO], "rejected": result[T.ReviewTier.REJECT],
                     "left": result[T.ReviewTier.LEAVE], "doubtful": result[T.ReviewTier.DOUBTFUL],
                     "failed": result["failed"], "edited": len(after), "skipped": p.skipped}
    batch.save(update_fields=["before", "after", "summary"])
    return batch


# --- undo ------------------------------------------------------------------------------------------------------------

def undo(batch: ReviewBatch) -> dict:
    """Take a batch back: its approved triples leave the graph and are pending again, its rejected ones are pending
    again, its edits are put back; tier, reason and batch are cleared. A triple a person decided or edited since is
    left as it is and listed."""
    if batch.status != ReviewBatch.Status.APPLIED:
        raise ReviewError(f"batch {batch.pk} is {batch.status}")
    mine = list(T.objects.select_related("document", "job").filter(review_batch=batch))
    kept, edit_kept, reverted, cleared = [], [], [], []
    committed = sorted((t for t in mine if t.review_tier == T.ReviewTier.AUTO and t.status == T.Status.APPROVED),
                       key=lambda t: t.committed_at or timezone.now(), reverse=True)   # measures before their hazards
    if committed:
        with session() as s:
            for t in committed:
                if t.mode == "schema":
                    triple_ops._follow_moves(s, t)
                s.execute_write(triple_ops._uncommit, t)
                # pending at once: the last of a manual's facts on a family takes the "documented by" link with it
                T.objects.filter(pk=t.pk).update(status=T.Status.PENDING)
    for t in mine:
        own = ((t.review_tier == T.ReviewTier.AUTO and t.status == T.Status.APPROVED)
               or (t.review_tier == T.ReviewTier.REJECT and t.status == T.Status.REJECTED)
               or (t.review_tier in (T.ReviewTier.LEAVE, T.ReviewTier.DOUBTFUL) and t.status == T.Status.PENDING))
        if not own:
            kept.append(t.pk)        # a person decided it since; their decision stays
            continue
        fields = {"status": T.Status.PENDING, "review_tier": "", "review_reason": "", "review_batch": None,
                  "applied_props": {}, "applied_aliases": {}, "replaced_props": {}, "layer": "", "committed_at": None}
        snap_after, snap_before = batch.after.get(str(t.pk)), batch.before.get(str(t.pk))
        if snap_before:
            if snap_after and _snap(t) != snap_after:
                edit_kept.append(t.pk)    # edited again since: the person's edit stays, only the status goes back
            else:
                fields.update(snap_before)
        if t.status == T.Status.PENDING:
            # never committed: only the tier (and an edit of the batch) goes
            fields = {k: v for k, v in fields.items() if k not in ("applied_props", "applied_aliases", "replaced_props",
                                                                   "layer", "committed_at")}
        T.objects.filter(pk=t.pk).update(updated_at=timezone.now(), **fields)
        (reverted if t.status != T.Status.PENDING else cleared).append(t.pk)
    batch.status, batch.undone_at = ReviewBatch.Status.UNDONE, timezone.now()
    batch.summary = {**batch.summary, "undo": {"reverted": len(reverted), "cleared": len(cleared), "kept": kept,
                                               "edit_kept": edit_kept}}
    batch.save(update_fields=["status", "undone_at", "summary"])
    return {"reverted": reverted, "cleared": cleared, "kept": kept, "edit_kept": edit_kept, "uncommitted": len(committed)}


def batch_json(b: ReviewBatch) -> dict:
    return {"id": b.pk, "status": b.status, "tool_version": b.tool_version, "scope": b.scope,
            "summary": {k: v for k, v in b.summary.items() if k != "skipped"},
            "skipped": len(b.summary.get("skipped") or {}), "author": b.author, "started_by": b.started_by,
            "created_at": b.created_at.isoformat(), "undone_at": b.undone_at.isoformat() if b.undone_at else None}
