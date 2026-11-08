"""P27: assisted review, report only. Nothing is approved, rejected or written to the graph.

    manage.py review_replay --safety-pending [--doc <title start>] [--compare sheet.json] --out report.json
    manage.py review_replay --decided 400 [--seed 1] --out report.json     # against Angel's decisions
    manage.py review_replay --ids ids.json [--compare sheet.json] --out report.json
    ... --resume earlier.json      # reuse an earlier report's complete votes (after the DGX went down)

`--compare`: a reference sheet {triple id: {"verdict": approve|fix|reject|leave|unsure}} to score the tiers against.
`--no-judge`: code checks and duplicate groups only (no DGX chat calls).
"""

import json
import random
import re
from collections import Counter

from django.core.management.base import BaseCommand, CommandError

from context_graph import review_assist as ra, triples as triple_ops
from context_graph.models import CandidateTriple as T


class Command(BaseCommand):
    help = "Assisted review (P27): tiers for staged or decided triples, scored; changes nothing"

    def add_arguments(self, parser):
        what = parser.add_mutually_exclusive_group(required=True)
        what.add_argument("--safety-pending", action="store_true")
        what.add_argument("--decided", type=int, default=0, help="a random sample of N approved / rejected triples")
        what.add_argument("--ids", default="", help="a JSON list of triple ids")
        parser.add_argument("--doc", default="", help="only documents whose title starts with this")
        parser.add_argument("--seed", type=int, default=20261027)
        parser.add_argument("--compare", default="")
        parser.add_argument("--no-judge", action="store_true")
        parser.add_argument("--workers", type=int, default=2, help="calls at once (2 is what extraction uses)")
        parser.add_argument("--resume", default="", help="an earlier report: its complete votes are reused")
        parser.add_argument("--out", required=True)

    def handle(self, *args, **o):
        qs = T.objects.select_related("document").prefetch_related("evidence")
        if o["safety_pending"]:
            rows = [t for t in qs.filter(status=T.Status.PENDING) if triple_ops.is_safety(t)]
        elif o["decided"]:
            ids = list(T.objects.filter(status__in=[T.Status.APPROVED, T.Status.REJECTED]).values_list("id", flat=True))
            random.seed(o["seed"])
            rows = list(qs.filter(id__in=random.sample(ids, min(o["decided"], len(ids)))))
        else:
            rows = list(qs.filter(id__in=json.load(open(o["ids"]))))
        if o["doc"]:
            rows = [t for t in rows if t.document and t.document.title.startswith(o["doc"])]
        if not rows:
            raise CommandError("no triples")
        rows.sort(key=lambda t: (str(t.document_id or ""), t.page_start or 0, t.id))
        self.stdout.write(f"{len(rows)} triples")

        # Facts already approved on the same anchors (pending runs only): a pending repeat of one is a repeat, and a
        # measure's hazard may be one of them. In a replay on decided facts they are the facts being scored.
        approved = []
        if o["safety_pending"]:
            anchors = {t.anchor_id for t in rows if t.anchor_id}
            approved = [t for t in qs.filter(status=T.Status.APPROVED, anchor_id__in=anchors) if triple_ops.is_safety(t)]
        sources = {t.id: ra.source_of(t) for t in rows + approved}
        code = ra.code_checks(rows)
        for i, verdict in ra.notice_checks(rows, hazards=rows + approved).items():
            code.setdefault(i, verdict)
        dups, alias = ({}, {}) if o["decided"] else ra.duplicate_groups(
            rows + approved, {i: s[0] for i, s in sources.items()}, kept_first={t.id for t in approved})
        self.stdout.write(f"code checks decided {len(code)}, repeats {len(dups)}")
        to_judge = [t for t in rows if not (code.get(t.id, ("",))[0] in ("reject", "leave") or t.id in dups)]
        done = {}
        if o["resume"]:
            done = {r["id"]: r["votes"] for r in json.load(open(o["resume"]))["rows"]
                    if r["votes"] and all(v["stated"] != "error" for v in r["votes"])}
        votes = {} if o["no_judge"] else {
            **{i: v for i, v in done.items() if i in {t.id for t in to_judge}},
            **ra.judge_all([t for t in to_judge if t.id not in done], sources, workers=o["workers"])}
        self.stdout.write(f"judged {len(votes)} ({sum(1 for v in votes.values() for x in v if x['stated'] == 'error')} failed calls)")

        tiers = {t.id: ra.tier(code.get(t.id), dups.get(t.id), votes.get(t.id) if not o["no_judge"] else None) for t in rows}
        tiers.update({t.id: ("approved", "already approved") for t in approved})
        # Hazards staged on the same anchors but not in this run: a measure of a rejected one is left.
        known = {t.id for t in rows + approved}
        outside = {}
        for h in T.objects.filter(predicate="HAS_HAZARD", anchor_id__in={t.anchor_id for t in rows if t.anchor_id}):
            if h.id not in known:
                key = (h.document_id, h.anchor_id, ra._norm(h.object_name))
                if outside.get(key) not in ("approved", "pending"):
                    outside[key] = h.status
        ra.follow_hazards(rows + approved, tiers, alias, outside)
        report = []
        for t in rows:
            tier, reason = tiers[t.id]
            renamed = alias.get(ra._norm(t.subject_name)) if t.subject_type == "Hazard" else None
            report.append({
                "id": t.id, "doc": t.document.title if t.document else "", "page": t.page_start,
                "status": t.status, "edited": t.edited, "prompt": t.preset_name,
                "fact": f"{t.subject_type}:{t.subject_name} -{t.predicate}-> {t.object_type}:{t.object_name}",
                "quote": sources[t.id][0][:300], "tier": tier, "reason": reason,
                "rename_subject_to": renamed, "votes": votes.get(t.id),
            })
        result = {"summary": self.summary(report, o["compare"]), "rows": report}
        json.dump(result, open(o["out"], "w"), indent=1, default=str)
        self.stdout.write(json.dumps(result["summary"], indent=1))

    def summary(self, report: list[dict], compare: str) -> dict:
        s = {"triples": len(report), "tiers": dict(Counter(r["tier"] for r in report))}
        decided = [r for r in report if r["status"] != "pending"]
        if decided:
            def angel(r):
                return "rejected" if r["status"] == "rejected" else ("approved_edited" if r["edited"] else "approved")
            s["vs_review"] = _matrix(decided, angel)
            s["vs_review_by_prompt"] = {p: _matrix([r for r in decided if (r["prompt"] or "(none)") == p], angel)
                                        for p in sorted({r["prompt"] or "(none)" for r in decided})}
        if compare:
            raw = json.load(open(compare))
            sheet = {int(k): v["verdict"] for k, v in raw.items()}
            # A repeat pair kept the other way round agrees: the sheet rejected this copy as a repeat of X and the
            # tool rejected X as a repeat of this one (the same fact stays, on the other page).
            tier_of = {r["id"]: r for r in report}
            for k, v in raw.items():
                k = int(k)
                dup_of = [int(x) for x in re.findall(r"\d{3,}", v.get("reason", ""))] if "duplicate" in v.get("reason", "") else []
                if sheet[k] == "reject" and k in tier_of and tier_of[k]["tier"] == "auto" and any(
                        d in tier_of and tier_of[d]["tier"] == "reject" and f"repeat of {k}" in tier_of[d]["reason"] for d in dup_of):
                    sheet[k] = "approve"
                    for d in dup_of:
                        if d in tier_of and f"repeat of {k}" in tier_of[d]["reason"]:
                            sheet[d] = "reject"
            mine = [r for r in report if r["id"] in sheet]
            s["vs_sheet"] = _matrix(mine, lambda r: sheet[r["id"]])
            s["vs_sheet_disagreements"] = [
                {"id": r["id"], "sheet": sheet[r["id"]], "tier": r["tier"], "reason": r["reason"], "fact": r["fact"]}
                for r in mine if (r["tier"] == "auto") != (sheet[r["id"]] == "approve")]
        return s


def _matrix(rows: list[dict], truth) -> dict:
    """Tier × reference counts, and the share of automatic approvals the reference approved as they were."""
    m = Counter((r["tier"], truth(r)) for r in rows)
    auto = [r for r in rows if r["tier"] == "auto"]
    good = sum(1 for r in auto if truth(r) == "approve" or truth(r) == "approved")
    plain = sum(1 for r in rows if truth(r) in ("approve", "approved"))
    return {"matrix": {f"{a} / {b}": n for (a, b), n in sorted(m.items())},
            "auto": len(auto), "auto_precision": round(good / len(auto), 3) if auto else None,
            "auto_recall": round(good / plain, 3) if plain else None,
            "left_for_reviewer": sum(1 for r in rows if r["tier"] == "reviewer")}
