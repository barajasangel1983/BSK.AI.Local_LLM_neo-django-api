"""P27c: apply a `review_replay` report (and Claude's or Angel's verdict sheet) as one batch, one manual per batch.

    manage.py review_apply --report p27b.json [--sheet verdicts.json] --doc <id | title or file name start> [--dry-run]
    manage.py review_apply --report p27b.json [--sheet verdicts.json] --dry-run          # every manual, counts only

No judging: the report's tiers are reused. Rows changed since the report are skipped and counted. Approved Safety facts
are written to the graph; `manage.py review_batch --undo <id>` takes the batch back.
"""

import json
import os
from collections import Counter

from django.core.management.base import BaseCommand, CommandError

from context_graph import review_batches as rb


class Command(BaseCommand):
    help = "Apply the automated review (P27c) to one manual as a batch with an undo"

    def add_arguments(self, parser):
        parser.add_argument("--report", required=True)
        parser.add_argument("--sheet", default="", help="verdicts {id: {verdict, reason, subject?, hazard?}}")
        parser.add_argument("--doc", default="", help="the document's id, or the start of its title or file name")
        parser.add_argument("--facet", default="Safety")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--started-by", default=os.getenv("USER", ""))

    def handle(self, *args, **o):
        report = json.load(open(o["report"]))
        sheet = json.load(open(o["sheet"])) if o["sheet"] else None
        if not o["dry_run"] and not o["doc"]:
            raise CommandError("one manual per batch: give --doc (or --dry-run for every manual)")
        try:
            p = rb.plan(report, sheet, o["doc"])
        except rb.ReviewError as exc:
            raise CommandError(str(exc))
        if not p.steps:
            raise CommandError(f"nothing to apply ({len(p.skipped)} rows changed since the report)")
        docs = {s.triple.document_id for s in p.steps}
        if not o["dry_run"] and len(docs) > 1:
            raise CommandError(f"--doc matches {len(docs)} documents: give its id")
        per_doc = Counter()
        for s in p.steps:
            d = s.triple.document
            per_doc[((d.title or d.filename) if d else "?")[:50], s.action] += 1
        self.stdout.write(json.dumps(p.counts()))
        for title in sorted({k[0] for k in per_doc}):
            self.stdout.write(f"  {title:50} " + " ".join(f"{a} {per_doc[title, a]}" for a in
                                                          ("approve", "fix", "reject", "leave", "doubtful")))
        if p.skipped:
            self.stdout.write("skipped: " + json.dumps(dict(Counter(p.skipped.values()))))
        if o["dry_run"]:
            return
        d = p.steps[0].triple.document
        scope = {"facet": o["facet"], "document_id": str(d.id), "document": d.title or d.filename,
                 "report": os.path.abspath(o["report"]), "sheet": os.path.abspath(o["sheet"]) if o["sheet"] else ""}
        batch = rb.apply(p, scope, started_by=o["started_by"])
        self.stdout.write(self.style.SUCCESS(f"batch {batch.pk}: " + json.dumps(rb.batch_json(batch)["summary"])))
