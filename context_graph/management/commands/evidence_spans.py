"""Spans for evidence and chunks written before P25b (P25c, 2026-10-25). Dry run unless --apply.

    manage.py evidence_spans [--doc DOC_KEY] [--chroma] [--apply] [--out report.json]
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from context_graph import evidence_spans
from ingestion.models import Document


class Command(BaseCommand):
    help = "Find source spans for existing evidence (and with --chroma, chunks); a dry run unless --apply."

    def add_arguments(self, parser):
        parser.add_argument("--doc", help="one document (doc key)")
        parser.add_argument("--chroma", action="store_true", help="the Chroma chunks instead of the evidence")
        parser.add_argument("--apply", action="store_true", help="write (default: report only)")
        parser.add_argument("--out", help="write the full report as JSON")

    def handle(self, *args, **opts):
        docs = Document.objects.filter(doc_key=opts["doc"]) if opts["doc"] else Document.objects.order_by("doc_key")
        verb = "written" if opts["apply"] else "dry run"
        if opts["chroma"]:
            result = evidence_spans.backfill_chunks(docs, apply=opts["apply"])
            for key, counts in result.items():
                if counts:
                    self.stdout.write(f"{key}: {dict(counts)}")
            total = {}
            for counts in result.values():
                for k, v in counts.items():
                    total[k] = total.get(k, 0) + v
            self.stdout.write(f"chunks ({verb}): {total}")
            report = {"chunks": {k: dict(v) for k, v in result.items()}}
        else:
            reports = evidence_spans.backfill(docs, apply=opts["apply"])
            for key, counts in evidence_spans.by_document(reports).items():
                self.stdout.write(f"{key}: {counts}")
            failures = sum(r.round_trip_failures for r in reports)
            self.stdout.write(f"evidence ({verb}): {dict(evidence_spans.summary(reports))}; round-trip failures {failures}")
            report = {"by_document": evidence_spans.by_document(reports), "not_exact": evidence_spans.changed_list(reports),
                      "round_trip_failures": failures}
        if opts["out"]:
            with open(opts["out"], "w") as f:
                json.dump(report, f, indent=1)
            self.stdout.write(f"report: {opts['out']}")
