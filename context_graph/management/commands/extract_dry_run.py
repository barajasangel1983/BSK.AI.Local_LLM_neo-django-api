"""P26b-2: run one facet on the real extraction path without staging anything, and score it against the review.

    manage.py extract_dry_run --doc <doc_key> --anchor <family / model id> --facet controls [--pages 88-100]
        [--prompt-file new.txt] [--old-code] [--runs 2] [--out result.json]
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from context_graph import dry_run, extraction
from ingestion.models import Document


class Command(BaseCommand):
    help = "A facet run that stages nothing, scored against the document's review"

    def add_arguments(self, parser):
        parser.add_argument("--doc", required=True, help="the document's doc_key")
        parser.add_argument("--anchor", required=True, help="the family, model or installed part it describes")
        parser.add_argument("--asset", default="", help="the asset, for an installed part")
        parser.add_argument("--facet", required=True, choices=[r["key"] for r in extraction.FACET_RUNS])
        parser.add_argument("--pages", default="", help="pages to read (default: the facet's own page choice)")
        parser.add_argument("--prompt-file", default="", help="a candidate prompt text (default: the live version)")
        parser.add_argument("--old-code", action="store_true", help="values joined, ids as before P26b-2")
        parser.add_argument("--runs", type=int, default=2)
        parser.add_argument("--out", default="", help="write the runs and scores as JSON")

    def handle(self, *args, **o):
        doc = Document.objects.filter(doc_key=o["doc"]).first()
        if doc is None:
            raise CommandError(f"no document {o['doc']!r}")
        template = Path(o["prompt_file"]).read_text() if o["prompt_file"] else None
        preset = dry_run.preset_for(o["facet"], template)
        review = dry_run.labels(doc, preset.name)
        runs, scores = [], []
        for n in range(1, o["runs"] + 1):
            result = dry_run.run(doc, facet=o["facet"], anchor_id=o["anchor"], asset_id=o["asset"], pages=o["pages"],
                                 template=template, old_code=o["old_code"])
            s = dry_run.score(result, review)
            runs.append(result)
            scores.append(s)
            self.stdout.write(f"run {n}: pages {result['pages']} · {s['triples']} triples · approved "
                              f"{s['approved_found']}/{s['approved_total']} · rejected back {s['rejected_back']}/"
                              f"{s['rejected_total']} · new {s['new']} · IOPoints {s['iopoints']} ({s['iopoint_ids']} ids)"
                              f" · bits {len(s['bits'])} · strips {len(s['strips'])} · generic {len(s['generic'])}"
                              f" · contacts {len(s['contacts'])} · joined {len(s['joined_values'])}"
                              f" · disagreed {s['disagreed']} · strips dropped {len(s['strips_dropped'])}"
                              f" · failed calls {s['failed_calls']}")
        if o["out"]:
            Path(o["out"]).write_text(json.dumps({"args": {k: v for k, v in o.items() if k in (
                "doc", "anchor", "facet", "pages", "prompt_file", "old_code", "runs")},
                "runs": runs, "scores": scores, "summary": dry_run.summary(scores)}, indent=1, default=str))
            self.stdout.write(f"written: {o['out']}")
