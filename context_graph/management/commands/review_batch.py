"""P27c: the automated review's batches.

    manage.py review_batch --list
    manage.py review_batch --show <id>
    manage.py review_batch --undo <id>     # approved ones leave the graph, every decision back to pending
"""

import json

from django.core.management.base import BaseCommand, CommandError

from context_graph import review_batches as rb
from context_graph.models import ReviewBatch


class Command(BaseCommand):
    help = "List, show or undo the automated review's batches (P27c)"

    def add_arguments(self, parser):
        what = parser.add_mutually_exclusive_group(required=True)
        what.add_argument("--list", action="store_true")
        what.add_argument("--show", type=int)
        what.add_argument("--undo", type=int)

    def handle(self, *args, **o):
        if o["list"]:
            for b in ReviewBatch.objects.all():
                j = rb.batch_json(b)
                self.stdout.write(f"{b.pk:4} {b.status:8} {b.created_at:%Y-%m-%d %H:%M} {j['scope'].get('document', '')[:40]:40} "
                                  f"approved {j['summary'].get('approved', 0)} rejected {j['summary'].get('rejected', 0)} "
                                  f"left {j['summary'].get('left', 0)} doubtful {j['summary'].get('doubtful', 0)}")
            return
        try:
            b = ReviewBatch.objects.get(pk=o["show"] or o["undo"])
        except ReviewBatch.DoesNotExist:
            raise CommandError("no such batch")
        if o["show"]:
            self.stdout.write(json.dumps({**rb.batch_json(b), "skipped": b.summary.get("skipped")}, indent=1, default=str))
            return
        try:
            result = rb.undo(b)
        except rb.ReviewError as exc:
            raise CommandError(str(exc))
        self.stdout.write(self.style.SUCCESS(
            f"batch {b.pk} undone: {len(result['reverted'])} back to pending ({result['uncommitted']} out of the graph), "
            f"{len(result['cleared'])} pending ones cleared"))
        if result["kept"]:
            self.stdout.write(f"decided by a person since, left as they are: {result['kept']}")
        if result["edit_kept"]:
            self.stdout.write(f"edited by a person since, their edit kept: {result['edit_kept']}")
