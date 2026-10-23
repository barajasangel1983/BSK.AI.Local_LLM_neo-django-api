"""Move today's manual facts off the installed parts to their family or model (P26c).

    manage.py move_facts                              every family: what would move, merge and conflict
    manage.py move_facts --target <id> [--json f]     one family's plan (nothing written)
    manage.py move_facts --target <id> --apply [--choices f.json] [--author name]
    manage.py move_facts --undo <batch id>

`--choices` is a JSON file {"choices": {group id: models | both | member id}, "stays": [old ids]}. Apply and undo
refuse while a library job runs (a job reading the graph mid-move would see half of it)."""

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from context_graph import fact_moves
from context_graph.driver import session
from context_graph.models import FactMoveBatch


class Command(BaseCommand):
    help = "Move manual facts from installed parts to their product family or device model (dry run by default)."

    def add_arguments(self, parser):
        parser.add_argument("--target", help="a product family id (or a device model without one)")
        parser.add_argument("--json", help="write the plan to this file")
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--choices", help="JSON file: {choices: {group: models|both|member id}, stays: [ids]}")
        parser.add_argument("--author", default="")
        parser.add_argument("--undo", type=int, help="a batch id to undo")

    def handle(self, *args, **opts):
        from ingestion.backup import active_jobs

        if (opts["apply"] or opts["undo"]) and active_jobs():
            raise CommandError("a library job is running: try again when it is done")
        if opts["undo"]:
            batch = FactMoveBatch.objects.filter(pk=opts["undo"]).first()
            if batch is None:
                raise CommandError(f"no batch {opts['undo']}")
            try:
                out = fact_moves.undo(batch)
            except fact_moves.MoveError as exc:
                raise CommandError(str(exc))
            self.stdout.write(self.style.SUCCESS(f"batch {batch.pk} undone: {out['nodes']} nodes, {out['links']} links back"))
            return
        with session() as s:
            g = s.execute_read(fact_moves.load_graph)
        if not opts["target"]:
            for t in fact_moves.targets(g):
                sm = fact_moves.plan(g, t["id"])["summary"]
                self.stdout.write(f"{t['name'][:44]:44} {t['id']}\n    parts {len(t['parts'])}, models {len(t['models'])}: "
                                  f"moved {sm['moved']}, merged away {sm['merged_away']}, conflicts {sm['conflicts']}, "
                                  f"shared (stay) {sm['shared']}")
            return
        choices = json.loads(Path(opts["choices"]).read_text()) if opts["choices"] else {}
        try:
            if opts["apply"]:
                batch = fact_moves.apply(opts["target"], choices.get("choices"), choices.get("stays"), author=opts["author"])
                self.stdout.write(self.style.SUCCESS(f"batch {batch.pk} applied: {json.dumps(batch.summary)}"))
                return
            p = fact_moves.plan(g, opts["target"], choices.get("choices"), choices.get("stays"))
        except fact_moves.MoveError as exc:
            raise CommandError(str(exc))
        if opts["json"]:
            Path(opts["json"]).write_text(json.dumps(p, indent=1, default=str))
        self.stdout.write(json.dumps({"summary": p["summary"], "counts": p["counts"]}, indent=1))
