"""Suggest which parameter of a device shows each PLC signal of its assembly (P22c-3c); staged for review.

    manage.py suggest_maps --part bsk:component:EXTR01/main-drive-vfd [--dry-run]
"""

from django.core.management.base import BaseCommand, CommandError

from context_graph import maps


class Command(BaseCommand):
    help = "Look up the readout parameter behind each PLC signal of a device's assembly and stage MAPS_TO for review."

    def add_arguments(self, parser):
        parser.add_argument("--part", required=True, help="the device (a component id with its manuals)")
        parser.add_argument("--dry-run", action="store_true", help="look up and report; stage nothing")

    def handle(self, *args, part="", dry_run=False, **options):
        try:
            results = maps.suggest(part, dry_run=dry_run)
        except ValueError as exc:
            raise CommandError(str(exc))
        for r in results:
            if r["state"] == "found":
                self.stdout.write(f"{r['signal']} ({r['unit']}) = {r['code']} {r['name']}  p.{r['page']}  "
                                  f"{'(dry run)' if dry_run else 'staged ' + str(r.get('staged'))}  “{r.get('quote', '')}”")
            else:
                self.stdout.write(f"{r['signal']} ({r['unit']}): {r['state']} — {r.get('reason', '')}")
