"""manage.py spc_evaluate: one pass of the SPC evaluator (run once a minute by neo-spc.timer).

    --tool <id>     only this tool
    --since <ISO>   judge again from this time (with --tool: its events from then on are replaced)
    --dry-run       say what would be written, write nothing

Ends with exit code 1 when the historian could not be read for any tool.
"""

from datetime import datetime, timezone

from django.core.management.base import BaseCommand, CommandError

from spc import evaluate


class Command(BaseCommand):
    help = "Judge the rows that arrived since the last pass with every enabled SPC tool, and write SPC events."

    def add_arguments(self, parser):
        parser.add_argument("--tool", type=str, default=None, help="Only this tool (its id).")
        parser.add_argument("--since", type=str, default=None, help="Judge again from this time (ISO, UTC).")
        parser.add_argument("--dry-run", action="store_true", help="Write nothing.")

    def handle(self, *args, **options):
        since = None
        if options["since"]:
            try:
                since = datetime.fromisoformat(options["since"].replace("Z", "+00:00"))
            except ValueError:
                raise CommandError("--since must be an ISO date and time, e.g. 2026-10-12T00:00:00Z")
            since = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        results = evaluate.evaluate(options["tool"], since, options["dry_run"])
        out = self.stdout.write
        for r in results:
            if r.get("error"):
                out(f"{r['tool']}: NOT judged ({r['error']})")
            elif r["skipped"]:
                out(f"{r['tool'] or 'SPC'}: {r['skipped']}")
            else:
                out(f"{r['tool']}: {r['judged']} samples judged, {r['created']} event(s) started, {r['extended']} continued"
                    + (" (dry run, nothing written)" if options["dry_run"] else ""))
                for e in r.get("events", []) if options["dry_run"] else []:
                    out(f"    {e['started_at']:%H:%M:%S} {e['type']} {e['rule']} {e['group']} value {e['value']:g} × {e['count']}")
        if not results:
            out("No SPC tool.")
        failed = [r for r in results if r.get("error")]
        if failed and len(failed) == len(results):
            raise SystemExit(1)
