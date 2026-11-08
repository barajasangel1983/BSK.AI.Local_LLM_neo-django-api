"""The problem-solving triggers and one due knowledge check, on their own (P29).

    python manage.py problems_tick            # what the monitor's minute pass also does
    python manage.py problems_tick --check-only

The live system runs these inside `monitor_tick`; this command is for a preview, a test or a catch-up by hand.
"""

from django.core.management.base import BaseCommand

from problems import llm, triggers


class Command(BaseCommand):
    help = "Run the problem-solving triggers for every asset, then one due knowledge check."

    def add_arguments(self, parser):
        parser.add_argument("--check-only", action="store_true", help="only the knowledge check")

    def handle(self, *args, check_only=False, **options):
        from monitor import agent
        if not check_only:
            for asset in agent.assets():
                try:
                    for ticket in triggers.tick(asset["id"]):
                        self.stdout.write(f"{asset['id']}: {ticket.number} opened ({ticket.trigger}): {ticket.title}")
                except Exception as exc:
                    self.stderr.write(f"{asset['id']}: triggers failed ({type(exc).__name__}: {str(exc)[:200]})")
        done = llm.run_due(limit=1)
        self.stdout.write(f"problems pass done: {done} check(s)")
