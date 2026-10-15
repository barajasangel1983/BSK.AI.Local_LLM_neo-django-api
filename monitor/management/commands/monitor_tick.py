"""One pass of the monitoring agent (P15a): every asset, its new rows, the posts they call for.

    python manage.py monitor_tick          # run by neo-monitor.timer once a minute
    python manage.py monitor_tick --force  # also when the agent is switched off

A one-shot, like spc_evaluate: no process stays in memory, and the timer does not start a pass while one
runs. Problems with one asset are logged and the next pass catches up.
"""

from django.core.management.base import BaseCommand

from monitor import agent
from monitor.models import MonitorSettings


class Command(BaseCommand):
    help = "One pass of the monitoring agent."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="run even when the agent is switched off")

    def handle(self, *args, force=False, **options):
        if not (force or MonitorSettings.get().agent_enabled):
            self.stdout.write("monitoring agent is off")
            return
        total = 0
        for asset in agent.assets():
            try:
                written = agent.tick(asset["id"])
            except Exception as exc:      # the historian, the graph or the model is down: the next pass catches up
                self.stderr.write(f"{asset['id']}: pass failed ({type(exc).__name__}: {str(exc)[:200]})")
                continue
            total += written
            if written:
                self.stdout.write(f"{asset['id']}: {written} post(s)")
        self.stdout.write(f"monitor pass done: {total} post(s)")
