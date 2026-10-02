"""Delete every curated Context Graph node (dev only).

    python manage.py graph_reset --yes
"""

from django.core.management.base import BaseCommand, CommandError

from context_graph import services


class Command(BaseCommand):
    help = "Delete all curated (:Entity) nodes and their relationships."

    def add_arguments(self, parser):
        parser.add_argument("--yes", action="store_true", help="Confirm deletion.")

    def handle(self, *args, yes=False, **options):
        if not yes:
            raise CommandError("This deletes the whole curated graph; re-run with --yes to confirm.")
        self.stdout.write(f"Deleted {services.reset_graph()} node(s).")
