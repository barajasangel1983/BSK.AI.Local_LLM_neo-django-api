"""Load a seed file into the Context Graph (validated, idempotent MERGE).

    python manage.py graph_seed context_graph/seeds/extr01.yaml [--dry-run]
"""

from django.core.management.base import BaseCommand, CommandError

from context_graph import services
from context_graph.schema import SchemaError


class Command(BaseCommand):
    help = "Validate and MERGE a YAML seed file into the Context Graph."

    def add_arguments(self, parser):
        parser.add_argument("path")
        parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing.")

    def handle(self, *args, path, dry_run=False, **options):
        try:
            result = services.load_seed(path, dry_run=dry_run)
        except SchemaError as exc:
            raise CommandError(f"Seed rejected by schema: {exc}")
        verb = "Validated" if dry_run else "Loaded"
        self.stdout.write(f"{verb} {result['name']!r}: {result['nodes']} nodes, {result['relationships']} "
                          f"relationships (schema v{result['schema_version']}).")
