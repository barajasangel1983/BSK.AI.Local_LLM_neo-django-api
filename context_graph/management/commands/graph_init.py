"""Initialize the Context Graph: schema v1 in the registry + Neo4j constraints/indexes (idempotent).

    python manage.py graph_init
"""

from django.core.management.base import BaseCommand

from context_graph import services


class Command(BaseCommand):
    help = "Store the default schema (if none) and apply Neo4j constraints and indexes."

    def handle(self, *args, **options):
        result = services.init_graph()
        if result["created_version"]:
            self.stdout.write(f"Stored default schema as version {result['created_version']}.")
        self.stdout.write(f"Active schema v{result['schema_version']}; applied {result['statements']} "
                          f"constraint/index statement(s).")
