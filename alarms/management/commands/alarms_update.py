from django.core.management.base import BaseCommand

from alarms import service


class Command(BaseCommand):
    help = "Read the historian's rows into alarm episodes (P31). --rebuild --days N: delete and read the last N days again."

    def add_arguments(self, parser):
        parser.add_argument("--asset", default="bsk:asset:EXTR01")
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--days", type=int, default=30)

    def handle(self, *args, asset, rebuild, days, **options):
        result = service.rebuild(asset, days) if rebuild else service.update(asset)
        self.stdout.write(f"{asset}: {result}")
