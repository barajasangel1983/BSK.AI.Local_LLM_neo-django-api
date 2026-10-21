"""Propose the device models and product families of an asset's parts (P26a): `manage.py device_models
--asset bsk:asset:EXTR01 --out <file.csv>`. Writes the CSV for the "Device models" import template and prints the
grouping; nothing is written to the graph (the file is imported and its triples approved like any data file)."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from context_graph import device_models, services


class Command(BaseCommand):
    help = "Propose device models and product families from the asset's components (a CSV to confirm, then import)."

    def add_arguments(self, parser):
        parser.add_argument("--asset", default="bsk:asset:EXTR01")
        parser.add_argument("--out", required=True, help="the CSV to write")

    def handle(self, *args, **opts):
        try:
            ctx = services.get_asset_context(opts["asset"])
        except Exception as exc:        # noqa: BLE001 — a missing asset or an unreachable graph
            raise CommandError(str(exc))
        components = []

        def walk(items):
            for c in items:
                components.append(c)
                walk(c["children"])
        walk(ctx["components"])
        rows = device_models.propose(components, opts["asset"].split(":", 2)[2])
        out = Path(opts["out"]).expanduser()
        out.write_text(device_models.to_csv(rows), encoding="utf-8")
        self.stdout.write(device_models.summary(rows))
        self.stdout.write(self.style.SUCCESS(f"written: {out}"))
