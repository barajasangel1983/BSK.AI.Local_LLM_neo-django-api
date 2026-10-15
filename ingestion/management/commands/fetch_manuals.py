"""Download the manuals a BOM names (P14d step 1): `manage.py fetch_manuals --bom <csv> [--dry-run]`.

Files go to data/manuals/<ASSET>/ with manifest.json (URL, date, size, SHA-256).
Nothing is added to the library; that is the next step, one parse job at a time.
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ingestion import manuals


class Command(BaseCommand):
    help = "Download the manufacturer PDFs named in a BOM's Manual URL column (no library upload)."

    def add_arguments(self, parser):
        parser.add_argument("--bom", required=True, help="BOM CSV with Item, Manufacturer, Manual URL, Manual Title")
        parser.add_argument("--asset", default="EXTR01", help="asset key: the folder under data/manuals/")
        parser.add_argument("--out", help="output folder (default data/manuals/<asset>)")
        parser.add_argument("--dry-run", action="store_true", help="list what would be fetched; download nothing")

    def handle(self, *args, **opts):
        bom = Path(opts["bom"]).expanduser()
        out = Path(opts["out"]).expanduser() if opts["out"] else Path(settings.BASE_DIR) / "data" / "manuals" / opts["asset"]
        try:
            items = manuals.read_bom(bom)
        except manuals.ManualsError as exc:
            raise CommandError(str(exc))
        self.stdout.write(f"{bom}: {len(items)} distinct manuals -> {out}")
        if opts["dry_run"]:
            for m in items:
                where = "fetch" if manuals.allowed(m.url) else "by hand (not a manufacturer's https server)"
                self.stdout.write(f"  {where:8} {m.file}  [{', '.join(m.items)}]")
            return

        out.mkdir(parents=True, exist_ok=True)
        known = manuals.load_manifest(out)
        for i, m in enumerate(items, 1):
            manuals.fetch(m, out, known.get(m.url), settings.LIBRARY_MAX_UPLOAD_BYTES)
            size = f"{m.size / 1e6:.1f} MB" if m.size else ""
            self.stdout.write(f"  [{i}/{len(items)}] {m.status:10} {m.file} {size} {m.reason}".rstrip())
            manuals.save_manifest(out, bom, items)   # after each file: an interrupted run keeps what it fetched

        by_status = {s: [m for m in items if m.status == s] for s in ("downloaded", "kept", "by hand")}
        total = sum(m.size for m in items) / 1e6
        self.stdout.write(f"downloaded {len(by_status['downloaded'])}, kept {len(by_status['kept'])}, "
                          f"by hand {len(by_status['by hand'])}; {total:.0f} MB in {out}")
        for m in by_status["by hand"]:
            self.stdout.write(f"  by hand: {m.title or m.url} ({m.reason}) {m.url}")
