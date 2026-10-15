"""Add the manuals downloaded by fetch_manuals to the library (P14d-2), then let the worker run.

    manage.py library_ingest --manifest data/manuals/EXTR01/manifest.json --asset bsk:asset:EXTR01 --embed
    manage.py library_ingest --manifest … --dry-run      list what would be added and where
    manage.py library_ingest --manifest … --status       where each document of the manifest is
    manage.py library_ingest --manifest … --figures      the figures pass: describe each manual's figures
                                                         (VLM on BSK) after its parse, then embed it again

Each manual goes into Manuals/<Manufacturer> (a datasheet into Datasheets/<Manufacturer>),
smallest first, with its title, source URL, download date and asset. Adding queues the
parse; with --embed the embeddings follow it. The worker does the rest, one job at a time,
and waits when BSK or the DGX is off. A file already in the library (same SHA-256) is skipped.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError

from ingestion import folders, library
from ingestion.models import Document, Job

DATASHEET = re.compile(r"\bdata ?sheet\b", re.I)


def folder_path(entry: dict) -> str:
    top = "Datasheets" if DATASHEET.search(entry.get("title") or entry.get("file") or "") else "Manuals"
    maker = re.sub(r"[/\s]+", " ", entry.get("manufacturer") or "").strip()
    return f"{top}/{maker}" if maker else top


class Command(BaseCommand):
    help = "Add the manuals of a fetch_manuals manifest to the library (parse, optionally embed)."

    def add_arguments(self, parser):
        parser.add_argument("--manifest", required=True, help="manifest.json written by fetch_manuals")
        parser.add_argument("--asset", default="", help="plant asset of the documents, e.g. bsk:asset:EXTR01")
        parser.add_argument("--embed", action="store_true", help="generate the embeddings after each parse")
        parser.add_argument("--dry-run", action="store_true", help="list only")
        parser.add_argument("--status", action="store_true", help="show where each document is")
        parser.add_argument("--figures", action="store_true",
                            help="describe the figures after the parse (VLM on BSK), then embed again")

    def handle(self, *args, **opts):
        manifest = Path(opts["manifest"]).expanduser()
        try:
            entries = json.loads(manifest.read_text())["manuals"]
        except (OSError, ValueError, KeyError) as exc:
            raise CommandError(f"{manifest}: {exc}")
        ready = sorted((e for e in entries if e.get("status") in ("downloaded", "kept")), key=lambda e: e.get("size") or 0)
        if opts["status"]:
            return self.status(ready)
        if opts["figures"] and not opts["dry_run"]:
            return self.figures_pass(ready)
        self.stdout.write(f"{manifest}: {len(ready)} downloaded manuals, smallest first")
        added = skipped = 0
        for e in ready:
            path, target = manifest.parent / e["file"], folder_path(e)
            if not path.exists():
                self.stdout.write(f"  missing    {e['file']}")
                continue
            if opts["dry_run"]:
                self.stdout.write(f"  {target:32} {e['file']} ({(e.get('size') or 0) / 1e6:.1f} MB)")
                continue
            folder = folders.get_by_path(target, create=True)
            upload = SimpleUploadedFile(e["file"], path.read_bytes(), content_type="application/pdf")
            doc, created = library.add_document(
                upload, folder=folder, then_embed=opts["embed"], title=e.get("title") or "",
                source_url=e["url"], fetched_at=datetime.fromisoformat(e["downloaded_at"]) if e.get("downloaded_at") else None,
                plant_asset=opts["asset"])
            if created:
                added += 1
                self.stdout.write(f"  added      {target}/{e['file']} -> {doc.doc_key}")
            else:
                skipped += 1
                self.stdout.write(f"  in library {e['file']} ({doc.doc_key})")
        if not opts["dry_run"]:
            self.stdout.write(f"added {added}, already in the library {skipped}; the library worker parses them one at a time")

    def figures_pass(self, entries):
        """Queue Describe figures (then a new embedding) for every manual; one not parsed yet gets it
        after its parse. The worker runs figures after parses and embeddings, oldest first."""
        queued = later = 0
        for e in entries:
            doc = Document.objects.filter(sha256=e.get("sha256")).first()
            if doc is None:
                self.stdout.write(f"  not added  {e['file']}")
                continue
            parse = doc.jobs.filter(kind=Job.Kind.PARSE, status__in=library.ACTIVE).first()
            if parse:
                parse.params = {**parse.params, "then_figures": True}
                parse.save(update_fields=["params"])
                later += 1
            else:
                library.enqueue(doc, Job.Kind.FIGURES, {"then_embed": True})
                queued += 1
        self.stdout.write(f"figures queued for {queued} manual(s), after the parse for {later}; "
                          "each is embedded again when its figures are done")

    def status(self, entries):
        totals = {}
        for e in entries:
            doc = Document.objects.filter(sha256=e.get("sha256")).first()
            if doc is None:
                state = "not added"
            else:
                job = doc.jobs.filter(status__in=library.ACTIVE).order_by("created_at").first()
                if job and job.kind == Job.Kind.PARSE:
                    state = f"parse {job.status}" + (f": {job.message}" if job.message else "")
                elif job:
                    state = f"{job.kind} {job.status}" + (f": {job.message}" if job.message else "")
                elif doc.parse_status == Document.ParseStatus.FAILED:
                    state = f"parse failed: {doc.parse_error[:120]}"
                elif doc.rag_status == Document.PipelineStatus.DONE:
                    state = f"embedded ({doc.page_count} pages, {doc.rag_chunk_count} chunks"
                    if doc.figures_status == Document.FiguresStatus.DONE:
                        from ingestion import figures as figs
                        state += f", {figs.summary(figs.load(doc))['described']} figures described"
                    state += ")"
                elif doc.rag_status == Document.PipelineStatus.FAILED:
                    state = f"embed failed: {doc.rag_error[:120]}"
                else:
                    state = f"parse {doc.parse_status}"
            totals[state.split(":")[0].split(" (")[0]] = totals.get(state.split(":")[0].split(" (")[0], 0) + 1
            self.stdout.write(f"  {e['file'][:70]:70} {state}")
        self.stdout.write("; ".join(f"{k}: {v}" for k, v in sorted(totals.items())))
