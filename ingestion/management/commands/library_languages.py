"""Keep only some languages of a multilingual document for RAG (P14d-2).

    manage.py library_languages LOW-VOLTAGE-MOTORS-… --keep en,es [--dry-run]
    manage.py library_languages LOW-VOLTAGE-MOTORS-… --keep all

Shows the document's pages per language and how many chunks are kept. The parsed text stays
whole; only the embedding (and Describe figures) leave the other pages out. An embedded
document is embedded again.
"""

from django.core.management.base import BaseCommand, CommandError

from ingestion import languages, library
from ingestion.chunking import DEFAULT_STRATEGY, chunk_parsed
from ingestion.models import Document, Job


class Command(BaseCommand):
    help = "Keep only some languages of a multilingual document for RAG."

    def add_arguments(self, parser):
        parser.add_argument("doc_key")
        parser.add_argument("--keep", required=True, help="ISO codes, e.g. en,es; 'all' to keep every language")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opts):
        doc = Document.objects.filter(doc_key=opts["doc_key"]).first()
        if doc is None:
            raise CommandError(f"no document {opts['doc_key']!r}")
        if doc.parse_status != Document.ParseStatus.PARSED:
            raise CommandError(f"{doc.doc_key} is not parsed yet ({doc.parse_status})")
        keep = [] if opts["keep"].strip().lower() == "all" else \
            sorted({c.strip().lower() for c in opts["keep"].split(",") if c.strip()})
        parsed = library.load_parsed(doc)
        md = parsed["document"].get("md_content") or ""
        pages = languages.kept_pages(md, keep)
        chunks = chunk_parsed(parsed, DEFAULT_STRATEGY)
        kept = [c for c in chunks if pages is None or not c.page_start or c.page_start in pages]
        self.stdout.write(f"{doc.doc_key}: pages per language {languages.summary(md)}")
        self.stdout.write(f"keep {', '.join(keep) or 'all'}: {len(pages) if pages is not None else doc.page_count} of "
                          f"{doc.page_count} pages, {len(kept)} of {len(chunks)} chunks (default chunking)")
        if opts["dry_run"]:
            return
        doc.languages = keep
        doc.save(update_fields=["languages"])
        if doc.rag_status == Document.PipelineStatus.DONE:
            library.enqueue(doc, Job.Kind.EMBED)
            self.stdout.write("embedding again (queued)")
        else:
            self.stdout.write("saved; its embedding will keep only these pages")
