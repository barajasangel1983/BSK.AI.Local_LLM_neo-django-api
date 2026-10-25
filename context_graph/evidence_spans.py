"""Spans for the evidence and chunks written before P25b (P25c, 2026-10-25), without the LLM.

Each document's text of today is its snapshot. An evidence row is resolved against it:

- a table part (chunk_index 200000 + table * 1000 + part): the table is found again (`tables.find_tables`) and the
  part rebuilt; accepted when its text equals the stored excerpt, then the span is the part's rows (exact);
- a text window: rebuilt with the chunking it was read with (`Evidence.window`) and its `chunk_index`; accepted when it
  starts like the stored excerpt (with or without its tables), then located; otherwise the excerpt alone is located,
  and a place found is `partial` (only the excerpt is known to be there);
- anything else (a figure's description, a suggest-maps quote): the excerpt is located.

Never a guess: several places are `ambiguous`, none `not_found`. A span already set is never overwritten.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ingestion import library, spans
from ingestion.chunking import chunk_parsed

from . import tables
from .drawings import FIGURE_CHUNK_BASE
from .models import Evidence

PREFIX = 300            # normalized characters compared between a rebuilt window and its stored excerpt


@dataclass
class DocReport:
    doc_key: str
    counts: Counter = field(default_factory=Counter)
    changed: list[tuple[int, str]] = field(default_factory=list)      # (evidence id, status) not exact
    round_trip_failures: int = 0


def _starts_alike(window_text: str, excerpt: str) -> bool:
    head = spans.normalize(excerpt)[:PREFIX]
    return bool(head) and (spans.normalize(window_text).startswith(head)
                           or spans.normalize(spans._without_tables(window_text)).startswith(head))


def resolve(doc, rows: list[Evidence]) -> tuple[spans.Snapshot | None, dict[int, dict]]:
    """{evidence id: span fields} for these rows of one document, against its text of today."""
    parsed = library.load_parsed(doc)
    md = (parsed.get("document") or {}).get("md_content") or ""
    if not md:
        return None, {}
    snap = spans.Snapshot(md)
    windows: dict[str, list] = {}
    found_tables = None
    out: dict[int, dict] = {}
    for e in rows:
        ci = e.chunk_index
        if ci is not None and ci >= tables.TABLE_CHUNK_BASE:
            if found_tables is None:
                found_tables = tables.find_tables(md)
            number, part = divmod(ci - tables.TABLE_CHUNK_BASE, 1000)
            parts = dict(tables.windows(found_tables[number], number)) if number < len(found_tables) else {}
            w = parts.get(ci)
            if w is not None and w.char_start is not None and \
                    spans.normalize(w.text[:len(e.excerpt)]) == spans.normalize(e.excerpt):
                out[e.pk] = {"text_hash": snap.hash, "char_start": w.char_start, "char_end": w.char_end,
                             "span_status": spans.EXACT, "row_start": w.row_start, "row_end": w.row_end}
            else:
                out[e.pk] = spans.NONE.fields(snap.hash)
            continue
        strategy = (e.window or {}).get("strategy")
        if ci is not None and ci < FIGURE_CHUNK_BASE and strategy:
            key = repr(sorted((e.window or {}).items(), key=str))
            if key not in windows:
                windows[key] = chunk_parsed(parsed, strategy, (e.window or {}).get("params"))
            chunk = windows[key][ci] if ci < len(windows[key]) else None
            if chunk is not None and _starts_alike(chunk.text, e.excerpt):
                out[e.pk] = snap.locate(chunk.text, chunk.page_start, chunk.page_end).fields(snap.hash)
                continue
            span = snap.locate(e.excerpt, e.page_start, e.page_end)
            if span.status == spans.EXACT:      # only the excerpt is known to be there
                span = spans.Span(span.start, span.end, spans.PARTIAL)
            out[e.pk] = span.fields(snap.hash)
            continue
        out[e.pk] = snap.locate(e.excerpt, e.page_start, e.page_end).fields(snap.hash)
    return snap, out


def backfill(docs, apply: bool = False) -> list[DocReport]:
    """Resolve every document's evidence without a span; with `apply`, save the snapshots and write the spans."""
    reports = []
    for doc in docs:
        rows = list(Evidence.objects.filter(document=doc, span_status=""))
        report = DocReport(doc.doc_key)
        if not rows:
            reports.append(report)
            continue
        snap, found = resolve(doc, rows)
        if snap is None:
            report.counts["no text"] = len(rows)
            reports.append(report)
            continue
        for e in rows:
            fields = found[e.pk]
            report.counts[fields["span_status"]] += 1
            if fields["span_status"] != spans.EXACT:
                report.changed.append((e.pk, fields["span_status"]))
            elif e.chunk_index is None or e.chunk_index < tables.TABLE_CHUNK_BASE:
                # round trip: the text at an exact span is what was located (whitespace-normalized)
                text = snap.text[fields["char_start"]:fields["char_end"]]
                if not spans.normalize(text) and fields["char_end"] > fields["char_start"]:
                    report.round_trip_failures += 1
        if apply:
            spans.Snapshot.save(library.doc_dir(doc) / "text", snap.text)
            for e in rows:
                for k, v in found[e.pk].items():
                    setattr(e, k, v)
            Evidence.objects.bulk_update(rows, ["text_hash", "char_start", "char_end", "span_status", "row_start",
                                                "row_end"], batch_size=500)
        reports.append(report)
    return reports


def backfill_chunks(docs, apply: bool = False) -> dict[str, Counter]:
    """The same for the documents' Chroma chunks: span metadata added, ids and vectors unchanged."""
    from ingestion.vector_store import VectorStore

    store, out = VectorStore(), {}
    for doc in docs:
        got = store.document_chunks(doc.doc_key)
        ids, texts, metas = got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
        counts: Counter = Counter()
        todo_ids, todo_metas = [], []
        snap = None
        for cid, text, meta in zip(ids, texts, metas):
            meta = meta or {}
            if meta.get("span_status"):
                counts["already"] += 1
                continue
            if snap is None:
                md = (library.load_parsed(doc).get("document") or {}).get("md_content") or ""
                snap = spans.Snapshot(md)
            fields = snap.locate(text, meta.get("page_start"), meta.get("page_end")).fields(snap.hash)
            counts[fields["span_status"]] += 1
            todo_ids.append(cid)
            todo_metas.append({**meta, **{k: v for k, v in fields.items() if v is not None}})
        if apply and todo_ids:
            spans.Snapshot.save(library.doc_dir(doc) / "text", snap.text)
            store.update_chunk_metadata(todo_ids, todo_metas)
        out[doc.doc_key] = counts
    return out


def backfill_quotes(docs, apply: bool = False) -> dict:
    """P25d: the quote of each evidence row with a span and no quote yet, for the one triple it supports (a row shared
    by several triples is left). With `apply`, written; a quote already set is never overwritten."""
    from django.db.models import Count

    from . import quotes

    counts: Counter = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    not_exact: list[dict] = []
    shared = 0
    for doc in docs:
        rows = list(Evidence.objects.filter(document=doc, quote_status="", char_start__isnull=False)
                    .annotate(n=Count("triples")).prefetch_related("triples"))
        texts: dict[str, str] = {}
        done = []
        for e in rows:
            if e.n != 1:
                shared += e.n > 1
                continue
            t = e.triples.all()[0]
            if e.text_hash not in texts:
                texts[e.text_hash] = quotes.snapshot_text(doc, e.text_hash)
            found = quotes.fields(e, t.object_name, t.object_props, texts[e.text_hash])
            if not found:
                continue
            counts[found["quote_status"]] += 1
            by_type[t.object_type][found["quote_status"]] += 1
            if found["quote_status"] != quotes.EXACT:
                not_exact.append({"evidence": e.pk, "triple": t.pk, "status": found["quote_status"],
                                  "document": doc.doc_key, "type": t.object_type, "name": t.object_name})
            for k, v in found.items():
                setattr(e, k, v)
            done.append(e)
        if apply and done:
            Evidence.objects.bulk_update(done, ["quote_status", "quote_start", "quote_end"], batch_size=500)
    return {"counts": counts, "by_type": {k: dict(v) for k, v in by_type.items()}, "not_exact": not_exact,
            "shared_skipped": shared}


def summary(reports: list[DocReport]) -> Counter:
    total: Counter = Counter()
    for r in reports:
        total.update(r.counts)
    return total


def by_document(reports) -> dict:
    return {r.doc_key: dict(r.counts) for r in reports if r.counts}


def changed_list(reports) -> dict:
    out = defaultdict(list)
    for r in reports:
        out[r.doc_key] = r.changed
    return dict(out)
