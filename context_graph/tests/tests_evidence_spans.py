"""P25b/P25c (2026-10-25): evidence knows where it sits in the document's text. A table part points at its rows; an
extraction run writes the spans and the text snapshot; the backfill resolves rows written before P25b, as a dry run
unless asked, and never overwrites a span."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import evidence_spans, extraction, registry, tables
from context_graph.evidence import evidence_json
from context_graph.models import Evidence
from context_service import service
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V7 = yaml.safe_load((SCHEMAS / "industrial_v7.yaml").read_text())

TABLE_PAGE = ("# 7. Lubrication\n\nTable 7.2 Regreasing intervals\n\n| Frame | Interval | Amount |\n|---|---|---|\n"
              "| 280 | 7 000 h | 40 g |\n| 315 | 6 000 h | 55 g |\n| 355 | 5 600 h | 70 g |\n")
PAGES = ["# 4. Terminals\n\n" + " ".join(f"Terminal {n} is digital input DI {n - 5}, 24 V DC, low below {n} V." for n in range(5, 25)),
         TABLE_PAGE]
MD = f"\n\n{PAGE_BREAK}\n\n".join(PAGES)


class TableOffsetTests(SimpleTestCase):
    def test_a_table_part_points_at_its_rows(self):
        [table] = tables.find_tables(MD)
        self.assertEqual(MD[table.char_start:table.char_end].splitlines()[0], "| Frame | Interval | Amount |")
        [(index, part)] = tables.windows(table, 0)
        self.assertEqual((part.row_start, part.row_end), (1, 3))
        self.assertEqual(MD[part.char_start:part.char_end], "| 280 | 7 000 h | 40 g |\n| 315 | 6 000 h | 55 g |\n"
                                                            "| 355 | 5 600 h | 70 g |")
        self.assertEqual(index, tables.TABLE_CHUNK_BASE)

    def test_rows_split_into_parts_keep_their_numbers(self):
        [table] = tables.find_tables(MD)
        parts = [w for _, w in tables.windows(table, 0, max_rows=2)]
        self.assertEqual([(w.row_start, w.row_end) for w in parts], [(1, 2), (3, 3)])
        self.assertEqual(MD[parts[1].char_start:parts[1].char_end], "| 355 | 5 600 h | 70 g |")


@override_settings(GRAPH_ENABLED=False)
class ExtractionSpanTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": MD}}

        def llm(messages):
            text = messages[-1]["content"]
            if "Row 3" in text:
                return json.dumps({"triples": [{"subject": "Motor", "subject_type": "Component", "predicate": "HAS_PROCEDURE",
                                                "object": "Regreasing frame 355", "object_type": "Procedure", "confidence": 0.9}]})
            return json.dumps({"triples": [{"subject": "Converter", "subject_type": "Component", "predicate": "HAS_IO",
                                            "object": "DI 0", "object_type": "IOPoint", "confidence": 0.9}]})
        p = patch("context_graph.extraction.call_llm", side_effect=llm)
        p.start()
        self.addCleanup(p.stop)
        registry.save_version(V7, note="v7")
        self.doc, _ = library.add_document(SimpleUploadedFile("motor.pdf", b"%PDF motor"))
        self.drain()
        self.doc.refresh_from_db()

    def run_extract(self):
        params = extraction.validate_params({"pages": "1-2"})
        params["chunking"] = {"strategy": "structure", "params": {"max_tokens": 200, "overlap_sentences": 0}}
        return extraction._extract(None, self.doc, params)

    def test_a_run_writes_spans_and_the_snapshot(self):
        self.run_extract()
        rows = list(Evidence.objects.filter(document=self.doc))
        self.assertTrue(rows)
        self.assertTrue(all(e.span_status == "exact" and e.text_hash for e in rows), [(e.chunk_index, e.span_status) for e in rows])
        table = next(e for e in rows if e.chunk_index >= tables.TABLE_CHUNK_BASE)
        self.assertEqual((table.row_start, table.row_end), (1, 3))
        self.assertTrue((library.doc_dir(self.doc) / "text" / f"{table.text_hash}.md").exists())
        window = next(e for e in rows if e.chunk_index < tables.TABLE_CHUNK_BASE)
        self.assertIn("is digital input DI", MD[window.char_start:window.char_end])

    @override_settings(EVIDENCE_SPANS=False)
    def test_spans_off_writes_none(self):
        self.run_extract()
        self.assertFalse(Evidence.objects.filter(document=self.doc).exclude(span_status="").exists())

    def test_the_backfill_resolves_old_rows_and_never_overwrites(self):
        with override_settings(EVIDENCE_SPANS=False):          # rows as written before P25b
            self.run_extract()
        reports = evidence_spans.backfill([self.doc])           # dry run: nothing written
        self.assertEqual(sum(reports[0].counts.values()), Evidence.objects.filter(document=self.doc).count())
        self.assertFalse(Evidence.objects.filter(document=self.doc).exclude(span_status="").exists())
        evidence_spans.backfill([self.doc], apply=True)
        rows = list(Evidence.objects.filter(document=self.doc))
        self.assertTrue(all(e.span_status == "exact" for e in rows), [(e.chunk_index, e.span_status) for e in rows])
        Evidence.objects.filter(document=self.doc).update(span_status="partial")     # a span already set stays
        evidence_spans.backfill([self.doc], apply=True)
        self.assertFalse(Evidence.objects.filter(document=self.doc).exclude(span_status="partial").exists())


@override_settings(GRAPH_ENABLED=False)
class QuoteTests(ExtractionSpanTests):
    """P25d/P25e: a run writes each place's quote; the backfill never overwrites one; the screens and the packet get
    the quote."""

    def test_a_run_writes_quotes_and_the_screens_get_them(self):
        self.run_extract()
        rows = [e for e in Evidence.objects.filter(document=self.doc) if e.chunk_index < tables.TABLE_CHUNK_BASE]
        exact = [e for e in rows if e.quote_status == "exact"]
        self.assertTrue(exact)
        self.assertTrue(all("DI 0," in MD[e.quote_start:e.quote_end] for e in exact))
        self.assertTrue(all(e.quote_start is None for e in rows if e.quote_status != "exact"))
        e = exact[0]
        body = APIClient().get(f"/api/graph/evidence/{e.pk}/text/").json()
        self.assertEqual((body["kind"], body["quote_status"]), ("span", "exact"))
        self.assertIn("DI 0,", body["text"][body["quote"]["start"]:body["quote"]["end"]])
        self.assertIn("DI 0", evidence_json(e)["quote"])
        facts = [{"evidence_id": e.pk, "source": {"label": "motor.pdf p.1"}}]
        service._quote_sources(facts)
        self.assertEqual((facts[0]["source"]["quote_status"], "evidence_id" in facts[0]), ("exact", False))

    def test_the_quote_backfill_never_overwrites(self):
        self.run_extract()
        Evidence.objects.filter(document=self.doc).update(quote_status="", quote_start=None, quote_end=None)
        report = evidence_spans.backfill_quotes([self.doc])                     # dry run
        self.assertTrue(report["counts"]["exact"])
        self.assertFalse(Evidence.objects.filter(document=self.doc).exclude(quote_status="").exists())
        evidence_spans.backfill_quotes([self.doc], apply=True)
        self.assertEqual(Evidence.objects.filter(document=self.doc, quote_status="exact").count(), report["counts"]["exact"])
        Evidence.objects.filter(document=self.doc).update(quote_status="ambiguous")
        self.assertFalse(evidence_spans.backfill_quotes([self.doc], apply=True)["counts"])


class RepeatedUnitTests(SimpleTestCase):
    """P25 D6: "6 ms" with unit "ms" showed as "6 ms ms"."""

    def test_the_unit_leaves_a_plain_value_only(self):
        w = extraction.without_repeated_unit
        self.assertEqual(w({"value": "6 ms", "unit": "ms"}), {"value": "6"})
        self.assertEqual(w({"value": "88.8 x 56.8 x 45 mm", "unit": "mm"}), {"value": "88.8 x 56.8 x 45"})
        self.assertEqual(w({"min": "≥ 0.001 t", "max": "12...16 t", "unit": "t"}), {"min": "≥ 0.001", "max": "12...16"})
        self.assertEqual(w({"value": "1 MB of RAM", "unit": "RAM"}), {})
        self.assertEqual(w({"value": "10 million cycles", "unit": "cycles"}), {})
        self.assertEqual(w({"value": "6", "unit": "ms"}), {})
