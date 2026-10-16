"""Suggest pages for Generate triples (2026-10-16): the model picks the pages from the document's outline.
The DGX is mocked; the graph is disabled."""

import json
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import page_suggest
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

from .tests_extraction_tables import ASSET, CONTEXT, MOTOR

EN = "Check the bearing temperature after the first hours of operation and regrease the bearings at the given interval. " * 3
ES = "Compruebe la temperatura de los rodamientos tras las primeras horas y engrase los rodamientos en el intervalo indicado. " * 3
PAGES = [
    f"# Contents\n\n{EN}",
    f"## 7 Maintenance\n\n{EN}",
    EN,                                                         # continues 7 Maintenance
    f"## 10 Troubleshooting\n\nTable 10.1: Troubleshooting\n\n{EN}",
    f"## 7 Mantenimiento\n\n{ES}",
]
MD = f"\n\n{PAGE_BREAK}\n\n".join(PAGES)


class OutlineTests(SimpleTestCase):
    def test_one_entry_per_heading_with_its_continuation_pages(self):
        lines = page_suggest.outline(MD, "en", ["en", "en", "en", "en", "es"])
        self.assertEqual([(a, b) for a, b, _ in lines], [(1, 1), (2, 3), (4, 4)])     # the Spanish page is left out
        self.assertTrue(lines[1][2].startswith("Pages 2-3: 7 Maintenance — Check the bearing"))
        self.assertTrue(lines[2][2].startswith("Page 4: 10 Troubleshooting | Table 10.1: Troubleshooting — "))

    def test_one_language(self):
        self.assertEqual(page_suggest.outline_language(["en", "en"], ["en"]), "")             # one language: every page
        self.assertEqual(page_suggest.outline_language(["de", "en", "es"], ["es", "en"]), "es")  # the first kept one
        self.assertEqual(page_suggest.outline_language(["de", "en", "es"], []), "en")
        self.assertEqual(page_suggest.outline_language(["de", "de", "fr"], []), "de")


@override_settings(GRAPH_ENABLED=False)
class SuggestPagesTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": MD}}
        self.messages = []
        self.answer = {"note": "Maintenance and troubleshooting.", "sections": [
            {"pages": "2", "title": "7 Maintenance", "why": "Intervals."},       # takes its continuation page 3
            {"pages": "4-9", "title": "10 Troubleshooting", "why": "Causes."},   # 5-9: not in the outline
            {"pages": "5", "title": "7 Mantenimiento"},                         # another language
            {"pages": "eighteen"}, "not a section"]}

        def llm(messages):
            self.messages.append(messages)
            return json.dumps(self.answer)
        for target, kwargs in (("context_graph.extraction.call_llm", {"side_effect": llm}),
                               ("context_graph.services.get_asset_context", {"return_value": CONTEXT})):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.doc, _ = library.add_document(SimpleUploadedFile("motors.pdf", b"%PDF motors"))
        self.drain()
        self.client = APIClient()

    def suggest(self, **body):
        return self.client.post("/api/graph/extract/suggest-pages/", {"document_id": str(self.doc.id), **body}, format="json")

    def test_pages_of_the_chosen_sections_with_their_continuations(self):
        r = self.suggest(asset_id=ASSET, anchor_id=MOTOR)
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["pages"], "2-4")
        self.assertEqual([(s["pages"], s["title"]) for s in body["sections"]], [("2-3", "7 Maintenance"), ("4", "10 Troubleshooting")])
        self.assertEqual((body["language"], body["outline_pages"], body["total_pages"]), ("en", 4, 5))
        system, outline = self.messages[0][0]["content"], self.messages[0][1]["content"]
        self.assertIn('It describes the part "Extruder motor" of the asset "Extruder EXTR01"', system)
        self.assertIn("You extract", system)                                   # the default preset's goal
        self.assertIn("Pages 2-3: 7 Maintenance", outline)
        self.assertNotIn("Mantenimiento", outline)

    def test_asset_only_and_nothing_found(self):
        self.answer = {"note": "Only other models.", "sections": []}
        body = self.suggest(asset_id=ASSET).json()
        self.assertEqual((body["pages"], body["sections"], body["note"]), ("", [], "Only other models."))
        self.assertIn("a generic manual for one of its parts", self.messages[0][0]["content"])

    def test_errors(self):
        self.assertEqual(self.client.post("/api/graph/extract/suggest-pages/", {"document_id": "0" * 32}, format="json")
                         .status_code, 404)
        self.assertEqual(self.suggest(preset_id=99999).status_code, 404)
        self.assertEqual(self.suggest(anchor_id=MOTOR).status_code, 400)       # a part needs its asset
        unparsed, _ = library.add_document(SimpleUploadedFile("later.pdf", b"%PDF later"))
        r = self.client.post("/api/graph/extract/suggest-pages/", {"document_id": str(unparsed.id)}, format="json")
        self.assertEqual((r.status_code, r.json()["error"]), (400, "parse the document first (Library → Parse)"))
        with patch("context_graph.extraction.call_llm", return_value="not json"):
            self.assertEqual(self.suggest().status_code, 400)
