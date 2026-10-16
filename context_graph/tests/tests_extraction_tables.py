"""Generate triples, step 3 of the schema review (2026-10-16): tables row by row, properties, the split retry
and the "Maintenance (v3)" preset. The DGX is mocked; the graph is disabled."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import extraction, registry, tables
from context_graph.extraction import clean_properties, merge_properties, split_text
from context_graph.models import CandidateTriple, PromptPreset
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

HEADER = "| TROUBLE | CAUSE | WHAT TO DO |\n|---|---|---|\n"
PAGE_1 = ("## 10 Troubleshooting\n\nThese instructions do not cover all details or variations in equipment.\n\n"
          "Table 10.1: Troubleshooting\n\n" + HEADER +
          "| Motor fails to start | Blown fuses | Replace fuses with proper type and rating. |\n"
          "| Motor fails to start | Overload trips | Check and reset overload in starter. |\n"
          "| Motor stalls | One phase may be open | Check lines for open phase. |\n")
PAGE_2 = "Nothing but a picture on this page."
PAGE_3 = (HEADER + "| Hot bearings | Insufficient grease | Maintain proper quality and amount of grease in bearing. |\n"
          "| Hot bearings | Excess lubricant | Reduce quantity of grease, bearing should not be more than half full. |\n")
LUBE = ("## 7.2.3 Lubrication intervals and amounts\n\nTable 7.2\n\n"
        "| Frame size | Amount of grease g/bearing | 1500 r/min |\n|--|--|--|\n"
        "| Ball bearings, lubrication intervals in duty hours | Ball bearings, lubrication intervals in duty hours "
        "| Ball bearings, lubrication intervals in duty hours |\n"
        "| 315 | 55 | 7 600 |\n| 355 | 70 | 5 600 |\n| 400 | - | 4 700 |\n")
MD = f"\n\n{PAGE_BREAK}\n\n".join([PAGE_1, PAGE_2, PAGE_3, LUBE])


class TableReaderTests(SimpleTestCase):
    def test_tables_captions_sections_and_continuations(self):
        found = tables.find_tables(MD)
        self.assertEqual([(t.page, t.caption, t.section, len(t.rows)) for t in found], [
            (1, "Table 10.1: Troubleshooting", "10 Troubleshooting", 3),
            (3, "Table 10.1: Troubleshooting", "10 Troubleshooting", 2),     # continued two pages on: same caption
            (4, "Table 7.2", "7.2.3 Lubrication intervals and amounts", 4)])

    def test_rows_with_the_same_first_cell_stay_together(self):
        table = tables.find_tables(MD)[0]
        self.assertEqual(len(tables.windows(table, 0)), 1)
        parts = tables.windows(table, 0, max_rows=2)
        self.assertEqual([w.text.count("TROUBLE = Motor") for _, w in parts], [2, 1])   # "fails to start" kept whole
        self.assertEqual([i for i, _ in parts], [200_000, 200_001])
        self.assertTrue(all(w.text.startswith("Table 10.1: Troubleshooting (section: 10 Troubleshooting), page 1.")
                            and "Columns: TROUBLE | CAUSE | WHAT TO DO" in w.text and '"note"' in w.text for _, w in parts))
        self.assertIn("Row 1: TROUBLE = Motor fails to start; CAUSE = Blown fuses; WHAT TO DO = Replace fuses", parts[0][1].text)

    def test_a_heading_row_is_repeated_above_its_rows(self):
        _, window = tables.windows(tables.find_tables(MD)[2], 2)[0]
        self.assertIn("Rows under: Ball bearings, lubrication intervals in duty hours", window.text)
        self.assertEqual(window.text.count("Ball bearings"), 1)
        self.assertIn("Row 2: Frame size = 355; Amount of grease g/bearing = 70; 1500 r/min = 5 600", window.text)
        self.assertIn("Row 3: Frame size = 400; 1500 r/min = 4 700", window.text)        # an empty cell is left out

    def test_text_without_tables_and_halves(self):
        self.assertEqual(tables.strip_tables("a\n| x | y |\nb"), "a\nb")
        text = "\n".join(f"Line {n} of a long window about bearings and grease." for n in range(20))
        first, second = split_text(text)
        self.assertEqual(first + "\n" + second, text)
        self.assertEqual(split_text("short"), [])

    def test_properties_are_cleaned_and_merged(self):
        self.assertEqual(clean_properties({"interval": " 5 600 h ", "Bad Name": "x", "name": "y", "causes": ["a", "", 3],
                                           "status": "approved",
                                           "amount": 70, "nested": {"a": 1}}),
                         {"interval": "5 600 h", "causes": ["a", "3"], "amount": 70})
        self.assertEqual(merge_properties({"causes": ["a"], "summary": "x"}, {"causes": ["b", "a"], "summary": "y", "interval": "1"}),
                         {"causes": ["a", "b"], "summary": "x; y", "interval": "1"})


ASSET = "bsk:asset:EXTR01"
MOTOR = "bsk:component:EXTR01/main-motor"
CONTEXT = {
    "asset": {"id": ASSET, "label": "Asset", "name": "Extruder EXTR01", "properties": {}},
    "components": [{"id": MOTOR, "label": "Component", "name": "Extruder motor", "children": [], "signals": [], "alarms": [],
                    "properties": {"name": "Extruder motor", "manufacturer": "ABB", "model": "M3BP 355MLA 4"}}],
    "signals": [], "alarms": [], "procedures": [], "documents": [], "connections": [], "counts": {},
}


def reply(text: str) -> str:
    """The mocked model: a failure mode with its causes for each trouble row; a procedure for the lubrication row."""
    out = []
    for line in text.splitlines():
        if not line.startswith("Row "):
            continue
        cells = [part.split(" = ", 1)[1] for part in line.split(": ", 1)[1].split("; ")]
        if len(cells) == 3 and cells[0] in ("Motor fails to start", "Motor stalls", "Hot bearings"):
            out.append({"subject": "Extruder motor", "subject_type": "Component", "predicate": "HAS_FAILURE_MODE",
                        "object": cells[0], "object_type": "FailureMode", "object_properties": {"causes": [cells[1]],
                                                                                              "colour": "red"},
                        "confidence": 0.9})
        if cells[:2] == ["355", "70"]:
            out.append({"subject": "Extruder motor", "subject_type": "Component", "predicate": "HAS_PROCEDURE",
                        "object": "Regreasing", "object_type": "Procedure", "object_subtype": "Lubrication",
                        "object_properties": {"interval": "5 600 h", "amount": "70 g per bearing"}, "confidence": 0.8})
    return json.dumps({"triples": out})


@override_settings(GRAPH_ENABLED=False)
class TablesPipelineTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": MD}}
        self.texts = []

        def llm(messages):
            self.texts.append(messages[-1]["content"])
            return reply(messages[-1]["content"])
        for target, kwargs in (("context_graph.extraction.call_llm", {"side_effect": llm}),
                               ("context_graph.services.get_asset_context", {"return_value": CONTEXT})):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.llm_reply = llm
        registry.save_version(yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v3.yaml").read_text()),
                              note="v3")
        self.doc, _ = library.add_document(SimpleUploadedFile("motors.pdf", b"%PDF motors"))
        self.drain()
        self.client = APIClient()

    def extract(self, **body):
        r = self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "mode": "schema",
                                                     "asset_id": ASSET, "anchor_id": MOTOR, **body}, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        self.texts.clear()
        self.drain()
        self.doc.refresh_from_db()
        return {t.object_name: t for t in CandidateTriple.objects.filter(document=self.doc)}

    def test_tables_are_read_row_by_row_with_their_properties(self):
        staged = self.extract()
        tables_read = [t for t in self.texts if "Read it row by row" in t]
        self.assertEqual(len(tables_read), 3)
        self.assertTrue(all("Columns: TROUBLE | CAUSE | WHAT TO DO" in t or "Frame size" in t for t in tables_read))
        self.assertFalse(any("| Motor fails to start" in t for t in self.texts if "Read it row by row" not in t))
        # Causes from two rows join on one failure mode; a property the schema does not document is dropped.
        self.assertEqual(staged["Motor fails to start"].object_props, {"causes": ["Blown fuses", "Overload trips"]})
        self.assertEqual(staged["Hot bearings"].object_props["causes"], ["Insufficient grease", "Excess lubricant"])
        self.assertEqual(staged["Hot bearings"].page_start, 3)
        self.assertEqual(staged["Regreasing"].object_props,
                         {"subtype": "Lubrication", "interval": "5 600 h", "amount": "70 g per bearing"})
        self.assertGreaterEqual(staged["Regreasing"].chunk_index, tables.TABLE_CHUNK_BASE)
        self.assertEqual(self.doc.graph_error, "")

    def test_off_reads_tables_inside_the_text_as_before(self):
        self.extract(tables_by_row=False)
        self.assertFalse(any("Read it row by row" in t for t in self.texts))
        self.assertTrue(any("| Motor fails to start" in t for t in self.texts))

    def test_an_answer_cut_off_twice_is_read_again_in_halves(self):
        calls = {"n": 0}

        def cut_off_once(messages):
            calls["n"] += 1
            if calls["n"] <= 2:        # the window's answer and its retry: cut off at the output limit
                return '{"triples": [{"subject": "Extruder motor", "subject_type": "Comp'
            return self.llm_reply(messages)
        long_page = "## 7 Maintenance\n\n" + "\n".join(
            f"Inspect the bearings and the shaft seals of the motor, step {n}, at regular intervals." for n in range(30))
        self.mock_convert.return_value = {"status": "success", "errors": [],
                                          "document": {"md_content": long_page + f"\n\n{PAGE_BREAK}\n\n" + PAGE_1}}
        self.doc, _ = library.add_document(SimpleUploadedFile("motors2.pdf", b"%PDF motors 2"))
        self.drain()
        with patch("context_graph.extraction.call_llm", side_effect=cut_off_once):
            staged = self.extract(pages="1-2")
        self.assertEqual(self.doc.graph_error, "")              # read again in two halves: not a failed window
        self.assertIn("Motor fails to start", staged)
        self.assertGreaterEqual(calls["n"], 5)


@override_settings(GRAPH_ENABLED=False)
class MaintenancePresetTests(LibraryTestCase):
    def test_the_preset_is_offered_with_the_properties(self):
        extraction.ensure_default_presets()
        preset = PromptPreset.objects.get(name="Maintenance (v3)")
        self.assertEqual((preset.mode, preset.is_default), ("schema", False))
        self.assertEqual(extraction.default_preset("schema").name, "Default schema-guided")
        schema = registry.save_version(yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v3.yaml")
                                                      .read_text()), note="v3")
        system = extraction.build_messages(preset, registry.active_schema(), "text", "motors.pdf", "7 Maintenance")[0]["content"]
        self.assertIn("- Procedure: procedure_type (the kind, as the source states it); summary", system)
        self.assertIn("- FailureMode: description (the failure as described); causes", system)
        self.assertNotIn("{properties}", system)
        self.assertIsNotNone(schema)
        # The built-in maintenance preset can be edited and restored, not deleted.
        r = APIClient().delete(f"/api/graph/presets/{preset.pk}/")
        self.assertEqual(r.status_code, 409)


@override_settings(GRAPH_ENABLED=False)
class ReviewByConfidenceTests(LibraryTestCase):
    """Step 4: the review list ranked by confidence; low confidence hidden unless asked."""

    def test_order_and_floor(self):
        doc, _ = self.upload()
        for name, confidence in (("a", 0.5), ("b", 0.95), ("c", None), ("d", 0.8)):
            CandidateTriple.objects.create(document=doc, mode="schema", subject_name=name, subject_type="Component",
                                           predicate="HAS_PROCEDURE", object_name=f"p{name}", object_type="Procedure",
                                           confidence=confidence)
        client = APIClient()
        names = lambda **qp: [t["subject"]["name"] for t in client.get("/api/graph/triples/", qp).json()["triples"]]  # noqa: E731
        self.assertEqual(names(), ["a", "b", "c", "d"])                         # unchanged without the options
        self.assertEqual(names(order="-confidence"), ["b", "d", "a", "c"])
        self.assertEqual(names(order="confidence"), ["a", "d", "b", "c"])
        self.assertEqual(names(order="-confidence", min_confidence="0.8"), ["b", "d", "c"])   # no confidence: always shown
        ids = client.get("/api/graph/triples/", {"fields": "ids", "min_confidence": "0.9"}).json()
        self.assertEqual(ids["total"], 2)
        self.assertEqual(client.get("/api/graph/triples/", {"min_confidence": "x"}).status_code, 400)
        self.assertEqual(client.get("/api/graph/triples/", {"order": "name"}).status_code, 400)
