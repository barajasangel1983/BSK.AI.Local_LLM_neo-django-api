"""Generate triples: pages, kept languages and the part a document describes (schema review step 2, 2026-10-15).
The DGX is mocked; the graph is disabled (the asset's context is a fixture)."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import extraction, registry
from context_graph.extraction import ExtractionError, parse_pages, validate_params
from context_graph.models import CandidateTriple
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.models import Document
from ingestion.tests.tests_library import LibraryTestCase

EN = "Inspect the motor at regular intervals, at least once a year. Check the bearings for noise and temperature. " * 4
IT = "Ispezionare il motore a intervalli regolari, almeno una volta all'anno. Controllare i cuscinetti per rumore e temperatura. " * 4
PAGES = [f"# Maintenance page {n}\n\n{EN if n in (1, 2, 4) else IT}" for n in range(1, 6)]
MANUAL = {"status": "success", "errors": [], "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(PAGES)}}

ASSET = "bsk:asset:EXTR01"
MOTOR = "bsk:component:EXTR01/main-motor"


def node(id_, label, name, **props):
    return {"id": id_, "label": label, "name": name, "properties": {"name": name, **props}}


CONTEXT = {
    "asset": node(ASSET, "Asset", "Extruder EXTR01"),
    "components": [{**node("bsk:component:EXTR01/main-drive", "Component", "Main Drive Motor"), "children": [
        {**node(MOTOR, "Component", "Extruder motor", component_type="Motor", manufacturer="ABB",
                model="M3BP 355MLA 4", part_number="3GBP352410-ADG"), "children": [], "signals": [], "alarms": []}],
        "signals": [node("bsk:signal:EXTR01/motor_power_kw", "Signal", "Motor power")], "alarms": []}],
    "signals": [],
    "alarms": [{**node("bsk:alarm:EXTR01/MOTOR_OVERLOAD", "Alarm", "Main drive motor overload", code="MOTOR_OVERLOAD"),
                "components": [], "procedures": []}],
    "procedures": [{**node("bsk:procedure:EXTR01/main-drive-overload-recovery", "Procedure", "Main drive overload recovery"),
                    "applies_to": [], "addresses": []}],
    "documents": [], "connections": [], "counts": {},
}


@override_settings(GRAPH_ENABLED=False)
class ExtractionScopeTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = MANUAL
        self.prompts = []

        def llm(messages):
            self.prompts.append(messages)
            return json.dumps({"triples": [
                {"subject": "M3BP 355MLA 4", "subject_type": "Component", "predicate": "HAS_PROCEDURE",
                 "object": "Motor general inspection", "object_type": "Procedure", "confidence": 0.9},
                {"subject": "Motor overheats", "subject_type": "FailureMode", "predicate": "INDICATED_BY",
                 "object": "MOTOR_OVERLOAD", "object_type": "Alarm", "confidence": 0.7}]})
        for target, kwargs in (("context_graph.extraction.call_llm", {"side_effect": llm}),
                               ("context_graph.services.get_asset_context", {"return_value": CONTEXT})):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        registry.save_version(yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v3.yaml").read_text()),
                              note="v3")
        self.doc, _ = library.add_document(SimpleUploadedFile("motors.pdf", b"%PDF motors"))
        self.drain()
        self.doc.refresh_from_db()
        self.client = APIClient()

    def extract(self, **body):
        r = self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "mode": "schema",
                                                     "chunking": {"strategy": "fixed", "params": {"size": 64, "overlap": 0}},
                                                     **body}, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        self.prompts.clear()
        self.drain()
        return [m[-1]["content"] for m in self.prompts]

    def test_pages_are_parsed_and_checked(self):
        self.assertEqual(parse_pages("14, 18-26; 3"), [[3, 3], [14, 14], [18, 26]])
        self.assertEqual(parse_pages(""), [])
        for bad in ("a-b", "5-2", "0", "1-2-3"):
            with self.assertRaises(ExtractionError):
                parse_pages(bad)
        self.assertEqual(validate_params({"pages": "2,4"})["pages"], [[2, 2], [4, 4]])
        with self.assertRaisesMessage(ExtractionError, "choose the asset"):
            validate_params({"anchor_id": MOTOR})
        with self.assertRaisesMessage(ExtractionError, "must be a component id"):
            validate_params({"anchor_id": ASSET, "asset_id": ASSET})

    def test_only_the_chosen_pages_and_the_kept_languages_are_read(self):
        everything = self.extract()
        self.assertTrue(any("Ispezionare" in t for t in everything))
        on_pages = self.extract(pages="1-3")
        self.assertTrue(on_pages and all("page 4" not in t and "page 5" not in t for t in on_pages))
        Document.objects.filter(pk=self.doc.pk).update(languages=["en"])
        english = self.extract()
        self.assertTrue(english and all("Ispezionare" not in t for t in english))
        self.assertLess(len(english), len(everything))
        all_languages = self.extract(follow_languages=False)
        self.assertEqual(len(all_languages), len(everything))
        self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "pages": "3", "mode": "schema"},
                             format="json")
        self.drain()
        self.doc.refresh_from_db()
        self.assertIn("no text on the chosen pages in the kept languages", self.doc.graph_error)

    def test_the_parts_configuration_is_in_the_anchor_block(self):
        """The options of the installed part (BOM column Configuration) tell the model which variants apply."""
        motor = CONTEXT["components"][0]["children"][0]
        configured = {**motor, "properties": {**motor["properties"], "configuration": "IE3, PTC thermistors, no brake",
                                              "configuration_basis": "assumed minimal, 2026-10-21"}}
        context = {**CONTEXT, "components": [{**CONTEXT["components"][0], "children": [configured]}]}
        with patch("context_graph.services.get_asset_context", return_value=context):
            text = extraction.anchor_context(ASSET, MOTOR)["text"]
        self.assertIn("extract only what applies to this part.\n- This part's configuration: IE3, PTC thermistors, no brake\n"
                      "- Existing entities", text)
        self.assertNotIn("This part's configuration", extraction.anchor_context(ASSET, MOTOR)["text"])

    def test_the_part_the_document_describes(self):
        self.extract(asset_id=ASSET, anchor_id=MOTOR, pages="1")
        system = self.prompts[0][0]["content"]
        self.assertIn('It describes the part "Extruder motor" of the asset "Extruder EXTR01" (Motor, ABB, M3BP 355MLA 4, '
                      'part number 3GBP352410-ADG)', system)
        self.assertIn("Alarms: MOTOR_OVERLOAD (Main drive motor overload)", system)
        self.assertIn("Signals: Motor power", system)
        self.assertIn("Procedures: Main drive overload recovery", system)
        staged = {(t.subject_name, t.predicate, t.object_name) for t in CandidateTriple.objects.filter(document=self.doc)}
        # The model's "M3BP 355MLA 4" is the anchor part, by its name.
        self.assertIn(("Extruder motor", "HAS_PROCEDURE", "Motor general inspection"), staged)
        self.assertIn(("Motor overheats", "INDICATED_BY", "MOTOR_OVERLOAD"), staged)
        self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "asset_id": ASSET,
                                                     "anchor_id": "bsk:component:EXTR01/nothing"}, format="json")
        self.drain()
        self.doc.refresh_from_db()
        self.assertIn("is not a part of", self.doc.graph_error)
