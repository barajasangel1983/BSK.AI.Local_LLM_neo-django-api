"""P22c-3c (2026-10-18): the readout parameter behind each PLC signal of a device's assembly, looked up and checked
before it is staged. Search, model and graph lookups mocked."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase, override_settings

from chat.retrieval import SearchResult, V2Chunk
from context_graph import extraction, maps, registry
from context_graph.models import CandidateTriple
from ingestion.tests.tests_library import LibraryTestCase

V6 = yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v6.yaml").read_text())
DRIVE = "bsk:component:EXTR01/main-drive-vfd"
SIGNALS = [{"id": "bsk:signal:EXTR01/screw_speed_rpm", "name": "Screw speed", "subtype": "Speed", "unit": "rpm", "on_parts": []},
           {"id": "bsk:signal:EXTR01/screw_torque_pct", "name": "Screw torque", "subtype": "Torque", "unit": "%", "on_parts": []},
           {"id": "bsk:signal:EXTR01/motor_power_kw", "name": "Motor power", "subtype": "Power", "unit": "kW", "on_parts": []}]
TEXT = ("01.01 Motor speed used. Estimated or measured motor speed. Unit: rpm. "
        "01.10 Motor torque. Motor torque in percent of the nominal motor torque. Unit: %. "
        "01.14 Output power. Converter output power. Unit: kW.")


class UnitTests(SimpleTestCase):
    def test_codes_by_their_numbers(self):
        self.assertEqual(maps.code_in("1.1", "see 01.01 Motor speed used"), "01.01")
        self.assertEqual(maps.code_in("1.1", "see 01.10 Motor torque"), "")             # not inside another code
        self.assertEqual(maps.code_in("01.14", "01.14 Output power"), "01.14")
        self.assertEqual(maps.code_in("p310", "set p0310 to the rated frequency"), "p0310")
        self.assertTrue(maps.NOT_A_READOUT.search("Torque ref ramp out"))
        self.assertFalse(maps.NOT_A_READOUT.search("Motor speed used"))

    def test_units(self):
        self.assertEqual(maps.unit_key("1/min"), "rpm")
        self.assertEqual(maps.unit_key(" kW "), "kw")
        self.assertNotEqual(maps.unit_key("%"), maps.unit_key("rpm"))


@override_settings(GRAPH_ENABLED=False)
class SuggestMapsTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        registry.save_version(V6, note="v6")
        self.doc, _ = self.upload("ACS880 firmware.pdf")
        chunk = V2Chunk(id="c1", text=TEXT, source=self.doc.filename, asset_id=self.doc.doc_key, page_start=150, page_end=150)
        self.answers = {"Speed": {"code": "01.01", "name": "Motor speed used", "unit": "rpm", "page": 150, "quote": "01.01 Motor speed used"},
                        "Torque": {"code": "01.10", "name": "Motor torque", "unit": "rpm", "page": 150, "quote": "x"},   # wrong unit
                        "Power": {"code": "01.99", "name": "Output power", "unit": "kW", "page": 150, "quote": "x"}}    # not in text

        def llm(messages):
            quantity = next(k for k in self.answers if f"— {k} in" in messages[-1]["content"])
            return json.dumps(self.answers[quantity])
        for target, kwargs in (("context_graph.maps.assembly_signals", {"return_value": SIGNALS}),
                               ("context_graph.maps.device", {"return_value": {"name": "Main drive frequency converter",
                                                                               "model": "ACS880-07", "maker": "ABB",
                                                                               "doc_keys": [self.doc.doc_key]}}),
                               ("chat.retrieval.search_v2", {"return_value": SearchResult(chunks=[chunk], candidates=1, reranker="ok", latency_ms=1)}),
                               ("context_graph.extraction.call_llm", {"side_effect": llm})):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def test_found_checked_and_staged(self):
        results = {r["signal"]: r for r in maps.suggest(DRIVE)}
        self.assertEqual((results["Screw speed"]["state"], results["Screw speed"]["code"]), ("found", "01.01"))
        self.assertIn("is not the signal's", results["Screw torque"]["reason"])
        self.assertIn("is not in the excerpts", results["Motor power"]["reason"])
        staged = {t.predicate: t for t in CandidateTriple.objects.filter(document=self.doc)}
        self.assertEqual(set(staged), {"HAS_PARAMETER", "MAPS_TO"})
        has, link = staged["HAS_PARAMETER"], staged["MAPS_TO"]
        self.assertEqual((has.object_type, has.object_props["code"], has.object_props["subtype"]), ("Parameter", "01.01", "Readout"))
        self.assertEqual((link.subject_id, link.object_id, link.issue, link.status),
                         (has.object_id, "bsk:signal:EXTR01/screw_speed_rpm", "", "pending"))
        self.assertEqual((link.page_start, link.evidence.count(), link.preset_name), (150, 1, "suggest_maps"))
        again = {r["signal"]: r for r in maps.suggest(DRIVE)}                              # a second run stages nothing new
        self.assertEqual(again["Screw speed"]["state"], "already")
        self.assertEqual(CandidateTriple.objects.filter(document=self.doc).count(), 2)

    def test_dry_run_stages_nothing(self):
        results = maps.suggest(DRIVE, dry_run=True)
        self.assertEqual([r["state"] for r in results], ["found", "none", "none"])
        self.assertFalse(CandidateTriple.objects.exists())
