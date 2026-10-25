"""P22d (2026-10-25): the Safety facet. Schema v8 is additive; the Safety prompt's sentence is checked word for word
in its source; a Safety fact is approved one at a time and only with an exact quote."""

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings

from context_graph import extraction, facets, quotes, registry, schema as schema_module, triples
from context_graph.models import CandidateTriple, Evidence
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V7 = yaml.safe_load((SCHEMAS / "industrial_v7.yaml").read_text())
V8 = yaml.safe_load((SCHEMAS / "industrial_v8.yaml").read_text())

WARNING = ("## WARNING\n\nHazardous voltage at the DC link.\n\nThe DC link capacitors hold a dangerous voltage after the "
           "supply is switched off. Contact causes death or serious injury. Wait 5 minutes after switching off before "
           "you open the device.\n\n<!-- image -->\n\nOnly qualified personnel may install the device.")


class SchemaV8Tests(SimpleTestCase):
    def test_v8_only_adds_to_v7(self):
        self.assertEqual(schema_module.validate_definition(V8), [])
        d = schema_module.diff(schema_module.Schema.from_definition(V7, 7), schema_module.Schema.from_definition(V8, 8))
        self.assertEqual((d["removed_entity_types"], d["removed_relationship_types"]), ([], []))
        self.assertEqual(sorted(d["added_entity_types"]), ["Hazard", "SafetyFunction", "SafetyMeasure"])
        self.assertTrue(all(not c["removed"] for c in d["changed_pairs"].values()))

    def test_the_safety_facet(self):
        mapping = V8["facets"]
        for label in ("Hazard", "SafetyMeasure", "SafetyFunction"):
            self.assertEqual(facets.compute(mapping, label), ["Safety"])
        self.assertIn("Safety", facets.compute(mapping, "Procedure", "Inspection", [("REQUIRES", "from")]))
        self.assertNotIn("Safety", facets.compute(mapping, "Procedure", "Inspection"))


class SentenceTests(SimpleTestCase):
    def test_word_for_word_with_whitespace_markers_and_entities_aside(self):
        found = quotes.find_sentence(WARNING, "Wait 5 minutes after switching off before you open the device.")
        self.assertEqual(WARNING[found[1]:found[2]], "Wait 5 minutes after switching off before you open the device.")
        text = "Contact\n  causes death <!-- page-break --> or serious injury. Above &gt; 60 °C derate."
        self.assertEqual(quotes.find_sentence(text, "Contact causes death or serious injury.")[0], "exact")
        self.assertEqual(quotes.find_sentence(text, "Above > 60 °C derate.")[0], "exact")

    def test_a_paraphrase_is_not_found_and_a_repeat_is_ambiguous(self):
        self.assertEqual(quotes.find_sentence(WARNING, "Wait five minutes before opening the device."),
                         ("not_found", None, None))
        self.assertEqual(quotes.find_sentence(WARNING + "\n\n" + WARNING, "Only qualified personnel may install the device.")[0],
                         "ambiguous")

    def test_the_parser_keeps_the_sentence(self):
        raw = json.dumps({"triples": [{"subject": "Drive", "subject_type": "Component", "predicate": "HAS_HAZARD",
                                       "object": "Electric shock from the DC link", "object_type": "Hazard",
                                       "quote": " Contact causes death or serious injury. ", "confidence": 0.9}]})
        self.assertEqual(extraction.parse_triples(raw)[0]["quote"], "Contact causes death or serious injury.")


@override_settings(GRAPH_ENABLED=False)
class SafetyStagingTests(LibraryTestCase):
    """A Safety run's quotes come from the model's sentence, located in the window's span."""

    def setUp(self):
        super().setUp()
        md = f"# 1. Safety\n\n{WARNING}\n\n{PAGE_BREAK}\n\n# 2. Terminals\n\nTerminal 5 is DI 0."
        self.mock_convert.return_value = {"status": "success", "errors": [], "document": {"md_content": md}}
        answer = {"triples": [
            {"subject": "Converter", "subject_type": "Component", "predicate": "HAS_HAZARD",
             "object": "Electric shock from the DC link capacitors", "object_type": "Hazard",
             "object_properties": {"severity": "WARNING"},
             "quote": "The DC link capacitors hold a dangerous voltage after the supply is switched off.", "confidence": 0.9},
            {"subject": "Electric shock from the DC link capacitors", "subject_type": "Hazard", "predicate": "PROTECTED_BY",
             "object": "Wait 5 minutes after switching off", "object_type": "SafetyMeasure",
             "object_properties": {"value": "5", "unit": "min"},
             "quote": "Wait five minutes before opening.", "confidence": 0.9}]}
        p = patch("context_graph.extraction.call_llm", return_value=json.dumps(answer))
        p.start()
        self.addCleanup(p.stop)
        registry.save_version(V8, note="v8")
        self.doc, _ = library.add_document(SimpleUploadedFile("drive.pdf", b"%PDF drive"))
        self.drain()
        self.doc.refresh_from_db()

    def test_the_sentence_is_checked_word_for_word(self):
        params = extraction.validate_params({"pages": "1"})
        params["chunking"] = {"strategy": "structure", "params": {"max_tokens": 400, "overlap_sentences": 0}}
        extraction._extract(None, self.doc, params)
        hazard = CandidateTriple.objects.get(document=self.doc, predicate="HAS_HAZARD")
        self.assertIn("Safety", hazard.facets)
        e = hazard.evidence.get()
        text = quotes.snapshot_text(self.doc, e.text_hash)
        self.assertEqual((e.quote_status, text[e.quote_start:e.quote_end]),
                         ("exact", "The DC link capacitors hold a dangerous voltage after the supply is switched off."))
        measure = CandidateTriple.objects.get(document=self.doc, predicate="PROTECTED_BY").evidence.get()
        self.assertEqual((measure.quote_status, measure.quote_start), ("not_found", None))     # a paraphrase


    def test_a_run_keeps_its_own_facts_links_to_types_another_run_owns(self):
        answer = {"triples": [
            {"subject": "Safe Torque Off", "subject_type": "SafetyFunction", "predicate": "HAS_IO", "object": "DI 0",
             "object_type": "IOPoint", "quote": "Terminal 5 is DI 0.", "confidence": 0.9},
            {"subject": "Converter", "subject_type": "Component", "predicate": "HAS_IO", "object": "DI 1",
             "object_type": "IOPoint", "confidence": 0.9}]}
        params = extraction.validate_params({"pages": "1-2"})
        params.update(leave_types=["IOPoint"], own_types=["Hazard", "SafetyMeasure", "SafetyFunction"])
        with patch("context_graph.extraction.call_llm", return_value=json.dumps(answer)):
            extraction._extract(None, self.doc, params)
        self.assertEqual(list(CandidateTriple.objects.filter(document=self.doc).values_list("subject_name", flat=True)),
                         ["Safe Torque Off"])


@override_settings(GRAPH_ENABLED=False)
class SafetyApprovalTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.doc, _ = self.upload()

    def stage(self, facets_=("Safety",), quote_status="exact", **fields):
        t = CandidateTriple.objects.create(
            document=self.doc, mode="schema", subject_name="Converter", subject_type="Component",
            subject_id="bsk:component:EXTR01/converter", predicate="HAS_HAZARD", object_name="Hot surface",
            object_type="Hazard", object_id="bsk:hazard:EXTR01/converter/hot-surface", facets=list(facets_), **fields)
        t.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, quote_status=quote_status))
        return t

    def test_safety_facts_are_approved_alone_and_with_an_exact_quote(self):
        a, b = self.stage(), self.stage()
        result = triples.approve([a, b])
        self.assertEqual((result["approved"], result["errors"]), ([], {a.pk: triples.SAFETY_ALONE, b.pk: triples.SAFETY_ALONE}))
        weak = self.stage(quote_status="not_found")
        self.assertEqual(triples.approve([weak])["errors"], {weak.pk: triples.SAFETY_QUOTE})
        self.assertEqual(triples.safety_blocker(a, alone=True), "")

    def test_other_facts_are_not_concerned(self):
        t = self.stage(facets_=("Maintenance",), quote_status="")
        self.assertEqual(triples.safety_blocker(t, alone=False), "")


class MeasureValueTests(SimpleTestCase):
    def test_a_measure_keeps_a_number_and_drops_a_sentence(self):
        self.assertEqual(extraction.without_wordy_value({"value": "5", "unit": "min"}), {"value": "5", "unit": "min"})
        self.assertEqual(extraction.without_wordy_value({"value": 0.4, "unit": "m"}), {"value": 0.4, "unit": "m"})
        self.assertEqual(extraction.without_wordy_value({"value": "remove this before starting up", "subtype": "Procedure step"}),
                         {"subtype": "Procedure step"})


class SignalWordTests(SimpleTestCase):
    def test_the_notice_above_the_text(self):
        text = ("## DANGER\n\n## Hazardous voltage\n\nElectric motors have high voltages.\n\n- Switch off first.\n\n"
                "## Health hazard\n\nChemical substances can present a health risk.\n\n"
                "**NOTICE**\n\nCondensation holes left open reduce the degree of protection.")
        at = text.index
        self.assertEqual(quotes.signal_word(text, at("Electric motors")), "DANGER")
        self.assertEqual(quotes.signal_word(text, at("Switch off")), "DANGER")
        self.assertEqual(quotes.signal_word(text, at("Chemical")), "")        # another section's title comes first
        self.assertEqual(quotes.signal_word(text, at("Condensation")), "NOTICE")
        self.assertEqual(quotes.signal_word(WARNING, WARNING.index("Wait 5")), "WARNING")


class AbbreviationTests(SimpleTestCase):
    def test_a_name_and_its_initials_are_one_name(self):
        names = extraction.abbreviations("The Safe Torque Off (STO) function and Safely-Limited Speed (SLS). See p1082 (MAX).")
        self.assertEqual({names[k] for k in ("safe torque off", "sto", "safe torque off sto")}, {"Safe Torque Off (STO)"})
        self.assertEqual(names["sls"], "Safely-Limited Speed (SLS)")
        self.assertNotIn("max", names)
        self.assertNotIn("sto", extraction.abbreviations("Safe Torque Off (STO) ... Store Transfer Order (STO)"))
