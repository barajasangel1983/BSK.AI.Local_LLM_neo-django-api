"""P22c (2026-10-18): Controls and Setup — parameters, terminals and interfaces (schema v6), the third facet run.

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph.tests.tests_controls_setup
"""

import os
import unittest
from pathlib import Path

import yaml
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver, extraction, facets, page_suggest, registry, services
from context_graph.identity import EntityIndex
from context_graph.models import CandidateTriple, PromptPreset
from context_graph.schema import Schema, diff, validate_definition
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V5 = yaml.safe_load((SCHEMAS / "industrial_v5.yaml").read_text())
V6 = yaml.safe_load((SCHEMAS / "industrial_v6.yaml").read_text())
SEED = Path(extraction.__file__).parent / "seeds" / "extr01.yaml"
TEST_URI = os.getenv("GRAPH_TEST_URI")
WINDOW = type("W", (), {"page_start": 134, "page_end": 134, "section_path": ["5.4 Quick commissioning"],
                        "text": "p1082 Maximum speed: set the maximum speed of the motor, default 1500 rpm."})()
PRESET = PromptPreset(name="Controls & Setup (v6)", version=1)


def triple(subject, stype, predicate, obj, otype, oprops=None, osub="", sprops=None, ssub=""):
    return {"subject": subject, "subject_type": stype, "subject_subtype": ssub, "subject_properties": sprops or {},
            "predicate": predicate, "object": obj, "object_type": otype, "object_subtype": osub,
            "object_properties": oprops or {}, "confidence": 0.9}


class SchemaV6Tests(SimpleTestCase):
    def test_v6_only_adds_to_v5(self):
        self.assertEqual(validate_definition(V6), [])
        change = diff(Schema.from_definition(V5), Schema.from_definition(V6))
        self.assertEqual(change["added_entity_types"], ["IOPoint", "Interface", "Parameter"])
        self.assertEqual(change["added_relationship_types"],
                         ["CARRIES", "COMMUNICATES_VIA", "CONFIGURED_BY", "HAS_IO", "HAS_PARAMETER", "MAPS_TO", "SETS"])
        self.assertEqual((change["removed_entity_types"], change["removed_relationship_types"], change["changed_pairs"]),
                         ([], [], {}))
        self.assertTrue(all(not c["removed"] for c in change["changed_vocabulary"].values()))

    def test_their_facets(self):
        m = V6["facets"]
        self.assertEqual(facets.compute(m, "Parameter", "Setting"), ["Setup"])
        self.assertEqual(facets.compute(m, "Parameter", "Readout"), ["Controls"])
        self.assertEqual(facets.compute(m, "Parameter"), ["Setup"])
        self.assertEqual(facets.compute(m, "IOPoint", "Digital input"), ["Installation", "Controls"])
        self.assertEqual(facets.compute(m, "Interface", "Fieldbus"), ["Controls"])
        self.assertEqual(facets.compute(m, "Alarm"), ["Troubleshooting"])                 # v5's rules kept

    def test_suggested_links_are_never_offered_to_the_model(self):
        v6 = Schema.from_definition(V6)
        _, rels = extraction.extractable(v6)
        self.assertIn("HAS_PARAMETER", rels)
        self.assertNotIn("MAPS_TO", rels)
        self.assertNotIn("CARRIES", rels)
        _, text = extraction.render_schema(v6)
        self.assertNotIn("MAPS_TO", text)

    def test_the_prompt_and_its_pages(self):
        mode, template = extraction.EXTRA_PRESETS["Controls & Setup (v6)"]
        self.assertEqual(mode, "schema")
        for phrase in ("HAS_PARAMETER", "CONFIGURED_BY", "Not every parameter mentioned in a\n  function diagram",
                       "Never create a Signal, an Alarm, a Specification"):
            self.assertIn(phrase, template)
        self.assertIn("terminal strips", page_suggest.WHERE_BY_PRESET["Controls & Setup (v6)"])


@override_settings(GRAPH_ENABLED=False, GRAPH_PART_SCOPED_IDS=True)
class StagingTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        registry.save_version(V6, note="v6")
        self.doc, _ = self.upload()
        self.schema = registry.active_schema()
        self.index = EntityIndex()
        self.index.add("bsk:asset:EXTR01", "Asset", "Extruder EXTR01")
        self.index.add("bsk:component:EXTR01/feeder-screw-drive", "Component", "Feed screw drive")
        self.index.add("bsk:component:EXTR01/cutter-drive", "Component", "Cutter drive")
        self.part = self.index.part_scope("bsk:component:EXTR01/feeder-screw-drive", "EXTR01")

    def stage(self, t, part="default"):
        return extraction._candidate(t, "schema", self.schema, self.index, "EXTR01", self.doc, None, WINDOW, 0, PRESET,
                                     part=self.part if part == "default" else part)

    def test_a_parameters_id_is_its_code_under_the_part(self):
        a = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Maximum speed", "Parameter",
                              {"code": "p1082", "unit": "rpm", "default": "1500"}, "Setting"))
        self.assertEqual((a.issue, a.object_id), ("", "bsk:parameter:EXTR01/feeder-screw-drive/p1082"))
        self.assertEqual(a.object_props, {"subtype": "Setting", "code": "p1082", "unit": "rpm", "default": "1500"})
        b = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Max. speed", "Parameter", {"code": "p1082"}))
        self.assertEqual(b.object_id, a.object_id)                                   # another name, the same parameter
        self.index.add(a.object_id, "Parameter", "Maximum speed", tags=["p1082"])     # once approved: the feed screw
        self.index.owners[a.object_id] = {"bsk:component:EXTR01/feeder-screw-drive"}  # drive's (HAS_PARAMETER)
        c = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Speed limit", "Parameter", {"code": "p1082"}))
        self.assertEqual((c.object_id, c.object_existing), (a.object_id, True))
        cutter = self.index.part_scope("bsk:component:EXTR01/cutter-drive", "EXTR01")
        d = self.stage(triple("Cutter drive", "Component", "HAS_PARAMETER", "Maximum speed", "Parameter", {"code": "p1082"}),
                       part=cutter)
        self.assertEqual((d.object_id, d.object_existing), ("bsk:parameter:EXTR01/cutter-drive/p1082", False))

    def test_terminals_links_and_facets(self):
        t = self.stage(triple("DI 0", "IOPoint", "CONFIGURED_BY", "ON/OFF1 command", "Parameter", {"code": "p0840"},
                              "Setting", {"connector": "-X138", "terminal": "5", "signal_type": "24 V DC"}, "Digital input"))
        self.assertEqual(t.issue, "")
        self.assertEqual(t.subject_id, "bsk:io-point:EXTR01/feeder-screw-drive/X138-5")
        facets.stamp([t], self.schema)
        self.assertEqual(t.facets, ["Installation", "Controls", "Setup"])
        m = self.stage(triple("Actual speed", "Parameter", "MAPS_TO", "Screw speed", "Signal", sprops={"code": "r0021"}))
        self.assertIn("suggested by the Studio", m.issue)
        self.assertEqual([r["key"] for r in extraction.facet_runs_available()], ["maintenance", "ratings", "controls"])
        registry.save_version(V5, note="v5")
        self.assertEqual([r["key"] for r in extraction.facet_runs_available()], ["maintenance", "ratings"])


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j", GRAPH_PART_SCOPED_IDS=True,
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class ParameterGraphTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        registry.save_version(V6, note="v6")
        self.client = APIClient()
        self.doc, _ = self.upload()

    def test_an_approved_parameter_is_found_again_by_its_code(self):
        schema = registry.active_schema()
        index = EntityIndex.load()
        part = index.part_scope("bsk:component:EXTR01/feeder", "EXTR01")
        t = extraction._candidate(triple("Feeder", "Component", "HAS_PARAMETER", "Maximum speed", "Parameter",
                                         {"code": "p1082", "unit": "rpm"}, "Setting"),
                                  "schema", schema, index, "EXTR01", self.doc, None, WINDOW, 0, PRESET, part=part)
        facets.stamp([t], schema)
        t.save()
        r = self.client.post("/api/graph/triples/approve/", {"ids": [t.pk]}, format="json").json()
        self.assertEqual(r["approved"], [t.pk], r)
        with driver.session() as s:
            node = s.run("MATCH (p:Parameter {id: $id}) RETURN p", id="bsk:parameter:EXTR01/feeder/p1082").single()["p"]
        self.assertEqual((node["name"], node["code"], node["facets"]), ("Maximum speed", "p1082", ["Setup"]))
        again = extraction._candidate(triple("Feeder", "Component", "HAS_PARAMETER", "Speed max", "Parameter", {"code": "p1082"}),
                                      "schema", schema, EntityIndex.load(), "EXTR01", self.doc, None, WINDOW, 0, PRESET,
                                      part=part)
        self.assertEqual((again.object_id, again.object_existing), ("bsk:parameter:EXTR01/feeder/p1082", True))


class CodeTests(SimpleTestCase):
    def test_one_form_of_a_code(self):
        n = extraction.normalize_code
        self.assertEqual(n("P310"), ("p0310", ""))
        self.assertEqual(n("p0310"), ("p0310", ""))
        self.assertEqual(n("P96"), ("p0096", ""))
        self.assertEqual(n("p1070[0]"), ("p1070", "0"))
        self.assertEqual(n("p1071[C]"), ("p1071", "C"))
        self.assertEqual(n("p1070-0"), ("p1070", "0"))
        self.assertEqual(n("r0021"), ("r0021", ""))
        self.assertEqual(n("01.01"), ("01.01", ""))          # group.index: two digits each
        self.assertEqual(n("1.1"), ("01.01", ""))            # a parsed table drops the zeros
        self.assertEqual(n("46.3"), ("46.03", ""))
        self.assertEqual(n("1.10"), ("01.10", ""))
        self.assertEqual(n("2-03"), ("2-03", ""))            # anything else, as written
        self.assertEqual(n("KP"), ("KP", ""))


@override_settings(GRAPH_ENABLED=False, GRAPH_PART_SCOPED_IDS=True)
class NormalizedStagingTests(StagingTests):
    def test_two_forms_one_parameter(self):
        a = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Rated motor frequency", "Parameter",
                              {"code": "P310"}))
        b = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Rated frequency", "Parameter",
                              {"code": "p0310"}))
        self.assertEqual(a.object_id, b.object_id)
        self.assertEqual(a.object_props["code"], "p0310")
        c = self.stage(triple("Feed screw drive", "Component", "HAS_PARAMETER", "Speed setpoint", "Parameter",
                              {"code": "p1070[0]"}))
        self.assertEqual((c.object_props["code"], c.object_props["data_set"], c.object_id.rsplit("/", 1)[-1]),
                         ("p1070", "0", "p1070"))
