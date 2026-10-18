"""P22a (2026-10-18): a part's ratings are Specifications (schema v4), apart from the signals' operating limits.

The overnight batch (2026-10-17) staged 29 `Component -HAS_LIMIT-> OperatingLimit` triples (the G120C's rated
currents, the gearbox's oil temperatures, the Optris and TM411 data sheets) that v3 refuses. With v4 a new run
stages them as `Component -HAS_SPEC-> Specification`, and "Make it a rating" converts the ones already staged.

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph.tests.tests_specifications
"""

import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from chat import asset_context
from chat.test_part_facts import build, part_context, MOTOR
from context_graph import driver, extraction, page_suggest, registry, services
from context_graph.identity import EntityIndex
from context_graph.models import CandidateTriple, PromptPreset
from context_graph.schema import Schema, SchemaError, diff, validate_definition
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V3 = yaml.safe_load((SCHEMAS / "industrial_v3.yaml").read_text())
V4 = yaml.safe_load((SCHEMAS / "industrial_v4.yaml").read_text())
SEED = Path(extraction.__file__).parent / "seeds" / "extr01.yaml"
TEST_URI = os.getenv("GRAPH_TEST_URI")
RTD = "bsk:component:EXTR01/die-head"
RANGE = {"kind": "specification", "low": "-200", "high": "600", "unit": "°C", "basis": "Table 10.1.2, Row 1"}


class SchemaV4Tests(SimpleTestCase):
    def test_v4_is_valid_and_only_adds_to_v3(self):
        self.assertEqual(validate_definition(V4), [])
        change = diff(Schema.from_definition(V3), Schema.from_definition(V4))
        self.assertEqual((change["removed_entity_types"], change["removed_relationship_types"]), ([], []))
        self.assertTrue(all(not c["removed"] for c in change["changed_pairs"].values()))
        self.assertEqual(change["added_entity_types"], ["Specification"])
        self.assertEqual(change["added_relationship_types"], ["HAS_SPEC"])
        self.assertEqual(change["changed_pairs"], {"DESCRIBES": {"added": [["DocumentSection", "Specification"]], "removed": []}})
        v3, v4 = Schema.from_definition(V3), Schema.from_definition(V4)
        self.assertEqual({k: v for k, v in v4.properties.items() if k != "Specification"}, v3.properties)

    def test_the_new_pair_and_a_part_still_has_no_limit(self):
        v4 = Schema.from_definition(V4)
        self.assertEqual(v4.check_relationship("Component", "HAS_SPEC", "Specification"), "HAS_SPEC")
        self.assertEqual(v4.check_relationship("Signal", "HAS_LIMIT", "OperatingLimit"), "HAS_LIMIT")
        with self.assertRaises(SchemaError):
            v4.check_relationship("Component", "HAS_LIMIT", "OperatingLimit")
        with self.assertRaises(SchemaError):
            v4.check_relationship("Signal", "HAS_SPEC", "Specification")
        self.assertEqual(v4.relationship_for("is rated", "Component", "Specification"), "HAS_SPEC")
        self.assertEqual(v4.subtype_for("Specification", "environmental"), "Environmental")
        self.assertEqual(set(v4.properties["Specification"]), {"value", "min", "max", "unit", "condition", "basis"})
        entity_types, relationships = extraction.render_schema(v4)
        self.assertIn("Specification", entity_types)
        self.assertIn("- HAS_SPEC: Component -> Specification", relationships)

    def test_a_limits_values_as_a_ratings(self):
        self.assertEqual(extraction.rating_props(RANGE), {"min": "-200", "max": "600", "unit": "°C", "basis": "Table 10.1.2, Row 1"})
        self.assertEqual(extraction.rating_props({"kind": "specification", "target": "±1.0 °C", "group": "LT15"}),
                         {"value": "±1.0 °C", "condition": "LT15"})
        self.assertEqual(extraction.rating_props({"low": "", "high": None}), {})
        v3, v4 = Schema.from_definition(V3), Schema.from_definition(V4)
        self.assertTrue(extraction.is_part_rating("Component", "HAS_LIMIT", "OperatingLimit", v4))
        self.assertFalse(extraction.is_part_rating("Component", "HAS_LIMIT", "OperatingLimit", v3))     # no Specification
        self.assertFalse(extraction.is_part_rating("Signal", "HAS_LIMIT", "OperatingLimit", v4))         # a process limit

    def test_a_rating_never_matches_a_process_limit(self):
        index = EntityIndex()
        index.add("bsk:operating-limit:EXTR01/barrel-zone-1-temperature-normal-range", "OperatingLimit",
                  "Barrel zone 1 temperature normal range")
        index.add("bsk:operating-limit:EXTR01/measuring-range", "OperatingLimit", "Measuring range")
        for name in ("Measuring range", "Barrel zone 1 temperature normal range"):
            m = index.resolve("Specification", name, "EXTR01")
            self.assertEqual((m.kind, m.candidates), ("new", []), name)
            self.assertTrue(m.id.startswith("bsk:specification:EXTR01/"))

    def test_the_spec_preset_and_its_pages(self):
        name, (mode, template) = "Specifications (v4)", extraction.EXTRA_PRESETS["Specifications (v4)"]
        self.assertEqual(mode, "schema")
        self.assertIn("HAS_SPEC", template)
        self.assertIn("never an OperatingLimit", template)
        self.assertIn("technical data and specifications", page_suggest.WHERE_BY_PRESET[name])
        self.assertIn("troubleshooting", page_suggest.WHERE)


class ChatRatingsTests(SimpleTestCase):
    def test_a_parts_ratings_are_in_its_facts(self):
        ctx = part_context()
        ctx["specifications"] = [
            {"id": "bsk:specification:EXTR01/main-motor/ambient", "label": "Specification", "name": "Ambient temperature",
             "properties": {"name": "Ambient temperature", "min": "-20", "max": "40", "unit": "°C", "subtype": "Environmental"},
             "components": [MOTOR]},
            {"id": "bsk:specification:EXTR01/main-motor/current", "label": "Specification", "name": "Rated current",
             "properties": {"name": "Rated current", "value": "16.5", "unit": "A", "condition": "at 2 kHz"},
             "components": [MOTOR]},
            {"id": "bsk:specification:EXTR01/main-motor/oil", "label": "Specification", "name": "Highest oil bath temperature",
             "properties": {"name": "Highest oil bath temperature", "max": "+65", "unit": "°C"}, "components": [MOTOR]},
        ]
        facts = build(ctx).facts
        self.assertEqual(asset_context.SECTION_TITLES["SPECIFICATIONS"], "Ratings of parts (datasheet values, not process limits)")
        self.assertIn("- Ambient temperature of Extruder motor: -20 to 40 °C (Environmental)", facts)
        self.assertIn("- Rated current of Extruder motor: 16.5 A (at 2 kHz)", facts)
        self.assertIn("- Highest oil bath temperature of Extruder motor: at most +65 °C", facts)
        self.assertFalse(any(" of Extruder motor: " in f for f in build(copy.deepcopy(part_context())).facts))


@override_settings(GRAPH_ENABLED=False)
class MakeRatingTests(LibraryTestCase):
    """Staging and the review action; the graph is off (new ids are proposed, nothing is matched)."""

    def setUp(self):
        super().setUp()
        self.doc, _ = self.upload()
        self.client = APIClient()

    def stage(self, **fields):
        defaults = dict(document=self.doc, mode="schema", subject_name="Die head", subject_type="Component",
                        subject_id=RTD, subject_existing=True, predicate="HAS_LIMIT", object_name="Measuring range",
                        object_type="OperatingLimit", object_id="bsk:operating-limit:EXTR01/measuring-range",
                        object_props=RANGE, issue="Component -HAS_LIMIT-> OperatingLimit is not allowed by the schema",
                        page_start=21, page_end=21, model="qwen", preset_name="Maintenance (v3)")
        defaults.update(fields)
        return CandidateTriple.objects.create(**defaults)

    def make(self, *ts):
        return self.client.post("/api/graph/triples/make-rating/", {"ids": [t.pk for t in ts]}, format="json")

    def test_needs_schema_v4(self):
        registry.save_version(V3, note="v3")
        r = self.make(self.stage())
        self.assertEqual(r.status_code, 400)
        self.assertIn("must have Specification (v4)", r.json()["error"])

    def test_a_rejected_part_limit_becomes_a_pending_rating(self):
        registry.save_version(V4, note="v4")
        rejected = self.stage(status=CandidateTriple.Status.REJECTED)
        signal_limit = self.stage(subject_name="Melt pressure", subject_type="Signal", subject_id="bsk:signal:EXTR01/p", issue="")
        approved = self.stage(status=CandidateTriple.Status.APPROVED)
        r = self.make(rejected, signal_limit, approved)
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual([t["id"] for t in body["triples"]], [rejected.pk])
        self.assertEqual(set(body["errors"]), {str(signal_limit.pk), str(approved.pk)})
        rejected.refresh_from_db()
        self.assertEqual((rejected.status, rejected.predicate, rejected.object_type, rejected.issue),
                         ("pending", "HAS_SPEC", "Specification", ""))
        self.assertEqual(rejected.object_props, {"min": "-200", "max": "600", "unit": "°C", "basis": "Table 10.1.2, Row 1"})
        self.assertTrue(rejected.object_id.startswith("bsk:specification:"), rejected.object_id)
        self.assertEqual((rejected.subject_id, rejected.page_start), (RTD, 21))          # its part and evidence kept
        self.assertIn("made a rating", rejected.auto_fix)

    def test_a_new_run_stages_a_part_limit_as_a_rating(self):
        schema = registry.save_version(V4, note="v4")
        t = {"subject": "Die head", "subject_type": "Component", "predicate": "has range", "object": "Measuring range",
             "object_type": "OperatingLimit", "object_properties": RANGE, "confidence": 0.9}
        window = type("W", (), {"page_start": 21, "page_end": 21, "section_path": [], "text": "Measuring range -200 to 600 °C"})()
        preset = PromptPreset(name="Maintenance (v3)", version=1)
        c = extraction._candidate(t, "schema", registry.active_schema(), EntityIndex(), "EXTR01", self.doc, None, window, 0,
                                  preset)
        self.assertEqual((c.predicate, c.object_type, c.issue), ("HAS_SPEC", "Specification", ""))
        self.assertEqual(c.object_props, {"min": "-200", "max": "600", "unit": "°C", "basis": "Table 10.1.2, Row 1"})
        self.assertIn("a part's rating", c.auto_fix)
        self.assertEqual(c.schema_version, schema.version)
        registry.save_version(V3, note="v3")
        c = extraction._candidate(t, "schema", registry.active_schema(), EntityIndex(), "EXTR01", self.doc, None, window, 0,
                                  preset)
        self.assertEqual(c.object_type, "OperatingLimit")
        self.assertTrue(c.issue)                       # v3 refuses it, as before


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class RatingGraphTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        registry.save_version(V4, note="v4")
        self.client = APIClient()
        self.doc, _ = self.upload()

    def test_an_approved_rating_is_on_its_part_and_in_the_asset_context(self):
        t = MakeRatingTests.stage(self, status=CandidateTriple.Status.REJECTED)
        self.assertEqual(self.client.post("/api/graph/triples/make-rating/", {"ids": [t.pk]}, format="json").status_code, 200)
        r = self.client.post("/api/graph/triples/approve/", {"ids": [t.pk]}, format="json").json()
        self.assertEqual(r["approved"], [t.pk], r)
        t.refresh_from_db()
        with driver.session() as s:
            row = s.run("MATCH (:Component {id: $c})-[:HAS_SPEC]->(sp:Specification) RETURN sp", c=RTD).single()
            limits = s.run("MATCH (:Component {id: $c})-[:HAS_LIMIT]->() RETURN count(*) AS n", c=RTD).single()["n"]
        self.assertEqual((row["sp"]["name"], row["sp"]["min"], row["sp"]["max"], row["sp"]["unit"]),
                         ("Measuring range", "-200", "600", "°C"))
        self.assertEqual(limits, 0)
        ctx = services.get_asset_context("bsk:asset:EXTR01")
        self.assertEqual([(sp["name"], sp["components"]) for sp in ctx["specifications"]], [("Measuring range", [RTD])])
        self.assertEqual(ctx["counts"]["specifications"], 1)
