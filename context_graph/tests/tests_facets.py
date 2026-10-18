"""P22b (2026-10-18): knowledge facets as tags derived from the schema's `facets:` mapping (schema v5).

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph.tests.tests_facets
"""

import io
import os
import unittest
from pathlib import Path

import yaml
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver, extraction, facets, registry, services, triples
from context_graph.models import CandidateTriple
from context_graph.schema import Schema, diff, is_empty_diff, validate_definition
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V4 = yaml.safe_load((SCHEMAS / "industrial_v4.yaml").read_text())
V5 = yaml.safe_load((SCHEMAS / "industrial_v5.yaml").read_text())
MAP = V5["facets"]
SEED = Path(extraction.__file__).parent / "seeds" / "extr01.yaml"
TEST_URI = os.getenv("GRAPH_TEST_URI")
FEEDER = "bsk:component:EXTR01/feeder"
ALARM = "bsk:alarm:EXTR01/SCREEN_BLOCK"


class SchemaV5Tests(SimpleTestCase):
    def test_v5_is_v4_plus_the_mapping(self):
        self.assertEqual(validate_definition(V5), [])
        change = diff(Schema.from_definition(V4), Schema.from_definition(V5))
        self.assertTrue(change["changed_facets"])
        self.assertEqual({k: v for k, v in change.items() if k != "changed_facets"},
                         {"added_entity_types": [], "removed_entity_types": [], "added_relationship_types": [],
                          "removed_relationship_types": [], "changed_pairs": {}, "changed_vocabulary": {}})
        self.assertFalse(is_empty_diff(change))
        self.assertEqual({k: v for k, v in V5.items() if k != "facets"}, V4)

    def test_a_bad_mapping_is_refused(self):
        for section, expected in (
                ({"types": {"Alarm": ["Safty"]}}, "'Safty' is not a facet"),
                ({"types": {"Gadget": ["Safety"]}}, "'Gadget' is not an entity type"),
                ({"subtypes": {"Procedure": {"Painting": ["Maintenance"]}}}, "'Painting' is not a subtype of Procedure"),
                ({"links": [{"rel": "NOPE", "end": "from", "facets": ["Safety"]}]}, "links[0] needs"),
                ({"links": [{"rel": "ADDRESSES", "end": "middle", "facets": ["Safety"]}]}, "links[0] needs"),
                ({"colours": {}}, "unknown key(s) colours")):
            errors = validate_definition({**V4, "facets": section})
            self.assertTrue(any(expected in e for e in errors), (section, errors))


class ComputeTests(SimpleTestCase):
    def test_by_type_subtype_default_and_link(self):
        self.assertEqual(facets.compute(MAP, "Alarm", "Device fault"), ["Troubleshooting"])
        self.assertEqual(facets.compute(MAP, "FailureMode"), ["Maintenance", "Troubleshooting"])
        self.assertEqual(facets.compute(MAP, "Procedure", "Calibration"), ["Setup", "Quality"])
        self.assertEqual(facets.compute(MAP, "Procedure", "Operation; Commissioning"), ["Operation", "Setup"])
        self.assertEqual(facets.compute(MAP, "Procedure", None), ["Maintenance"])          # D3: no subtype
        self.assertEqual(facets.compute(MAP, "Procedure", "Corrective", [("ADDRESSES", "from")]),
                         ["Maintenance", "Troubleshooting"])
        self.assertEqual(facets.compute(MAP, "Procedure", "Inspection", [("MITIGATED_BY", "to")]),
                         ["Maintenance", "Troubleshooting"])
        self.assertEqual(facets.compute(MAP, "Procedure", "Inspection", [("MITIGATED_BY", "from")]), ["Maintenance"])
        self.assertEqual(facets.compute(MAP, "Specification", "Electrical"), ["Installation", "Controls"])
        self.assertEqual(facets.compute(MAP, "Signal", "Temperature"), ["Controls"])
        for core in ("Component", "Asset", "Document", "DocumentSection", "Plant"):
            self.assertEqual(facets.compute(MAP, core, "Motor", [("HAS_ALARM", "from")]), [], core)
        self.assertEqual(facets.compute(None, "Alarm"), [])                                  # v4: no mapping

    def test_a_triples_facets_are_its_ends(self):
        t = CandidateTriple(mode="schema", subject_type="Procedure", subject_props={"subtype": "Corrective"},
                            predicate="ADDRESSES", object_type="Alarm", object_props={})
        self.assertEqual(facets.for_triple(MAP, t), ["Maintenance", "Troubleshooting"])
        t = CandidateTriple(mode="schema", subject_type="Component", predicate="HAS_SPEC", object_type="Specification",
                            object_props={"subtype": "Measurement"})
        self.assertEqual(facets.for_triple(MAP, t), ["Quality"])
        t = CandidateTriple(mode="schema", subject_type="Component", predicate="HAS_COMPONENT", object_type="Component")
        self.assertEqual(facets.for_triple(MAP, t), [])
        t.mode = "freeform"
        self.assertEqual(facets.for_triple(MAP, t), [])


class TripleFacetTests(LibraryTestCase):
    """Staged triples get their facets when staged and edited; review filters by facet. Graph off."""

    def setUp(self):
        super().setUp()
        self.doc, _ = self.upload()
        self.client = APIClient()

    def stage(self, **fields):
        defaults = dict(document=self.doc, mode="schema", subject_name="Feeder", subject_type="Component",
                        subject_id=FEEDER, subject_existing=True, predicate="HAS_PROCEDURE", object_name="Clean the screen",
                        object_type="Procedure", object_id="bsk:procedure:EXTR01/feeder/clean-the-screen",
                        object_props={"subtype": "Cleaning"}, page_start=3, page_end=3)
        defaults.update(fields)
        t = CandidateTriple(**defaults)
        facets.stamp([t], registry.active_schema())
        t.save()
        return t

    def test_staged_edited_and_filtered(self):
        registry.save_version(V5, note="v5")
        clean = self.stage()
        alarm = self.stage(predicate="HAS_ALARM", object_name="Screen blocked", object_type="Alarm", object_id=ALARM,
                           object_props={})
        part = self.stage(predicate="HAS_COMPONENT", object_name="Hopper", object_type="Component",
                          object_id="bsk:component:EXTR01/feeder/hopper", object_props={})
        self.assertEqual((clean.facets, alarm.facets, part.facets), (["Maintenance"], ["Troubleshooting"], []))

        def ids(facet):
            r = self.client.get("/api/graph/triples/", {"document": str(self.doc.id), "facet": facet})
            self.assertEqual(r.status_code, 200, r.content)
            return sorted(t["id"] for t in r.json()["triples"])
        self.assertEqual(ids("Maintenance"), [clean.pk])
        self.assertEqual(ids("Troubleshooting"), [alarm.pk])
        self.assertEqual(ids("none"), [part.pk])
        self.assertEqual(self.client.get("/api/graph/triples/", {"facet": "Gardening"}).status_code, 400)
        self.assertEqual(self.client.get("/api/graph/triples/", {"document": str(self.doc.id)}).json()["triples"][0]["facets"],
                         clean.facets)

        triples.edit(clean, {"object_type": "Alarm"})       # a reviewer's change re-derives them
        clean.refresh_from_db()
        self.assertEqual(clean.facets, ["Troubleshooting"])
        config = self.client.get("/api/graph/extract/config/").json()
        self.assertEqual(config["schema"]["facets"], list(facets.FACETS))
        registry.save_version(V4, note="v4")
        self.assertEqual(self.client.get("/api/graph/extract/config/").json()["schema"]["facets"], [])

    def test_the_command_needs_a_mapping(self):
        registry.save_version(V4, note="v4")
        with self.assertRaisesRegex(CommandError, "no facets: mapping"):
            call_command("graph_facets", stdout=io.StringIO())
        with self.assertRaisesRegex(CommandError, "is for a dry run"):
            call_command("graph_facets", apply=True, schema_file=str(SCHEMAS / "industrial_v5.yaml"), stdout=io.StringIO())


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class FacetGraphTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        self.client = APIClient()
        self.doc, _ = self.upload()

    def node_facets(self, node_id):
        with driver.session() as s:
            rec = s.run("MATCH (n:Entity {id: $id}) RETURN n.facets AS f", id=node_id).single()
        return rec["f"] if rec else "gone"

    def counts(self):
        with driver.session() as s:
            return s.run(facets_counts()).single().data()

    def stage(self, **fields):
        return TripleFacetTests.stage(self, **fields)

    def test_approve_and_delete_keep_the_tags_current(self):
        registry.save_version(V5, note="v5")
        proc = "bsk:procedure:EXTR01/feeder/clear-the-screen"
        a = self.stage(object_name="Clear the screen", object_id=proc, object_props={"subtype": "Corrective"})
        b = self.stage(subject_name="Clear the screen", subject_type="Procedure", subject_id=proc,
                       subject_props={"subtype": "Corrective"}, subject_existing=False, predicate="ADDRESSES",
                       object_name="Screen blocked", object_type="Alarm", object_id=ALARM, object_existing=True,
                       object_props={})
        r = self.client.post("/api/graph/triples/approve/", {"ids": [a.pk]}, format="json").json()
        self.assertEqual(r["approved"], [a.pk], r)
        self.assertEqual(self.node_facets(proc), ["Maintenance"])
        self.assertIsNone(self.node_facets(FEEDER))                               # core: no property
        r = self.client.post("/api/graph/triples/approve/", {"ids": [b.pk]}, format="json").json()
        self.assertEqual(r["approved"], [b.pk], r)
        self.assertEqual(self.node_facets(proc), ["Maintenance", "Troubleshooting"])   # it now answers an alarm
        self.client.post("/api/graph/triples/delete/", {"ids": [b.pk]}, format="json")
        self.assertEqual(self.node_facets(proc), ["Maintenance"])                 # the link went, so did the tag

    def test_the_command_dry_run_then_apply(self):
        registry.save_version(V5, note="v5")
        before = self.counts()
        out = io.StringIO()
        call_command("graph_facets", stdout=out)
        self.assertIn("Dry run: nothing written", out.getvalue())
        self.assertIn("Troubleshooting", out.getvalue())
        self.assertIsNone(self.node_facets(ALARM))                                # dry run wrote nothing
        call_command("graph_facets", apply=True, stdout=io.StringIO())
        self.assertEqual(self.node_facets(ALARM), ["Troubleshooting"])
        self.assertEqual(self.node_facets("bsk:signal:EXTR01/screw_speed_rpm"), ["Controls"])
        self.assertIsNone(self.node_facets(FEEDER))
        self.assertEqual(self.counts(), before)                                   # only tags changed
        out = io.StringIO()
        call_command("graph_facets", stdout=out)                                  # idempotent
        self.assertIn("nodes whose stored facets would change: 0", out.getvalue())


def facets_counts():
    from context_graph.management.commands.graph_facets import COUNTS
    return COUNTS
