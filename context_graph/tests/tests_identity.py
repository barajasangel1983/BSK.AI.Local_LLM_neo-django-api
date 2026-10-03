"""P8b: identity resolution, possible matches, schema subtypes / relationship aliases, units.

Integration (the brief's lab fixture, schema v2) needs the throwaway Neo4j:
    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver, registry, services, structured, triples
from context_graph.identity import EntityIndex, tag_key, tags_in
from context_graph.models import CandidateTriple
from context_graph.schema import Schema, validate_definition
from context_graph.units import normalize_unit
from ingestion import library
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

TEST_URI = os.getenv("GRAPH_TEST_URI")
SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"
V2 = yaml.safe_load((SCHEMAS / "industrial_v2.yaml").read_text())


def index():
    i = EntityIndex()
    i.add("bsk:asset:EXTR01", "Asset", "Extruder EXTR01")
    i.add("bsk:component:EXTR01/M101", "Component", "Main Motor", aliases=["Extruder main drive motor"])
    i.add("bsk:component:EXTR01/VFD101", "Component", "Variable Frequency Drive")
    i.add("bsk:component:EXTR01/barrel-zone-1", "Component", "Barrel Zone 1")
    i.add("bsk:component:EXTR01/barrel-zone-2", "Component", "Barrel Zone 2")
    i.add("bsk:component:EXTR02/M101", "Component", "Main Motor")            # same tag, other asset
    i.add("bsk:alarm:EXTR01/DIE_PLUG", "Alarm", "DIE_PLUG", tags=["DIE_PLUG"])
    i.add("bsk:signal:EXTR01/die_pressure_bar", "Signal", "Die pressure")
    return i


class TagTests(SimpleTestCase):
    def test_tags(self):
        self.assertEqual(tags_in("Motor M101 (VFD-101 driven)"), ["M101", "VFD-101"])
        self.assertEqual(tags_in("DIE_PLUG alarm"), ["DIE_PLUG"])
        self.assertEqual(tags_in("signal die_pressure_bar"), ["die_pressure_bar"])
        self.assertEqual(tags_in("the main motor"), [])
        self.assertEqual(tags_in("p. 3"), [])                                 # too short to be a tag
        self.assertEqual({tag_key(t) for t in ("VFD-101", "vfd_101", "VFD 101")}, {"VFD101"})


class ResolverTests(SimpleTestCase):
    def r(self, label, name, scope="EXTR01", **kw):
        return index().resolve(label, name, scope, **kw)

    def test_order(self):
        self.assertEqual((m := self.r("Component", "bsk:component:EXTR01/M101")).kind, "id")
        self.assertEqual(self.r("Component", "M101").kind, "id")          # its proposed id already exists
        for name in ("Motor M101", "Main Motor (M101)", "m-101 motor"):
            m = self.r("Component", name)
            self.assertEqual((m.id, m.kind), ("bsk:component:EXTR01/M101", "tag"), name)
        self.assertEqual(self.r("Component", "M101", scope="EXTR02").id, "bsk:component:EXTR02/M101")  # asset scope first
        self.assertEqual(self.r("Component", "Main Motor").kind, "name")          # first exact name in this label
        self.assertEqual(self.r("Component", "extruder main drive motor").kind, "alias")
        self.assertEqual(self.r("Alarm", "DIE_PLUG alarm").id, "bsk:alarm:EXTR01/DIE_PLUG")
        self.assertEqual(self.r("Signal", "die_pressure_bar").kind, "id")           # historian column = its id key
        self.assertEqual(self.r("Signal", "the die_pressure_bar reading").kind, "tag")

    def test_fuzzy_is_only_a_suggestion_and_numbers_never_merge(self):
        m = self.r("Component", "Barrel Zone 1 heater")
        self.assertEqual(m.kind, "possible")
        self.assertEqual(m.id, "bsk:component:EXTR01/Barrel-Zone-1-heater")       # proposed, not merged
        self.assertEqual(m.candidates[0]["id"], "bsk:component:EXTR01/barrel-zone-1")
        zone3 = self.r("Component", "Barrel Zone 3")
        self.assertNotIn("barrel-zone", zone3.id.rsplit("/", 1)[0] + "x")          # never auto-assigned
        self.assertIn(zone3.kind, ("possible", "new"))
        self.assertEqual(self.r("Component", "Hopper").kind, "new")
        self.assertEqual(self.r("Component", "Barrel Zone 1 heater", fuzzy=False).kind, "new")

    def test_explicit_tag_from_an_import_key(self):
        m = self.r("Component", "Drive", tag="vfd_101")
        self.assertEqual((m.id, m.kind), ("bsk:component:EXTR01/VFD101", "tag"))


class SchemaVocabularyTests(SimpleTestCase):
    def test_v2_loads_and_answers(self):
        schema = Schema.from_definition(V2)
        self.assertEqual(schema.subtype_for("Component", "overload  RELAY"), "Overload relay")
        self.assertIsNone(schema.subtype_for("Component", "Spaceship"))
        self.assertEqual(schema.relationship_for("runs", "Component", "Component"), "DRIVES")
        self.assertEqual(schema.relationship_for("Measured by", "Component", "Signal"), "MONITORED_BY")
        self.assertIsNone(schema.relationship_for("runs", "Signal", "Component"))   # pair not allowed
        self.assertEqual(Schema.from_yaml(SCHEMAS / "industrial_v1.yaml").subtypes, {})   # v1 still valid

    def test_validation_and_diff(self):
        bad = json.loads(json.dumps(V2))
        bad["relationship_types"]["CONTROLS"]["aliases"].append("drives")             # same phrase, two types
        bad["entity_types"]["Signal"]["subtypes"] = "Temperature"
        errors = " | ".join(validate_definition(bad))
        self.assertIn("alias 'drives' is used by both DRIVES and CONTROLS", errors)
        self.assertIn("subtypes must be a list", errors)

    def test_units(self):
        self.assertEqual([normalize_unit(u) for u in ("degC", "kg/hr", "bar(g)", "KW", "furlongs")],
                         ["°C", "kg/h", "bar", "kW", "furlongs"])


def fake_llm(messages):
    return json.dumps({"triples": [
        {"subject": "VFD101", "subject_type": "Component", "subject_subtype": "drive", "predicate": "drives",
         "object": "Motor M101", "object_type": "Component", "object_subtype": "MOTOR", "confidence": 1},
        {"subject": "Barrel Zone 1 heater", "subject_type": "Component", "predicate": "MONITORED_BY",
         "object": "Die pressure", "object_type": "Signal", "confidence": 1},
    ]})


@override_settings(GRAPH_ENABLED=False)
class StagingTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        for target, value in (("context_graph.extraction.call_llm", None),
                              ("context_graph.extraction.EntityIndex.load", index())):
            p = patch(target, side_effect=fake_llm) if value is None else patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("context_graph.registry.active_schema", return_value=Schema.from_definition(V2, version=2))
        p.start()
        self.addCleanup(p.stop)
        self.doc, _ = self.upload()
        self.drain()
        library.enqueue(self.doc, Job.Kind.EXTRACT, {"mode": "schema", "asset_id": "bsk:asset:EXTR01"})
        self.drain()

    def test_synonyms_subtypes_and_matches_are_staged(self):
        drives = self.doc.triples.get(predicate="DRIVES")                      # "drives" -> DRIVES (allowed pair)
        self.assertEqual((drives.subject_id, drives.subject_match, drives.subject_props),
                         ("bsk:component:EXTR01/VFD101", "id", {"subtype": "Drive"}))   # proposed id exists
        self.assertEqual((drives.object_id, drives.object_match, drives.object_props),
                         ("bsk:component:EXTR01/M101", "tag", {"subtype": "Motor"}))
        self.assertEqual(drives.issue, "")
        heater = self.doc.triples.get(predicate="MONITORED_BY")
        self.assertEqual(heater.subject_match, "possible")
        self.assertEqual(heater.subject_candidates[0]["id"], "bsk:component:EXTR01/barrel-zone-1")

    def test_possible_match_blocks_approval_until_resolved(self):
        heater = self.doc.triples.get(predicate="MONITORED_BY")
        self.assertIn("may be an existing entity", triples.unresolved(heater))
        result = triples.approve([heater])                                       # graph disabled, but blocked first
        self.assertIn("pick one of the suggestions", result["errors"][heater.pk])

        client = APIClient()
        body = client.get(f"/api/graph/triples/?document={self.doc.id}&q=heater").json()["triples"][0]
        # Only zone 1 is suggested: "zone 2" has a different number, so it scores too low.
        self.assertEqual((body["subject"]["match"], len(body["subject"]["candidates"])), ("possible", 1))
        r = client.patch(f"/api/graph/triples/{heater.pk}/", {"subject_id": "bsk:component:EXTR01/barrel-zone-1"},
                         format="json").json()
        self.assertEqual((r["subject"]["match"], r["subject"]["existing"], r["subject"]["candidates"]), ("id", True, []))
        heater.refresh_from_db()
        self.assertEqual(triples.unresolved(heater), "")

        drives = self.doc.triples.get(predicate="DRIVES")
        CandidateTriple.objects.filter(pk=drives.pk).update(object_match="possible")
        r = client.patch(f"/api/graph/triples/{drives.pk}/", {"object_match": "new"}, format="json").json()
        self.assertEqual((r["object"]["match"], r["object"]["existing"]), ("new", False))
        self.assertEqual(client.patch(f"/api/graph/triples/{drives.pk}/", {"object_match": "tag"},
                                      format="json").status_code, 400)

    def test_extract_config_lists_vocabulary(self):
        body = APIClient().get("/api/graph/extract/config/").json()["schema"]
        self.assertIn("Overload relay", body["subtypes"]["Component"])
        self.assertIn("runs", next(r for r in body["relationships"] if r["name"] == "DRIVES")["aliases"])


@override_settings(GRAPH_ENABLED=False)
class ImportMatchTests(LibraryTestCase):
    def test_import_keys_are_tags_and_names_may_be_possible(self):
        csv = b"Item,Description,Type,Unit\nvfd_101,Drive,drive,degC\nBarrel Zone 1 heater,Heater,heater,C\n"
        mapping = {"entities": [{"key": "asset", "type": "Asset", "scope": True},
                                {"key": "c", "type": "Component", "id_column": "Item", "name_column": "Item",
                                 "properties": {"subtype": "Type", "unit": "Unit"}}],
                   "relationships": [{"from": "asset", "type": "HAS_COMPONENT", "to": "c"}]}
        with override_settings(GRAPH_DATA_BASE=os.path.join(self.tmp, "gd")), \
                patch("context_graph.structured.EntityIndex.load", return_value=index()), \
                patch("context_graph.registry.active_schema", return_value=Schema.from_definition(V2, version=2)):
            df, _ = structured.add_data_file(SimpleUploadedFile("c.csv", csv))
            result = structured.build(df, mapping, "bsk:asset:EXTR01")
        vfd, heater = sorted(result.triples, key=lambda t: t.row_number)
        self.assertEqual((vfd.object_id, vfd.object_match, vfd.object_props),
                         ("bsk:component:EXTR01/VFD101", "tag", {"subtype": "Drive", "unit": "°C"}))
        self.assertEqual((heater.object_match, heater.object_props["unit"]), ("possible", "°C"))
        self.assertEqual(result.stats()["possible_matches"], 1)


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class LabFixtureIntegrationTests(LibraryTestCase):
    """The brief's fixture: VFD101 DRIVES M101 and OL101 PROTECTS M101, found in text and a drawing."""

    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        services.save_schema(V2, note="P8b test")
        self.doc, _ = self.upload()

    def stage(self, s_name, pred, o_name, page, kind="text"):
        idx = EntityIndex.load()
        sm, om = idx.resolve("Component", s_name, "EXTR01"), idx.resolve("Component", o_name, "EXTR01")
        t = CandidateTriple.objects.create(
            document=self.doc, mode="schema", subject_name=s_name, subject_type="Component", subject_id=sm.id,
            subject_match=sm.kind, subject_existing=sm.existing, predicate=pred, object_name=o_name,
            object_type="Component", object_id=om.id, object_match=om.kind, object_existing=om.existing, page_start=page)
        t.evidence.create(source_kind=kind, document=self.doc, page_start=page, excerpt=f"{s_name} {pred} {o_name}")
        return t

    def test_one_entity_per_tag_with_both_evidences(self):
        with driver.session() as s:
            for key, name, sub in (("M101", "Main Motor", "Motor"), ("VFD101", "Variable Frequency Drive", "Drive"),
                                   ("OL101", "Overload Relay", "Overload relay")):
                s.run("MATCH (a:Entity {id: 'bsk:asset:EXTR01'}) "
                      "CREATE (c:Entity:Component {id: $id, name: $name, subtype: $sub, tag: $tag})<-[:HAS_COMPONENT]-(a)",
                      id=f"bsk:component:EXTR01/{key}", name=name, sub=sub, tag=key).consume()
        text = self.stage("VFD101", "DRIVES", "Motor M101", 17)
        vision = self.stage("VFD-101", "DRIVES", "M101 main motor", 17, kind="vision")
        protects = self.stage("Overload relay OL101", "PROTECTS", "M101", 17, kind="vision")
        for t in (text, vision, protects):   # every variant resolved to the existing entity (by id or tag)
            self.assertIn(t.subject_match, ("id", "tag"))
            self.assertIn(t.object_match, ("id", "tag"))
        self.assertEqual(len(triples.approve([text, vision, protects])["approved"]), 3)

        edge = self.q("MATCH ({id: 'bsk:component:EXTR01/VFD101'})-[r:DRIVES]->(m {id: 'bsk:component:EXTR01/M101'}) "
                      "RETURN size(r.triple_ids) AS n, size(r.evidence_ids) AS ev, m.aliases AS aliases")
        self.assertEqual(edge, [{"n": 2, "ev": 2, "aliases": ["Motor M101", "M101 main motor"]}])   # one edge, both sources
        self.assertEqual(self.q("MATCH (c:Component) WHERE toLower(c.id) CONTAINS 'm101' "
                                "RETURN collect(c.id) AS ids")[0]["ids"], ["bsk:component:EXTR01/M101"])   # no duplicate
        self.assertEqual(self.q("MATCH (:Component {id: 'bsk:component:EXTR01/OL101'})-[r:PROTECTS]->() "
                                "RETURN count(r) AS n")[0]["n"], 1)

    def q(self, cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()
