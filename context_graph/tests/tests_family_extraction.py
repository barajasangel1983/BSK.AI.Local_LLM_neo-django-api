"""P26b: extraction anchored on a product family or a device model, moving a triple to another level, and `instance`
on a family's machine links.

Without a graph: the parameters, the anchor's names, ids under the family's key, a foreign family refused. Against a
throwaway Neo4j (GRAPH_TEST_URI): the levels of a part ("auto": Ratings on the model, the rest on the family), the
refusal of a family run on a manual already reviewed on its parts (D1), matching within the family, the move with
the fact's linked triples (D3), `instance` required and written, the family's manual, suggest_maps through the
family."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
from django.test import TestCase, override_settings

from context_graph import driver, extraction, maps, registry, triples
from context_graph.identity import EntityIndex, PartScope
from context_graph.models import CandidateTriple
from context_graph.schema import Schema
from ingestion.models import Document

V7 = yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v7.yaml").read_text())
ASSET = "bsk:asset:EXTR01"
FAMILY = "bsk:product-family:siemens/sinamics-g120c"
MODEL = "bsk:device-model:siemens/g120c-7.5"
OTHER_MODEL = "bsk:device-model:siemens/g120c-5.5"
CUTTER = "bsk:component:EXTR01/cutter-drive"
FEED = "bsk:component:EXTR01/feeder-screw-drive"
WINDOW = SimpleNamespace(page_start=3, page_end=3, section_path=["Parameters"], text="p1082 Maximum speed")
PRESET = SimpleNamespace(name="Controls & Setup (v6)", version=1)


def raw(subject, subject_type, predicate, obj, object_type, **extra):
    return {"subject": subject, "subject_type": subject_type, "predicate": predicate, "object": obj,
            "object_type": object_type, "confidence": 0.9, **extra}


@override_settings(GRAPH_ENABLED=False)
class FamilyRunTests(TestCase):
    def setUp(self):
        self.schema = Schema.from_definition(V7)
        self.index = EntityIndex()
        self.index.add(FAMILY, "ProductFamily", "SINAMICS G120C")
        self.index.add("bsk:product-family:watlow/ez-zone-pm-plus", "ProductFamily", "EZ-ZONE PM PLUS")
        self.scope = PartScope(FAMILY, "siemens/sinamics-g120c", frozenset({FAMILY, MODEL}), "family")
        self.doc = Document.objects.create(filename="g120c.pdf", doc_key="G120C", sha256="1" * 64, size=1)

    def test_params_take_a_family_without_an_asset(self):
        params = extraction.validate_params({"anchor_id": FAMILY})
        self.assertEqual((params["anchor_id"], params["anchor_level"], params["asset_id"]), (FAMILY, "part", ""))
        with self.assertRaisesMessage(extraction.ExtractionError, "choose the asset"):
            extraction.validate_params({"anchor_id": CUTTER})
        with self.assertRaisesMessage(extraction.ExtractionError, "anchor_level must be one of"):
            extraction.validate_params({"anchor_id": FAMILY, "anchor_level": "machine"})

    def test_the_products_names_mean_the_family_itself(self):
        anchor = {"name": "SINAMICS G120C", "label": "ProductFamily", "generic": {"g120c", "siemens sinamics g120c"}}
        t = extraction.map_anchor(raw("G120C", "Component", "HAS_PARAMETER", "Maximum speed", "Parameter"), anchor)
        self.assertEqual((t["subject"], t["subject_type"], t["subject_anchor"]), ("SINAMICS G120C", "ProductFamily", True))
        other = extraction.map_anchor(raw("Operator panel", "Component", "HAS_COMPONENT", "G120C", "Component"), anchor)
        self.assertEqual((other["subject"], other["subject_type"]), ("Operator panel", "Component"))

    def test_a_familys_facts_get_ids_under_its_key(self):
        t = extraction.map_anchor(raw("G120C", "Component", "HAS_PARAMETER", "Maximum speed", "Parameter",
                                      object_properties={"code": "P1082"}),
                                  {"name": "SINAMICS G120C", "label": "ProductFamily", "generic": {"g120c"}})
        c = extraction._candidate(t, "schema", self.schema, self.index, "siemens/sinamics-g120c", self.doc, None, WINDOW, 0,
                                  PRESET, part=self.scope)
        self.assertEqual((c.subject_id, c.subject_match, c.object_id, c.issue),
                         (FAMILY, "id", "bsk:parameter:siemens/sinamics-g120c/p1082", ""))

    def test_another_family_is_still_not_extracted(self):
        t = raw("EZ-ZONE PM PLUS", "ProductFamily", "HAS_PARAMETER", "Set point", "Parameter")
        c = extraction._candidate(t, "schema", self.schema, self.index, "siemens/sinamics-g120c", self.doc, None, WINDOW, 0,
                                  PRESET, part=self.scope)
        self.assertIn("not extracted", c.issue)

    def test_a_familys_limit_is_its_rating(self):
        self.assertTrue(extraction.is_part_rating("ProductFamily", "HAS_LIMIT", "OperatingLimit", self.schema))
        self.assertFalse(extraction.is_part_rating("Signal", "HAS_LIMIT", "OperatingLimit", self.schema))

    def test_instance_must_be_an_installed_part(self):
        t = CandidateTriple.objects.create(
            document=self.doc, mode="schema", subject_name="Speed actual", subject_type="Parameter",
            subject_id="bsk:parameter:siemens/sinamics-g120c/r0021", predicate="MAPS_TO", object_name="Cutter speed",
            object_type="Signal", object_id="bsk:signal:EXTR01/cutter-speed", subject_match="id", object_match="id")
        with self.assertRaisesMessage(triples.TripleError, "instance must be an installed part"):
            triples.edit(t, {"instance": FAMILY})
        triples.edit(t, {"instance": CUTTER})
        self.assertEqual(t.rel_props, {"instance": CUTTER})


TEST_URI = os.getenv("GRAPH_TEST_URI")


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class FamilyGraphTests(TestCase):
    def setUp(self):
        registry.save_version(V7, note="v7")
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
            s.run(f"""
                CREATE (a:Entity:Asset {{id: '{ASSET}', name: 'Extruder EXTR01'}})
                CREATE (c1:Entity:Component {{id: '{CUTTER}', name: 'Cutter drive', source: 'structured'}})
                CREATE (c2:Entity:Component {{id: '{FEED}', name: 'Feed screw drive', source: 'structured'}})
                CREATE (m1:Entity:DeviceModel {{id: '{MODEL}', name: 'G120C 7.5 kW', designation: '7.5 kW, 400 V'}})
                CREATE (m2:Entity:DeviceModel {{id: '{OTHER_MODEL}', name: 'G120C 5.5 kW'}})
                CREATE (f:Entity:ProductFamily {{id: '{FAMILY}', name: 'SINAMICS G120C', manufacturer: 'Siemens'}})
                CREATE (a)-[:HAS_COMPONENT]->(c1), (a)-[:HAS_COMPONENT]->(c2)
                CREATE (c1)-[:INSTANCE_OF]->(m1), (c2)-[:INSTANCE_OF]->(m2)
                CREATE (m1)-[:VARIANT_OF]->(f), (m2)-[:VARIANT_OF]->(f)
                CREATE (f)-[:HAS_PARAMETER]->(:Entity:Parameter {{id: 'bsk:parameter:siemens/sinamics-g120c/r0021',
                                                                 name: 'Speed actual value', code: 'r0021'}})
                CREATE (c1)-[:HAS_ALARM]->(:Entity:Alarm {{id: 'bsk:alarm:EXTR01/cutter-drive/f30001', name: 'Overcurrent'}})
                CREATE (c1)-[:MONITORED_BY]->(:Entity:Signal {{id: 'bsk:signal:EXTR01/cutter-speed', name: 'Cutter speed', unit: 'rpm'}})
                CREATE (c2)-[:MONITORED_BY]->(:Entity:Signal {{id: 'bsk:signal:EXTR01/feed-speed', name: 'Feed speed', unit: 'rpm'}})
            """).consume()
        self.doc = Document.objects.create(filename="g120c.pdf", doc_key="G120C", sha256="2" * 64, size=1)

    def test_auto_runs_ratings_on_the_model_and_the_rest_on_the_family(self):
        params = extraction.validate_params({"asset_id": ASSET, "anchor_id": CUTTER, "anchor_level": "family"})
        self.assertEqual((params["anchor_id"], params["anchor_part"]), (FAMILY, CUTTER))
        self.assertEqual(extraction.validate_params(params)["anchor_id"], FAMILY)       # validated again: the same
        self.assertEqual(extraction.validate_params({"asset_id": ASSET, "anchor_id": CUTTER, "anchor_level": "model"})
                         ["anchor_id"], MODEL)
        anchor, per_facet = extraction._anchor_for_level(CUTTER, "auto", ["maintenance", "ratings"], {})
        self.assertEqual((anchor, per_facet), (CUTTER, {"maintenance": FAMILY, "ratings": MODEL}))
        self.assertEqual(extraction.anchors_of({"anchor_id": CUTTER, "facet_anchors": per_facet}), [FAMILY, MODEL])

    def test_a_manual_reviewed_on_the_parts_waits_for_p26c(self):
        self.assertEqual(extraction.reviewed_on_parts(self.doc, FAMILY), "")
        CandidateTriple.objects.create(document=self.doc, mode="schema", status="approved", anchor_id=CUTTER,
                                       subject_name="Cutter drive", subject_type="Component", subject_id=CUTTER,
                                       predicate="HAS_ALARM", object_name="Overcurrent", object_type="Alarm")
        refused = extraction.reviewed_on_parts(self.doc, MODEL)
        self.assertIn("already reviewed on Cutter drive (1)", refused)
        self.assertEqual(extraction.reviewed_on_parts(self.doc, CUTTER), "")       # a part run is never refused

    def test_a_family_run_sees_the_familys_facts_not_the_parts(self):
        index = EntityIndex.load()
        family = index.part_scope(FAMILY, "siemens/sinamics-g120c")
        self.assertEqual((family.level, family.family), ("family", frozenset({FAMILY, MODEL, OTHER_MODEL})))
        self.assertEqual(index.resolve("Parameter", "Speed actual value", "siemens/sinamics-g120c", part=family).kind, "name")
        self.assertEqual(index.resolve("Alarm", "Overcurrent", "siemens/sinamics-g120c", part=family).kind, "new")
        self.assertEqual(index.resolve("Component", "Cutter drive", "siemens/sinamics-g120c", part=family).kind, "new")
        model = index.part_scope(MODEL, "siemens/g120c-7.5")      # a model run matches its family's facts
        self.assertEqual(index.resolve("Parameter", "Speed actual value", "siemens/g120c-7.5", part=model).kind, "name")

    def test_the_familys_prompt_names_it_and_its_facts(self):
        ctx = extraction.anchor_context("", FAMILY)
        self.assertEqual((ctx["label"], ctx["name"]), ("ProductFamily", "SINAMICS G120C"))
        self.assertIn('product family "SINAMICS G120C" (Siemens, models: G120C 5.5 kW; G120C 7.5 kW)', ctx["text"])
        self.assertIn("Parameters: r0021 (Speed actual value)", ctx["text"])
        self.assertIn("g120c 7 5 kw", ctx["generic"])

    def staged(self, subject, subject_type, subject_id, predicate, obj, object_type, object_id, **kw):
        return CandidateTriple.objects.create(
            document=self.doc, mode="schema", anchor_id=FAMILY, subject_name=subject, subject_type=subject_type,
            subject_id=subject_id, predicate=predicate, object_name=obj, object_type=object_type, object_id=object_id,
            subject_match="id" if subject_type == "ProductFamily" else "new", object_match="new", **kw)

    def test_a_move_takes_the_facts_linked_triples_along(self):
        alarm_id = "bsk:alarm:siemens/sinamics-g120c/motor-overtemperature"
        has = self.staged("SINAMICS G120C", "ProductFamily", FAMILY, "HAS_ALARM", "Motor overtemperature", "Alarm", alarm_id,
                          subject_existing=True)
        fix = self.staged("Check the motor fan", "Procedure", "bsk:procedure:siemens/sinamics-g120c/check-the-motor-fan",
                          "ADDRESSES", "Motor overtemperature", "Alarm", alarm_id)
        targets = triples.move_targets(FAMILY)
        self.assertEqual([p["id"] for p in targets["parts"]], [CUTTER, FEED])
        changed = triples.move(has, CUTTER)
        self.assertEqual({t.pk for t in changed}, {has.pk, fix.pk})
        has.refresh_from_db()
        fix.refresh_from_db()
        moved = "bsk:alarm:EXTR01/cutter-drive/Motor-overtemperature"
        self.assertEqual((has.subject_id, has.subject_type, has.object_id, has.moved_from, has.issue),
                         (CUTTER, "Component", moved, FAMILY, ""))
        self.assertEqual(fix.object_id, moved)
        has.status = "approved"
        has.save()
        with self.assertRaisesMessage(triples.TripleError, "P26c"):
            triples.move(has, FAMILY)

    def test_a_familys_machine_link_needs_its_instance(self):
        maps_to = self.staged("Speed actual value", "Parameter", "bsk:parameter:siemens/sinamics-g120c/r0021", "MAPS_TO",
                              "Cutter speed", "Signal", "bsk:signal:EXTR01/cutter-speed", subject_existing=True,
                              object_existing=True)
        CandidateTriple.objects.filter(pk=maps_to.pk).update(subject_match="id", object_match="id")
        maps_to.refresh_from_db()
        result = triples.approve([maps_to])
        self.assertIn("choose the installed part", result["errors"][maps_to.pk])
        triples.edit(maps_to, {"instance": CUTTER})
        maps_to.refresh_from_db()
        self.assertEqual(triples.approve([maps_to])["errors"], {})
        with driver.session() as s:
            rel = s.run("MATCH (:Parameter)-[r:MAPS_TO]->(:Signal {id: 'bsk:signal:EXTR01/cutter-speed'}) "
                        "RETURN r.instance AS instance").single()
            documented = s.run("MATCH (:ProductFamily {id: $id})-[:DOCUMENTED_BY]->(d:Document) RETURN d.doc_key AS k",
                               id=FAMILY).single()
        self.assertEqual((rel["instance"], documented["k"]), (CUTTER, "G120C"))

    def test_suggest_maps_uses_the_familys_parameter(self):
        dev = maps.device(FEED)
        self.assertEqual(set(dev["products"]), {OTHER_MODEL, FAMILY})
        self.assertEqual(maps.product_parameter(dev, "r0021")["id"], "bsk:parameter:siemens/sinamics-g120c/r0021")
        self.assertIsNone(maps.product_parameter(dev, "r0022"))
