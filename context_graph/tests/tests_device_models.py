"""P26a (schema v7): device models and product families apart from the installed parts.

The proposal from the components, the "Device models" import template with global ids, families and models kept
out of extraction, the self-link guard, and (against a throwaway Neo4j, GRAPH_TEST_URI) the asset context that
gives a model's or a family's facts to every installed part of it."""

import os
import unittest
from pathlib import Path

import yaml
from django.test import SimpleTestCase, TestCase, override_settings

from context_graph import device_models, driver, extraction, registry, services, structured, triples
from context_graph.identity import EntityIndex
from context_graph.models import CandidateTriple, DataFile
from context_graph.schema import Schema
from context_graph.tests.tests_structured import ImportTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V7 = yaml.safe_load((SCHEMAS / "industrial_v7.yaml").read_text())
ASSET = "bsk:asset:EXTR01"


def component(key, name, **props):
    return {"id": f"bsk:component:EXTR01/{key}", "name": name, "properties": {"name": name, **props}, "children": []}


class ProposalTests(SimpleTestCase):
    def test_twins_share_a_model_and_variants_a_family(self):
        rows = device_models.propose([
            component("cutter-drive", "Cutter drive", manufacturer="Siemens", model="SINAMICS G120C, 7.5 kW, 400 V, PROFINET",
                      part_number="6SL3210-1KE21-7UF1"),
            component("feeder-screw-drive", "Feed screw drive", manufacturer="Siemens",
                      model="SINAMICS G120C, 5.5 kW, 400 V, PROFINET", part_number="6SL3210-1KE21-3UF1"),
            component("barrel-zone-1-controller", "Zone 1 controller", manufacturer="Watlow",
                      model="EZ-ZONE PM PLUS, 1/16 DIN (PM6), integrated", part_number="PM6 (configure)"),
            component("barrel-zone-2-controller", "Zone 2 controller", manufacturer="Watlow",
                      model="EZ-ZONE PM PLUS, 1/16 DIN (PM6), integrated", part_number="PM6 (configure)"),
            component("belt-scale-integrator", "Belt scale integrator", manufacturer="Siemens", model="Milltronics BW500",
                      part_number="7MH7152"),
            component("screw", "Screw", model="Single screw, segmented (OEM part)"),                 # no manufacturer
            component("plc/ATECC608B-Crypto-Microchip", "ATECC608B", manufacturer="Microchip", model="ATECC608B"),
        ], "EXTR01")
        by_part = {r["Component"]: r for r in rows}
        self.assertEqual(set(by_part), {"cutter-drive", "feeder-screw-drive", "barrel-zone-1-controller",
                                        "barrel-zone-2-controller", "belt-scale-integrator"})   # no OEM part, no sub-part
        self.assertEqual((by_part["cutter-drive"]["Family Key"], by_part["cutter-drive"]["Model Key"]),
                         ("siemens/sinamics-g120c", "siemens/sinamics-g120c-7.5-kw-400-v-profinet"))
        self.assertEqual(by_part["feeder-screw-drive"]["Family Key"], "siemens/sinamics-g120c")
        self.assertEqual(by_part["barrel-zone-1-controller"]["Model Key"], by_part["barrel-zone-2-controller"]["Model Key"])
        self.assertEqual((by_part["barrel-zone-1-controller"]["Order Code"], by_part["cutter-drive"]["Order Code"]),
                         ("", "6SL3210-1KE21-7UF1"))        # "PM6 (configure)" names no model
        self.assertEqual((by_part["belt-scale-integrator"]["Family Key"], by_part["belt-scale-integrator"]["Model Key"]),
                         ("", "siemens/milltronics-bw500"))  # one model, no family
        summary = device_models.summary(rows)
        self.assertIn("4 models, 5 installed parts", summary)
        self.assertIn("EZ-ZONE PM PLUS, 1/16 DIN (PM6), integrated: barrel-zone-1-controller, barrel-zone-2-controller",
                      summary)


@override_settings(GRAPH_ENABLED=False)
class DeviceModelImportTests(ImportTestCase):
    CSV = ("Component,Name,Manufacturer,Family Key,Family,Model Key,Model,Order Code\n"
           "cutter-drive,Cutter drive,Siemens,siemens/sinamics-g120c,SINAMICS G120C,"
           "siemens/sinamics-g120c-7.5-kw-400-v-profinet,\"SINAMICS G120C, 7.5 kW, 400 V, PROFINET\",6SL3210-1KE21-7UF1\n"
           "feeder-screw-drive,Feed screw drive,Siemens,siemens/sinamics-g120c,SINAMICS G120C,"
           "siemens/sinamics-g120c-5.5-kw-400-v-profinet,\"SINAMICS G120C, 5.5 kW, 400 V, PROFINET\",6SL3210-1KE21-3UF1\n"
           "belt-scale-integrator,Belt scale integrator,Siemens,,,siemens/milltronics-bw500,Milltronics BW500,7MH7152\n")

    def test_the_template_links_parts_to_global_models_and_families(self):
        registry.save_version(V7, note="v7")
        file_id = self.upload("EXTR01_device_models.csv", self.CSV.encode())["data_file"]["id"]
        best = self.detail(file_id)["suggestions"][0]
        self.assertEqual((best["key"], best["complete"]), ("device_models", True))
        self.assertEqual(structured.check(DataFile.objects.get(pk=file_id), best["mapping"], ASSET), [])
        index = EntityIndex()
        for key, name in (("cutter-drive", "Cutter drive"), ("feeder-screw-drive", "Feed screw drive"),
                          ("belt-scale-integrator", "Belt scale integrator")):
            index.add(f"bsk:component:EXTR01/{key}", "Component", name)
        # An existing 5.5 kW model is not a match for the 7.5 kW one: models match by id only.
        index.add("bsk:device-model:siemens/sinamics-g120c-5.5-kw-400-v-profinet", "DeviceModel",
                  "SINAMICS G120C, 5.5 kW, 400 V, PROFINET")
        result = structured.build(DataFile.objects.get(pk=file_id), best["mapping"], ASSET, index=index)
        links = sorted((t.subject_id, t.predicate, t.object_id, t.object_match) for t in result.triples)
        self.assertEqual(links, [
            ("bsk:component:EXTR01/belt-scale-integrator", "INSTANCE_OF", "bsk:device-model:siemens/milltronics-bw500", "new"),
            ("bsk:component:EXTR01/cutter-drive", "INSTANCE_OF",
             "bsk:device-model:siemens/sinamics-g120c-7.5-kw-400-v-profinet", "new"),
            ("bsk:component:EXTR01/feeder-screw-drive", "INSTANCE_OF",
             "bsk:device-model:siemens/sinamics-g120c-5.5-kw-400-v-profinet", "id"),
            ("bsk:device-model:siemens/sinamics-g120c-5.5-kw-400-v-profinet", "VARIANT_OF",
             "bsk:product-family:siemens/sinamics-g120c", "new"),
            ("bsk:device-model:siemens/sinamics-g120c-7.5-kw-400-v-profinet", "VARIANT_OF",
             "bsk:product-family:siemens/sinamics-g120c", "new"),
        ])
        drive = next(t for t in result.triples if t.object_id.endswith("7.5-kw-400-v-profinet") and t.predicate == "INSTANCE_OF")
        self.assertEqual((drive.object_name, drive.object_props),
                         ("SINAMICS G120C, 7.5 kW, 400 V, PROFINET",
                          {"manufacturer": "Siemens", "order_code": "6SL3210-1KE21-7UF1"}))


@override_settings(GRAPH_ENABLED=False)
class NotExtractedTests(TestCase):
    def test_families_and_models_are_not_offered_to_the_model(self):
        schema = Schema.from_definition(V7)
        entity_types, relationships = extraction.render_schema(schema)
        self.assertNotIn("ProductFamily", entity_types + relationships)
        self.assertNotIn("DeviceModel", entity_types + relationships)
        self.assertIn("- HAS_SPEC: Component -> Specification", relationships)

    def test_a_link_to_itself_is_never_approved(self):
        t = CandidateTriple.objects.create(
            mode="schema", subject_name="Barrel zone 1 power controller", subject_type="Component",
            subject_id="bsk:component:EXTR01/barrel-zone-1-ssr", predicate="HAS_COMPONENT",
            object_name="Barrel zone 1 power controller", object_type="Component",
            object_id="bsk:component:EXTR01/barrel-zone-1-ssr", subject_match="id", object_match="id")
        result = triples.approve([t])
        self.assertEqual((result["approved"], result["errors"]), ([], {t.pk: extraction.SELF_LINK}))


TEST_URI = os.getenv("GRAPH_TEST_URI")


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class AssetContextModelTests(TestCase):
    def setUp(self):
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
            s.run("""
                CREATE (a:Entity:Asset {id: 'bsk:asset:EXTR01', name: 'Extruder EXTR01'})
                CREATE (c1:Entity:Component {id: 'bsk:component:EXTR01/cutter-drive', name: 'Cutter drive'})
                CREATE (c2:Entity:Component {id: 'bsk:component:EXTR01/feeder-screw-drive', name: 'Feed screw drive'})
                CREATE (m1:Entity:DeviceModel {id: 'bsk:device-model:siemens/g120c-7.5', name: 'G120C 7.5 kW'})
                CREATE (m2:Entity:DeviceModel {id: 'bsk:device-model:siemens/g120c-5.5', name: 'G120C 5.5 kW'})
                CREATE (f:Entity:ProductFamily {id: 'bsk:product-family:siemens/sinamics-g120c', name: 'SINAMICS G120C'})
                CREATE (a)-[:HAS_COMPONENT]->(c1), (a)-[:HAS_COMPONENT]->(c2)
                CREATE (c1)-[:INSTANCE_OF]->(m1), (c2)-[:INSTANCE_OF]->(m2)
                CREATE (m1)-[:VARIANT_OF]->(f), (m2)-[:VARIANT_OF]->(f)
                CREATE (m1)-[:HAS_SPEC]->(:Entity:Specification {id: 'bsk:specification:x/rated', name: 'Rated output current', value: '16.5'})
                CREATE (f)-[:HAS_ALARM]->(:Entity:Alarm {id: 'bsk:alarm:x/F30001', name: 'Overcurrent', code: 'F30001'})
                CREATE (c1)-[:HAS_SPEC]->(:Entity:Specification {id: 'bsk:specification:x/own', name: 'Own rating'})
            """).consume()

    def test_a_models_and_a_familys_facts_belong_to_their_installed_parts(self):
        ctx = services.get_asset_context(ASSET)
        cutter = next(c for c in ctx["components"] if c["id"].endswith("/cutter-drive"))
        self.assertEqual((cutter["device_model"]["name"], cutter["device_model"]["family"]["name"]),
                         ("G120C 7.5 kW", "SINAMICS G120C"))
        specs = {s["name"]: s["components"] for s in ctx["specifications"]}
        self.assertEqual(specs, {"Rated output current": ["bsk:component:EXTR01/cutter-drive"],
                                 "Own rating": ["bsk:component:EXTR01/cutter-drive"]})
        alarm = next(a for a in ctx["part_alarms"] if a["name"] == "Overcurrent")
        self.assertEqual(alarm["components"], ["bsk:component:EXTR01/cutter-drive", "bsk:component:EXTR01/feeder-screw-drive"])
