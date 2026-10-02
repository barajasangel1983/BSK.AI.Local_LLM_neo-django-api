"""Context Graph P1 integration tests against a throwaway Neo4j.

    docker compose -f deploy/neo4j/docker-compose.yml --env-file .env --profile test up -d neo4j-test
    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph

The test instance is wiped before every test; the real graph is never touched.
"""

import os
import unittest
from pathlib import Path

from django.test import TestCase, override_settings

from context_graph import driver, repository, services
from context_graph.repository import NodeNotFound
from context_graph.schema import Schema, SchemaError

TEST_URI = os.getenv("GRAPH_TEST_URI")
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see module docstring) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class GraphIntegrationTests(TestCase):
    def setUp(self):
        driver.close_driver()  # connect to the test instance, not a cached dev driver
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()

    def _count(self, query="MATCH (n:Entity) RETURN count(n) AS n"):
        with driver.session() as s:
            return s.run(query).single()["n"]

    def test_constraints_exist(self):
        with driver.session() as s:
            names = {row["name"] for row in s.run("SHOW CONSTRAINTS YIELD name")}
        self.assertIn("entity_id", names)

    def test_seed_is_idempotent(self):
        services.load_seed(SEED)
        services.load_seed(SEED)
        self.assertEqual(self._count(), 87)
        self.assertEqual(self._count("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n"), 136)

    def test_asset_context_answers_the_core_questions(self):
        services.load_seed(SEED)
        ctx = services.get_asset_context("bsk:asset:EXTR01")

        self.assertEqual([n["name"] for n in ctx["hierarchy"]], ["BSKLAB Demo Plant", "Extrusion", "Extrusion Line 1"])
        self.assertEqual(ctx["counts"], {"components": 9, "signals": 30, "alarms": 20, "procedures": 8, "documents": 0})
        screw = next(c for c in ctx["components"] if c["id"].endswith("/screw-barrel"))
        self.assertEqual([c["name"] for c in screw["children"]], ["Barrel Zone 1", "Barrel Zone 2", "Barrel Zone 3"])
        die = next(c for c in ctx["components"] if c["id"].endswith("/die-head"))
        pressure = next(s for s in die["signals"] if s["id"].endswith("/die_pressure_bar"))
        self.assertEqual((pressure["limits"][0]["properties"]["low"], pressure["limits"][0]["properties"]["high"]),
                         (84.3, 99.2))
        plug = next(a for a in ctx["alarms"] if a["properties"]["code"] == "DIE_PLUG")
        self.assertEqual([p["name"] for p in plug["procedures"]], ["Die plug / screen block clearing"])
        self.assertIn(("bsk:component:EXTR01/screw-barrel", "material_flow", "bsk:component:EXTR01/die-head"),
                      {(c["source"], c["kind"], c["target"]) for c in ctx["connections"]})

    def test_alarm_procedures_and_search(self):
        services.load_seed(SEED)
        result = services.alarm_procedures("bsk:alarm:EXTR01/MOTOR_OVERLOAD")
        self.assertEqual([(p["name"], [t["name"] for t in p["applies_to"]]) for p in result["procedures"]],
                         [("Main drive overload recovery", ["Main Drive Motor"])])
        found = services.search_nodes("Signal", "temp")
        self.assertEqual(found["total"], 5)
        with self.assertRaises(NodeNotFound):
            services.alarm_procedures("bsk:asset:EXTR01")  # exists, but isn't an alarm

    def test_graph_data_around_a_node(self):
        services.load_seed(SEED)
        data = services.graph_data("bsk:component:EXTR01/die-head", depth=1)
        ids = {n["id"] for n in data["nodes"]}
        self.assertIn("bsk:alarm:EXTR01/DIE_PLUG", ids)
        self.assertTrue(all(l["source"] in ids and l["target"] in ids for l in data["links"]))
        only_signals = services.graph_data("bsk:component:EXTR01/die-head", depth=1, labels=["Signal"])
        self.assertEqual({n["label"] for n in only_signals["nodes"][1:]}, {"Signal"})
        with self.assertRaises(NodeNotFound):
            services.graph_data("bsk:asset:NOPE")

    def test_schema_is_enforced_on_writes(self):
        schema = Schema.from_yaml()
        with driver.session() as s:
            s.execute_write(repository.upsert_node, schema, "Asset", "bsk:asset:A1", {"name": "A1"})
            s.execute_write(repository.upsert_node, schema, "Signal", "bsk:signal:A1/s", {"name": "s"})
            with self.assertRaisesMessage(SchemaError, "not allowed"):
                s.execute_write(repository.upsert_relationship, schema, "bsk:signal:A1/s", "HAS_COMPONENT", "bsk:asset:A1")
            with self.assertRaisesMessage(SchemaError, "doesn't match entity type"):
                s.execute_write(repository.upsert_node, schema, "Signal", "bsk:asset:A2", {})
            with self.assertRaises(NodeNotFound):
                s.execute_write(repository.upsert_relationship, schema, "bsk:asset:A1", "MONITORED_BY", "bsk:signal:A1/x")
        self.assertEqual(self._count("MATCH ()-[r]->() RETURN count(r) AS n"), 0)

    def test_schema_edits_respect_data_in_use(self):
        import copy

        from context_graph import registry
        from context_graph.services import SchemaConflict, activate_schema, check_schema, save_schema

        services.load_seed(SEED)
        base = copy.deepcopy(registry.active_schema().definition)

        no_procedures = copy.deepcopy(base)
        del no_procedures["entity_types"]["Procedure"]
        for rel in ("HAS_PROCEDURE", "APPLIES_TO", "ADDRESSES"):
            del no_procedures["relationship_types"][rel]
        result = check_schema(no_procedures)
        self.assertEqual({c["name"] for c in result["conflicts"]},
                         {"Procedure", "HAS_PROCEDURE", "APPLIES_TO", "ADDRESSES"})
        with self.assertRaises(SchemaConflict):
            save_schema(no_procedures)

        with_sensor = copy.deepcopy(base)
        with_sensor["entity_types"]["Sensor"] = {"description": "Physical sensor"}
        saved = save_schema(with_sensor, note="add Sensor")
        self.assertEqual(saved["diff"]["added_entity_types"], ["Sensor"])
        with driver.session() as s:
            indexes = {row["name"] for row in s.run("SHOW INDEXES YIELD name")}
        self.assertIn("sensor_name", indexes)

        # back to v1 is fine: Sensor is unused
        self.assertEqual(activate_schema(1)["diff"]["removed_entity_types"], ["Sensor"])
