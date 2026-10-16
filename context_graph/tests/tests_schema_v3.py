"""Schema `Industrial` v3 (2026-10-15): valid, additive over v2, and saving it changes nothing in the graph.

The integration test needs the throwaway Neo4j:

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph.tests.tests_schema_v3
"""

import os
import unittest
from pathlib import Path

import yaml
from django.test import SimpleTestCase, TestCase, override_settings

from context_graph import driver, extraction, registry, services
from context_graph.schema import Schema, SchemaError, diff, validate_definition

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"
V2 = yaml.safe_load((SCHEMAS / "industrial_v2.yaml").read_text())
V3 = yaml.safe_load((SCHEMAS / "industrial_v3.yaml").read_text())
TEST_URI = os.getenv("GRAPH_TEST_URI")


class SchemaV3Tests(SimpleTestCase):
    def test_v3_is_valid_and_only_adds_to_v2(self):
        self.assertEqual(validate_definition(V3), [])
        change = diff(Schema.from_definition(V2), Schema.from_definition(V3))
        self.assertEqual((change["removed_entity_types"], change["removed_relationship_types"]), ([], []))
        self.assertTrue(all(not c["removed"] for c in change["changed_pairs"].values()))
        self.assertTrue(all(not c["removed"] for c in change["changed_vocabulary"].values()))
        self.assertEqual(change["added_entity_types"], ["FailureMode"])
        self.assertEqual(change["added_relationship_types"], ["HAS_FAILURE_MODE", "INDICATED_BY", "MITIGATED_BY"])
        # Every v2 type, relation and pair is still there (two descriptions are reworded, nothing else).
        v2, v3 = Schema.from_definition(V2), Schema.from_definition(V3)
        self.assertLessEqual(set(v2.entity_types), set(v3.entity_types))
        for name, rel in v2.relationship_types.items():
            self.assertLessEqual(rel.pairs, v3.relationship_types[name].pairs)
            self.assertLessEqual(set(rel.aliases), set(v3.relationship_types[name].aliases))
        self.assertEqual([label for label in v2.entity_types if v2.entity_types[label] != v3.entity_types[label]],
                         ["Alarm", "Procedure"])

    def test_the_new_pairs(self):
        v3 = Schema.from_definition(V3)
        for subject, rel, obj in (("Component", "HAS_ALARM", "Alarm"), ("Signal", "HAS_ALARM", "Alarm"),
                                  ("Asset", "HAS_ALARM", "Alarm"), ("Component", "HAS_FAILURE_MODE", "FailureMode"),
                                  ("FailureMode", "INDICATED_BY", "Alarm"), ("FailureMode", "INDICATED_BY", "Signal"),
                                  ("FailureMode", "MITIGATED_BY", "Procedure"), ("Component", "HAS_PROCEDURE", "Procedure"),
                                  ("Procedure", "DOCUMENTED_BY", "Document"), ("DocumentSection", "DESCRIBES", "FailureMode")):
            self.assertEqual(v3.check_relationship(subject, rel, obj), rel)
        with self.assertRaises(SchemaError):
            v3.check_relationship("FailureMode", "MITIGATED_BY", "Alarm")
        self.assertEqual(v3.subtype_for("Alarm", "device fault"), "Device fault")
        self.assertEqual(v3.subtype_for("Document", "operating instructions"), "Operating instructions")
        self.assertEqual(v3.relationship_for("what to do", "FailureMode", "Procedure"), "MITIGATED_BY")

    def test_properties_are_documented_and_checked(self):
        v3 = Schema.from_definition(V3)
        self.assertIn("manufacturer", v3.properties["Component"])
        self.assertEqual(v3.properties["OperatingLimit"]["kind"], "normal | specification | alarm | trip")
        self.assertEqual(Schema.from_definition(V2).properties, {})
        bad = {**V3, "entity_types": {**V3["entity_types"], "Asset": {"description": "x", "properties": {"Bad Name": "x"}}}}
        self.assertIn("entity type 'Asset': property 'Bad Name' must be snake_case", validate_definition(bad))
        bad["entity_types"]["Asset"]["properties"] = ["a", "b"]
        self.assertIn("entity type 'Asset': properties must map a property name to its meaning", validate_definition(bad))

    def test_extraction_offers_the_new_types(self):
        entity_types, relationships = extraction.render_schema(Schema.from_definition(V3))
        self.assertIn("FailureMode", entity_types)
        self.assertIn("Mechanical", entity_types)
        self.assertIn("- MITIGATED_BY: FailureMode -> Procedure", relationships)
        self.assertIn("Signal -> Alarm", relationships)


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class SchemaV3GraphTests(TestCase):
    def setUp(self):
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.save_schema(V2, note="v2")
        services.load_seed(SEED)

    def snapshot(self):
        with driver.session() as s:
            nodes = s.run("MATCH (n) RETURN n.id AS id, labels(n) AS l, properties(n) AS p ORDER BY id").data()
            rels = s.run("MATCH (a)-[r]->(b) RETURN a.id AS a, type(r) AS t, b.id AS b, properties(r) AS p ORDER BY a, t, b").data()
        return nodes, rels

    def test_saving_v3_changes_nothing_in_the_graph(self):
        before = self.snapshot()
        result = services.save_schema(V3, note="v3")
        self.assertEqual(registry.active_schema().version, result["version"])
        self.assertEqual(self.snapshot(), before)
        # v2 stays a saved version that can be made active again.
        previous = result["version"] - 1
        services.activate_schema(previous)
        self.assertEqual(registry.active_schema().definition, V2)


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class V3FillInsTests(TestCase):
    """The fill-ins only add: a missing property, a new relation, a new Document; deleting them restores the graph."""

    def setUp(self):
        import shutil
        import tempfile
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.save_schema(V3, note="v3")
        services.load_seed(SEED)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.settings_override = override_settings(GRAPH_DATA_BASE=tmp)
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def q(self, cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    def stage_and_approve(self, name, csv_text, mapping):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from context_graph import structured, triples
        from context_graph.models import CandidateTriple
        df, _ = structured.add_data_file(SimpleUploadedFile(name, csv_text.encode()))
        structured.stage(df, mapping, "bsk:asset:EXTR01")
        staged = list(CandidateTriple.objects.filter(data_file=df))
        triples.approve(staged)
        return [CandidateTriple.objects.get(pk=t.pk) for t in staged]

    def test_subtypes_links_and_documents_are_added_and_removed_again(self):
        from context_graph import triples
        from context_graph.management.commands.v3_fillins import mappings
        maps = mappings()
        before = self.q("MATCH (n) RETURN count(n) AS n")[0]["n"], self.q("MATCH ()-[r]->() RETURN count(r) AS r")[0]["r"]
        die = self.q("MATCH (c {id: 'bsk:component:EXTR01/die-head'}) RETURN c.component_type AS t")[0]["t"]

        done = self.stage_and_approve("parts.csv", "Item,Parent,Subtype\ndie-head,,Die\n", maps["component_subtypes"])
        done += self.stage_and_approve("alarms.csv", "Code,Tag,Subtype\nDIE_PLUG,die_pressure_bar,PLC alarm\n", maps["alarm_signals"])
        done += self.stage_and_approve("manuals.csv", "Item,Document key,Title,Class,Source URL,Library id\n"
                                       "die-head,DIE-MANUAL,Die head manual,Operating instructions,https://x.example/die.pdf,1\n",
                                       maps["manuals"])
        part = self.q("MATCH (c {id: 'bsk:component:EXTR01/die-head'}) RETURN c.component_type AS t, c.subtype AS s")[0]
        self.assertEqual(part, {"t": die, "s": "Die"})                           # added beside, not instead of
        self.assertEqual(self.q("MATCH (s {id: 'bsk:signal:EXTR01/die_pressure_bar'})-[:HAS_ALARM]->(a {id: 'bsk:alarm:EXTR01/DIE_PLUG'}) "
                                "RETURN a.subtype AS s")[0]["s"], "PLC alarm")
        doc = self.q("MATCH (:Component {id: 'bsk:component:EXTR01/die-head'})-[:DOCUMENTED_BY]->(d:Document) RETURN d.id AS id, d.subtype AS s")[0]
        self.assertEqual(doc, {"id": "bsk:document:DIE-MANUAL", "s": "Operating instructions"})   # the id extraction uses

        triples.delete(done)
        after = self.q("MATCH (n) RETURN count(n) AS n")[0]["n"], self.q("MATCH ()-[r]->() RETURN count(r) AS r")[0]["r"]
        self.assertEqual(after, before)
        self.assertEqual(self.q("MATCH (c {id: 'bsk:component:EXTR01/die-head'}) RETURN c.component_type AS t, c.subtype AS s")[0],
                         {"t": die, "s": None})

    def test_deleting_text_triples_keeps_the_imported_document(self):
        """2026-10-16: deleting the ABB run's triples took the manuals file's Document and its DOCUMENTED_BY with it."""
        from context_graph import triples
        from context_graph.management.commands.v3_fillins import mappings
        from context_graph.models import CandidateTriple
        from ingestion.models import Document
        lib = Document.objects.create(filename="die.pdf", doc_key="DIE-MANUAL", sha256="0" * 64, size=1, title="Die head manual")
        self.stage_and_approve("manuals.csv", "Item,Document key,Title,Class,Source URL,Library id\n"
                               f"die-head,DIE-MANUAL,Die head manual,Operating instructions,https://x.example/die.pdf,{lib.id}\n",
                               mappings()["manuals"])
        text = CandidateTriple.objects.create(
            document=lib, mode="schema", subject_name="Die Head", subject_type="Component",
            subject_id="bsk:component:EXTR01/die-head", predicate="MONITORED_BY", object_name="Die pressure",
            object_type="Signal", object_id="bsk:signal:EXTR01/die_pressure_bar", page_start=3, model="m",
            preset_name="P", preset_version=1)
        triples.approve([text])
        doc = "MATCH (d:Document {id: 'bsk:document:DIE-MANUAL'}) RETURN d.name AS name, d.source AS source"
        self.assertEqual(self.q(doc), [{"name": "Die head manual", "source": "structured"}])   # not renamed by the text
        self.assertEqual(self.q("MATCH (:Document {id: 'bsk:document:DIE-MANUAL'})-[:HAS_SECTION]->(s) RETURN count(s) AS n")[0]["n"], 1)

        triples.delete([CandidateTriple.objects.get(pk=text.pk)])
        self.assertEqual(self.q(doc), [{"name": "Die head manual", "source": "structured"}])
        self.assertEqual(self.q("MATCH ({id: 'bsk:component:EXTR01/die-head'})-[:DOCUMENTED_BY]->(d:Document {id: 'bsk:document:DIE-MANUAL'}) "
                                "RETURN count(d) AS n")[0]["n"], 1)
        self.assertEqual(self.q("MATCH (s:DocumentSection {doc_key: 'DIE-MANUAL'}) RETURN count(s) AS n")[0]["n"], 0)
