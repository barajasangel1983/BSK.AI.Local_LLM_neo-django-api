"""P8c: sources of entities and relationships, and of asset-chat graph facts.

Integration needs the throwaway Neo4j:
    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver, provenance, services, structured, triples
from context_graph.models import CandidateTriple, Evidence
from ingestion.tests.tests_library import LibraryTestCase

TEST_URI = os.getenv("GRAPH_TEST_URI")
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"


class LabelTests(SimpleTestCase):
    def test_origin_labels(self):
        self.assertEqual(provenance.origin_label("seed:extr01-v1"), "Seed (extr01-v1)")
        self.assertEqual(provenance.origin_label("text"), "Document extraction")
        self.assertEqual(provenance.origin_label("structured"), "Structured import")
        self.assertEqual(provenance.origin_label(None), "Unknown")


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class SourcesIntegrationTests(LibraryTestCase):
    DIE = "bsk:component:EXTR01/die-head"
    MELT = "bsk:signal:EXTR01/melt_temp_c"

    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        self.doc, _ = self.upload("Extruder manual.pdf")
        self.client = APIClient()

    def approve_text_triple(self, page, subject_name="Die head assembly"):
        t = CandidateTriple.objects.create(
            document=self.doc, mode="schema", subject_name=subject_name, subject_type="Component", subject_id=self.DIE,
            subject_match="id", predicate="MONITORED_BY", object_name="Melt temperature", object_type="Signal",
            object_id=self.MELT, object_match="new", page_start=page, model="qwen", preset_name="P", preset_version=1)
        t.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, page_start=page,
                                               excerpt="The melt temperature is measured at the die head.",
                                               extractor="llm-text", model="qwen", prompt="P v1"))
        self.assertEqual(triples.approve([t])["approved"], [t.pk])
        return t

    def test_seeded_entity_and_edge(self):
        body = self.client.get(f"/api/graph/nodes/{self.DIE}/sources/").json()
        self.assertEqual((body["name"], body["origin"], body["sources"]), ("Die Head", "Seed (extr01-v1)", []))
        edge = self.client.get("/api/graph/edges/sources/", {"from": "bsk:asset:EXTR01", "type": "HAS_COMPONENT",
                                                              "to": self.DIE}).json()
        self.assertEqual(edge["origin"], "Seed (extr01-v1)")

    def test_extracted_facts_trace_to_the_page(self):
        t = self.approve_text_triple(4)
        body = self.client.get(f"/api/graph/nodes/{self.DIE}/sources/").json()
        self.assertEqual(body["aliases"], ["Die head assembly"])
        src = body["sources"][0]
        self.assertEqual((src["label"], src["document"]["filename"], src["page_start"], src["extractor"]),
                         ("Extruder manual.pdf p.4", "Extruder manual.pdf", 4, "llm-text"))
        self.assertEqual(src["triples"][0]["id"], t.pk)
        new = self.client.get(f"/api/graph/nodes/{self.MELT}/sources/").json()
        self.assertEqual((new["origin"], len(new["sources"])), ("Document extraction", 1))
        edge = self.client.get("/api/graph/edges/sources/", {"from": self.DIE, "type": "MONITORED_BY", "to": self.MELT}).json()
        self.assertEqual((edge["origin"], edge["sources"][0]["label"]), ("Document extraction", "Extruder manual.pdf p.4"))

        self.assertEqual(self.client.get("/api/graph/nodes/bsk:component:EXTR01/nope/sources/").status_code, 404)
        self.assertEqual(self.client.get("/api/graph/edges/sources/", {"from": self.DIE}).status_code, 400)

    def test_import_rows_and_lab_relationships(self):
        with override_settings(GRAPH_DATA_BASE=os.path.join(self.tmp, "gd")):
            df, _ = structured.add_data_file(SimpleUploadedFile("bom.csv", b"Item,Description\ngearbox,Gearbox\n"))
            structured.stage(df, structured.suggestions(df.columns)[0]["mapping"], "bsk:asset:EXTR01")
        triples.approve(list(df.triples.all()))
        body = self.client.get("/api/graph/nodes/bsk:component:EXTR01/gearbox/sources/").json()
        self.assertEqual((body["origin"], body["sources"][0]["label"]), ("Structured import", "bom.csv row 2"))

        lab = CandidateTriple.objects.create(document=self.doc, mode="freeform", subject_name="Die pressure",
                                             subject_type="Parameter", predicate="must stay below", object_name="99 bar",
                                             object_type="Value", page_start=2)
        lab.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, page_start=2, excerpt="x"))
        triples.approve([lab])
        edge = self.client.get("/api/graph/edges/sources/", {"triple_id": lab.pk}).json()
        self.assertEqual((edge["type"], edge["sources"][0]["label"]), ("must stay below", "Extruder manual.pdf p.2"))
        node = self.client.get(f"/api/graph/nodes/lab:{self.doc.doc_key}:die-pressure/sources/").json()
        self.assertEqual((node["label"], len(node["sources"])), ("Lab", 1))

    @patch("chat.asset_context.historian_snapshot", return_value=None)
    def test_asset_chat_facts_carry_their_source(self, _):
        from chat import asset_context
        self.approve_text_triple(4)
        ctx = asset_context.build("bsk:asset:EXTR01", "die head melt temperature", 100000)
        self.assertEqual(len(ctx.facts), len(ctx.fact_sources))
        self.assertTrue(all(ctx.fact_sources), "every cited fact has a source")
        by_fact = dict(zip(ctx.facts, ctx.fact_sources))
        die_line = next(f for f in by_fact if f.startswith("- Die Head"))
        self.assertEqual(by_fact[die_line], "Extruder manual.pdf p.4")     # extracted evidence wins over the seed label
        feeder_line = next(f for f in by_fact if f.startswith("- Feeder"))
        self.assertEqual(by_fact[feeder_line], "Seed (extr01-v1)")
        graph = asset_context.citations(ctx)[0]
        self.assertEqual(graph["fact_sources"], ctx.fact_sources)
