"""P8a: evidence records, all occurrences, aliases and evidence IDs in the graph.

Integration tests need the throwaway Neo4j (see tests_integration.py):
    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
"""

import importlib
import json
import os
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.apps import apps
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import driver, services, structured, triples
from context_graph.models import CandidateTriple, Evidence
from context_graph.tests.tests_extraction import fake_llm
from ingestion import library
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

TEST_URI = os.getenv("GRAPH_TEST_URI")
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"


@override_settings(GRAPH_ENABLED=False)
class StagingEvidenceTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch("context_graph.extraction.call_llm", side_effect=fake_llm)
        self.llm = patcher.start()
        self.addCleanup(patcher.stop)
        self.doc, _ = self.upload()
        self.drain()

    def extract(self, **params):
        library.enqueue(self.doc, Job.Kind.EXTRACT, {"mode": "schema", **params})
        self.drain()

    def test_every_window_is_kept_as_evidence(self):
        self.extract()
        windows = self.llm.call_count
        self.assertGreater(windows, 1)
        t = self.doc.triples.get(predicate="MONITORED_BY")
        evidence = list(t.evidence.order_by("chunk_index"))
        self.assertEqual((len(evidence), t.occurrences), (windows, windows))
        self.assertEqual([e.chunk_index for e in evidence], list(range(windows)))
        first = evidence[0]
        self.assertEqual((first.source_kind, first.extractor, first.document_id, first.page_start),
                         ("text", "llm-text", self.doc.pk, 1))
        self.assertEqual((first.prompt, first.window), ("Default schema-guided v1",
                                                        {"strategy": "fixed", "params": {"size": 800, "overlap": 100}}))
        self.assertTrue(first.excerpt)

        body = APIClient().get(f"/api/graph/triples/?document={self.doc.id}&q=MONITORED").json()
        sources = body["triples"][0]["sources"]
        self.assertEqual(len(sources), windows)
        detail = APIClient().get(f"/api/graph/evidence/{sources[0]['id']}/").json()
        self.assertEqual(detail["triple_ids"], [t.pk])

    def test_rerun_and_delete_leave_no_orphan_evidence(self):
        self.extract()
        before = Evidence.objects.count()
        self.extract()                                        # pending triples replaced
        self.assertEqual(Evidence.objects.count(), before)
        triples.delete(list(self.doc.triples.all()))
        self.assertEqual(Evidence.objects.count(), 0)

    def test_structured_rows_are_evidence(self):
        csv = b"Item,Description\nfeeder,Feeder\nfeeder,Feeder\ngearbox,Gearbox\n"
        with override_settings(GRAPH_DATA_BASE=os.path.join(self.tmp, "graph_data")):
            df, _ = structured.add_data_file(SimpleUploadedFile("bom.csv", csv))
            mapping = structured.suggestions(df.columns)[0]["mapping"]
            structured.stage(df, mapping, "bsk:asset:EXTR01")
        feeder = df.triples.get(object_id="bsk:component:EXTR01/feeder")
        self.assertEqual(feeder.occurrences, 2)
        self.assertEqual(sorted(feeder.evidence.values_list("row_number", flat=True)), [2, 3])
        self.assertEqual(set(feeder.evidence.values_list("source_kind", "extractor")), {("structured", "column-mapping")})

    def test_migration_backfills_one_evidence_per_triple(self):
        self.extract()
        CandidateTriple.evidence.through.objects.all().delete()
        Evidence.objects.all().delete()
        t = self.doc.triples.get(predicate="MONITORED_BY")
        migration = importlib.import_module("context_graph.migrations.0005_backfill_evidence")
        migration.forwards(apps, None)
        migration.forwards(apps, None)                        # idempotent
        e = t.evidence.get()
        self.assertEqual((e.source_kind, e.page_start, e.excerpt, e.prompt),
                         ("text", t.page_start, t.evidence_text, "Default schema-guided v1"))
        self.assertEqual(Evidence.objects.count(), self.doc.triples.count())


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class GraphEvidenceIntegrationTests(LibraryTestCase):
    DIE = "bsk:component:EXTR01/die-head"
    SIG = "bsk:signal:EXTR01/die_pressure_bar"

    def setUp(self):
        super().setUp()
        driver.close_driver()
        self.addCleanup(driver.close_driver)
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
        services.init_graph()
        services.load_seed(SEED)
        self.doc, _ = self.upload()

    def q(self, cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    def stage(self, subject_name, page):
        t = CandidateTriple.objects.create(
            document=self.doc, mode="schema", subject_name=subject_name, subject_type="Component", subject_id=self.DIE,
            predicate="MONITORED_BY", object_name="Die pressure", object_type="Signal", object_id=self.SIG,
            page_start=page, model="qwen", preset_name="P", preset_version=1)
        t.evidence.add(Evidence.objects.create(source_kind="text", document=self.doc, page_start=page, excerpt=f"p{page}"))
        return t

    def node(self, node_id):
        return self.q("MATCH (n {id: $id}) RETURN n.name AS name, coalesce(n.aliases, []) AS aliases, "
                      "coalesce(n.evidence_ids, []) AS ev", id=node_id)[0]

    def test_aliases_and_evidence_ids_are_added_and_removed_exactly(self):
        a = self.stage("Die head assembly", 3)
        b = self.stage("die head assembly", 7)                # same alias (normalized) from another page
        c = self.stage("Die Head", 9)                         # the node's own name: no alias
        self.assertEqual(len(triples.approve([a, b, c])["approved"]), 3)
        die = self.node(self.DIE)
        self.assertEqual((die["name"], die["aliases"]), ("Die Head", ["Die head assembly"]))   # never renamed
        ev = [e.pk for t in (a, b, c) for e in t.evidence.all()]
        self.assertEqual(sorted(die["ev"]), sorted(ev))
        edge = self.q("MATCH ({id: $s})-[r:MONITORED_BY]->({id: $o}) RETURN r.evidence_ids AS ev, r.source AS src",
                      s=self.DIE, o=self.SIG)[0]
        self.assertEqual(sorted(edge["ev"]), sorted(ev))
        self.assertNotEqual(edge["src"], "text")             # seeded edge keeps its provenance

        triples.delete([CandidateTriple.objects.get(pk=a.pk)])
        die = self.node(self.DIE)
        self.assertEqual(die["aliases"], ["Die head assembly"])   # b still names it that way… (normalized match)
        triples.delete([CandidateTriple.objects.get(pk=b.pk), CandidateTriple.objects.get(pk=c.pk)])
        die = self.node(self.DIE)
        self.assertEqual((die["aliases"], die["ev"]), ([], []))
        self.assertEqual(self.q("MATCH ({id: $s})-[r:MONITORED_BY]->({id: $o}) RETURN count(r) AS n",
                                s=self.DIE, o=self.SIG)[0]["n"], 1)   # the seeded edge survives

    def test_backfill_command_restores_graph_links_idempotently(self):
        t = self.stage("Die head assembly", 3)
        triples.approve([t])
        self.q("MATCH (n {id: $id}) REMOVE n.evidence_ids, n.aliases", id=self.DIE)
        CandidateTriple.objects.filter(pk=t.pk).update(applied_aliases={})
        out = StringIO()
        call_command("graph_backfill_evidence", stdout=out)
        call_command("graph_backfill_evidence", stdout=StringIO())
        die = self.node(self.DIE)
        self.assertEqual((die["aliases"], die["ev"]), (["Die head assembly"], [t.evidence.get().pk]))
        self.assertIn("1 aliases added", out.getvalue())
        self.assertEqual(CandidateTriple.objects.get(pk=t.pk).applied_aliases, {self.DIE: ["Die head assembly"]})
