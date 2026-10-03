"""Triple extraction + review (P4).

Unit tests: the DGX LLM is mocked and the graph is disabled (entity matching
falls back to proposed ids). Integration tests (commit / delete / promote
against Neo4j) need the throwaway test instance — see tests_integration.py:

    GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from context_graph import driver, extraction, registry, services, triples
from context_graph.extraction import ExtractionError, parse_triples, render_template, validate_params
from context_graph.models import CandidateTriple, PromptPreset
from ingestion import library
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

TEST_URI = os.getenv("GRAPH_TEST_URI")
SEED = Path(__file__).resolve().parent.parent / "seeds" / "extr01.yaml"

SCHEMA_REPLY = {"triples": [
    {"subject": "Die Head", "subject_type": "Component", "predicate": "MONITORED_BY",
     "object": "Die Pressure", "object_type": "Signal", "confidence": 0.9},
    {"subject": "Die Head", "subject_type": "Component", "predicate": "HAS_ALARM",   # pair not allowed
     "object": "Die Plug", "object_type": "Alarm", "confidence": 1},
    {"subject": "", "subject_type": "Component", "predicate": "X", "object": "y", "object_type": "Signal"},  # incomplete
]}
FREEFORM_REPLY = {"triples": [
    {"subject": "Die pressure", "subject_type": "Parameter", "predicate": "must stay below",
     "object": "99 bar", "object_type": "Value", "confidence": 1},
]}


def fake_llm(messages):
    """Schema-guided prompts mention the allowed relationships; free-form ones don't."""
    return json.dumps(SCHEMA_REPLY if "MONITORED_BY" in messages[0]["content"] else FREEFORM_REPLY)


class ParsingTests(SimpleTestCase):
    def test_parse_triples_strips_thinking_skips_incomplete_and_clamps(self):
        content = "<think>hmm {not json}</think>\n" + json.dumps({"triples": [
            {"subject": "A", "subject_type": "Component", "predicate": "P", "object": "B", "object_type": "Signal",
             "confidence": 7},
            {"subject": "A", "predicate": "P"},
            "junk",
        ]})
        self.assertEqual(parse_triples(content), [{"subject": "A", "subject_type": "Component", "predicate": "P",
                                                   "object": "B", "object_type": "Signal", "confidence": 1.0}])
        for bad, message in [("no json here", "no JSON object"), ('{"triples": 3}', "no 'triples' list"),
                             ("{oops}", "invalid JSON")]:
            with self.assertRaisesMessage(ExtractionError, message):
                parse_triples(bad)

    def test_render_template_only_replaces_known_placeholders(self):
        out = render_template('Types: {entity_types}\nReturn {"triples": []} for {document} {unknown}',
                              entity_types="Asset", document="manual.pdf")
        self.assertEqual(out, 'Types: Asset\nReturn {"triples": []} for manual.pdf {unknown}')

    @patch("context_graph.extraction.call_llm", side_effect=["not json", json.dumps(FREEFORM_REPLY)])
    def test_extract_window_retries_once(self, llm):
        self.assertEqual(len(extraction.extract_window([{"role": "system", "content": "x"}])), 1)
        self.assertEqual(llm.call_count, 2)
        self.assertIn("ONLY the JSON object", llm.call_args.args[0][-1]["content"])


@override_settings(GRAPH_ENABLED=False)
class ExtractionPipelineTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch("context_graph.extraction.call_llm", side_effect=fake_llm)
        self.llm = patcher.start()
        self.addCleanup(patcher.stop)
        self.doc, _ = self.upload()
        self.drain()  # parse

    def extract(self, **params):
        library.enqueue(self.doc, Job.Kind.EXTRACT, params)
        self.drain()
        self.doc.refresh_from_db()

    def test_validate_params(self):
        params = validate_params({"mode": "both"})
        self.assertEqual(params["chunking"], extraction.DEFAULT_CHUNKING)
        self.assertEqual(set(params["presets"]), {"schema", "freeform"})
        for bad, message in [({"mode": "magic"}, "mode must be one of"),
                             ({"chunking": {"strategy": "fixed", "params": {"size": 100, "overlap": 100}}}, "smaller than size"),
                             ({"presets": {"schema": 999}}, "unknown schema prompt preset"),
                             ({"asset_id": "EXTR01"}, "canonical id")]:
            with self.assertRaisesMessage(ExtractionError, message):
                validate_params(bad)

    def test_schema_extraction_stages_deduplicated_checked_triples(self):
        self.extract(mode="schema", asset_id="bsk:asset:EXTR01")
        self.assertEqual(self.doc.graph_status, "done")
        windows = self.llm.call_count
        staged = {t.predicate: t for t in self.doc.triples.all()}
        self.assertEqual(set(staged), {"MONITORED_BY", "HAS_ALARM"})
        ok = staged["MONITORED_BY"]
        self.assertEqual((ok.status, ok.mode, ok.issue, ok.occurrences), ("pending", "schema", "", windows))
        self.assertEqual((ok.subject_id, ok.object_id),  # graph disabled -> proposed ids in the asset scope
                         ("bsk:component:EXTR01/Die-Head", "bsk:signal:EXTR01/Die-Pressure"))
        self.assertEqual((ok.preset_name, ok.preset_version, ok.page_start), ("Default schema-guided", 1, 1))
        self.assertTrue(ok.evidence_text)
        self.assertIn("not allowed by the schema", staged["HAS_ALARM"].issue)

    def test_freeform_and_rerun_keep_reviewed_triples(self):
        self.extract(mode="both")
        self.assertEqual(set(self.doc.triples.values_list("mode", flat=True)), {"freeform", "schema"})
        freeform = self.doc.triples.get(mode="freeform")
        self.assertEqual((freeform.subject_id, freeform.issue), ("", ""))  # no schema checks / ids in free-form
        triples.reject([self.doc.triples.get(predicate="HAS_ALARM")])

        self.extract(mode="both")
        self.assertEqual(self.doc.triples.filter(predicate="HAS_ALARM").count(), 1)   # not re-staged
        self.assertEqual(self.doc.triples.get(predicate="HAS_ALARM").status, "rejected")
        self.assertEqual(self.doc.triples.filter(status="pending").count(), 2)        # replaced, not duplicated
        self.assertIn("already reviewed", self.doc.jobs.filter(kind="extract").latest("created_at").message)

        self.extract(mode="schema")   # a schema-only re-run leaves pending free-form triples alone
        self.assertEqual(self.doc.triples.filter(mode="freeform", status="pending").count(), 1)

    def test_failed_window_is_noted_not_fatal(self):
        self.llm.side_effect = [RuntimeError("DGX down")] + [json.dumps(SCHEMA_REPLY)] * 50  # transport errors: no retry
        self.extract(mode="schema")
        self.assertEqual(self.doc.graph_status, "done")
        self.assertEqual(self.doc.graph_error, "1 window call(s) failed")

    def test_timeout_is_retried_once(self):
        self.llm.side_effect = [requests.Timeout("busy")] + [json.dumps(SCHEMA_REPLY)] * 50
        self.extract(mode="schema")
        self.assertEqual(self.doc.graph_error, "")
        self.llm.side_effect = [requests.Timeout("busy")] * 2 + [json.dumps(SCHEMA_REPLY)] * 50
        self.extract(mode="schema")
        self.assertEqual(self.doc.graph_error, "1 window call(s) failed")

    def test_evidence_types_are_neither_prompted_nor_staged(self):
        entity_types, relationships = extraction.render_schema(registry.active_schema())
        self.assertNotIn("Document", entity_types)
        self.assertNotIn("HAS_SECTION", relationships)
        self.assertIn("Component -> Signal", relationships)
        self.llm.side_effect = lambda m: json.dumps({"triples": [
            {"subject": "EXTR01", "subject_type": "Asset", "predicate": "DOCUMENTED_BY",
             "object": "manual.pdf", "object_type": "Document"}]})
        self.extract(mode="schema")
        self.assertFalse(self.doc.triples.exists())

    def test_edit_reresolves_ids_and_flags_issues(self):
        self.extract(mode="schema")
        t = self.doc.triples.get(predicate="HAS_ALARM")
        triples.edit(t, {"subject_type": "Asset", "subject_name": "EXTR01"})
        t.refresh_from_db()
        self.assertEqual((t.subject_id, t.issue, t.edited), ("bsk:asset:EXTR01", "", True))
        triples.edit(t, {"object_id": "bsk:signal:EXTR01/x"})
        self.assertIn("not a valid Alarm id", t.issue)
        with self.assertRaisesMessage(triples.TripleError, "not editable"):
            triples.edit(t, {"status": "approved"})
        t.status = "approved"
        with self.assertRaisesMessage(triples.TripleError, "can't be edited"):
            triples.edit(t, {"predicate": "X"})


@override_settings(GRAPH_ENABLED=False)
class ExtractionApiTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.client = APIClient()
        patcher = patch("context_graph.extraction.call_llm", side_effect=fake_llm)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.doc, _ = self.upload()

    def test_extract_config(self):
        body = self.client.get("/api/graph/extract/config/").json()
        self.assertEqual((body["provider"], body["modes"]), ("DGX", ["schema", "freeform", "both"]))
        self.assertEqual(body["default_chunking"], extraction.DEFAULT_CHUNKING)
        self.assertIn("fixed", body["chunking_strategies"])   # same shape as /api/rag/config/
        self.assertNotIn("Document", body["schema"]["entity_types"])
        rels = {r["name"]: r["pairs"] for r in body["schema"]["relationships"]}
        self.assertIn(["Component", "Signal"], rels["MONITORED_BY"])
        self.assertNotIn("HAS_SECTION", rels)

    def test_presets_crud_and_versioning(self):
        listed = self.client.get("/api/graph/presets/").json()["presets"]
        self.assertEqual({(p["mode"], p["is_default"], p["builtin"]) for p in listed},
                         {("schema", True, True), ("freeform", True, True)})
        r = self.client.post("/api/graph/presets/", {"name": "Alarms only", "mode": "schema", "template": "T {text}"},
                             format="json")
        self.assertEqual(r.status_code, 201)
        pid = r.json()["id"]
        self.assertEqual(self.client.post("/api/graph/presets/", {"name": "Alarms only", "mode": "schema",
                                                                  "template": "x"}, format="json").status_code, 409)
        r = self.client.patch(f"/api/graph/presets/{pid}/", {"template": "T2 {text}", "is_default": True}, format="json")
        self.assertEqual((r.json()["version"], r.json()["is_default"]), (2, True))
        self.assertEqual(extraction.default_preset("schema").pk, pid)
        builtin = next(p for p in listed if p["mode"] == "schema")
        self.assertEqual(self.client.delete(f"/api/graph/presets/{builtin['id']}/").status_code, 409)
        self.assertEqual(self.client.delete(f"/api/graph/presets/{pid}/").status_code, 204)
        self.assertTrue(PromptPreset.objects.get(pk=builtin["id"]).is_default)   # default falls back

    def test_extract_review_flow(self):
        r = self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "mode": "both"}, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        self.assertEqual(r.json()["jobs"][0]["kind"], "extract")
        self.assertEqual(self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "mode": "x"},
                                          format="json").status_code, 400)
        self.drain()

        body = self.client.get(f"/api/graph/triples/?document={self.doc.id}&status=pending").json()
        self.assertEqual(body["counts"], {"pending": 3, "approved": 0, "rejected": 0})
        by_pred = {t["predicate"]: t for t in body["triples"]}
        self.assertEqual(by_pred["must stay below"]["layer"], None)
        self.assertEqual(self.client.get("/api/graph/triples/?issues=1").json()["total"], 1)
        self.assertEqual(self.client.get("/api/graph/triples/?q=pressure").json()["total"], 2)

        bad = by_pred["HAS_ALARM"]["id"]
        r = self.client.patch(f"/api/graph/triples/{bad}/", {"subject_type": "Asset", "subject_name": "EXTR01"},
                              format="json")
        self.assertEqual((r.json()["subject"]["id"], r.json()["issue"]), ("bsk:asset:EXTR01", ""))
        self.assertEqual(self.client.post("/api/graph/triples/reject/", {"ids": [bad]}, format="json").json(),
                         {"rejected": 1})
        doc = self.client.get(f"/api/documents/{self.doc.id}/").json()
        self.assertEqual(doc["graph"]["triples"], {"pending": 2, "approved": 0, "rejected": 1})

        # Approving needs the graph.
        r = self.client.post("/api/graph/triples/approve/", {"ids": [by_pred["MONITORED_BY"]["id"]]}, format="json")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.client.post("/api/graph/triples/delete/", {"ids": [bad, 99999]},
                                          format="json").status_code, 404)
        r = self.client.post("/api/graph/triples/delete/", {"ids": [t["id"] for t in body["triples"]]}, format="json")
        self.assertEqual(r.json(), {"deleted": 3})
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.graph_status, "none")


@unittest.skipUnless(TEST_URI, "set GRAPH_TEST_URI (see tests_integration.py) to run against a test Neo4j")
@override_settings(GRAPH_ENABLED=True, NEO4J_URI=TEST_URI or "", NEO4J_USER="neo4j",
                   NEO4J_PASSWORD=os.getenv("NEO4J_TEST_PASSWORD", "bsklab-test-only"), NEO4J_DATABASE="neo4j")
class ReviewIntegrationTests(LibraryTestCase):
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

    def q(self, cypher, **params):
        with driver.session() as s:
            return s.run(cypher, **params).data()

    def stage(self, mode="schema", **fields):
        defaults = dict(document=self.doc, mode=mode, subject_name="Die Head", subject_type="Component",
                        subject_id="bsk:component:EXTR01/die-head", predicate="MONITORED_BY",
                        object_name="Melt viscosity", object_type="Signal", object_id="bsk:signal:EXTR01/melt-viscosity",
                        page_start=3, page_end=3, section_path=["Die Head", "Sensors"], model="qwen", preset_name="P")
        defaults.update(fields)
        return CandidateTriple.objects.create(**defaults)

    def approve(self, *ts):
        return self.client.post("/api/graph/triples/approve/", {"ids": [t.pk for t in ts]}, format="json").json()

    def test_approve_writes_provenance_and_evidence_then_delete_cleans_up(self):
        base_nodes = self.q("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"]
        t = self.stage()
        self.assertEqual(self.approve(t), {"approved": [t.pk], "errors": {}, "layers": {"curated": 1, "lab": 0}})
        t.refresh_from_db()
        self.assertEqual((t.status, t.layer), ("approved", "curated"))

        rel = self.q("MATCH (:Entity {id: $s})-[r:MONITORED_BY]->(n:Entity {id: $o}) RETURN properties(r) AS r, n",
                     s=t.subject_id, o=t.object_id)[0]
        self.assertEqual((rel["r"]["source"], rel["r"]["triple_ids"], rel["r"]["page_start"]), ("text", [t.pk], 3))
        self.assertEqual((rel["n"]["name"], rel["n"]["source"]), ("Melt viscosity", "text"))
        # Existing entity keeps its name; evidence: Document -> Section -> DESCRIBES both ends.
        self.assertEqual(self.q("MATCH (n {id: 'bsk:component:EXTR01/die-head'}) RETURN n.name AS n")[0]["n"], "Die Head")
        described = self.q("MATCH (:Document {doc_key: $k})-[:HAS_SECTION]->(s)-[:DESCRIBES]->(n) RETURN n.id AS id",
                           k=self.doc.doc_key)
        self.assertEqual({r["id"] for r in described}, {t.subject_id, t.object_id})

        self.assertEqual(self.client.patch(f"/api/graph/triples/{t.pk}/", {"predicate": "X"}, format="json").status_code, 400)
        self.client.post("/api/graph/triples/delete/", {"ids": [t.pk]}, format="json")
        self.assertEqual(self.q("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"], base_nodes)  # node, doc, section gone
        self.assertEqual(self.q("MATCH (n {id: 'bsk:component:EXTR01/die-head'}) RETURN count(n) AS n")[0]["n"], 1)

    def test_restating_a_seed_edge_never_deletes_it(self):
        seed = self.q("MATCH (a:Asset {id: 'bsk:asset:EXTR01'})-[r:HAS_COMPONENT]->(c {id: 'bsk:component:EXTR01/die-head'}) "
                      "RETURN properties(r) AS p")[0]["p"]
        t1 = self.stage(subject_name="EXTR01", subject_type="Asset", subject_id="bsk:asset:EXTR01",
                        predicate="HAS_COMPONENT", object_name="Die Head", object_type="Component",
                        object_id="bsk:component:EXTR01/die-head")
        t2 = self.stage(subject_name="Extruder", subject_type="Asset", subject_id="bsk:asset:EXTR01",
                        predicate="HAS_COMPONENT", object_name="die head", object_type="Component",
                        object_id="bsk:component:EXTR01/die-head")
        self.assertEqual(len(self.approve(t1, t2)["approved"]), 2)
        rel = self.q("MATCH (:Asset {id: 'bsk:asset:EXTR01'})-[r:HAS_COMPONENT]->({id: 'bsk:component:EXTR01/die-head'}) "
                     "RETURN properties(r) AS p")
        self.assertEqual(len(rel), 1)
        self.assertEqual(rel[0]["p"]["source"], seed.get("source"))    # seed provenance untouched
        self.assertEqual(sorted(rel[0]["p"]["triple_ids"]), sorted([t1.pk, t2.pk]))
        self.client.post("/api/graph/triples/delete/", {"ids": [t1.pk, t2.pk]}, format="json")
        rel = self.q("MATCH (:Asset {id: 'bsk:asset:EXTR01'})-[r:HAS_COMPONENT]->({id: 'bsk:component:EXTR01/die-head'}) "
                     "RETURN r.triple_ids AS ids")
        self.assertEqual(rel, [{"ids": []}])

    def test_invalid_schema_triple_is_reported_not_written(self):
        t = self.stage(predicate="HAS_ALARM")
        result = self.approve(t)
        self.assertEqual(result["approved"], [])
        self.assertIn("not allowed by the schema", result["errors"][str(t.pk)])

    def test_lab_layer_commit_promote_and_document_delete(self):
        lab = self.stage(mode="freeform", subject_name="Die pressure", subject_type="Parameter", subject_id="",
                         predicate="must stay below", object_name="99 bar", object_type="Value", object_id="")
        self.assertEqual(self.approve(lab)["layers"], {"curated": 0, "lab": 1})
        self.assertEqual(self.client.get("/api/graph/lab/").json(), {"nodes": 2, "relationships": 1, "documents": [
            {"doc_key": self.doc.doc_key, "nodes": 2, "filename": self.doc.filename}]})
        data = self.client.get(f"/api/graph/data/?layer=lab&doc={self.doc.doc_key}").json()
        self.assertEqual([link["type"] for link in data["links"]], ["must stay below"])
        self.assertEqual(self.q("MATCH (n:Lab:Entity) RETURN count(n) AS n")[0]["n"], 0)   # never curated
        curated = self.client.get("/api/graph/data/?limit=1000").json()
        self.assertFalse(any(n["id"].startswith("lab:") for n in curated["nodes"]))

        r = self.client.post(f"/api/graph/triples/{lab.pk}/promote/",
                             {"subject_type": "Component", "subject_name": "Die Head", "predicate": "MONITORED_BY",
                              "object_type": "Signal", "object_name": "Die Pressure"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual((body["mode"], body["layer"], body["status"]), ("schema", "curated", "approved"))
        self.assertEqual(body["subject"]["id"], "bsk:component:EXTR01/die-head")    # matched the seeded entity
        self.assertTrue(body["subject"]["existing"])
        self.assertEqual(self.q("MATCH (n:Lab) RETURN count(n) AS n")[0]["n"], 0)

        before = self.q("MATCH ()-[r]->() WHERE $id IN coalesce(r.triple_ids, []) RETURN count(r) AS n", id=lab.pk)[0]["n"]
        self.assertGreater(before, 0)
        self.assertEqual(self.client.delete(f"/api/documents/{self.doc.id}/").status_code, 200)
        self.assertEqual(self.q("MATCH ()-[r]->() WHERE $id IN coalesce(r.triple_ids, []) RETURN count(r) AS n",
                                id=lab.pk)[0]["n"], 0)
        self.assertEqual(self.q("MATCH (d:Document {doc_key: $k}) RETURN count(d) AS n", k=self.doc.doc_key)[0]["n"], 0)
