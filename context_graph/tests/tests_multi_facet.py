"""P22c-0 (2026-10-18): "All facets" — one extract job per document goes through the facet prompts in turn; each
picks its own pages and stages its triples. The DGX is mocked; the graph is disabled."""

import json
from pathlib import Path
from unittest.mock import patch

import requests
import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import extraction, registry
from context_graph.extraction import ExtractionError, validate_params
from context_graph.models import CandidateTriple
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

SCHEMAS = Path(extraction.__file__).parent / "schemas"
V3 = yaml.safe_load((SCHEMAS / "industrial_v3.yaml").read_text())
V5 = yaml.safe_load((SCHEMAS / "industrial_v5.yaml").read_text())
TEXT = "Check the pump diaphragm every 5000 hours and replace it when worn. The rated pressure is 7 bar. " * 4
MANUAL = {"status": "success", "errors": [], "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(
    [f"# 8. Maintenance\n\n{TEXT}", f"# 8.1 Troubleshooting\n\n{TEXT}", f"# 4. Technical data\n\n{TEXT}"])}}


@override_settings(GRAPH_ENABLED=False)
class MultiFacetRunTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = MANUAL
        self.suggested = {"Maintenance": "1-2", "Specifications": "3"}   # the pages each facet's prompt picks
        self.down = set()                                                 # prompts whose calls fail
        self.read = []                                                    # (prompt, text) of each extraction call

        def llm(messages):
            system = messages[0]["content"]
            if system.startswith("You choose the pages"):
                which = "Specifications" if "technical data and specifications" in system else "Maintenance"
                if which in self.down:
                    raise requests.ConnectionError("vLLM refused")
                pages = self.suggested[which]
                return json.dumps({"sections": [{"pages": pages, "title": "x", "why": "y"}] if pages else [],
                                   "note": "" if pages else "no technical data"})
            which = "Specifications" if "never an OperatingLimit" in system else "Maintenance"
            if which in self.down:
                raise requests.ConnectionError("vLLM refused")
            self.read.append((which, messages[-1]["content"]))
            if which == "Specifications":
                return json.dumps({"triples": [{"subject": "Dosing pump", "subject_type": "Component", "predicate": "HAS_SPEC",
                                                "object": "Max. operating pressure", "object_type": "Specification",
                                                "object_properties": {"value": "7", "unit": "bar"}, "confidence": 0.9}]})
            return json.dumps({"triples": [{"subject": "Dosing pump", "subject_type": "Component", "predicate": "HAS_PROCEDURE",
                                            "object": "Replace the diaphragm", "object_type": "Procedure",
                                            "object_properties": {"subtype": "Replacement"}, "confidence": 0.9}]})
        p = patch("context_graph.extraction.call_llm", side_effect=llm)
        p.start()
        self.addCleanup(p.stop)
        registry.save_version(V5, note="v5")
        self.doc, _ = library.add_document(SimpleUploadedFile("dda.pdf", b"%PDF dda"))
        self.drain()
        self.doc.refresh_from_db()
        self.client = APIClient()

    def run_all(self, **body):
        r = self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "facet_runs": "all", **body},
                             format="json")
        self.assertEqual(r.status_code, 202, r.content)
        self.read.clear()
        self.drain()
        return Job.objects.filter(document=self.doc, kind=Job.Kind.EXTRACT).latest("created_at")

    def by_facet_run(self, job):
        return {r["key"]: r for r in job.params["facet_results"]}

    def test_the_runs_offered_follow_the_schema(self):
        self.assertEqual([r["key"] for r in extraction.facet_runs_available()], ["maintenance", "ratings"])
        self.assertEqual(validate_params({"facet_runs": "all"})["facet_runs"], ["maintenance", "ratings"])
        self.assertEqual(validate_params({"facet_runs": ["ratings", "maintenance"]})["facet_runs"],
                         ["maintenance", "ratings"])                         # always in the facets' order
        with self.assertRaisesMessage(ExtractionError, "not available: safety"):
            validate_params({"facet_runs": ["safety"]})
        registry.save_version(V3, note="v3")                                 # no Specification: no ratings run
        self.assertEqual([r["key"] for r in extraction.facet_runs_available()], ["maintenance"])
        config = self.client.get("/api/graph/extract/config/").json()
        self.assertEqual([r["key"] for r in config["facet_runs"]], ["maintenance"])

    def test_each_facet_reads_its_own_pages_and_stages_its_triples(self):
        job = self.run_all()
        self.assertEqual(job.status, Job.Status.DONE, job.error)
        results = self.by_facet_run(job)
        self.assertEqual((results["maintenance"]["state"], results["maintenance"]["pages"], results["maintenance"]["staged"]),
                         ("done", "1-2", 1))
        self.assertEqual((results["ratings"]["state"], results["ratings"]["pages"], results["ratings"]["staged"]),
                         ("done", "3", 1))
        self.assertTrue(all("Technical data" not in text for which, text in self.read if which == "Maintenance"))
        self.assertTrue(any(which == "Specifications" for which, _ in self.read))
        self.assertTrue(all("Maintenance" not in text and "Troubleshooting" not in text
                            for which, text in self.read if which == "Specifications"))
        self.assertEqual(job.message, "Maintenance: 1 · Ratings: 1")
        staged = {t.preset_name: t for t in CandidateTriple.objects.filter(document=self.doc)}
        self.assertEqual(staged["Maintenance (v3)"].facets, ["Maintenance"])
        self.assertEqual((staged["Specifications (v4)"].object_type, staged["Specifications (v4)"].status),
                         ("Specification", "pending"))

    def test_a_run_replaces_only_its_own_prompts_pending_triples(self):
        self.run_all()
        self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "facet_runs": ["ratings"]}, format="json")
        self.drain()
        self.assertEqual(sorted(CandidateTriple.objects.filter(document=self.doc, status="pending")
                                .values_list("preset_name", flat=True)), ["Maintenance (v3)", "Specifications (v4)"])

    def test_reviewed_facets_are_skipped_nothing_found_and_failures_are_per_facet(self):
        self.run_all()
        CandidateTriple.objects.filter(document=self.doc, preset_name="Maintenance (v3)").update(status="approved")
        self.suggested["Specifications"] = ""                                  # no technical data in this manual
        job = self.run_all()
        results = self.by_facet_run(job)
        self.assertEqual(results["maintenance"]["state"], "skipped")
        self.assertIn("1 triples already reviewed", results["maintenance"]["note"])
        self.assertEqual((results["ratings"]["state"], results["ratings"]["note"]), ("none", "no technical data"))
        self.assertEqual(job.status, Job.Status.DONE)
        self.assertEqual(job.message, "Maintenance: skipped · Ratings: nothing found")

        self.suggested["Specifications"] = "3"
        self.down = {"Specifications"}                                         # the DGX drops during one facet
        job = self.run_all(rerun_reviewed=True)
        results = self.by_facet_run(job)
        self.assertEqual(results["maintenance"]["state"], "done")
        self.assertEqual(results["ratings"]["state"], "failed")
        self.assertIn("choosing pages failed", results["ratings"]["note"])
        self.assertEqual(job.status, Job.Status.DONE)                          # one facet done: the job is done

        self.down = {"Specifications", "Maintenance"}
        job = self.run_all(rerun_reviewed=True)
        self.assertEqual(job.status, Job.Status.FAILED)                        # every facet failed: the job failed
        self.assertIn("every facet failed", job.error)

    def test_facet_counts_for_the_review_tabs(self):
        self.run_all()
        r = self.client.get("/api/graph/triples/", {"document": str(self.doc.id), "facet_counts": "1", "facet": "Maintenance"}).json()
        self.assertEqual(r["total"], 1)
        self.assertEqual({k: v for k, v in r["facet_counts"].items() if v},
                         {"Maintenance": 1, "none": 1})          # the rating has no subtype: no facet
        self.assertNotIn("facet_counts", self.client.get("/api/graph/triples/", {"document": str(self.doc.id)}).json())
