"""Room for long answers (2026-10-18): windows in parallel (GRAPH_EXTRACT_PARALLEL) and the Controls & Setup run's
smaller windows. Angel: no cap on the triples — more compute and a more specific prompt instead. DGX mocked."""

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework.test import APIClient

from context_graph import extraction, registry
from context_graph.models import CandidateTriple
from ingestion import library
from ingestion.chunker import PAGE_BREAK
from ingestion.tests.tests_library import LibraryTestCase

V6 = yaml.safe_load((Path(extraction.__file__).parent / "schemas" / "industrial_v6.yaml").read_text())
PAGE = "Set p{n:04d} to the value of step {n}. Terminal {n} is DI {n} and its function is set in p{n:04d}. "
MANUAL = {"status": "success", "errors": [], "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(
    [f"# 5.{k} Commissioning\n\n" + "".join(PAGE.format(n=k * 100 + j) for j in range(40)) for k in range(1, 4)])}}


@override_settings(GRAPH_ENABLED=False)
class ExtractRoomTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.mock_convert.return_value = MANUAL
        registry.save_version(V6, note="v6")
        self.doc, _ = library.add_document(SimpleUploadedFile("g120c.pdf", b"%PDF g120c"))
        self.drain()
        self.client = APIClient()
        self.texts, self.threads, self.lock = [], set(), threading.Lock()

        def llm(messages):
            text = messages[-1]["content"]
            with self.lock:
                self.texts.append(text)
                self.threads.add(threading.get_ident())
            time.sleep(0.01)
            codes = sorted(set(__import__("re").findall(r"p\d{4}", text)))
            return json.dumps({"triples": [
                {"subject": "Drive", "subject_type": "Component", "predicate": "HAS_PARAMETER", "object": f"Step {c}",
                 "object_type": "Parameter", "object_subtype": "Setting", "object_properties": {"code": c}, "confidence": 0.9}
                for c in codes]})
        p = patch("context_graph.extraction.call_llm", side_effect=llm)
        p.start()
        self.addCleanup(p.stop)

    def run_extract(self, **body):
        CandidateTriple.objects.filter(document=self.doc).delete()
        self.texts.clear()
        self.threads.clear()
        r = self.client.post("/api/graph/extract/", {"document_ids": [str(self.doc.id)], "mode": "schema", **body}, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        self.drain()
        return list(CandidateTriple.objects.filter(document=self.doc).order_by("id")
                    .values_list("object_id", "chunk_index", "occurrences"))

    def test_parallel_windows_give_the_same_triples(self):
        one = self.run_extract()
        self.assertEqual(len(self.threads), 1)
        with override_settings(GRAPH_EXTRACT_PARALLEL=3):
            three = self.run_extract()
        self.assertGreater(len(self.threads), 1)                                  # the windows went out together
        self.assertEqual(three, one)                                              # same triples, same order
        self.assertEqual(len(one), 120)

    def test_the_controls_run_reads_smaller_windows(self):
        with patch("context_graph.page_suggest.suggest", return_value={"pages": "1-3", "sections": [], "note": ""}):
            self.run_extract(facet_runs=["controls"])
            small = list(self.texts)
            self.run_extract(facet_runs=["controls"], chunking={"strategy": "fixed", "params": {"size": 800, "overlap": 100}})
            chosen = list(self.texts)                                               # the reviewer's windows are kept
        self.assertGreater(len(small), len(chosen))
        self.assertLess(max(len(t) for t in small), max(len(t) for t in chosen))
