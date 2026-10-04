"""Drawings to triples (P9a.3): Generate triples with `include_drawings` sends the described
drawings to the VLM and stages the result with vision evidence. The DGX and the VLM are mocked."""

import json
from unittest.mock import patch

from context_graph import drawings, extraction, provenance
from context_graph.models import CandidateTriple, Evidence
from context_graph.tests.tests_extraction import fake_llm
from gpu import orchestrator as gpu
from ingestion import figures, library
from ingestion.models import Job
from ingestion.tests.tests_library import LibraryTestCase

DRAWING_REPLY = {"triples": [
    {"subject": "VFD101", "subject_type": "Component", "predicate": "DRIVES", "object": "M101", "object_type": "Component"},
    {"subject": "Die Head", "subject_type": "Component", "predicate": "MONITORED_BY",      # also found in the text
     "object": "Die Pressure", "object_type": "Signal"},
]}
FIGURES = {"source": "pdf", "document_revision": "1", "figures": [
    {"index": 0, "page": 2, "bbox": {"l": 10, "t": 700, "r": 300, "b": 400, "coord_origin": "BOTTOMLEFT"},
     "page_size": {"width": 600, "height": 800}, "caption": "Fig. 3 Drive train", "area": 0.2, "status": "done",
     "skip_reason": "", "kind": "schematic", "description": "A VFD feeds motor M101."},
    {"index": 1, "page": 2, "bbox": None, "page_size": None, "caption": "", "area": 0.1, "status": "done",
     "skip_reason": "", "kind": "photo", "description": "A photo of the gearbox."},
    {"index": 2, "page": 3, "bbox": None, "page_size": None, "caption": "", "area": 0.1, "status": "pending", "skip_reason": ""},
]}


class DrawingTriplesTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        for target, kwargs in (("context_graph.extraction.call_llm", {"side_effect": fake_llm}),
                               ("ingestion.figures.crop", {"return_value": (b"JPEG", 100, 80)})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("context_graph.drawings.vlm.complete", return_value=json.dumps(DRAWING_REPLY))
        self.vlm = patcher.start()
        self.addCleanup(patcher.stop)
        self.doc, _ = self.upload()
        library.run_job(library.claim_next())            # parse
        figures.save(self.doc, FIGURES)

    def extract(self, **params):
        job = library.enqueue(self.doc, Job.Kind.EXTRACT, {"mode": "schema", **params})
        library.run_job(library.claim_next())
        job.refresh_from_db()
        self.doc.refresh_from_db()
        return job

    def test_drawings_are_only_read_when_asked(self):
        self.assertFalse(extraction.validate_params({})["include_drawings"])
        self.extract()
        self.vlm.assert_not_called()
        self.assertFalse(Evidence.objects.filter(source_kind="vision").exists())

    def test_drawing_triples_are_staged_with_vision_evidence(self):
        job = self.extract(include_drawings=True)
        self.assertEqual(job.status, "done")
        self.assertEqual(self.vlm.call_count, 1)                    # the schematic only: not the photo, not the pending figure
        jpeg, prompt = self.vlm.call_args.args
        self.assertIn("MONITORED_BY", prompt)                       # the active schema is in the prompt
        self.assertIn("Caption in the document: Fig. 3 Drive train", prompt)
        self.assertEqual((self.vlm.call_args.kwargs["json_mode"], self.vlm.call_args.kwargs["purpose"],
                          self.vlm.call_args.kwargs["max_tokens"]), (True, "vlm-drawing", 1000))

        drives = CandidateTriple.objects.get(predicate="DRIVES")
        self.assertEqual((drives.subject_name, drives.object_name, drives.status, drives.page_start),
                         ("VFD101", "M101", "pending", 2))
        self.assertEqual(drives.model, "qwen3-vl-4b-instruct")
        ev = drives.evidence.get()
        self.assertEqual((ev.source_kind, ev.figure_id, ev.extractor, ev.prompt), ("vision", "0", "vlm", "Drawing v1"))
        self.assertEqual(ev.region["page"], 2)
        self.assertEqual(ev.excerpt, "A VFD feeds motor M101.")
        self.assertEqual(provenance.short_label(ev), "Extruder Manual.pdf p.2 (figure)")

        # The fact found in the text and in the drawing is one triple with two places.
        shared = CandidateTriple.objects.get(predicate="MONITORED_BY")
        self.assertIn("vision", shared.evidence.values_list("source_kind", flat=True))

    def test_invalid_json_is_retried_once_then_the_figure_is_skipped(self):
        self.vlm.side_effect = ['{"triples":[{"subject":"VFD1', json.dumps(DRAWING_REPLY)]
        self.extract(include_drawings=True)
        self.assertEqual(self.vlm.call_count, 2)
        self.assertIn("at most 12 triples", self.vlm.call_args.args[1])
        self.assertTrue(CandidateTriple.objects.filter(predicate="DRIVES").exists())

        CandidateTriple.objects.all().delete()
        self.vlm.side_effect = ["nope", "still not json"]
        job = self.extract(include_drawings=True)
        self.assertEqual(job.status, "done")
        self.assertFalse(CandidateTriple.objects.filter(predicate="DRIVES").exists())
        self.assertIn("1 window call(s) failed", self.doc.graph_error)
        self.assertTrue(CandidateTriple.objects.filter(predicate="MONITORED_BY").exists())   # text triples are kept

    def test_bsk_unreachable_keeps_the_text_triples(self):
        self.vlm.side_effect = gpu.GpuUnavailable("BSK asleep")
        job = self.extract(include_drawings=True)
        self.assertEqual((job.status, self.doc.graph_status), ("done", "done"))
        self.assertIn("drawings were not read", self.doc.graph_error)
        self.assertTrue(CandidateTriple.objects.filter(predicate="MONITORED_BY").exists())

    def test_eligible_and_summary_count_drawings(self):
        self.assertEqual([f["index"] for f in drawings.eligible(FIGURES)], [0])
        self.assertEqual(figures.summary(FIGURES)["drawings"], 1)
