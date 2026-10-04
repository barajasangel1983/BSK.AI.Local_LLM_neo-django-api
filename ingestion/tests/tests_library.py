"""Document library tests (Django runner: `manage.py test ingestion`).

Docling and the DGX embedder are mocked; Chroma is real, in a temp dir.
"""

import io
import json
import os
import shutil
import tempfile
from datetime import timedelta
from unittest.mock import patch

import chromadb
from chromadb.config import Settings as ChromaSettings
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from ingestion import library
from ingestion.chunker import PAGE_BREAK, chunk_document
from ingestion.chunking import ChunkingError, chunk_parsed, resolve_params, variant_key
from ingestion.models import Document, IngestionJob, Job
from ingestion.vector_store import VectorStore

SENT = "The die pressure must stay below 99 bar during normal operation of the extruder line."
MD = (
    "# Die Head\n\n"
    + " ".join([SENT] * 6) + "\n\n"
    + "| Pin | Signal |\n|---|---|\n| 1 | GND |\n| 2 | +24V |\n\n"
    + f"{PAGE_BREAK}\n\n"
    + "# Screw & Barrel\n\n"
    + "Check zone 2 at 98.5 °C, e.g. per Fig. 3 before restarting the screw drive after a stop.\n\n"
    + "- Stop the feeder\n- Clear the screen\n- Restart slowly\n"
)
PARSED = {"document": {"md_content": MD}, "status": "success", "errors": []}


class ChunkingTests(SimpleTestCase):
    def test_resolve_params_defaults_and_bounds(self):
        self.assertEqual(resolve_params("sentence", {}), {"max_tokens": 300, "overlap_sentences": 1})
        self.assertEqual(resolve_params("fixed", {"size": "256"}), {"size": 256, "overlap": 64})
        for strategy, params, message in [
            ("magic", {}, "unknown chunking strategy"),
            ("sentence", {"max_tokens": 5000}, "between 64 and 800"),
            ("sentence", {"size": 10}, "unknown parameter"),
            ("fixed", {"size": 100, "overlap": 100}, "smaller than size"),
            ("structure", {"max_tokens": "big"}, "must be an integer"),
        ]:
            with self.assertRaisesMessage(ChunkingError, message):
                resolve_params(strategy, params)

    def test_variant_key_is_stable_and_param_sensitive(self):
        a = variant_key("sentence", {"max_tokens": 300, "overlap_sentences": 1})
        self.assertEqual(a, variant_key("sentence", {"overlap_sentences": 1, "max_tokens": 300}))
        self.assertNotEqual(a, variant_key("sentence", {"max_tokens": 128, "overlap_sentences": 1}))

    def test_structure_default_matches_existing_chunker(self):
        self.assertEqual(chunk_parsed(PARSED, "structure"), chunk_document(PARSED))

    def test_structure_chunks_never_exceed_max_tokens(self):
        """A table flattened into one very long 'sentence' used to come out as one oversized chunk."""
        row = "| Zone 1 | 185 C | 190 C | 195 C | alarm at 210 C "
        md = "# Limits\n\nIntro sentence about the limits of the machine in normal operation. " + row * 400 + "\n\nNext paragraph with a normal sentence about the screw."
        chunks = chunk_document({"md_content": md}, max_tokens=800, overlap_sentences=1)
        self.assertGreater(len(chunks), 3)
        self.assertLessEqual(max(c.token_count for c in chunks), 800)

    def test_sentence_chunks(self):
        chunks = chunk_parsed(PARSED, "sentence", {"max_tokens": 64, "overlap_sentences": 1})
        paragraphs = [c for c in chunks if c.content_type == "paragraph"]
        self.assertTrue(all(c.token_count <= 64 for c in chunks))
        self.assertTrue(all(c.text.rstrip().endswith((".", "!", "?")) for c in paragraphs))  # whole sentences
        self.assertIn(SENT, paragraphs[1].text.split(" " + SENT, 1)[0] + " " + SENT)  # overlap carried
        table = next(c for c in chunks if c.content_type == "table")
        self.assertIn("| 2 | +24V |", table.text)
        self.assertEqual(table.page_start, 1)
        lst = next(c for c in chunks if c.content_type == "list")
        self.assertEqual((lst.section_path, lst.page_start), (["Screw & Barrel"], 2))
        # "e.g." and "Fig. 3" don't break the sentence
        self.assertTrue(any("e.g. per Fig. 3 before restarting" in c.text for c in paragraphs))

    def test_overlap_never_crosses_sections(self):
        chunks = chunk_parsed(PARSED, "sentence", {"max_tokens": 64, "overlap_sentences": 2})
        barrel = [c for c in chunks if c.section_path == ["Screw & Barrel"] and c.content_type == "paragraph"]
        self.assertTrue(all("die pressure" not in c.text for c in barrel))

    def test_fixed_windows(self):
        chunks = chunk_parsed(PARSED, "fixed", {"size": 64, "overlap": 16})
        self.assertTrue(all(c.token_count <= 64 for c in chunks))
        self.assertGreaterEqual(len([c for c in chunks if c.section_path == ["Die Head"]]), 2)


class LibraryTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.override = override_settings(
            LIBRARY_BASE=os.path.join(self.tmp, "library"),
            CHROMA_DIR=os.path.join(self.tmp, "chroma"),
            INGESTION_RAW_BASE=os.path.join(self.tmp, "incoming"),
        )
        self.override.enable()
        os.makedirs(settings.INGESTION_RAW_BASE)
        patcher = patch("ingestion.library.convert_file", return_value=PARSED)
        self.mock_convert = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("ingestion.library.Embedder")
        self.mock_embedder = patcher.start()
        self.mock_embedder.return_value.embed_passages.side_effect = lambda texts: [[0.6, 0.8]] * len(texts)
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, name="Extruder Manual.pdf", content=b"%PDF manual"):
        return library.add_document(SimpleUploadedFile(name, content))

    def drain(self):
        while (job := library.claim_next()) is not None:
            library.run_job(job)

    def chunks(self, doc_key):
        data = VectorStore()._get_collection().get(where={"asset_id": doc_key}, include=["metadatas"])
        return data["ids"], data["metadatas"]


class LibraryPipelineTests(LibraryTestCase):
    def test_upload_stores_file_dedupes_and_queues_parse(self):
        doc, created = self.upload()
        self.assertTrue(created)
        self.assertEqual(doc.doc_key, "EXTRUDER-MANUAL")
        self.assertTrue(library.file_path(doc).exists())
        self.assertEqual(list(doc.jobs.values_list("kind", "status")), [("parse", "queued")])

        again, created = self.upload("copy.pdf")  # same bytes
        self.assertEqual((again.pk, created), (doc.pk, False))
        other, _ = self.upload(content=b"%PDF other")  # same name, different bytes
        self.assertEqual(other.doc_key, "EXTRUDER-MANUAL-2")

    def test_parse_caches_docling_output(self):
        doc, _ = self.upload()
        self.drain()
        doc.refresh_from_db()
        self.assertEqual((doc.parse_status, doc.page_count), ("parsed", 2))
        self.assertEqual(json.loads(library.parsed_path(doc).read_text()), PARSED)
        self.assertEqual(self.mock_convert.call_count, 1)

    def test_embed_writes_chunks_with_strategy_metadata(self):
        doc, _ = self.upload()
        job = library.enqueue(doc, Job.Kind.EMBED, {"strategy": "sentence", "params": {"max_tokens": 64}})
        self.assertEqual(job.params, {"strategy": "sentence", "params": {"max_tokens": 64, "overlap_sentences": 1}})
        self.drain()

        doc.refresh_from_db()
        ids, metas = self.chunks(doc.doc_key)
        self.assertEqual((doc.rag_status, doc.rag_strategy, doc.rag_chunk_count), ("done", "sentence", len(ids)))
        self.assertGreater(len(ids), 0)
        self.assertEqual({m["document_id"] for m in metas}, {str(doc.id)})
        self.assertEqual({m["chunk_strategy"] for m in metas}, {"sentence"})
        self.assertEqual(self.mock_convert.call_count, 1)  # parsed once, reused by embed
        job.refresh_from_db()
        self.assertEqual((job.status, job.progress_done), ("done", job.progress_total))

    def test_re_embedding_replaces_the_documents_chunks(self):
        doc, _ = self.upload()
        library.enqueue(doc, Job.Kind.EMBED, {"strategy": "sentence", "params": {"max_tokens": 64}})
        self.drain()
        first, _ = self.chunks(doc.doc_key)
        library.enqueue(doc, Job.Kind.EMBED, {"strategy": "structure"})
        self.drain()
        second, metas = self.chunks(doc.doc_key)
        doc.refresh_from_db()
        self.assertTrue(set(first).isdisjoint(second))
        self.assertEqual({m["chunk_strategy"] for m in metas}, {"structure"})
        self.assertEqual(doc.rag_chunk_count, len(second))

    def test_enqueue_returns_the_active_job_and_validates_params(self):
        doc, _ = self.upload()
        a = library.enqueue(doc, Job.Kind.EMBED, {})
        self.assertEqual(library.enqueue(doc, Job.Kind.EMBED, {"strategy": "fixed"}).pk, a.pk)
        other, _ = self.upload(content=b"%PDF other")
        with self.assertRaises(ChunkingError):
            library.enqueue(other, Job.Kind.EMBED, {"strategy": "sentence", "params": {"max_tokens": 1}})

    def test_failure_marks_job_and_document(self):
        self.mock_embedder.return_value.embed_passages.side_effect = RuntimeError("DGX down")
        doc, _ = self.upload()
        job = library.enqueue(doc, Job.Kind.EMBED, {})
        with self.assertLogs("chat", level="ERROR"):
            self.drain()
        job.refresh_from_db()
        doc.refresh_from_db()
        self.assertEqual((job.status, doc.rag_status), ("failed", "failed"))
        self.assertIn("DGX down", job.error)

    def test_cancel_queued_and_running(self):
        doc, _ = self.upload()
        queued = library.enqueue(doc, Job.Kind.EMBED, {})
        self.assertTrue(library.cancel(queued))
        doc.refresh_from_db()
        self.assertEqual(doc.rag_status, "none")

        running = library.enqueue(doc, Job.Kind.EMBED, {})
        def cancel_midway(texts):
            Job.objects.filter(pk=running.pk).update(status=Job.Status.CANCELLED)
            return [[0.6, 0.8]] * len(texts)
        self.mock_embedder.return_value.embed_passages.side_effect = cancel_midway
        self.drain()
        running.refresh_from_db()
        self.assertEqual(running.status, "cancelled")
        self.assertEqual(self.chunks(doc.doc_key)[0], [])

    def test_stale_running_jobs_are_requeued(self):
        doc, _ = self.upload()
        job = library.claim_next()
        Job.objects.filter(pk=job.pk).update(heartbeat_at=timezone.now() - timedelta(hours=1))
        self.assertEqual(library.requeue_stale(), 1)
        job.refresh_from_db()
        self.assertEqual(job.status, "queued")
        self.assertEqual(library.claim_next().attempts, 2)

    def test_remove_embeddings_and_delete_document(self):
        doc, _ = self.upload()
        library.enqueue(doc, Job.Kind.EMBED, {})
        self.drain()
        self.assertGreater(library.remove_embeddings(doc), 0)
        doc.refresh_from_db()
        self.assertEqual((doc.rag_status, doc.rag_chunk_count), ("none", 0))
        self.assertTrue(library.file_path(doc).exists())  # still in the library

        folder = library.doc_dir(doc)
        library.delete_document(doc)
        self.assertFalse(Document.objects.exists())
        self.assertFalse(folder.exists())

    def test_worker_command_drains_the_queue(self):
        doc, _ = self.upload()
        library.enqueue(doc, Job.Kind.EMBED, {})
        out = io.StringIO()
        call_command("library_worker", "--once", stdout=out)
        self.assertIn("running parse", out.getvalue())
        self.assertIn("running embed", out.getvalue())
        self.assertFalse(Job.objects.filter(status__in=library.ACTIVE).exists())


class LibraryApiTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        self.client = APIClient()

    def test_upload_list_detail(self):
        r = self.client.post("/api/documents/", {"files": [SimpleUploadedFile("a.pdf", b"%PDF a"),
                                                           SimpleUploadedFile("b.pdf", b"%PDF b")]})
        self.assertEqual(r.status_code, 201)
        self.assertEqual([x["created"] for x in r.data["results"]], [True, True])
        docs = self.client.get("/api/documents/").data["documents"]
        self.assertEqual({d["doc_key"] for d in docs}, {"A", "B"})
        self.assertEqual(docs[0]["active_jobs"][0]["kind"], "parse")
        detail = self.client.get(f"/api/documents/{docs[0]['id']}/").data
        self.assertEqual(detail["rag"]["status"], "none")
        self.assertEqual(self.client.post("/api/documents/", {}).status_code, 400)

    def test_embed_single_and_bulk(self):
        a, _ = self.upload("a.pdf", b"%PDF a")
        b, _ = self.upload("b.pdf", b"%PDF b")
        r = self.client.post(f"/api/documents/{a.id}/embed/", {"strategy": "sentence", "params": {"max_tokens": 128}},
                             format="json")
        self.assertEqual((r.status_code, r.data["kind"]), (202, "embed"))
        r = self.client.post("/api/documents/embed/", {"ids": [str(a.id), str(b.id)], "strategy": "fixed"},
                             format="json")
        self.assertEqual(r.status_code, 202)
        self.assertEqual(len(r.data["jobs"]), 2)
        bad = self.client.post(f"/api/documents/{b.id}/embed/", {"strategy": "sentence", "params": {"max_tokens": 1}},
                               format="json")
        self.assertEqual(bad.status_code, 400)
        missing = self.client.post("/api/documents/embed/", {"ids": ["00000000-0000-4000-8000-000000000000"]},
                                   format="json")
        self.assertEqual(missing.status_code, 404)

    def test_jobs_cancel_and_remove_embeddings(self):
        doc, _ = self.upload()
        job = library.enqueue(doc, Job.Kind.EMBED, {})
        listed = self.client.get("/api/jobs/", {"document": str(doc.id), "active": "1"}).data["jobs"]
        self.assertEqual({j["kind"] for j in listed}, {"parse", "embed"})
        self.assertEqual(self.client.post(f"/api/jobs/{job.id}/cancel/").data["status"], "cancelled")
        self.assertEqual(self.client.post(f"/api/jobs/{job.id}/cancel/").status_code, 409)

        library.enqueue(doc, Job.Kind.EMBED, {})
        self.drain()
        r = self.client.delete(f"/api/documents/{doc.id}/embeddings/")
        self.assertGreater(r.data["chunks_deleted"], 0)
        self.assertEqual(self.client.delete(f"/api/documents/{doc.id}/").status_code, 200)
        self.assertEqual(self.client.get(f"/api/documents/{doc.id}/").status_code, 404)

    def test_rag_lab_delete_keeps_library_document(self):
        doc, _ = self.upload()
        library.enqueue(doc, Job.Kind.EMBED, {})
        self.drain()
        with self.assertLogs("chat", level="INFO"):
            r = self.client.delete(f"/api/rag/docs/{doc.doc_key}/")
        self.assertEqual(r.status_code, 200)
        doc.refresh_from_db()
        self.assertEqual(doc.rag_status, "none")

    def test_config_exposes_chunking_strategies(self):
        with patch("chat.rag_lab_views.get_v2_collection", return_value=None):
            chunking = self.client.get("/api/rag/config/").data["chunking"]
        self.assertEqual(chunking["default_strategy"], "structure")
        self.assertEqual(chunking["strategies"]["sentence"]["params"]["max_tokens"]["default"], 300)


class LegacyImportAndRebuildTests(LibraryTestCase):
    def setUp(self):
        super().setUp()
        with open(os.path.join(settings.INGESTION_RAW_BASE, "old.pdf"), "wb") as f:
            f.write(b"%PDF legacy")
        IngestionJob.objects.create(source_filename="old.pdf", source_sha256="x", document_id="OLD", asset_id="OLD",
                                    status=IngestionJob.Status.DONE, chunk_count=2)
        VectorStore().write_chunks(["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]], [{"asset_id": "OLD"}] * 2,
                                   source_sha256="x", document_revision="1", config_version="v2")
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
        client.get_or_create_collection("bsk_rag").add(ids=["h"], documents=["historian"], embeddings=[[1.0, 0.0]])

    def test_import_creates_library_documents_once(self):
        out = io.StringIO()
        call_command("library_import_legacy", stdout=out)
        doc = Document.objects.get(doc_key="OLD")
        self.assertEqual((doc.rag_status, doc.rag_strategy, doc.rag_chunk_count), ("done", "structure", 2))
        self.assertTrue(library.file_path(doc).exists())
        _, metas = self.chunks("OLD")
        self.assertEqual({m["document_id"] for m in metas}, {str(doc.id)})
        self.assertEqual(list(doc.jobs.values_list("kind", flat=True)), ["parse"])
        call_command("library_import_legacy", stdout=io.StringIO())
        self.assertEqual(Document.objects.count(), 1)

    def test_rebuild_reembeds_library_documents(self):
        call_command("library_import_legacy", "--no-parse", stdout=io.StringIO())
        doc = Document.objects.get(doc_key="OLD")
        Document.objects.filter(pk=doc.pk).update(rag_strategy="sentence", rag_params={"max_tokens": 64,
                                                                                         "overlap_sentences": 0})
        call_command("rebuild_rag_v2", "--dry-run", stdout=io.StringIO())
        self.assertEqual(len(self.chunks("OLD")[0]), 2)  # dry run: unchanged

        out = io.StringIO()
        call_command("rebuild_rag_v2", stdout=out)
        doc.refresh_from_db()
        ids, metas = self.chunks("OLD")
        self.assertEqual({m["chunk_strategy"] for m in metas}, {"sentence"})
        self.assertEqual(doc.rag_chunk_count, len(ids))
        coll = VectorStore()._get_collection()
        self.assertEqual(coll.configuration_json["hnsw"]["space"], "cosine")
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
        self.assertEqual(client.get_collection("bsk_rag").count(), 1)
        self.assertTrue(any(n.startswith("chroma.bak-") for n in os.listdir(self.tmp)))
        self.assertIn("0 failure(s)", out.getvalue())
