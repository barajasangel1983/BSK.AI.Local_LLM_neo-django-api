"""Django TestCase tests (run with `manage.py test ingestion`, not pytest):
ingest retry, cosine collections and the rebuild_rag_v2 command."""

import hashlib
import io
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
from django.test import TestCase, override_settings
from django.utils import timezone

from ingestion.models import IngestionJob
from ingestion.vector_store import VectorStore

PDF = b"%PDF-1.4 fake test document"
SHA = hashlib.sha256(PDF).hexdigest()


class TempDirsMixin:
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.incoming = os.path.join(self.tmp, "incoming")
        self.chroma = os.path.join(self.tmp, "chroma")
        os.makedirs(self.incoming)
        self.settings_override = override_settings(INGESTION_RAW_BASE=self.incoming, CHROMA_DIR=self.chroma)
        self.settings_override.enable()

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()


def _job(status, created_at=None, **extra):
    job = IngestionJob.objects.create(
        source_filename="doc.pdf", source_sha256=SHA, document_id="DOC", asset_id="DOC",
        document_revision="1", config_version=settings.INGESTION_CONFIG_VERSION, status=status, **extra,
    )
    if created_at:
        IngestionJob.objects.filter(id=job.id).update(created_at=created_at)
    return job


@patch("ingestion.views.threading.Thread")
class IngestRetryTests(TempDirsMixin, TestCase):
    def _upload(self):
        return self.client.post(
            "/api/rag/ingest/",
            {"file": SimpleUploadedFile("doc.pdf", PDF), "asset_id": "DOC", "document_revision": "1"},
        )

    def test_done_job_is_returned_as_is(self, mock_thread):
        job = _job(IngestionJob.Status.DONE, chunk_count=5)
        r = self._upload()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], str(job.id))
        mock_thread.assert_not_called()

    def test_failed_job_is_retried(self, mock_thread):
        job = _job(IngestionJob.Status.FAILED, error="DoclingError: boom")
        r = self._upload()

        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["job_id"], str(job.id))
        job.refresh_from_db()
        self.assertEqual((job.status, job.error), (IngestionJob.Status.QUEUED, None))
        mock_thread.return_value.start.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(self.incoming, "doc.pdf")))  # re-saved
        self.assertEqual(IngestionJob.objects.count(), 1)

    def test_stale_in_flight_job_is_retried(self, mock_thread):
        old = timezone.now() - timedelta(minutes=settings.INGESTION_STALE_MINUTES + 5)
        job = _job(IngestionJob.Status.PARSING, created_at=old)
        r = self._upload()

        self.assertEqual(r.status_code, 202)
        job.refresh_from_db()
        self.assertEqual(job.status, IngestionJob.Status.QUEUED)
        self.assertGreater(job.created_at, old)  # stale clock restarts with the retry

    def test_recent_in_flight_job_is_not_restarted(self, mock_thread):
        _job(IngestionJob.Status.EMBEDDING)
        r = self._upload()
        self.assertEqual(r.status_code, 200)
        mock_thread.assert_not_called()


class VectorStoreCosineTests(TempDirsMixin, TestCase):
    def test_new_collection_uses_cosine(self):
        store = VectorStore(chroma_path=self.chroma, collection_name="test_cosine")
        store.write_chunks(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"asset_id": "A"}, {"asset_id": "A"}],
                           source_sha256="s", document_revision="1", config_version="v")
        self.assertEqual(store._get_collection().configuration_json["hnsw"]["space"], "cosine")


def _fake_process_job(job_id):
    """Stand-in for the pipeline: write one chunk and mark the job done."""
    job = IngestionJob.objects.get(id=job_id)
    VectorStore().write_chunks(
        [f"chunk of {job.asset_id}"], [[0.6, 0.8]], [{"asset_id": job.asset_id, "source": job.source_filename}],
        source_sha256=job.source_sha256, document_revision=job.document_revision,
        config_version=job.config_version,
    )
    job.status, job.chunk_count = IngestionJob.Status.DONE, 1
    job.save()


@patch("ingestion.management.commands.rebuild_rag_v2.process_job", side_effect=_fake_process_job)
class RebuildCommandTests(TempDirsMixin, TestCase):
    def setUp(self):
        super().setUp()
        # Existing L2 collection with a stale chunk, plus an unrelated legacy collection.
        client = chromadb.PersistentClient(path=self.chroma, settings=ChromaSettings(anonymized_telemetry=False))
        client.create_collection(settings.RAG_COLLECTION_V2).add(ids=["old"], documents=["old"], embeddings=[[1.0, 0.0]])
        client.create_collection("bsk_rag").add(ids=["h"], documents=["historian"], embeddings=[[1.0, 0.0]])

        with open(os.path.join(self.incoming, "doc.pdf"), "wb") as f:
            f.write(PDF)
        IngestionJob.objects.create(source_filename="doc.pdf", source_sha256=SHA, document_id="DOC", asset_id="DOC",
                                    config_version="v1-old", status=IngestionJob.Status.DONE, chunk_count=9)
        IngestionJob.objects.create(source_filename="gone.pdf", source_sha256="x", document_id="GONE", asset_id="GONE",
                                    config_version="v1-old", status=IngestionJob.Status.DONE, chunk_count=3)

    def _collections(self):
        client = chromadb.PersistentClient(path=self.chroma, settings=ChromaSettings(anonymized_telemetry=False))
        return {c.name if hasattr(c, "name") else c: client.get_collection(c.name if hasattr(c, "name") else c)
                for c in client.list_collections()}

    def test_dry_run_changes_nothing(self, mock_process):
        out = io.StringIO()
        call_command("rebuild_rag_v2", "--dry-run", stdout=out)
        self.assertIn("rebuild DOC", out.getvalue())
        self.assertIn("skip    GONE: source file missing", out.getvalue())
        mock_process.assert_not_called()
        self.assertEqual(self._collections()[settings.RAG_COLLECTION_V2].get()["ids"], ["old"])

    def test_rebuilds_with_cosine_and_current_config(self, mock_process):
        out = io.StringIO()
        call_command("rebuild_rag_v2", stdout=out)

        collections = self._collections()
        v2 = collections[settings.RAG_COLLECTION_V2]
        self.assertEqual(v2.configuration_json["hnsw"]["space"], "cosine")
        self.assertEqual(v2.get()["documents"], ["chunk of DOC"])  # old chunk gone
        self.assertEqual(collections["bsk_rag"].count(), 1)  # legacy collection untouched

        job = IngestionJob.objects.get(asset_id="DOC")
        self.assertEqual((job.status, job.config_version), (IngestionJob.Status.DONE, settings.INGESTION_CONFIG_VERSION))
        self.assertEqual(IngestionJob.objects.filter(asset_id="GONE").count(), 1)  # skipped, kept
        self.assertTrue(any(name.startswith("chroma.bak-") for name in os.listdir(self.tmp)))
        self.assertIn("1 chunks; 0 failure(s)", out.getvalue())
