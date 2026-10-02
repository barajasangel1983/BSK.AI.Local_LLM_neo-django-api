"""Django TestCase tests (run with `manage.py test ingestion`, not pytest):
legacy ingest retry and cosine collections. Rebuild tests live in tests_library.py."""

import hashlib
import os
import shutil
import tempfile
from datetime import timedelta
from unittest.mock import patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
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
