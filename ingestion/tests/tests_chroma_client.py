"""Chroma client selection, legacy historian retrieval, and the multi-process regression.

The regression test needs a Chroma server:
    docker compose -f deploy/chroma/docker-compose.yml --env-file .env --profile test up -d chroma-test
    CHROMA_TEST_URL=http://127.0.0.1:8102 .venv/bin/python manage.py test ingestion
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

from django.conf import settings
from django.test import SimpleTestCase, override_settings

from chat.legacy_retrieval import query_chunks
from ingestion import chroma_client
from ingestion.vector_store import VectorStore

TEST_URL = os.getenv("CHROMA_TEST_URL")
BACKEND_DIR = Path(__file__).resolve().parents[2]


class ClientSelectionTests(SimpleTestCase):
    def test_tests_never_use_the_live_server(self):
        self.assertEqual(settings.CHROMA_HOST, "")  # forced by neo_llm_api.test_runner

    @override_settings(CHROMA_HOST="127.0.0.1", CHROMA_PORT=8100)
    @patch("ingestion.chroma_client.chromadb.HttpClient")
    def test_server_mode_uses_http_client(self, mock_http):
        self.assertTrue(chroma_client.server_mode())
        chroma_client.get_client()
        self.assertEqual(mock_http.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(mock_http.call_args.kwargs["port"], 8100)
        self.assertEqual(chroma_client.describe(), "http://127.0.0.1:8100")

    @patch("ingestion.chroma_client.chromadb.PersistentClient")
    def test_embedded_mode_without_host(self, mock_persistent):
        chroma_client.get_client()
        self.assertEqual(mock_persistent.call_args.kwargs["path"], str(settings.CHROMA_DIR))


class LegacyRetrievalTests(SimpleTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.override = override_settings(CHROMA_DIR=self.tmp)
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_historian_query_filters_by_source(self):
        chroma_client.legacy_collection().add(
            ids=["h1", "d1"],
            documents=["On 2026-03-28 shift B extruder EXTR01 OEE was 81 percent", "A resume of a software engineer"],
            metadatas=[{"source": "plc_historian"}, {"source": "resume.txt"}],
        )
        results = query_chunks("OEE of extruder EXTR01", top_k=2, where={"source": "plc_historian"})
        self.assertEqual([r.source for r in results], ["plc_historian"])
        self.assertIn("EXTR01", results[0].text)
        self.assertEqual(query_chunks("   "), [])


WRITER = """
import sys, chromadb
host, port, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
coll = chromadb.HttpClient(host=host, port=port).get_or_create_collection(name)
coll.add(ids=["fresh"], documents=["written by another process"], embeddings=[[0.6, 0.8]],
         metadatas=[{"asset_id": "DOC", "source": "fresh.pdf"}])
"""


@unittest.skipUnless(TEST_URL, "set CHROMA_TEST_URL (see module docstring) to run against a Chroma server")
class MultiProcessRegressionTests(SimpleTestCase):
    """Data written by another process (the library worker) is visible to this one immediately."""

    def test_reader_sees_other_process_writes(self):
        url = urlparse(TEST_URL)
        name = f"regression_{uuid.uuid4().hex[:8]}"
        with override_settings(CHROMA_HOST=url.hostname, CHROMA_PORT=url.port, RAG_COLLECTION_V2=name):
            store = VectorStore()
            store.write_chunks(["first"], [[1.0, 0.0]], [{"asset_id": "DOC", "source": "a.pdf"}],
                               source_sha256="s", document_revision="1", config_version="v")
            coll = store._get_collection()
            self.assertEqual(coll.count(), 1)  # this process has the collection open (and cached)

            subprocess.run([sys.executable, "-c", WRITER, url.hostname, str(url.port), name],
                           check=True, cwd=BACKEND_DIR, timeout=60)

            data = chroma_client.get_client().get_collection(name).get(where={"asset_id": "DOC"},
                                                                        include=["documents", "metadatas"])
            by_id = dict(zip(data["ids"], zip(data["documents"], data["metadatas"])))
            self.assertEqual(by_id["fresh"], ("written by another process", {"asset_id": "DOC", "source": "fresh.pdf"}))
            chroma_client.get_client().delete_collection(name)
