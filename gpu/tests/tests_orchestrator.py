"""BSK GPU orchestrator client (contract v1.2 in the frontend repo's claude/VLM_service_brief.md)."""

import fcntl
import os
import tempfile
from unittest.mock import MagicMock, patch

import requests
from django.test import TestCase, override_settings

from gpu import orchestrator as gpu
from usage.models import ModelCall


def resp(status, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or {}
    r.raise_for_status.side_effect = None if status < 400 else requests.HTTPError(str(status))
    return r


OK_DOCLING = resp(200, {"active": "docling", "health": "ok", "activation_ms": 21000})


class GpuTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.override = override_settings(GPU_LOCK_PATH=os.path.join(self.tmp, "gpu.lock"), GPU_ORCHESTRATOR_ENABLED=True,
                                          GPU_ORCHESTRATOR_URL="http://bsk:5003", DOCLING_URL="http://bsk:5001",
                                          GPU_ACTIVATE_TIMEOUT=90)
        self.override.enable()
        self.addCleanup(self.override.disable)
        patcher = patch("gpu.orchestrator.time.sleep")      # no real waiting
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)


@patch("gpu.orchestrator.requests.get")
@patch("gpu.orchestrator.requests.post")
class ActivateTests(GpuTestCase):
    def test_ready_immediately(self, post, get):
        post.return_value = OK_DOCLING
        self.assertEqual(gpu.activate("docling")["health"], "ok")
        post.assert_called_once_with("http://bsk:5003/gpu/activate", json={"service": "docling"}, timeout=(5.0, 45.0))
        get.assert_not_called()

    def test_starting_then_polls_status_every_2s(self, post, get):
        post.return_value = resp(200, {"active": "vl", "health": "starting", "activation_ms": None})
        get.side_effect = [resp(200, {"active": "vl", "health": "starting"}), resp(200, {"active": "vl", "health": "ok"})]
        self.assertEqual(gpu.activate("vl")["active"], "vl")
        self.assertEqual(get.call_count, 2)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2.0, 2.0])

    def test_conflict_backs_off_and_retries(self, post, get):
        post.side_effect = [resp(409, {"error": "transition in progress", "active": "docling"}), OK_DOCLING]
        gpu.activate("docling")
        self.assertEqual(post.call_count, 2)

    def test_error_retried_once_then_unavailable(self, post, get):
        failed = resp(200, {"active": "vl", "health": "error", "error_detail": "CUDA out of memory"})
        post.side_effect = [failed, resp(200, {"active": "vl", "health": "ok"})]
        gpu.activate("vl")                                   # one failure, then ok
        post.side_effect = [failed, failed]
        with self.assertRaisesMessage(gpu.GpuUnavailable, "CUDA out of memory"):
            gpu.activate("vl")

    def test_invalid_service_and_unreachable(self, post, get):
        with self.assertRaises(ValueError):
            gpu.activate("nope")
        post.return_value = resp(422, {"error": "invalid service"})
        with self.assertRaises(ValueError):
            gpu.activate("idle")
        post.side_effect = requests.ConnectionError("no route to host")
        with self.assertRaises(gpu.OrchestratorUnreachable):
            gpu.activate("docling")

    @override_settings(GPU_ACTIVATE_TIMEOUT=0)
    def test_timeout(self, post, get):
        with self.assertRaisesMessage(gpu.GpuUnavailable, "did not become healthy"):
            gpu.activate("docling")
        post.assert_not_called()


@patch("gpu.orchestrator.requests.get")
@patch("gpu.orchestrator.requests.post")
class UseTests(GpuTestCase):
    def test_activates_inside_the_lock_and_records_it(self, post, get):
        post.return_value = OK_DOCLING
        with gpu.use("docling"):
            post.assert_called_once()
        call = ModelCall.objects.get()
        self.assertEqual((call.purpose, call.model_id, call.status), ("gpu", "gpu:docling", "ok"))

    @override_settings(GPU_ORCHESTRATOR_ENABLED=False)
    def test_disabled_calls_nothing(self, post, get):
        with gpu.use("docling"):
            pass
        post.assert_not_called()
        self.assertFalse(ModelCall.objects.exists())

    def test_unreachable_orchestrator_falls_back_to_docling_if_it_answers(self, post, get):
        post.side_effect = requests.ConnectionError("down")
        get.return_value = resp(200)                         # Docling /health
        with self.assertLogs("gpu", level="WARNING"):
            with gpu.use("docling"):
                pass
        get.assert_called_once_with("http://bsk:5001/health", timeout=5)
        get.return_value = resp(503)
        with self.assertRaises(gpu.OrchestratorUnreachable):
            with gpu.use("docling"):
                pass
        with self.assertRaises(gpu.OrchestratorUnreachable):   # no fallback for the VLM
            with gpu.use("vl"):
                pass

    def test_busy_when_another_holder_has_the_lock(self, post, get):
        post.return_value = OK_DOCLING
        fd = os.open(os.path.join(self.tmp, "gpu.lock"), os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)                       # e.g. the worker mid-parse
        try:
            with self.assertRaisesMessage(gpu.GpuBusy, "GPU busy"):
                with gpu.use("vl", wait=0.2):
                    pass
            post.assert_not_called()                         # never switches under another holder
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        with gpu.use("docling", wait=1):                     # free again
            pass

    def test_lock_released_when_the_call_fails(self, post, get):
        post.return_value = OK_DOCLING
        with self.assertRaises(RuntimeError):
            with gpu.use("docling"):
                raise RuntimeError("conversion crashed")
        with gpu.use("docling", wait=0):
            pass


class DoclingThroughOrchestratorTests(GpuTestCase):
    @patch("requests.post")   # one `requests` module: dispatch by URL
    def test_parse_activates_docling_first(self, post):
        from ingestion.docling_client import DoclingClient, DoclingError
        order = []
        activation = [OK_DOCLING]

        def fake(url, **kwargs):
            if url.endswith("/gpu/activate"):
                order.append("activate")
                return activation.pop(0) if activation else OK_DOCLING
            order.append("convert")
            return resp(200, {"status": "success", "document": {"md_content": "x"}})
        post.side_effect = fake
        self.assertEqual(DoclingClient(base_url="http://bsk:5001").convert_file(b"%PDF", "a.pdf")["document"]["md_content"], "x")
        self.assertEqual(order, ["activate", "convert"])
        self.assertEqual(sorted(ModelCall.objects.values_list("purpose", flat=True)), ["gpu", "parse"])

        failed = resp(200, {"active": "docling", "health": "error", "error_detail": "container exited"})
        activation[:] = [failed, failed]
        with self.assertRaisesMessage(DoclingError, "Docling unavailable"):
            DoclingClient(base_url="http://bsk:5001").convert_file(b"%PDF", "a.pdf")


@override_settings(GPU_ORCHESTRATOR_URL="http://bsk:5003", DOCLING_URL="http://bsk:5001")
class HealthTests(TestCase):
    @override_settings(GPU_ORCHESTRATOR_ENABLED=True)
    @patch("gpu.orchestrator.requests.get", return_value=resp(200, {"active": "vl", "health": "ok"}))
    def test_bsk_entries_follow_the_orchestrator(self, get):
        from chat.views import _bsk_status
        self.assertEqual(_bsk_status("gpu-orchestrator")[0], "online")
        self.assertEqual(_bsk_status("vlm-bsk")[0], "online")
        self.assertEqual(_bsk_status("docling-bsk")[0], "idle")          # stopped by design
        get.side_effect = requests.ConnectionError("asleep")
        self.assertEqual(_bsk_status("vlm-bsk")[0], "offline")

    @override_settings(GPU_ORCHESTRATOR_ENABLED=False)
    @patch("chat.views.requests.get", return_value=resp(200))
    def test_disabled_keeps_direct_docling(self, get):
        from chat.views import _bsk_status
        self.assertEqual(_bsk_status("docling-bsk")[0], "online")
        self.assertEqual(_bsk_status("gpu-orchestrator")[0], "idle")
        self.assertEqual(_bsk_status("vlm-bsk")[0], "idle")
