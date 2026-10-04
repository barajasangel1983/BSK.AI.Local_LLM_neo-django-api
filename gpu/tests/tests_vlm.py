"""VLM client, image preparation and the gpu_smoke command (P9a.1)."""

import io
import tempfile
from unittest.mock import patch

import requests
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, override_settings
from PIL import Image

from gpu import images, vlm
from gpu import orchestrator as gpu
from gpu.tests.tests_orchestrator import GpuTestCase, resp
from ingestion.models import Document, Job
from usage.models import ModelCall

OK_VL = resp(200, {"active": "vl", "health": "ok", "activation_ms": 0})


def png(width, height, mode="RGB", color="red"):
    buf = io.BytesIO()
    Image.new(mode, (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def pdf(pages=2, size=(600, 800)):
    buf = io.BytesIO()
    sheets = [Image.new("RGB", size, "white") for _ in range(pages)]
    sheets[0].save(buf, format="PDF", save_all=True, append_images=sheets[1:])
    return buf.getvalue()


class ImageTests(SimpleTestCase):
    def test_large_image_is_downscaled_to_1280(self):
        jpeg, width, height = images.to_jpeg(png(3000, 1500))
        self.assertEqual((width, height), (1280, 640))
        self.assertEqual(Image.open(io.BytesIO(jpeg)).format, "JPEG")

    def test_small_image_is_never_upscaled(self):
        _, width, height = images.to_jpeg(png(300, 200))
        self.assertEqual((width, height), (300, 200))

    def test_transparency_becomes_white(self):
        jpeg, _, _ = images.to_jpeg(png(50, 50, mode="RGBA", color=(0, 0, 0, 0)))
        self.assertGreater(Image.open(io.BytesIO(jpeg)).getpixel((25, 25))[0], 240)

    def test_not_an_image(self):
        with self.assertRaises(images.ImageError):
            images.to_jpeg(b"%PDF-1.4 not an image")

    def test_pdf_page_count_and_render(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf(pages=3))
            f.flush()
            self.assertEqual(images.pdf_page_count(f.name), 3)
            jpeg, width, height = images.render_pdf_page(f.name, 2)
            self.assertEqual(max(width, height), 1280)
            self.assertEqual(Image.open(io.BytesIO(jpeg)).format, "JPEG")
            with self.assertRaises(images.ImageError):
                images.render_pdf_page(f.name, 4)

    def test_pdf_rendering_is_safe_from_several_threads(self):
        """PDFium isn't thread-safe; concurrent figure thumbnails once crashed the API process."""
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf(pages=4))
            f.flush()
            box = {"l": 50, "t": 700, "r": 400, "b": 300, "coord_origin": "BOTTOMLEFT"}
            size = {"width": 600, "height": 800}

            def work(i):
                images.pdf_page_count(f.name)
                images.render_pdf_page(f.name, i % 4 + 1)
                return images.crop_pdf_figure(f.name, i % 4 + 1, box, size)[1]
            with ThreadPoolExecutor(max_workers=12) as pool:
                widths = list(pool.map(work, range(48)))
        self.assertEqual(len(set(widths)), 1)

    def test_unreadable_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"not a pdf")
            f.flush()
            with self.assertRaises(images.ImageError):
                images.pdf_page_count(f.name)


@override_settings(VLM_URL="http://bsk:5002/v1", VLM_MODEL="qwen3-vl-4b-instruct", VLM_TIMEOUT=90)
class VlmClientTests(GpuTestCase):
    def _fake(self, order, completion=None):
        def fake(url, **kwargs):
            if url.endswith("/gpu/activate"):
                order.append(("activate", kwargs["json"]["service"]))
                return OK_VL
            if url.endswith("/gpu/touch"):
                order.append(("touch", None))
                return resp(200, {"ok": True})
            order.append(("completion", kwargs["json"]))
            if isinstance(completion, Exception):
                raise completion
            return resp(200, {"choices": [{"message": {"content": "A pump."}}],
                              "usage": {"prompt_tokens": 988, "completion_tokens": 3}})
        return fake

    @patch("requests.post")
    def test_one_image_request_through_the_gpu_lock(self, post):
        order = []
        post.side_effect = self._fake(order)
        self.assertEqual(vlm.complete(b"JPEG", "Describe", max_tokens=400, purpose="vlm-figure"), "A pump.")
        self.assertEqual([step for step, _ in order], ["activate", "completion", "touch"])
        self.assertEqual(order[0][1], "vl")
        body = order[1][1]
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]), ("qwen3-vl-4b-instruct", 400, 0))
        parts = body["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "Describe"})
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("response_format", body)
        call = ModelCall.objects.get(purpose="vlm-figure")
        self.assertEqual((call.model_id, call.prompt_tokens, call.completion_tokens), ("bsk-qwen3-vl-4b", 988, 3))

    @patch("requests.post")
    def test_json_mode(self, post):
        order = []
        post.side_effect = self._fake(order)
        vlm.complete(b"JPEG", "List", json_mode=True)
        self.assertEqual(order[1][1]["response_format"], {"type": "json_object"})

    @patch("requests.post")
    def test_vlm_unreachable_is_gpu_unavailable(self, post):
        post.side_effect = self._fake([], completion=requests.ConnectionError("refused"))
        with self.assertRaises(gpu.GpuUnavailable):
            vlm.complete(b"JPEG", "Describe")

    @patch("requests.post")
    def test_idle_health_idle_counts_as_activated(self, post):
        post.return_value = resp(200, {"active": "idle", "health": "idle"})
        self.assertEqual(gpu.activate("idle")["active"], "idle")


class GpuSmokeTests(GpuTestCase):
    def test_refuses_when_the_orchestrator_switch_is_off(self):
        with override_settings(GPU_ORCHESTRATOR_ENABLED=False), self.assertRaisesMessage(CommandError, "is off"):
            call_command("gpu_smoke")

    def test_refuses_while_library_jobs_are_active(self):
        doc = Document.objects.create(filename="a.pdf", doc_key="A", sha256="0" * 64)
        Job.objects.create(document=doc, kind=Job.Kind.PARSE)
        with self.assertRaisesMessage(CommandError, "1 library job(s)"):
            call_command("gpu_smoke")

    @patch("gpu.management.commands.gpu_smoke.Command._status", return_value={})
    @patch("gpu.management.commands.gpu_smoke.Command._vlm", side_effect=RuntimeError("no reply"))
    @patch("gpu.management.commands.gpu_smoke.Command._parse", return_value="ok")
    def test_reports_pass_and_fail(self, parse, vlm_check, status):
        out = io.StringIO()
        with self.assertRaisesMessage(CommandError, "gpu_smoke failed"):
            call_command("gpu_smoke", stdout=out)
        self.assertIn("PASS  parse: ok", out.getvalue())
        self.assertIn("FAIL  vlm: RuntimeError: no reply", out.getvalue())
