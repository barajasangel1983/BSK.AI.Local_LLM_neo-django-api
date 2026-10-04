"""Describe figures (P9a.2): figure list from Docling's JSON, crop, VLM description,
descriptions in the text. Docling and the VLM are mocked."""

import io
import json
import shutil
import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone
from PIL import Image, ImageDraw
from rest_framework.test import APIClient

from gpu import images
from gpu import orchestrator as gpu
from ingestion import figures, library
from ingestion.chunker import PAGE_BREAK
from ingestion.chunking import chunk_parsed, resolve_params
from ingestion.models import Document, Job

PAGE = {"width": 600.0, "height": 800.0}


def pdf_with_box():
    """One 600 x 800 pt page: white, with a red box at x 100–300, y 100–300 from the TOP."""
    page = Image.new("RGB", (600, 800), "white")
    ImageDraw.Draw(page).rectangle((100, 100, 300, 300), fill="red")
    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=72.0)
    return buf.getvalue()


def picture(page, l, t, r, b, origin="BOTTOMLEFT", captions=()):
    return {"prov": [{"page_no": page, "bbox": {"l": l, "t": t, "r": r, "b": b, "coord_origin": origin}}],
            "captions": [{"$ref": f"#/texts/{i}"} for i in captions]}


# The red box (BOTTOMLEFT: top y = 800 - 100), a 20 x 20 pt icon, and a full-page background.
LAYOUT = {
    "pages": {"1": {"size": PAGE}},
    "texts": [{"text": "Figure 1: Drive train"}],
    "pictures": [picture(1, 100, 700, 300, 500, captions=[0]), picture(1, 10, 790, 30, 770), picture(1, 0, 800, 600, 0)],
}
MD = f"# Drive\n\n<!-- image -->\n\nThe motor drives the screw.\n\n<!-- image -->\n\n<!-- image -->\n\n{PAGE_BREAK}\n\n# Page two\n\nMore text about the gearbox and its lubrication schedule."
REPLY = "Kind: schematic\nDescription: A VFD feeds motor M101.\nIt drives the screw.\nText: VFD101; M101; 45 kW"


def parsed(md=MD, layout=LAYOUT):
    return {"status": "success", "errors": [], "document": {"md_content": md, "json_content": layout}}


class CropTests(SimpleTestCase):
    def crop(self, bbox, **kwargs):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf_with_box())
            f.flush()
            jpeg, w, h, raw_w, raw_h = images.crop_pdf_figure(f.name, 1, bbox, PAGE, **kwargs)
        return Image.open(io.BytesIO(jpeg)), (w, h), (raw_w, raw_h)

    def assert_red(self, img, where):
        r, g, b = img.getpixel(where)
        self.assertTrue(r > 200 and g < 60 and b < 60, (r, g, b))

    def test_bottomleft_box_is_flipped_and_padded(self):
        img, (w, h), raw = self.crop({"l": 100, "t": 700, "r": 300, "b": 500, "coord_origin": "BOTTOMLEFT"})
        self.assertEqual(max(w, h), 880)                 # 200 pt + 5 % each side = 220 pt at 4x; never upscaled
        self.assert_red(img, (w // 2, h // 2))
        self.assertGreater(min(img.getpixel((4, 4))), 200)          # the padding is white page
        self.assertGreater(min(img.getpixel((w - 5, h - 5))), 200)

    def test_topleft_box(self):
        img, (w, h), _ = self.crop({"l": 100, "t": 100, "r": 300, "b": 300, "coord_origin": "TOPLEFT"}, padding=0)
        for point in ((5, 5), (w - 6, h - 6), (w // 2, h // 2)):
            self.assert_red(img, point)

    def test_padding_is_clamped_to_the_page_and_large_crops_are_downscaled(self):
        img, (w, h), (raw_w, raw_h) = self.crop({"l": 0, "t": 800, "r": 600, "b": 0, "coord_origin": "BOTTOMLEFT"})
        self.assertEqual((w, h), (960, 1280))
        self.assertEqual((raw_w, raw_h), (960, 1280))

    def test_empty_box(self):
        with self.assertRaises(images.ImageError):
            self.crop({"l": 100, "t": 500, "r": 100, "b": 500, "coord_origin": "BOTTOMLEFT"})


class ReplyTests(SimpleTestCase):
    def test_kind_description_and_text(self):
        self.assertEqual(figures.parse_reply(REPLY),
                         ("schematic", "A VFD feeds motor M101. It drives the screw. Text: VFD101; M101; 45 kW"))

    def test_markdown_bold_and_no_text(self):
        self.assertEqual(figures.parse_reply("**Kind:** photo\n**Description:** A gearbox.\n**Text:** none"),
                         ("photo", "A gearbox."))

    def test_free_text_answer_is_kept(self):
        self.assertEqual(figures.parse_reply("This is a pump with two valves."), ("", "This is a pump with two valves."))


class FigureTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        override = override_settings(LIBRARY_BASE=self.tmp, FIGURE_RETRY_MINUTES=[5, 15])
        override.enable()
        self.addCleanup(override.disable)
        patcher = patch("ingestion.library.convert_file", side_effect=lambda *a, **k: parsed())
        self.convert = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("ingestion.figures.vlm.complete", return_value=REPLY)
        self.vlm = patcher.start()
        self.addCleanup(patcher.stop)

    def upload(self, name="Drive.pdf", content=None):
        doc, _ = library.add_document(SimpleUploadedFile(name, content or pdf_with_box()))
        return doc

    def run_jobs(self):
        while (job := library.claim_next()) is not None:
            library.run_job(job)

    def describe(self, doc, **params):
        library.enqueue(doc, Job.Kind.FIGURES, params)
        self.run_jobs()
        doc.refresh_from_db()
        return figures.load(doc)


class ParseTests(FigureTestCase):
    def test_parse_asks_for_layout_lists_figures_and_drops_the_json(self):
        doc = self.upload()
        self.run_jobs()
        self.assertTrue(self.convert.call_args.kwargs["with_layout"])
        data = figures.load(doc)
        self.assertEqual([(f["status"], f["skip_reason"]) for f in data["figures"]], [
            ("pending", ""), ("skipped", "too small (under 2 % of the page)"), ("skipped", "covers the whole page")])
        self.assertEqual(data["figures"][0]["caption"], "Figure 1: Drive train")
        self.assertEqual(data["figures"][0]["page"], 1)
        self.assertNotIn("json_content", json.loads(library.parsed_path(doc).read_text())["document"])
        self.assertEqual(figures.summary(data), {"found": 3, "eligible": 1, "described": 0, "pending": 1,
                                                 "failed": 0, "skipped": 2, "drawings": 0})


class DescribeTests(FigureTestCase):
    def test_describes_eligible_figures_once(self):
        doc = self.upload()
        self.run_jobs()
        data = self.describe(doc)
        self.assertEqual(self.vlm.call_count, 1)
        jpeg, prompt = self.vlm.call_args.args
        self.assertEqual(Image.open(io.BytesIO(jpeg)).format, "JPEG")
        self.assertIn("Caption in the document: Figure 1: Drive train", prompt)
        self.assertEqual(self.vlm.call_args.kwargs["purpose"], "vlm-figure")
        first = data["figures"][0]
        self.assertEqual((first["status"], first["kind"]), ("done", "schematic"))
        self.assertIn("VFD101", first["description"])
        self.assertEqual(doc.figures_status, "done")
        self.assertTrue(figures.image_path(doc, 0).exists())

        self.describe(doc)                              # nothing left: the VLM isn't called again
        self.assertEqual(self.vlm.call_count, 1)
        self.describe(doc, figures=[0])                 # "Describe again" for one figure
        self.assertEqual(self.vlm.call_count, 2)
        data = self.describe(doc, figures=[1])          # skipped for its size, asked for by number: described anyway
        self.assertEqual(self.vlm.call_count, 3)
        self.assertEqual(data["figures"][1]["status"], "done")

    def test_old_documents_are_parsed_again_first(self):
        doc = self.upload()
        self.run_jobs()
        figures.figures_path(doc).unlink()              # parsed before figures existed
        self.convert.reset_mock()
        data = self.describe(doc)
        self.convert.assert_called_once()
        self.assertEqual(data["figures"][0]["status"], "done")

    def test_decorative_figures_are_skipped_and_left_out_of_the_text(self):
        self.vlm.return_value = "Kind: decorative\nDescription: A company logo.\nText: none"
        doc = self.upload()
        self.run_jobs()
        data = self.describe(doc)
        self.assertEqual((data["figures"][0]["status"], data["figures"][0]["skip_reason"]),
                         ("skipped", "decorative (logo, icon or background)"))
        self.assertNotIn("[Figure", library.load_parsed(doc)["document"]["md_content"])

    def test_one_failed_figure_does_not_fail_the_job(self):
        self.vlm.side_effect = RuntimeError("HTTP 500")
        doc = self.upload()
        self.run_jobs()
        data = self.describe(doc)
        self.assertEqual(data["figures"][0]["status"], "failed")
        self.assertEqual((doc.figures_status, doc.figures_error), ("done", "1 figure(s) could not be described"))
        self.assertEqual(Job.objects.get(kind="figures").status, "done")

    def test_bsk_unreachable_is_retried_with_backoff_then_left_pending(self):
        self.vlm.side_effect = gpu.GpuUnavailable("BSK asleep")
        doc = self.upload()
        self.run_jobs()
        library.enqueue(doc, Job.Kind.FIGURES)
        for attempt, minutes in ((1, 5), (2, 15)):
            self.run_jobs()
            job = Job.objects.get(kind="figures")
            self.assertEqual((job.status, job.attempts), ("queued", attempt))
            self.assertIn(f"trying again in {minutes} min", job.message)
            self.assertIsNone(library.claim_next())      # not before run_after
            Job.objects.filter(pk=job.pk).update(run_after=timezone.now())
        self.run_jobs()
        job = Job.objects.get(kind="figures")
        doc.refresh_from_db()
        self.assertEqual((job.status, doc.figures_status), ("done", "pending"))
        self.assertIn("BSK was not reachable", doc.figures_error)
        self.assertEqual(figures.load(doc)["figures"][0]["status"], "pending")
        self.assertEqual(doc.parse_status, "parsed")     # the parse is never failed by the VLM

    def test_parse_jobs_run_before_figure_jobs(self):
        first = self.upload()
        self.run_jobs()
        library.enqueue(first, Job.Kind.FIGURES)
        self.upload("Second.pdf", pdf_with_box() + b"\n%second")      # queues its parse after the figures job
        self.assertEqual(library.claim_next().kind, "parse")

    def test_office_files_are_refused(self):
        doc = self.upload("notes.docx", b"PK docx")
        with self.assertRaisesMessage(library.LibraryError, "figures need a PDF or an image"):
            library.enqueue(doc, Job.Kind.FIGURES)

    def test_image_document_is_one_figure(self):
        buf = io.BytesIO()
        Image.new("RGB", (2000, 1000), "blue").save(buf, format="PNG")
        self.convert.side_effect = lambda *a, **k: parsed(md="<!-- image -->\n\nPUMP P-101", layout={"pictures": [picture(1, 0, 10, 10, 0)]})
        doc = self.upload("pid.png", buf.getvalue())
        self.run_jobs()
        data = self.describe(doc)
        self.assertEqual(len(data["figures"]), 1)
        self.assertEqual(Image.open(io.BytesIO(self.vlm.call_args.args[0])).size, (1280, 640))
        md = library.load_parsed(doc)["document"]["md_content"]
        self.assertTrue(md.startswith("[Figure p.1 (schematic): A VFD feeds motor M101."))


class TextTests(FigureTestCase):
    def test_description_replaces_its_placeholder_and_becomes_a_figure_chunk(self):
        doc = self.upload()
        self.run_jobs()
        self.describe(doc)
        md = library.load_parsed(doc)["document"]["md_content"]
        line = "[Figure p.1 (schematic): A VFD feeds motor M101. It drives the screw. Text: VFD101; M101; 45 kW]"
        self.assertIn(f"# Drive\n\n\n\n{line}\n\n\n\nThe motor drives the screw.", md)
        self.assertEqual(md.count("<!-- image -->"), 2)           # skipped figures keep their placeholder
        self.assertNotIn("[Figure", library.load_parsed(doc, with_figures=False)["document"]["md_content"])
        chunks = chunk_parsed(library.load_parsed(doc), "sentence", resolve_params("sentence", {}))
        figure_chunks = [c for c in chunks if c.content_type == "figure"]
        self.assertEqual(len(figure_chunks), 1)
        self.assertEqual((figure_chunks[0].page_start, figure_chunks[0].text.strip()), (1, line))


class ApiTests(FigureTestCase):
    def test_list_image_and_describe(self):
        doc = self.upload()
        self.run_jobs()
        client = APIClient()
        listed = client.get("/api/documents/").data["documents"][0]["figures"]
        self.assertEqual((listed["found"], listed["eligible"], listed["supported"], listed["needs_parse"]), (3, 1, True, False))

        self.assertEqual(client.get(f"/api/documents/{doc.id}/figures/0/image/")["Content-Type"], "image/jpeg")
        self.assertEqual(client.get(f"/api/documents/{doc.id}/figures/9/image/").status_code, 404)

        r = client.post(f"/api/documents/{doc.id}/figures/describe/", {}, format="json")
        self.assertEqual((r.status_code, r.data["kind"]), (202, "figures"))
        self.run_jobs()
        data = client.get(f"/api/documents/{doc.id}/figures/").data
        self.assertEqual((data["status"], data["described"]), ("done", 1))
        self.assertEqual(data["figures"][0]["kind"], "schematic")
        self.assertNotIn("bbox", data["figures"][0])

        r = client.post(f"/api/documents/{doc.id}/figures/describe/", {"figures": "all"}, format="json")
        self.assertEqual(r.status_code, 400)
