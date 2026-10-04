"""Chat attachments (P9a.1): images and PDFs stay with the conversation; images and PDF
pages go to the vision model, PDFs read as text go to the text models."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from chat import attachments
from chat.models import ChatAttachment, Conversation
from chat.tests import _ok_response
from gpu import orchestrator as gpu
from gpu.tests.tests_vlm import pdf, png
from ingestion.chunker import PAGE_BREAK
from ingestion.docling_client import DoclingError
from ingestion.models import Document

VLM = "bsk-qwen3-vl-4b"
DGX = "dgx-qwen38-27b-fp8"


def parsed(*pages):
    return {"status": "success", "document": {"md_content": f"\n\n{PAGE_BREAK}\n\n".join(pages)}}


class AttachmentTestCase(TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        override = override_settings(CHAT_FILES_DIR=self.tmp, CHAT_CONTEXT_MAX_CHARS=24000)
        override.enable()
        self.addCleanup(override.disable)
        self.client = APIClient()
        for target, value in (("chat.views.generate_title", ("Title", "llm")),):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def upload(self, name, data, **extra):
        return self.client.post("/api/chat/attachments/", {"file": SimpleUploadedFile(name, data), **extra},
                                format="multipart")

    def chat(self, message, model, **extra):
        return self.client.post("/api/chat/", {"message": message, "model": model, "use_rag": False, **extra},
                                format="json")


class UploadTests(AttachmentTestCase):
    def test_image_is_stored_as_downscaled_jpeg(self):
        r = self.upload("photo.png", png(2560, 1280))
        self.assertEqual(r.status_code, 201)
        self.assertEqual((r.data["kind"], r.data["width"], r.data["height"]), ("image", 1280, 640))
        f = self.client.get(f"/api/chat/attachments/{r.data['id']}/file/")
        self.assertEqual((f.status_code, f["Content-Type"]), (200, "image/jpeg"))

    def test_pdf_reports_pages_and_serves_page_previews(self):
        r = self.upload("drawing.pdf", pdf(pages=3))
        self.assertEqual((r.status_code, r.data["kind"], r.data["page_count"]), (201, "pdf", 3))
        page = self.client.get(f"/api/chat/attachments/{r.data['id']}/pages/2/")
        self.assertEqual((page.status_code, page["Content-Type"]), (200, "image/jpeg"))
        self.assertEqual(self.client.get(f"/api/chat/attachments/{r.data['id']}/pages/9/").status_code, 400)

    def test_rejects_other_types_unreadable_files_and_long_pdfs(self):
        self.assertEqual(self.upload("notes.docx", b"PK").status_code, 400)
        self.assertEqual(self.upload("broken.png", b"not an image").status_code, 400)
        with override_settings(CHAT_ATTACHMENT_MAX_PAGES=2):
            r = self.upload("long.pdf", pdf(pages=3))
        self.assertEqual(r.status_code, 400)
        self.assertIn("RAG Lab", r.data["error"])
        with override_settings(CHAT_ATTACHMENT_MAX_BYTES=10):
            self.assertEqual(self.upload("big.png", png(50, 50)).status_code, 400)
        self.assertEqual(ChatAttachment.objects.count(), 0)
        self.assertEqual(list(Path(self.tmp).iterdir()), [])

    def test_unsent_upload_can_be_removed(self):
        r = self.upload("photo.png", png(50, 50))
        self.assertEqual(self.client.delete(f"/api/chat/attachments/{r.data['id']}/").status_code, 204)
        self.assertFalse((Path(self.tmp) / r.data["id"]).exists())


class ImageChatTests(AttachmentTestCase):
    @patch("chat.views.vlm.chat", return_value="A motor nameplate.")
    def test_image_goes_to_the_vision_model_and_stays_in_the_conversation(self, vlm_chat):
        att = self.upload("plate.png", png(100, 80)).data
        r = self.chat("What is this?", VLM, attachment_id=att["id"])
        self.assertEqual(r.status_code, 200)
        user, assistant = r.data["messages"]
        self.assertEqual(user["attachment"]["id"], att["id"])
        self.assertEqual(assistant["content"], "A motor nameplate.")
        content = vlm_chat.call_args.args[0][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "What is this?"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(str(ChatAttachment.objects.get().conversation_id), r.data["id"])
        self.assertEqual(Document.objects.count(), 0)          # never added to the library

    @patch("chat.views.vlm.chat", return_value="ok")
    def test_follow_up_resends_only_the_latest_image(self, vlm_chat):
        first = self.upload("one.png", png(100, 80, color="red")).data
        r = self.chat("first?", VLM, attachment_id=first["id"])
        second = self.upload("two.png", png(100, 80, color="blue"), conversation_id=r.data["id"]).data
        self.chat("second?", VLM, attachment_id=second["id"], conversation_id=r.data["id"])
        self.chat("and its colour?", VLM, conversation_id=r.data["id"])
        messages = vlm_chat.call_args.args[0]
        images = [m for m in messages if isinstance(m["content"], list)]
        self.assertEqual(len(images), 1)
        self.assertIs(images[0], messages[-1])
        url_latest = images[0]["content"][1]["image_url"]["url"]
        url_second = vlm_chat.call_args_list[1].args[0][-1]["content"][1]["image_url"]["url"]
        self.assertEqual(url_latest, url_second)

    @patch("chat.views.vlm.chat", return_value="Hello")
    def test_text_only_message_to_the_vision_model(self, vlm_chat):
        self.assertEqual(self.chat("hi", VLM).status_code, 200)
        self.assertEqual(vlm_chat.call_args.args[0][-1], {"role": "user", "content": "hi"})

    def test_image_with_a_text_model_is_refused_before_anything_is_saved(self):
        att = self.upload("plate.png", png(100, 80)).data
        r = self.chat("What is this?", DGX, attachment_id=att["id"])
        self.assertEqual((r.status_code, r.data["code"]), (400, "vision_model_required"))
        self.assertEqual(Conversation.objects.count(), 0)

    def test_gpu_busy_and_unavailable_are_503_with_a_code(self):
        att = self.upload("plate.png", png(100, 80)).data
        with patch("chat.views.vlm.chat", side_effect=gpu.GpuBusy("GPU busy")):
            r = self.chat("What is this?", VLM, attachment_id=att["id"])
        self.assertEqual((r.status_code, r.data["code"]), (503, "gpu_busy"))
        with patch("chat.views.vlm.chat", side_effect=gpu.GpuUnavailable("BSK asleep")):
            r = self.chat("What is this?", VLM, attachment_id=att["id"],
                          conversation_id=str(Conversation.objects.get().id))
        self.assertEqual((r.status_code, r.data["code"]), (503, "gpu_unavailable"))

    @patch("chat.views.vlm.chat", return_value="ok")
    def test_deleting_the_conversation_removes_its_files(self, vlm_chat):
        att = self.upload("plate.png", png(100, 80)).data
        r = self.chat("What is this?", VLM, attachment_id=att["id"])
        self.assertEqual(self.client.delete(f"/api/chat/attachments/{att['id']}/").status_code, 409)
        self.assertTrue((Path(self.tmp) / att["id"]).exists())
        self.client.delete(f"/api/conversations/{r.data['id']}/")
        self.assertEqual(ChatAttachment.objects.count(), 0)
        self.assertFalse((Path(self.tmp) / att["id"]).exists())

    def test_attachment_of_another_conversation_is_refused(self):
        other = Conversation.objects.create(owner=None, title="x")
        att = self.upload("plate.png", png(100, 80)).data
        ChatAttachment.objects.update(conversation=other)
        self.assertEqual(self.chat("?", VLM, attachment_id=att["id"]).status_code, 400)


class PdfChatTests(AttachmentTestCase):
    @patch("chat.views.vlm.chat", return_value="A P&ID with pump P-101.")
    def test_a_pdf_page_goes_to_the_vision_model(self, vlm_chat):
        att = self.upload("drawing.pdf", pdf(pages=3)).data
        r = self.chat("What is on this page?", VLM, attachment_id=att["id"], page=2)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["messages"][0]["attachment"]["page"], 2)
        self.assertTrue(vlm_chat.call_args.args[0][-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg"))
        self.assertEqual(self.chat("?", VLM, attachment_id=att["id"], page=7).status_code, 400)

    def test_a_whole_pdf_needs_a_text_model(self):
        att = self.upload("manual.pdf", pdf(pages=2)).data
        r = self.chat("Summarize", VLM, attachment_id=att["id"])
        self.assertEqual((r.status_code, r.data["code"]), (400, "text_model_required"))

    @patch("chat.views.requests.post")
    @patch("chat.attachments.convert_file", return_value=parsed("Torque is 45 Nm.", "Grease every 500 h."))
    def test_pdf_text_is_in_the_prompt_for_the_message_and_the_follow_up(self, convert, post):
        post.side_effect = [_ok_response("45 Nm"), _ok_response("Every 500 h")]
        att = self.upload("manual.pdf", pdf(pages=2)).data
        r = self.chat("What is the torque?", DGX, attachment_id=att["id"])
        self.assertEqual(r.status_code, 200)
        system = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
        self.assertIn("ATTACHED DOCUMENT: manual.pdf", system)
        self.assertIn("[page 1]\nTorque is 45 Nm.", system)
        self.assertIn("[page 2]\nGrease every 500 h.", system)
        self.assertEqual(r.data["messages"][1]["sources"][0],
                         {"kind": "attachment", "source": "manual.pdf", "attachment_id": att["id"],
                          "pages_read": 2, "page_count": 2, "truncated": False})

        self.chat("And the greasing interval?", DGX, conversation_id=r.data["id"])
        self.assertIn("Grease every 500 h.", post.call_args_list[1].kwargs["json"]["messages"][0]["content"])
        convert.assert_called_once()                            # parsed once, then cached
        self.assertEqual(convert.call_args.kwargs["wait"], 120)  # chat waits a short time for the GPU
        self.assertEqual(Document.objects.count(), 0)           # never added to the library

    @patch("chat.views.requests.post")
    @patch("chat.attachments.convert_file", return_value=parsed("A" * 4000, "B" * 4000, "C" * 4000))
    def test_long_pdf_is_cut_at_whole_pages_and_says_so(self, convert, post):
        post.return_value = _ok_response("ok")
        att = self.upload("long.pdf", pdf(pages=3)).data
        with override_settings(CHAT_CONTEXT_MAX_CHARS=20000, CHAT_ATTACHMENT_SHARE=0.5):
            r = self.chat("Summarize", DGX, attachment_id=att["id"])
        system = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("B" * 4000, system)
        self.assertNotIn("C" * 10, system)
        source = r.data["messages"][1]["sources"][0]
        self.assertEqual((source["pages_read"], source["page_count"], source["truncated"]), (2, 3, True))

    @patch("chat.attachments.convert_file")
    def test_placeholder_model_does_not_claim_to_read_the_pdf(self, convert):
        att = self.upload("manual.pdf", pdf(pages=2)).data
        r = self.chat("Summarize", "local-small", attachment_id=att["id"])
        self.assertEqual((r.status_code, r.data["messages"][1]["sources"]), (200, []))
        convert.assert_not_called()

    def test_gpu_busy_while_parsing_is_503(self):
        att = self.upload("manual.pdf", pdf(pages=2)).data
        error = DoclingError("Docling unavailable: GPU busy")
        error.__cause__ = gpu.GpuBusy("GPU busy")
        with patch("chat.attachments.convert_file", side_effect=error):
            r = self.chat("Summarize", DGX, attachment_id=att["id"])
        self.assertEqual((r.status_code, r.data["code"]), (503, "gpu_busy"))

    def test_unsent_uploads_are_removed_after_a_day(self):
        from datetime import timedelta
        from django.utils import timezone
        att = self.upload("old.png", png(50, 50)).data
        ChatAttachment.objects.update(created_at=timezone.now() - timedelta(hours=30))
        self.assertEqual(attachments.remove_orphans(), 1)
        self.assertFalse((Path(self.tmp) / att["id"]).exists())
