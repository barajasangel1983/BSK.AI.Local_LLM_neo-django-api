"""Speech to text and Settings → Chat (P15c). The provider is mocked; no audio leaves the test."""

from unittest.mock import MagicMock, patch

import requests
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from chat import speech
from chat.models import ChatSettings


def answer(status=200, body=None):
    resp = MagicMock(status_code=status, text="x")
    resp.json.return_value = body or {"text": " Clear the DIE PLUG alarm on EXTR01 ", "language": "en", "duration": 2.1}
    return resp


class SpeechTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        speech._terms_cache.clear()
        p = patch("chat.speech.key_terms", return_value=["Extruder EXTR01", "EXTR01", "DIE PLUG"])
        p.start()
        self.addCleanup(p.stop)

    def test_settings_keep_the_key_write_only(self):
        self.assertEqual(self.client.get("/api/chat/settings/").json(),
                         {"speech_enabled": False, "speech_language": "auto", "speech_api_key_set": False, "speech_available": False})
        r = self.client.put("/api/chat/settings/", {"speech_enabled": True, "speech_language": "es", "speech_api_key": "xai-123"},
                            format="json").json()
        self.assertEqual(r, {"speech_enabled": True, "speech_language": "es", "speech_api_key_set": True, "speech_available": True})
        self.assertNotIn("xai-123", str(self.client.get("/api/chat/settings/").content))
        self.assertEqual(self.client.put("/api/chat/settings/", {"speech_language": "fr"}, format="json").status_code, 400)
        self.assertEqual(self.client.put("/api/chat/settings/", {"speech_enabled": "yes"}, format="json").status_code, 400)
        self.client.put("/api/chat/settings/", {"speech_api_key": ""}, format="json")
        self.assertFalse(ChatSettings.get().speech_api_key)

    @patch("chat.speech.requests.post", return_value=answer())
    def test_a_recording_becomes_text_with_the_key_terms(self, post):
        recording = SimpleUploadedFile("speech.webm", b"opus", content_type="audio/webm")
        self.assertEqual(self.client.post("/api/chat/speech/", {"file": recording}, format="multipart").json()["kind"], "off")
        ChatSettings.objects.update_or_create(pk=1, defaults={"speech_enabled": True, "speech_api_key": "xai-123", "speech_language": "es"})
        recording = SimpleUploadedFile("speech.webm", b"opus", content_type="audio/webm")
        r = self.client.post("/api/chat/speech/", {"file": recording, "asset_id": "bsk:asset:EXTR01"}, format="multipart")
        self.assertEqual(r.json(), {"text": "Clear the DIE PLUG alarm on EXTR01", "language": "en", "duration": 2.1})
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer xai-123"})
        self.assertEqual(kwargs["data"][:3], [("model", "grok-voice-transcribe-2.0"), ("language", "es"), ("format", "true")])
        self.assertEqual([v for k, v in kwargs["data"] if k == "keyterm"], ["Extruder EXTR01", "EXTR01", "DIE PLUG"])
        self.assertEqual(kwargs["files"]["file"][0], "speech.webm")       # requests sends the file after the fields

    def test_provider_problems_are_said(self):
        ChatSettings.objects.update_or_create(pk=1, defaults={"speech_enabled": True, "speech_api_key": "xai-123"})
        for resp, kind, code in ((answer(401), "key", 400), (answer(500), "provider", 502)):
            with patch("chat.speech.requests.post", return_value=resp):
                r = self.client.post("/api/chat/speech/", {"file": SimpleUploadedFile("s.webm", b"x")}, format="multipart")
            self.assertEqual((r.status_code, r.json()["kind"]), (code, kind))
        with patch("chat.speech.requests.post", side_effect=requests.ConnectTimeout("x")):
            r = self.client.post("/api/chat/speech/", {"file": SimpleUploadedFile("s.webm", b"x")}, format="multipart")
        self.assertEqual(r.json()["kind"], "provider")
        big = SimpleUploadedFile("s.webm", b"x" * (speech.MAX_BYTES + 1))
        self.assertEqual(self.client.post("/api/chat/speech/", {"file": big}, format="multipart").json()["kind"], "audio")

    def test_the_key_is_tested_with_silence(self):
        with patch("chat.speech.requests.post", return_value=answer(body={"text": ""})) as post:
            self.assertEqual(self.client.post("/api/chat/speech/test/", {"api_key": "xai-new"}, format="json").json(), {"ok": True})
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer xai-new")
        self.assertTrue(post.call_args.kwargs["files"]["file"][1].startswith(b"RIFF"))
        with patch("chat.speech.requests.post", return_value=answer(403)):
            r = self.client.post("/api/chat/speech/test/", {"api_key": "bad"}, format="json").json()
        self.assertEqual(r["ok"], False)
        self.assertIn("refused the key", r["message"])

    def test_key_terms_come_from_the_graph(self):
        patch.stopall()
        speech._terms_cache.clear()
        entities = [{"label": "Signal", "name": "Die pressure", "keys": ["Die pressure", "die pressure bar"]},
                    {"label": "Alarm", "name": "Die plugged", "keys": ["Die plugged", "DIE_PLUG"]}]
        with patch("context_service.service.list_assets", return_value=[{"id": "bsk:asset:EXTR01", "name": "Extruder EXTR01"}]), \
                patch("chat.asset_context.entities", return_value=entities):
            terms = speech.key_terms("bsk:asset:EXTR01")
        self.assertEqual(terms, ["Extruder EXTR01", "EXTR01", "Die plugged", "DIE PLUG", "Die pressure"])
