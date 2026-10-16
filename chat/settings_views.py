"""Settings → Chat, and speech to text (P15c).

    GET  /api/chat/settings/        {speech_enabled, speech_language, speech_api_key_set, speech_available}
    PUT  /api/chat/settings/        any of {speech_enabled, speech_language, speech_api_key}; the key is
                                    write-only ("" clears it) and never returned
    POST /api/chat/speech/          multipart "file" (a recording, WebM/Opus), optional "asset_id": {text, language, duration}
    POST /api/chat/speech/test/     {api_key?}: sends half a second of silence with that key (or the saved one)
"""

from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from . import speech
from .models import ChatSettings

ERROR_STATUS = {"off": status.HTTP_409_CONFLICT, "key": status.HTTP_400_BAD_REQUEST,
                "audio": status.HTTP_400_BAD_REQUEST, "provider": status.HTTP_502_BAD_GATEWAY}


def _json(s: ChatSettings) -> dict:
    return {"speech_enabled": s.speech_enabled, "speech_language": s.speech_language,
            "speech_api_key_set": bool(s.speech_api_key), "speech_available": bool(s.speech_enabled and s.speech_api_key)}


@api_view(["GET", "PUT"])
def chat_settings(request):
    s = ChatSettings.get()
    if request.method == "PUT":
        data = request.data
        if "speech_enabled" in data:
            if not isinstance(data["speech_enabled"], bool):
                return Response({"error": "speech_enabled must be true or false"}, status=status.HTTP_400_BAD_REQUEST)
            s.speech_enabled = data["speech_enabled"]
        if "speech_language" in data:
            if data["speech_language"] not in ChatSettings.LANGUAGES:
                return Response({"error": f"speech_language must be one of {', '.join(ChatSettings.LANGUAGES)}"},
                                status=status.HTTP_400_BAD_REQUEST)
            s.speech_language = data["speech_language"]
        if "speech_api_key" in data:
            s.speech_api_key = str(data["speech_api_key"] or "").strip()[:300]
        s.save()
    return Response(_json(s))


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def transcribe(request):
    upload = request.FILES.get("file")
    if upload is None:
        return Response({"error": "no recording (multipart field 'file')"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        return Response(speech.transcribe(upload.read(), upload.name or "speech.webm",
                                          upload.content_type or "audio/webm", str(request.data.get("asset_id") or "")))
    except speech.SpeechError as exc:
        return Response({"error": str(exc), "kind": exc.kind}, status=ERROR_STATUS.get(exc.kind, 400))


@api_view(["POST"])
@parser_classes([JSONParser])
def speech_test(request):
    key = request.data.get("api_key")
    try:
        speech.transcribe(speech.silence(), "silence.wav", "audio/wav",
                          api_key=str(key).strip() if key else ChatSettings.get().speech_api_key)
    except speech.SpeechError as exc:
        return Response({"ok": False, "message": str(exc)})
    return Response({"ok": True})
