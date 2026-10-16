"""Speech to text for the chat's question box (P15c). The only module that talks to the speech provider.

The same provider and rules as BSKLAB EDGE (server/edge/speech.py): xAI Grok Voice Transcribe
(POST https://api.x.ai/v1/stt, multipart; the file must be the last field). The recording is sent on,
not kept. The machine names, alarm codes and tags of the Context Graph are passed as key terms so
that tags such as DIE_PLUG or EXTR01 are recognised. The key is entered in Settings → Chat.
"""

from __future__ import annotations

import io
import logging
import struct
import time

import requests

from usage import recorder as usage

from .models import ChatSettings

logger = logging.getLogger("chat")

URL = "https://api.x.ai/v1/stt"
MODEL = "grok-voice-transcribe-2.0"
MAX_BYTES = 10 * 1024 * 1024
MAX_TERMS, MAX_TERM_CHARS = 100, 50
TERMS_TTL = 600         # seconds an asset's key terms are remembered

_terms_cache: dict[str, tuple[float, list[str]]] = {}


class SpeechError(Exception):
    """kind: off | key | audio | provider"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def available() -> bool:
    s = ChatSettings.get()
    return bool(s.speech_enabled and s.speech_api_key)


def key_terms(asset_id: str = "") -> list[str]:
    """Names worth recognising: every asset, and the selected asset's alarms, components, procedures and
    signals. Problems here never stop a transcription."""
    cached = _terms_cache.get(asset_id)
    if cached and time.monotonic() - cached[0] < TERMS_TTL:
        return cached[1]
    terms: list[str] = []
    try:
        from context_service import service
        from chat import asset_context
        for asset in service.list_assets():
            terms += [asset["name"], asset["id"].split(":", 2)[-1]]
        if asset_id:
            order = {"Alarm": 0, "Component": 1, "Procedure": 2, "Signal": 3}     # what people say on the floor first
            for entity in sorted(asset_context.entities(asset_id), key=lambda e: order.get(e["label"], 9)):
                terms += [entity["name"], *(k for k in entity["keys"] if any(c.isdigit() for c in k) or k.isupper())]
    except Exception as exc:          # the graph is down: transcribe without key terms
        logger.warning("speech: key terms unavailable: %s", exc)
    seen, out = set(), []
    for term in terms:
        term = " ".join(str(term).replace("_", " ").split())[:MAX_TERM_CHARS]
        if len(term) >= 3 and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    out = out[:MAX_TERMS]
    _terms_cache[asset_id] = (time.monotonic(), out)
    return out


def transcribe(audio: bytes, filename: str = "speech.webm", content_type: str = "audio/webm",
               asset_id: str = "", api_key: str | None = None) -> dict:
    """Text of a short recording: {text, language, duration}. `api_key` checks a key not saved yet."""
    s = ChatSettings.get()
    key = api_key if api_key is not None else s.speech_api_key
    if api_key is None and not s.speech_enabled:
        raise SpeechError("off", "Speech to text is switched off (Settings → Chat).")
    if not key:
        raise SpeechError("key", "No speech key is set (Settings → Chat).")
    if not audio:
        raise SpeechError("audio", "The recording is empty.")
    if len(audio) > MAX_BYTES:
        raise SpeechError("audio", "The recording is too long (limit 10 MB).")

    fields: list[tuple[str, str]] = [("model", MODEL)]
    if s.speech_language != "auto":
        fields += [("language", s.speech_language), ("format", "true")]      # formatting needs a language
    fields += [("keyterm", term) for term in key_terms(asset_id)]
    try:
        with usage.track("speech", "xai-stt") as call:
            # requests writes the form fields before the file: the provider requires the file last.
            resp = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, data=fields,
                                 files={"file": (filename, audio, content_type)}, timeout=(10, 60))
            if resp.status_code < 400:
                call.from_response(resp.json())
    except requests.RequestException as exc:
        raise SpeechError("provider", f"The speech service did not answer: {type(exc).__name__}") from exc
    if resp.status_code in (401, 403):
        raise SpeechError("key", "The speech service refused the key. Check it in Settings → Chat.")
    if resp.status_code >= 400:
        logger.warning("speech: provider error %s: %s", resp.status_code, resp.text[:300])
        raise SpeechError("provider", f"The speech service could not transcribe the recording (HTTP {resp.status_code}).")
    data = resp.json()
    return {"text": (data.get("text") or "").strip(), "language": data.get("language") or "",
            "duration": data.get("duration") or 0}


def silence(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A short silent WAV, to check a key without recording anything."""
    frames = int(seconds * rate)
    out = io.BytesIO()
    out.write(b"RIFF" + struct.pack("<I", 36 + frames * 2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    out.write(b"data" + struct.pack("<I", frames * 2) + b"\x00\x00" * frames)
    return out.getvalue()
