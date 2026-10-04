"""Client for the VLM on the BSK desktop (Qwen3-VL-4B, OpenAI-compatible; contract v1.2).

Every request holds the Hub-wide GPU lock and activates `vl` through the
orchestrator first (`gpu.orchestrator.use`), so it never interrupts a Docling
parse and nothing interrupts it. One image per request, one request at a time.
"""

from __future__ import annotations

import base64

import requests
from django.conf import settings

from usage import recorder as usage

from . import orchestrator as gpu


def image_part(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}


def chat(messages: list[dict], max_tokens: int = 400, purpose: str | None = None, wait: float | None = None,
         json_mode: bool = False, timeout: float | None = None) -> str:
    """POST /v1/chat/completions on the VLM and return the reply text.

    `purpose` labels the call in Analytics (default: the surrounding usage scope).
    Raises gpu.GpuBusy (lock wait passed) or gpu.GpuUnavailable (BSK asleep / VLM failed to start).
    """
    payload = {"model": settings.VLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    with gpu.use("vl", wait=wait), usage.track(purpose, settings.VLM_CHAT_MODEL_ID) as call:
        try:
            resp = requests.post(settings.VLM_URL.rstrip("/") + "/chat/completions", json=payload,
                                 timeout=(gpu.CONNECT_TIMEOUT, timeout or settings.VLM_TIMEOUT))
        except requests.ConnectionError as exc:
            raise gpu.GpuUnavailable(f"VLM unreachable: {exc}") from exc
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        call.from_response(data)
        call.estimate(usage.messages_text(messages), reply)
    return reply


def complete(jpeg: bytes, prompt: str, max_tokens: int = 400, **kwargs) -> str:
    """One image + one prompt (the pipeline's request shape)."""
    return chat([{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(jpeg)]}],
                max_tokens=max_tokens, **kwargs)
