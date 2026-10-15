"""The model that writes the monitoring agent's sentences: qwen3:8b on Ollama, the DGX when it is away.

The same choice as EDGE's agent. Titles and numbers never come from here; `write` returns ("", "") when
no model can be used, and the post is then written from its template.
"""

from __future__ import annotations

import logging

import requests
from django.conf import settings

from spc import llm as spc_llm
from usage import recorder as usage

logger = logging.getLogger("chat")

OUTPUT_TOKENS = 300
PRIMARY = ("ollama-qwen3-8b", "qwen3:8b (Ollama)")
FALLBACK = ("dgx-qwen38-27b-fp8", "Qwen3 27B (DGX, fallback)")


def _ollama_answers() -> bool:
    try:
        tags = requests.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3).json()
    except (requests.RequestException, ValueError):
        return False
    return any(m.get("name") == "qwen3:8b" for m in tags.get("models", []))


def pick() -> tuple[str, str] | None:
    """(model id, name shown on the post): the small model when its server answers, else the DGX."""
    return PRIMARY if _ollama_answers() else FALLBACK


def write(model: tuple[str, str] | None, system: str, body: str) -> tuple[str, str]:
    """(the model's sentences, its name), or ("", "") when it cannot be used."""
    if model is None:
        return "", ""
    model_id, name = model
    messages = [{"role": "system", "content": system}, {"role": "user", "content": body}]
    try:
        with usage.track("monitor", model_id) as call:
            if model_id == PRIMARY[0]:
                raw = spc_llm._ollama(messages, settings.SPC_LLM_NUM_CTX, OUTPUT_TOKENS)
                text = (raw.get("message") or {}).get("content") or ""
            else:
                raw = spc_llm._openai(model_id, messages, OUTPUT_TOKENS)
                text = raw["choices"][0]["message"]["content"] or ""
            call.from_response(raw)
            call.estimate(usage.messages_text(messages), text)
    except (requests.RequestException, KeyError, IndexError, ValueError, spc_llm.ModelError) as exc:
        logger.warning("monitor: the model was not used (%s)", exc)
        return "", ""
    text = spc_llm.THINK.sub("", text).strip()
    return (text[:1200], name) if text else ("", "")
