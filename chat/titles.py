"""Conversation titles: generated from the first exchange, renamable by the user.

The title is generated once, on the first user/assistant exchange, with the
DGX model (local, fast, independent of the chat model). If that fails or
times out, the title falls back to the start of the user's first message.
"""

from __future__ import annotations

import logging
import re

import requests
from django.conf import settings

logger = logging.getLogger("chat")

DEFAULT_TITLE = "New Conversation"
TITLE_MAX_CHARS = 60
FALLBACK_MAX_CHARS = 50
RENAME_MAX_CHARS = 120

TITLE_PROMPT = (
    "Write a short title (3 to 6 words) for the topic of this conversation. "
    "Reply with the title only: no quotes, no trailing punctuation."
)


def _truncate_words(text: str, limit: int) -> str:
    """Cut at a word boundary, adding an ellipsis when shortened."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(" ,;:-") + "…"


def normalize_title(text: str) -> str:
    """Collapse whitespace (titles are single-line)."""
    return re.sub(r"\s+", " ", text or "").strip()


def fallback_title(user_message: str) -> str:
    return _truncate_words(normalize_title(user_message), FALLBACK_MAX_CHARS) or DEFAULT_TITLE


def clean_title(raw: str) -> str:
    """Turn raw model output into a title, or '' if nothing usable is left."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    if "<think>" in text:  # unterminated reasoning: nothing usable
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    title = re.sub(r"^(title\s*:\s*)", "", lines[0], flags=re.IGNORECASE)
    title = title.strip(" \"'`*#_").rstrip(".!?:;,")
    return _truncate_words(normalize_title(title), TITLE_MAX_CHARS)


def _llm_title(user_message: str, reply: str) -> str:
    resp = requests.post(
        f"{settings.DGX_API_BASE.rstrip('/')}/v1/chat/completions",
        json={
            "model": settings.DGX_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": f"User: {user_message[:1500]}\nAssistant: {reply[:1500]}"},
            ],
            "max_tokens": 32,
            "temperature": 0.2,
            # Qwen chat templates: skip reasoning for this one-liner.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=settings.CHAT_TITLE_TIMEOUT,
    )
    resp.raise_for_status()
    return clean_title(resp.json()["choices"][0]["message"]["content"])


def generate_title(user_message: str, reply: str) -> tuple[str, str]:
    """Return (title, source) where source is 'llm' or 'fallback'."""
    try:
        title = _llm_title(user_message, reply)
        if title:
            return title, "llm"
    except Exception as exc:
        logger.warning("title generation failed, using fallback: %s", exc)
    return fallback_title(user_message), "fallback"
