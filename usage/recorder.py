"""Record every model / AI-service call for the Analytics page.

    with usage.track("chat", "dgx-qwen38-27b-fp8") as call:
        resp = requests.post(...)
        call.from_response(resp.json())      # OpenAI / vLLM / Grok / Ollama usage formats

`purpose` and the conversation default to the surrounding `usage.scope(...)`
(set by the chat view), so call sites deep in the stack need no extra
arguments. Recording never changes the call: exceptions propagate unchanged
and a failure to save the record is only logged.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import requests

logger = logging.getLogger("usage")

_scope: contextvars.ContextVar[dict] = contextvars.ContextVar("usage_scope", default={})


@contextmanager
def scope(purpose: str | None = None, conversation_id=None):
    """Default purpose / conversation for calls made inside the block."""
    token = _scope.set({**_scope.get(), **{k: v for k, v in
                                           (("purpose", purpose), ("conversation_id", conversation_id)) if v}})
    try:
        yield
    finally:
        _scope.reset(token)


def set_conversation(conversation_id) -> None:
    """Attach the conversation to the current scope once it is known."""
    current = _scope.get()
    if current:
        current["conversation_id"] = conversation_id


def current_purpose(default: str) -> str:
    return _scope.get().get("purpose", default)


@dataclass
class Call:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated: bool = False

    def tokens(self, prompt: int | None, completion: int | None, estimated: bool = False) -> None:
        self.prompt_tokens, self.completion_tokens, self.estimated = prompt, completion, estimated

    def from_response(self, data: dict) -> None:
        """Token counts from an OpenAI-style (`usage`) or Ollama (`*_eval_count`) response."""
        if not isinstance(data, dict):
            return
        u = data.get("usage") or {}
        if u.get("prompt_tokens") is not None or u.get("completion_tokens") is not None:
            self.tokens(u.get("prompt_tokens"), u.get("completion_tokens"))
        elif data.get("prompt_eval_count") is not None or data.get("eval_count") is not None:
            self.tokens(data.get("prompt_eval_count"), data.get("eval_count"))

    def estimate(self, prompt_text: str, completion_text: str) -> None:
        """Fallback when the provider reports no usage (cl100k token counts)."""
        if self.prompt_tokens is not None or self.completion_tokens is not None:
            return
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            self.tokens(len(enc.encode(prompt_text or "")), len(enc.encode(completion_text or "")), estimated=True)
        except Exception:  # pragma: no cover - tiktoken missing / offline
            self.tokens(len(prompt_text or "") // 4, len(completion_text or "") // 4, estimated=True)


def messages_text(messages: list[dict]) -> str:
    """Plain text of a chat messages list (for token estimates)."""
    parts = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
        else:
            parts.append(str(content or ""))
    return "\n".join(parts)


@contextmanager
def track(purpose: str | None, model_id: str, conversation_id=None):
    """Time the block and save a ModelCall (ok / error / timeout)."""
    ctx = _scope.get()
    call = Call()
    start = time.monotonic()
    status, error = "ok", ""
    try:
        yield call
    except requests.Timeout as exc:
        status, error = "timeout", str(exc)
        raise
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            from .models import ModelCall
            ModelCall.objects.create(
                purpose=purpose or ctx.get("purpose") or "other",
                model_id=model_id[:128],
                conversation_id=conversation_id or ctx.get("conversation_id"),
                prompt_tokens=call.prompt_tokens, completion_tokens=call.completion_tokens,
                tokens_estimated=call.estimated,
                latency_ms=round((time.monotonic() - start) * 1000),
                status=status, error=error[:255],
            )
        except Exception:
            logger.exception("could not record model call purpose=%s model=%s", purpose, model_id)
