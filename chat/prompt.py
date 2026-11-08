"""The system prompt (P24-2): one for the Studio chat, EDGE and the Context Service.

    editable part   the standard one (Settings → Chat, an admin), or a conversation's own (engineers and admins in
                    EDGE; the Studio has one user today) — who the assistant is, how it answers
    fixed rules     always applied, not editable: answer from the context, cite it, never invent

The fixed rules for an asset question are in chat/asset_context.py (ASSET_PROMPT); this module holds the editable part.
"""

from __future__ import annotations

# The built-in standard (Settings → Chat can replace it).
DEFAULT_EDITABLE = ("You are Neo, the assistant for the BSKLAB plant. Answer practically, in the language of the "
                    "question.")
MAX_CHARS = 4000

PICTURE_RULE = ("\nWhat the attached picture shows was read by a vision model and may contain reading errors: say so "
                "when it matters.")


def standard() -> str:
    """The standard editable part: Settings → Chat's, else the built-in one."""
    from .models import ChatSettings
    return (ChatSettings.get().system_prompt or "").strip()[:MAX_CHARS] or DEFAULT_EDITABLE


def for_conversation(conversation) -> str:
    """The conversation's own editable part, else the standard one."""
    own = (getattr(conversation, "system_prompt", None) or "").strip() if conversation is not None else ""
    return own[:MAX_CHARS] or standard()


def picture_block(description: str) -> str:
    return f"PICTURE (what the vision model read in the attached picture):\n{description.strip()}" if description else ""
