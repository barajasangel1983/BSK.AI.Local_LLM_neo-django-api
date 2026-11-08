# Claude Session Log — 11/08/2026 — P24-2 (part 2): one system prompt, pictures EDGE's way

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p24-2b-prompt-pictures` (off `main` @ `f0d77ce`)
- **Plan:** frontend `claude/P24_studio_edge_alignment_plan.md`, Revision 2, P24-2 (Angel "go": pictures EDGE's way;
  one prompt with the standard editable part in Studio Settings). Paired with frontend `feat/p24-2b-prompt-pictures`.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | `chat/prompt.py`: the editable part — a conversation's own, else Settings → Chat's, else the built-in one | `chat/prompt.py` |
| 2 | Fields: `ChatSettings.system_prompt`, `Conversation.system_prompt`, `ChatAttachment.description`; migration 0010 | `chat/models.py`, `chat/migrations/0010_p24_2_prompt_pictures.py` |
| 3 | Asset prompt: editable part first, then fixed rules (+ EDGE's "values are the last recorded ones…" and the picture rule); library and plain chats start with the editable part too | `chat/asset_context.py`, `chat/views.py` |
| 4 | Pictures: a text model may get an image; the vision model reads it first (once, kept on the attachment), the text model answers with it, the facts and the manuals; a follow-up uses the latest picture; a new picture that can't be read stops the question (gpu_busy / gpu_unavailable); a PDF page still needs the vision model | `chat/views.py` |
| 5 | API: Settings → Chat `system_prompt` (+ `default_system_prompt`); `PATCH /api/conversations/<id>/ {system_prompt}`; a draft's prompt with its first message; the Context Service's ask uses the standard prompt | `chat/settings_views.py`, `chat/views.py`, `chat/serializers.py`, `context_service/service.py` |
| 6 | Tests | `chat/test_stream.py`, `chat/test_attachments.py`, `chat/test_speech.py`, `chat/test_asset_chat.py` |

## Verification

- Suite 820 OK. Evals 8B (agent on): safety 9/10, hard 20/30 (17 with the old prompt and the agent).
- Preview (worktree API :8003, Studio :8081): a picture question with the browser's headers — the vision model read it
  in 2 s, the 8B answered with it and the manuals in 31 s, noting reading errors are possible.
