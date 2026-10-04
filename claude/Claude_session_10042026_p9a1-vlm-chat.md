# Claude Session Log — 10/04/2026 — P9a.1 VLM client, gpu_smoke, chat attachments

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p9a1-vlm-chat` (off `main` @ `792f0ff`)
- **Plan:** frontend repo `claude/P9_vlm_integration_plan.md` (P9a = backend, P9b = frontend); paired with frontend `feat/p9b1-chat-attachments`
- **Why:** use the Qwen3-VL-4B model on the BSK PC in chat. Angel's decisions: images and PDFs can be attached; they stay in the chat session and never go to RAG; a PDF page can be shown to the vision model (page picker).

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | VLM client: one function for chat and pipelines; holds the GPU lock, activates `vl`, one image per request, recorded in Analytics as model `bsk-qwen3-vl-4b` | `gpu/vlm.py` |
| 2 | Image preparation: JPEG 85, long side ≤ 1280 px, never upscaled, transparency → white; PDF page count and page rendering | `gpu/images.py` |
| 3 | `manage.py gpu_smoke [--parse] [--vlm] [--cold] [--file]`: the two live checks from the runbook, PASS / FAIL; refuses while library jobs are active or the switch is off | `gpu/management/commands/gpu_smoke.py` |
| 4 | `activate("idle")` accepts the orchestrator's `health: "idle"` answer | `gpu/orchestrator.py` |
| 5 | `convert_file(..., wait=)`: chat passes a short GPU-lock wait instead of the pipeline's 30 min | `ingestion/docling_client.py` |
| 6 | `ChatAttachment` model; `Message.attachment` + `Message.attachment_page`. Files under `data/chat_files/<attachment id>/`, removed with the conversation; unsent uploads removed after 24 h | `chat/models.py`, `chat/migrations/0007_chat_attachments.py`, `chat/apps.py` |
| 7 | Attachment storage, PDF text context and endpoints (upload, delete unsent, file, page preview) | `chat/attachments.py`, `chat/urls.py` |
| 8 | Chat: `attachment_id` + `page`; model `bsk-qwen3-vl-4b`; 503 `gpu_busy` / `gpu_unavailable`; 400 `vision_model_required` / `text_model_required` | `chat/views.py`, `chat/serializers.py` |
| 9 | Settings: `VLM_CHAT_*`, `VLM_CONTEXT_MAX_CHARS`, `VLM_IMAGE_*`, `CHAT_FILES_DIR`, `CHAT_ATTACHMENT_*` | `neo_llm_api/settings.py` |
| 10 | Dependencies: `pillow==12.3.0`, `pypdfium2==5.13.0` | `requirements.txt` |

## Behaviour

- **Upload first, then send.** `POST /api/chat/attachments/` returns the attachment (kind, pages); `POST /api/chat/` references it with `attachment_id`. An upload made before the conversation exists is bound to it by the first message.
- **Image, or PDF + `page`:** answered by the vision model only. A follow-up to the vision model re-sends the most recent image / page only. No asset facts or RAG (8192-token context).
- **PDF without `page`:** parsed once with Docling (through the GPU lock, wait 120 s), cached as `text.md` next to the file. Whole pages are added to the system prompt of DGX / Ollama / Grok for that message and the following ones, within `CHAT_ATTACHMENT_SHARE` (50 %) of the model's budget. The reply's sources start with `{kind: "attachment", pages_read, page_count, truncated}`. Asset facts and RAG share what is left.
- **Limits:** 20 MB, 50 pages, one file per message.
- **Nothing is written to the library, Chroma or the graph.**

## Verification

- `manage.py test` (with the test Neo4j and Chroma): **245 OK** (1 skipped); `pytest ingestion/tests`: 15 passed. New: `gpu/tests/tests_vlm.py` (13), `chat/test_attachments.py` (18).
- Migration `chat.0007` applied on the live DB right after the model change (backup `data/backups/db-before-chat-0007-20261004.sqlite3`; no active jobs; worker restarted).
- **Live, `gpu_smoke --cold`:** PASS parse (20.9 s from idle) and PASS vlm (activate 24.0 s; read "BSK SMOKE 4721" in 1.4 s).
- **Live, API:**
  - image → vision model: 9.0 s; follow-up without attachment: 2.4 s, answered from the same image
  - PDF page 4 → vision model: correct page title
  - same PDF as text → DGX: 47 s including the `vl` → `docling` swap and the parse; source "6 of 6 pages"; a second conversation with the cached GPU state: 29 s
  - image with a text model: 400 `vision_model_required`
- One DGX reply timed out at 180 s during the browser check; the same request passed in 4.6 s right after (the shared DGX stalls at times, see `SERVICES_MAP.md`).
- Test conversations and their files were deleted afterwards.

## Not done / notes

- Attachments are not available in Compare, and regenerate does not re-send an attachment.
- Analytics: chat VLM calls keep the purpose `chat` (model `bsk-qwen3-vl-4b`); `vlm-figure` / `vlm-drawing` are for P9a.2 / P9a.3.
- **Commit:** — (waiting for Angel)
