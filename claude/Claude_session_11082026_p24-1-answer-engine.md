# Claude Session Log — 11/08/2026 — P24-1: one answer engine, streamed

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p24-1-answer-engine` (off `main` @ `0981b15`)
- **Plan:** frontend repo `claude/P24_studio_edge_alignment_plan.md`, "Revision 2" (Angel: "edge needs to answer exactly
  the same as studio"; "Stream from the start ... modify the context studio to do the same"). Paired with frontend
  `feat/p24-1-stream-chat`.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | Every text backend streams (DGX vLLM SSE with usage, Ollama NDJSON, Grok SSE); one ModelCall per reply; a server that ignores `stream` is read as one JSON object | `chat/views.py` (`_stream_call`, `_backend_stream`, `stream_reply_backend`) |
| 2 | `generate_reply_backend` = the streamed reply joined (one code path) | `chat/views.py` |
| 3 | `_chat` split: `_start` (checks, conversation, user message), `answer_context` (the engine's context: attachments as text, historian window, planner, facts, safety facet, excerpts, prompt; `ContextError`), `_reply_pieces`, `_save_reply` | `chat/views.py` |
| 4 | `POST /api/chat/stream/`: server-sent events `model`, `token`, `sources`, `done {conversation}`, `error {kind, message, status}` (EDGE's event names) | `chat/views.py`, `chat/urls.py` |
| 5 | Tests: streamed backends, events, same reply streamed and not, errors; the shared test reply also streams | `chat/test_stream.py`, `chat/tests.py` |

## Verification

- Suite 812 OK (47 skipped). Live stream from the worktree: 8B first token 10.4 s, 13 s total; DGX 10.1 s, 59 s.
- chat_eval on the 8B (safety S1, S3, S5): 3/3.
- Not yet in the engine: the Context Service's own answer path (`context_service/service.py`, model calls) and EDGE's
  features (question agent, nothing found, needs_choice, picture, editable prompt): P24-2.
