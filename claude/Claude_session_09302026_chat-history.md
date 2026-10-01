# Claude Session Log — 09/30/2026 — chat history + logging

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `fix/chat-history` (off `main` @ `38f3c1c`)
- **Cross-ref:** frontend session log `BSK.AI_neo-llm-hub/claude/Claude_session_09302026.md` (change 2 / problem 3)
- **Why a separate file:** `claude/Claude_session_09302026.md` exists only on branch `chore/gitignore-data-gitkeep` (another session); a same-named file here would conflict on merge.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Send conversation history to the LLM + structured chat logging | `chat/views.py`, `chat/tests.py`, `neo_llm_api/settings.py`, `.gitignore` | `ac5db0a` — PR [#1](https://github.com/barajasangel1983/BSK.AI.Local_LLM_neo-django-api/pull/1) |

## Details

- **Problem:** every model call sent only system prompt + current message; prior turns were stored but never sent.
- **Fix (`chat/views.py`):**
  - `build_chat_messages()` (pure): system + recent history + current user. History is fetched *before* saving the new user message, oldest dropped first, whole messages only, never starts with an assistant message, unanswered user messages (failed calls) skipped.
  - One **total** budget per model covers system prompt (incl. RAG context) + history + current message, so larger `bsk_rag_v2` chunks shrink history instead of overflowing the model.
  - `format_rag_footer()` / `strip_rag_footer()` keep the "Sources (RAG)" footer in one place; footer is stripped from history.
  - `call_dgx_gpt_oss_20b` / `call_grok_chat` / `call_ollama_qwen3_8b` take a `messages` list; default system prompts in `DEFAULT_SYSTEM_PROMPTS`.
  - Model backend failures return **502 JSON** `{"error": ...}` instead of an unhandled 500 HTML traceback.
  - `print()` calls in this file replaced by the `chat` logger.
- **Settings (env-overridable):** `CHAT_HISTORY_MAX_MESSAGES=20`, `CHAT_CONTEXT_MAX_CHARS=24000`, `CHAT_CONTEXT_MAX_CHARS_OLLAMA=6000` (Ollama's ~4k-token default context silently truncates the prompt start). `LOGGING`: `chat` logger → console + rotating `logs/chat.log` (5 MB × 5). `logs/` git-ignored.
- **Log line per model call:** `chat ok|failed conversation=<id> model=<id> use_rag=<bool> history_sent=<sent>/<available> rag_context_chars=<n> rag_sources=[...] latency_ms=<n>` (failures include traceback).

## Verification

- `python manage.py test chat`: 11 passing (builder ordering/limits/budget/orphans/footer; footer round-trip; `chat_view` second turn sends first exchange; backend failure → 502 and orphan skipped next turn). View tests use `assertLogs`, so they don't write to `logs/chat.log`.
- Live 2-turn test ("my favorite color is teal" → "what is my favorite color?"): `ollama-qwen3-8b` → "teal", `dgx-qwen38-27b-fp8` → "Teal"; log shows `history_sent=2/2`. Test conversations deleted.

## Notes

- **Incident:** a `git stash` / `stash pop` in this live checkout (to compare ingestion tests against `main`) made `runserver` reload the old code and not reload back. The first live attempt hit old code (DGX 60 s read timeout → 500; Ollama answered without history) and left 2 stray test conversations (`876ee8ca…`, `d07292c6…`), since deleted. Fixed by touching `chat/views.py`. Lesson: don't stash in the live checkout.
- **Pre-existing, unrelated:** `ingestion/tests` has 7 failures (`DoclingClient.submit_file` / `poll_status` don't exist; 2 chunker assertions) and needs `DJANGO_SETTINGS_MODULE=neo_llm_api.settings` to collect.
- **For the `bsk_rag_v2` switch:** v2 chunks are 600–1000 tokens (max 1500) vs ~120 today → RAG context needs its own cap (top_k / char budget), and `c.source` must map to v2 metadata.
