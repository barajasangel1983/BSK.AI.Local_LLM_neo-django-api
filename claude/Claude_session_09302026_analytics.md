# Claude Session Log — 10/03/2026 — Usage Analytics with real measurements (backend)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/analytics` (off `main` @ `f70b4f0`)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/analytics` (frontend log entry 28)
- **Decision (user):** no cost tracking; prompt and completion tokens matter most.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | New `usage` app: `ModelCall` (migration `usage/0001_initial`, applied; DB backup `data/backups/db.sqlite3.pre-analytics.bak`; worker restarted right after) | `usage/*`, `neo_llm_api/settings.py` | — |
| 2 | Recorder (`usage.recorder.track` / `scope`): tokens from OpenAI-style `usage` or Ollama `*_eval_count`, estimated (cl100k) when absent; status ok / error / timeout; never breaks the call | `usage/recorder.py` | — |
| 3 | All 8 call sites wrapped | `chat/views.py` (Grok, DGX, Ollama), `chat/titles.py`, `chat/retrieval.py` (rerank), `ingestion/embedder.py`, `ingestion/docling_client.py`, `context_graph/extraction.py` | — |
| 4 | Chat view: `purpose` (chat / compare / regenerate) and the conversation applied to all calls of a request | `chat/views.py` | — |
| 5 | `GET /api/usage/summary/?days=&purpose=` | `usage/summary.py`, `chat/views.py` | — |
| 6 | Tests, README | `usage/tests/tests_usage.py`, `README.md` | — |

## Verification

- `manage.py test`: all passing. `usage` adds 8 tests:
  - token formats and estimates
  - ok / error / timeout re-raised
  - a broken recorder doesn't break the call
  - p95
  - chat / compare / Ollama recording with the conversation
  - failed chat recorded as timeout
  - title calls
  - summary maths and filters
- **Live:**
  - an asset chat recorded embed (13 prompt tokens), rerank (1,483), chat (DGX) and title calls with provider token counts
  - an Ollama compare call recorded 43 prompt / 1,609 completion tokens (its reasoning counts as completion)

## Notes

- No history before deployment; the page states "Measured since …".
- Rows are kept indefinitely (small); a prune command can come later if needed.
