# Claude Session Log — 10/03/2026 — P7 Asset-scoped chat (backend)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/asset-chat` (off `main` @ `aef926d`)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/asset-chat` (frontend log entry 27)
- **Decisions (user):** asset facts are independent of the RAG toggle; documents: the asset's first, then the library; include the latest historian values vs limits, timestamped.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | `Conversation.asset_id` (migration `chat/0006_conversation_asset`, applied; DB backup `data/backups/db.sqlite3.pre-asset-chat.bak`; worker restarted right after) | `chat/models.py`, `chat/migrations/0006_conversation_asset.py`, `chat/serializers.py` | — |
| 2 | Asset fact sheet + historian snapshot + prompt + citations | `chat/asset_context.py` | — |
| 3 | Chat view: validate / store / keep the scope; facts independent of RAG; asset documents first, then library; graph + historian + document citations (`kind`) | `chat/views.py`, `chat/retrieval.py` (`doc_keys` filter) | — |
| 4 | Settings: `ASSET_CONTEXT_SHARE` 0.3, `ASSET_RAG_SHARE` 0.25, `DGX_CHAT_TIMEOUT` 180 s (was a hard-coded 60 s) | `neo_llm_api/settings.py` | — |
| 5 | Neo4j driver: unknown-relationship-type notifications disabled (log noise on every asset query) | `context_graph/driver.py` | — |
| 6 | Tests + README | `chat/test_asset_chat.py`, `context_graph/tests/tests_health.py`, `README.md` | — |

## Design notes

- **Fact sheet** (EXTR01: 72 facts, ~9.5k chars in full): location, component tree (nested), signals (unit, normal range, last RUNNING value and in/above/below), alarms (severity, components, procedures), procedures (type, status, summary, applies to, addresses), connections, linked documents. Over budget → pinned header lines + the lines that share the most words with the question, in original order (DGX: 7.2k chars; Ollama: 1.8k).
- **Historian:** latest row for `extruder_id = <asset key>`; if not RUNNING, the state, since when, and the last RUNNING row (limits are P5–P95 while running). Postgres down → answer without values.
- **Prompt:** facts are authoritative, cited [G]; documents [1]…; say when unknown, never invent values, limits or procedures. Out-of-range signals are listed explicitly.
- **Lab layer** never used.

## Verification

- `manage.py test` with the test Neo4j: **165 passing** (2 skipped). New `test_asset_chat.py` (8):
  - fact sheet content
  - historian status and out-of-range
  - budget trimming
  - no historian
  - scope stored, sticky and cleared
  - facts without RAG
  - asset documents first, then library
  - invalid scope (400), graph down (503)
  - graph down mid-conversation
  - integration: EXTR01 sheet from the seed
- **Live** (DGX Qwen):
  - "What should I do on a DIE_PLUG alarm, and which signals should I watch?" → the seeded procedure steps and the Die Head signals with their normal ranges, all cited [G], in 55 s. The first attempt hit the old 60 s timeout, hence `DGX_CHAT_TIMEOUT`.
  - "Is anything out of range right now?" (UI) → STOPPED since 23:06; last RUNNING sample 19:13 all in range; the HI_MOISTURE alarm noted.

## Open points

- For factory-keyword questions the legacy `plc_historian` shift summaries (old `bsk_rag`) are still added as documents; in asset chats they are usually noise and use document budget.
- No documents are linked to EXTR01 yet (no schema-mode extraction approved with the asset scope), so documents come from the whole library.
