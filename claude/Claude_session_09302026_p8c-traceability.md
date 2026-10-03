# Claude Session Log — 10/03/2026 — P8c Traceability

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p8c-traceability` (off `main` @ `8df2957`, after P8b PR #20 merged)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/p8c-traceability` (frontend log entry 34)

## Before the code

- Schema v2 (`context_graph/schemas/industrial_v2.yaml`) saved through the schema API as version 2 and activated. It validated with no conflicts.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Provenance service: origin labels, evidence items with file names, short labels and approved triples; node sources (curated + lab), edge sources, lab-triple sources; fact labels for the chat | `context_graph/provenance.py` (new) | — |
| 2 | API: `GET nodes/<id>/sources/`, `GET edges/sources/?from&type&to` or `?triple_id` (404 unknown, 400 incomplete) | `context_graph/views.py`, `context_graph/urls.py` | — |
| 3 | Asset chat: each cited fact has a node (or connection) behind it, and the graph citation carries `fact_sources`. Component connections now return their `source` / `evidence_ids`. | `chat/asset_context.py`, `context_graph/services.py` | — |
| 4 | Tests, README | `context_graph/tests/tests_provenance.py` (new), `README.md` | — |

## Verification

- `manage.py test` with the test Neo4j: **213 passing** (2 skipped). New tests (5):
  - origin labels
  - a seeded node and edge
  - an extracted fact traced to its page (node, new node, edge; 404 / 400)
  - an import row and a lab relationship / lab node
  - chat facts carrying their source (extracted evidence wins over the seed label; every fact labelled)
- Re-run of `chat` + `context_graph` after adding the connection labels: 158 OK.
- Live (runserver reloaded):
  - Die Head → `Seed (extr01-v1)`, no records
  - seeded edge → seed origin
  - lab triple → "Attention is all you need.pdf p.1"
  - EXTR01 chat context: 73 of 73 facts labelled

## Notes

- No migration and no worker restart needed (read-only endpoints).
