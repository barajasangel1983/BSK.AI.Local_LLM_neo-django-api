# Claude Session Log — 10/04/2026 — P10a Context Service and Context Packet v1

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p10a-context-service` (off `main` @ `9395809`)
- **Plan:** frontend repo `claude/P10_context_service_mcp_plan.md` (approved 2026-10-04)
- **Why:** one governed way to get the context for a question, for the Studio, REST clients and the MCP server that BSKLAB EDGE and other agents will use.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | New app `context_service`: `list_assets`, `resolve`, `entity_context`, `entity_sources`, `operational_state`, `search_documents`, `assemble` (the packet), `ask` | `context_service/service.py` |
| 2 | REST under `/api/context/`: `assets/`, `resolve/`, `entities/<id>/`, `entities/<id>/sources/`, `state/<asset>/`, `search/`, `assemble/`, `ask/` | `context_service/views.py`, `context_service/urls.py`, `neo_llm_api/urls.py`, `neo_llm_api/settings.py` |
| 3 | Fact-sheet builder: optional `focus_ids` (facts about the entities a question names are kept first), structured `items` and `values`, and `entities(asset_id)` | `chat/asset_context.py` |

## Behaviour

- **Resolution:** an explicit `asset_id` wins; otherwise the only asset named in the question (name or key, e.g. "EXTR01"). Components, signals, alarms and procedures named in the question become the **focus**; the longest name wins ("die head temperature" over "die head"). Several assets, or one phrase matching several entities → `ambiguous` with candidates, never a guess.
- **Context Packet v1:** `packet_version`, `query`, `resolved_scope`, `graph_facts` (id, text, section, entity id, focus, source), `relationships`, `document_evidence` (document, page, kind `text` / `figure`, score, scope `asset` / `library`), `operational_context` (state, values vs normal range, reserved `recent_events` / `trends`), `provenance`, `limits`, `warnings`, `timings_ms`, and `context_text` (the packet as prompt text, for small clients).
- **Budget:** `budget_chars` (2,000–60,000; default the chat budget). Facts get 60 % when documents are included; focus facts are kept first; document excerpts fill what is left, whole chunks only (the first one may be cut).
- **Any machine:** no asset in scope → a documents-only packet with the warning `no asset in scope: documents only`.
- **Curated facts only:** the lab layer and pending triples are never included.
- **Sources down:** `graph unavailable`, `documents unavailable`, `reranker unavailable…` as warnings; the rest of the packet is returned.
- **`ask`:** answers from the packet with a Studio model (default DGX); returns the answer, the model and the packet. Usage is recorded with purpose `context-ask`.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **288 OK** (1 skipped; 16 new in `context_service/tests/tests_service.py`). The asset-chat tests pass unchanged.
- **Live (EXTR01):** resolve finds the asset, the Die pressure signal and the DIE_PLUG alarm; the full packet has 74 facts with their origin and the historian state in about 1.5 s; with a 3,000-character budget it keeps 10 facts, the focus first.
- **Live (not modeled):** "How do I start up the BX80?" → documents-only packet with two BX80 Manual excerpts in 0.65 s.
- **Live (`ask`):** "Is anything out of range on EXTR01 right now?" answered by the DGX from the packet, citing [G].

## Not done / notes

- **The Studio chat still builds its own prompt.** It uses the same fact-sheet builder as the service, so the facts agree; moving the chat onto the packet is left as a follow-up to keep this step free of chat regressions.
- Focus matching is by name, alias, code and tag inside the asset. It does not yet use the fuzzy "possible match" of P8b for typos.
- `relationships` carries the asset's connections as text; typed relationships of a focus entity come with the Graph viewer tools (P10c).
- No migration.
- **Commit:** — (waiting for Angel)
