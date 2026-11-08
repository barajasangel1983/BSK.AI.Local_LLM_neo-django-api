# Claude Session Log — 11/08/2026 — P22g: Safety facts in the asset chat

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p22g-chat-safety` (off `main` @ `412fb5d`)
- **Plan:** frontend repo `claude/P22g_chat_facets_plan.md` (D1–D4 as recommended, Angel "go")
- **Why:** the 635 approved Safety facts (P27d) never reached the chat: `get_asset_context` read no Hazard,
  SafetyMeasure or SafetyFunction.

## Changes

| # | Change | Files |
|---|--------|-------|
| G1 | `get_asset_context`: `hazards` (with measures, owners), `safety_functions`, each part procedure's `requires`; counts | `context_graph/services.py` |
| G2 | Fact lines: "Safety notices of parts" (hazard (signal word) on parts: consequence; measures), "Safety functions of parts" ("(safety function)" in the line so the words match), procedure `; requires: …`; each part line carries its node's `facets` | `chat/asset_context.py` |
| G3 | `Plan.facet` from the question (Safety word list); a Safety question with no other kind gets `PLANNER_SAFETY_FACTS_SHARE` (0.4). Ranking: a fact of the facet comes right after the facts in focus — only the focused parts' when a part is in focus, else those sharing a question word — DANGER > WARNING > CAUTION on ties. In a question of another facet the safety lines rank last and don't count in the word weights, and a procedure's `requires` is shown only in safety questions, so those questions keep the same facts as with the switch off | `chat/planner.py`, `chat/asset_context.py`, `chat/views.py` |
| G4 | Answer rule in the asset prompt (measures first, with the signal word, no invented PPE or steps) | `chat/asset_context.py` |
| G5 | `CHAT_SAFETY_FACTS` (default off; off = today's fact sheet and prompt), `PLANNER_SAFETY_FACTS_SHARE` | `neo_llm_api/settings.py` |
| G6 | Eval set `extr01_safety.json` (10: 8 from approved facts, 2 abstain) | `chat/evals/extr01_safety.json` |
| G7 | Tests (6) | `chat/test_safety_facts.py` |

## Notes

- The Context Service (MCP) calls `asset_context.build` without a facet: with the switch on its packets also carry
  the safety lines, ranked by words only.
- "warning" is not a Safety word: alarm questions ("warning A581") stay alarm questions.

## Verification

- Backend suite: 798 tests OK (47 skipped); chat 169 OK after the ranking fixes.
- Evals (`~/backups/p22g-validation/`), switch off (round 1) → on (round 2, after the fixes):
  - Safety set: DGX 6/10 → 9/10 (S9 is a right "no safety notices" the scorer misses: 10/10); 8B 6/10 → 10/10.
  - Hard set (30): DGX 21 → 22 ✓; 8B 20 → 19 ✓ (D4: no more than 1 worse).
- Round 1 with the first ranking had the 8B hard set at 17: the safety lines changed which facts other questions kept
  (42 over 30). Fixed as in G3; checked on the fact sheets without model calls.
