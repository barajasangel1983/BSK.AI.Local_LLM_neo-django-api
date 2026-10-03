# Claude Session Log — 10/03/2026 — P8b Identity resolution, schema vocabulary

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p8b-identity` (off `main` @ `36da700`, after P8a PR #19 merged)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/p8b-identity` (frontend log entry 33)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Resolver: id → asset + tag → tag + type → name / alias → possible (fuzzy, suggestions only) → new. Tag extraction and normalization; numbers never merge. | `context_graph/identity.py` (new; replaces the old `EntityIndex` in `extraction.py`) | — |
| 2 | Triple ends record `match` + `candidates` (migration `0006_identity_matches`, applied live with backup; worker restarted). Approve is blocked on an unresolved possible match (checked before any graph connection). Edit accepts a picked id or `*_match: "new"`. | `context_graph/models.py`, `context_graph/triples.py` | — |
| 3 | Extraction uses the resolver; synonym predicates are mapped to schema relationships (allowed pairs only); subtypes are kept; prompts list subtypes; default prompt asks to keep engineering tags | `context_graph/extraction.py` | — |
| 4 | Imports: the key column is an explicit tag; names may be possible matches (`possible_matches` in the preview); units normalized; subtype spelling from the schema | `context_graph/structured.py`, `context_graph/units.py`, `context_graph/import_views.py` | — |
| 5 | Schema: optional `subtypes` / relationship `aliases`, validation (duplicate alias), lookups, vocabulary diff (counts as a change) | `context_graph/schema.py`, `context_graph/services.py` | — |
| 6 | Approving no longer adds a node's own tag as an alias ("M101" for …/M101) | `context_graph/triples.py` | — |
| 7 | API: `match` / `candidates` in triples and import previews; vocabulary in `extract/config/` and `schema/` | `context_graph/extraction_views.py` | — |
| 8 | Draft schema v2 (subtypes, aliases, DRIVES / PROTECTS / CONTROLS): validates against the live graph, no conflicts; **not saved** (Angel saves it) | `context_graph/schemas/industrial_v2.yaml` | — |
| 9 | Tests, README | `context_graph/tests/tests_identity.py`, `tests_structured.py`, `README.md` | — |

## Verification

- `manage.py test` with the test Neo4j: **208 passing** (2 skipped). New `tests_identity.py` (12):
  - tags
  - resolver order
  - fuzzy as suggestion only; zones never merge
  - import key as tag
  - schema v2 lookups and validation
  - units
  - staging (synonym → DRIVES, subtypes, matches)
  - approval blocked until resolved; pick / accept as new
  - extract config vocabulary
  - import possible matches
  - **integration: the ChatGPT lab fixture.** VFD101 / VFD-101 / Motor M101 / M101 main motor / OL101 from text and a vision source → one DRIVES edge with both triples and both evidences, aliases kept, no duplicate M101, PROTECTS edge.
- **Live:**
  - re-preview of the staged EXTR01 tag list: same 6 new / 56 existing / 0 possible as before; components now matched by tag (e.g. "Barrel Zone 1" → `barrel-zone-1`)
  - schema v2 validated against the live graph: no conflicts

## Notes

- Built-in prompt presets in the DB keep their stored text. The improved default text (keep tags, optional subtypes) appears as "Restore original" in the prompt editor.
- The old `extraction.EntityIndex` / `resolve_id` are now thin wrappers over `identity`.
