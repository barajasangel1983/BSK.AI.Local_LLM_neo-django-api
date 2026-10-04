# Claude Session Log — 10/04/2026 — P9a.3 Drawings to candidate triples

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p9a3-drawing-triples` (off `main` @ `2e5f91c`)
- **Plan:** frontend repo `claude/P9_vlm_integration_plan.md`, step 3; paired with frontend `feat/p9b3-drawing-triples`
- **Why:** facts that only a drawing shows (what drives, feeds or connects to what) should reach GraphLab's review like facts from the text.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | Drawing request: prompts for schema and free-form mode (relationships only, compact JSON, at most 25, the active schema in the prompt), one retry on invalid JSON, vision evidence | `context_graph/drawings.py` |
| 2 | Generate triples takes `include_drawings`; described drawings / schematics / diagrams are read after the text windows and staged through the same code path | `context_graph/extraction.py`, `context_graph/extraction_views.py` |
| 3 | Provenance label of vision evidence: "file.pdf p.17 (figure)" | `context_graph/provenance.py` |
| 4 | Figure summary counts `drawings` (described figures of kind drawing / schematic / diagram) | `ingestion/figures.py` |
| 5 | `DRAWING_MAX_TOKENS` (1000) | `neo_llm_api/settings.py` |

## Behaviour

- **Opt-in:** `include_drawings` is off by default. Only figures that Describe figures marked as drawing, schematic or diagram are sent; photos, charts, tables and undescribed figures are not.
- **Same pipeline as text:** identity resolution (P8b), schema checks, de-duplication and review. A fact found in the text and in a drawing is one triple with both places as evidence.
- **Evidence:** `source_kind: vision`, page, figure number (`figure_id`), box (`region`), model, prompt "Drawing v1", and the figure's description as the excerpt. Figure evidence uses `chunk_index` 100000 + figure number.
- **Failures:** invalid JSON → one retry asking for at most 12 triples, then the figure is counted as a failed call. BSK unreachable → the text triples are staged and the run notes "drawings were not read".
- No migration.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **269 OK** (1 skipped; 5 new in `context_graph/tests/tests_drawings.py`).
- **Live, request only:** figure 1 of "Attention is all you need.pdf": free-form 25 valid triples in 10.0 s; schema mode 0 triples in 1.9 s (correct: nothing industrial in it).
- **Live, whole path on a test image** (run in-process, because the worker was busy with Angel's extraction and must not be restarted): parse 28 s, Describe figures 28 s (kind diagram), Generate triples free-form with drawings 117 s → 37 triples, 4 of them from the drawing with vision evidence (e.g. PLC — CONNECTED_TO → OPC UA). Reviewed in the browser; test document deleted.

## Notes

- The library worker was restarted on this code after Angel's extraction finished (no active jobs), so "Include drawings" works from the UI.
- Not done: starting drawing triples for a single figure from the Figures view (the plan listed it as an option).
- **Commit:** — (waiting for Angel)
