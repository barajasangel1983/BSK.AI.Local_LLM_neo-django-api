# Claude Session Log — 10/04/2026 — P9a.2 Describe figures, images in the library

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p9a2-figures` (off `main` @ `f3143f2`)
- **Plan:** frontend repo `claude/P9_vlm_integration_plan.md`, step 2; paired with frontend `feat/p9b2-figures`
- **Why:** make the pictures, drawings and diagrams of library documents usable: described once by the VLM on BSK, searchable through RAG, visible in RAG Lab. Angel's decision: images go into the existing shared library (no separate VLM tab).

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | Library parses ask Docling for Markdown + JSON (`with_layout`); the figure list (page, box, caption) is saved as `figures.json`; the JSON itself is not kept | `ingestion/docling_client.py`, `ingestion/library.py` |
| 2 | Figures module: figure list from Docling's JSON, skip rules, crop, VLM prompt and reply parsing, descriptions into the Markdown | `ingestion/figures.py` |
| 3 | Crop per contract v1.2: PDF points, BOTTOMLEFT / TOPLEFT, 5 % padding clamped to the page, rendered up to 4x (288 DPI), only downscaled, JPEG ≤ 1280 px | `gpu/images.py` |
| 4 | Job kind `figures` ("Describe figures"); `Document.figures_status / figures_error / figures_updated_at`; `Job.run_after` for retries | `ingestion/models.py`, `ingestion/migrations/0004_figures.py` |
| 5 | Worker: one GPU lock per figure; results saved after each figure; parse jobs are claimed before other jobs; BSK unreachable → retry after 5 and 15 min, then "pending" | `ingestion/library.py` |
| 6 | API: `GET /documents/<id>/figures/`, `GET …/figures/<n>/image/`, `POST …/figures/describe/ {figures?, force?}`; `figures` block in the document JSON | `ingestion/library_views.py`, `ingestion/urls.py` |
| 7 | Chunking: a figure description is its own chunk with `content_type: "figure"` | `ingestion/chunker.py`, `ingestion/chunking.py` |
| 8 | Chat citations carry `content_type` | `chat/views.py` |
| 9 | Settings `FIGURE_*` | `neo_llm_api/settings.py` |

## Behaviour

- **Opt-in:** nothing is described until Describe figures is asked for. A figure already described is not sent again unless asked (`force`, or by number).
- **Skipped without a VLM call:** box under 2 % or over 95 % of the page; a side under 80 px after rendering; no position. **Skipped after the VLM call:** kind `decorative` (logo, icon, background). A figure asked for by number is described anyway.
- **Images in the library:** PNG / JPEG / WebP / GIF / BMP. One figure: the whole picture. Docling still parses the image, so text in it is the document's Markdown.
- **Into the text:** when embeddings or triples are generated, the n-th `<!-- image -->` placeholder becomes `[Figure p.N (kind): description]` (Docling writes one placeholder per picture, in the order of its picture list; verified on three files). The cached Markdown is not changed. For an image document the line is put at the top.
- **Old documents** (parsed before this change) have no figure list: Describe figures parses them again first.
- **BSK off:** the job is re-queued after 5 and 15 minutes, then ends with the document at "pending". Parse and embeddings are never failed by it.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **264 OK** (1 skipped; 18 new in `ingestion/tests/tests_figures.py`); `pytest ingestion/tests`: 15 passed.
- Migration `ingestion.0004` applied on the live DB right after the model change (backup `data/backups/db-before-ingestion-figures-20261004.sqlite3`; no active jobs; worker restarted).
- **Live:**
  - "Attention is all you need.pdf": parsed again, 6 figures found, 5 described as diagrams in about 50 s, 1 skipped as too small. The crops match the page.
  - A test image uploaded to the library: parsed, 1 figure described, embedded (2 chunks). A chat question about the diagram was answered from it, with the source marked `content_type: figure`. Test document and conversation deleted afterwards.

## Notes

- The 2 % rule skipped one real diagram in the Attention paper ("Scaled Dot-Product Attention", 1.7 % of the page). It can be described with "Describe anyway"; the threshold is `FIGURE_MIN_AREA`.
- The plan's "tables as Markdown" in the description was dropped: descriptions are kept to one paragraph so they stay one chunk, and Docling already extracts tables.
- "Attention is all you need.pdf" in the live library now has 5 figure descriptions; its embeddings were not regenerated.
- **Commit:** — (waiting for Angel)
