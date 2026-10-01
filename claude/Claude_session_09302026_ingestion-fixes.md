# Claude Session Log — 09/30/2026 — ingestion fixes (PR D)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `fix/ingestion-pages-cosine` (off `main` @ `d0c6472`)
- **Plan:** frontend repo `claude/Claude_session_09302026.md` (entry 8)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Real page numbers, content types, Docling options, cosine collections, retry failed/stale jobs, rebuild command, fixed ingestion tests | `ingestion/chunker.py`, `ingestion/docling_client.py`, `ingestion/vector_store.py`, `ingestion/views.py`, `ingestion/management/commands/rebuild_rag_v2.py` (new), `neo_llm_api/settings.py`, `ingestion/tests/test_chunker.py`, `ingestion/tests/test_docling_client.py`, `ingestion/tests/tests_retry_rebuild.py` (new), `conftest.py` (new) | _not committed yet_ |

## Details

- **Pages:** Docling is asked for `md_page_break_placeholder = PAGE_BREAK` (`<!-- page-break -->`); the chunker tags each markdown line with its page, paragraphs carry `(page_start, page_end)`, chunks take the range of their paragraph (overlap prefix ignored; sentence-split pieces inherit the paragraph range). Formats without pages stay on page 1. Replaces the hardcoded `page_start=1, page_end=1`.
- **Content types:** `table` (majority of lines are markdown table rows), `list` (bullets / numbered), else `paragraph` — previously almost every chunk was labeled "heading".
- **Docling options:** `image_export_mode: placeholder` (no more base64 figures over the wire; regex stripping kept as a safety net), `to_formats: ["md"]`.
- **Config version:** `INGESTION_CONFIG_VERSION` → `v2-2026-10-01`.
- **Cosine:** `VectorStore` creates collections with `configuration={"hnsw": {"space": "cosine"}}` (Chroma 1.x ignored the old `metadata={"hnsw:space": ...}`).
- **Retry:** `POST /api/rag/ingest/` re-runs an existing job (same row, unique per sha/revision/config) if it **failed** or is **in flight for more than `INGESTION_STALE_MINUTES` (30)** — e.g. its daemon thread was killed by a `runserver` reload. `created_at` resets to the retry start. Done jobs are still returned as-is (200).
- **`manage.py rebuild_rag_v2 [--dry-run] [--no-backup]`:** for each asset whose latest job is done and whose source file is in `INGESTION_RAW_BASE`: backs up the Chroma dir, drops/recreates `bsk_rag_v2` (cosine), replaces the asset's jobs with a new job at the current config and runs the pipeline synchronously. Other collections (legacy `bsk_rag`) untouched; missing files / non-done assets skipped and reported.
- **Tests:** the 7 pre-existing `ingestion/tests` failures fixed — `test_docling_client.py` rewritten for the sync `convert_file` API (payload incl. options, error status, HTTP error, connection error), `test_chunker.py` rewritten for markdown input (pages across breaks, paragraph spanning a break, no-break docs, content types, token limits, image stripping); root `conftest.py` sets `DJANGO_SETTINGS_MODULE` so `pytest ingestion/tests` runs without env vars. New Django tests in `ingestion/tests/tests_retry_rebuild.py` (named so pytest doesn't collect them): retry failed / stale / not-stale / done, cosine creation (real Chroma in a temp dir), rebuild dry-run and full rebuild (legacy collection untouched, backup made, missing file skipped).

## Verification

- `python manage.py test ingestion chat`: 46 passing. `pytest ingestion/tests`: 15 passing (was 7 failed / 5 passed).
- Real Docling (no writes): Attention paper → 14 page breaks, pages 1–15, 86 paragraph / 6 list / 6 table chunks, 0 base64 images; BSKLAB → pages 3, 4, 6.
- **Live rebuild** (73 s): backup `…/data/chroma_index.bak-20261001-075159`; `bsk_rag_v2` recreated with `space=cosine`; ATTENTION-IS-ALL-YOU-NEED 98, BSKLAB-COPILOTV0-01-PDF 8, KIRKWOOD1998-1 446 chunks; 0 failures.
- Live API: `/rag/config/` → cosine, 552 chunks, `v2-2026-10-01`; chunks show real pages; `/rag/query/` ranking and vector scores identical to before (confirms the earlier L2 conversion and that ranking was unaffected); chat citations e.g. "§ 3.5 Positional Encoding, p.6".

## Notes

- Retry of a failed job was verified by tests only (no live failure induced).
- The Chroma backup dir can be deleted once the rebuild is confirmed.
- Old chat messages keep the citations stored at the time (page 1).
