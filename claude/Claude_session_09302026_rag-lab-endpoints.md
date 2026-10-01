# Claude Session Log — 09/30/2026 — RAG Lab endpoints (Phase 5, PR B)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/rag-lab-endpoints` (off `main` @ `ec3bc85`, after PR #2 merge)
- **Plan:** frontend repo `claude/Claude_session_09302026.md` (entry 3)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | v2 document list, chunk browser, delete by asset, read-only config | `chat/rag_lab_views.py` (new), `chat/urls.py`, `chat/views.py`, `neo_llm_api/settings.py`, `chat/test_rag_lab.py` (new) | _not committed yet_ |

## Details

- **`GET /api/rag/docs/`** — now lists `bsk_rag_v2` documents grouped by `asset_id` (`name`, `document_revision`, `config_version`, `chunk_count`, `token_count` via tiktoken `cl100k_base`), joined with the latest `IngestionJob` (`status`, `job_id`, `ingested_at`, `error`), plus jobs that have no chunks yet (queued / in progress / failed). `size` from `data/incoming/<file>`. Aliases `chunks` / `tokens` kept for the current RAG Lab UI.
- **`DELETE /api/rag/docs/<asset_id>/`** — deletes the asset's chunks from `bsk_rag_v2` **and** its `IngestionJob` rows (otherwise ingest idempotency would return the old job and re-uploading wouldn't re-ingest). 404 if nothing matched. Returns `{asset_id, chunks_deleted, jobs_deleted}`.
- **`GET /api/rag/chunks/?asset_id=&offset=&limit=`** — chunks in reading order (index parsed from the chunk id), `limit` 1..100 (default 20); each chunk has `index, text, section_path, content_type, page_start/end, token_count, source, asset_id`; plus `total`.
- **`GET /api/rag/config/`** — chunker (`target_tokens` 600, `max_tokens` 800, `overlap_sentences` 1, tokenizer, `config_version`, read from `ingestion.chunker` constants), embedding model + dimensions (from a stored vector), reranker model, collection name/count/distance space, retrieval settings incl. `query_default_top_k` (5, new `RAG_QUERY_DEFAULT_TOP_K`) and `query_max_top_k` (10) for the Top-K slider marker/max.
- **Removed** the old file-based `rag_docs` / `rag_delete_doc` (listed `data/raw` files and deleted from `bsk_rag`). Legacy `/api/rag/upload/` kept (brief).
- `/api/rag/query/` default `top_k` now uses `RAG_QUERY_DEFAULT_TOP_K`.

## Verification

- `python manage.py test chat`: 39 passing (10 new: docs grouping/job join/job-only and missing collection; delete scoped to asset + 404; chunks ordering/paging/clamp/validation; config values + missing collection; chunk id parsing).
- Live: docs → `ATTENTION-PAPER-005`, 98 chunks, 12,758 tokens, `done`, 2.2 MB; chunks → reading order with token counts; config → real values (l2, 2048 dims); delete unknown asset → 404. No live data deleted.

## Notes

- **Temporary UI gap until PR C:** the current RAG Lab calls `DELETE /rag/docs/<filename>/`; it now expects an `asset_id`, so delete from the old UI returns 404 (the UI swallows it). The document list keeps working via the `chunks`/`tokens`/`size` aliases.
- The plain-text RAG sources footer is still appended to replies; remove it together with PR C (when the UI renders `Message.sources`).
