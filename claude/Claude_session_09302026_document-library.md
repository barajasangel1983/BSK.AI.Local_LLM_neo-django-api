# Claude Session Log — 10/02/2026 — P2 shared document library (backend)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/document-library` (off `main` @ `0f74328`, after P1 PR #9 merged)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/document-library`
- **Plan:** frontend repo `claude/Claude_session_09302026.md` (entry 17)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Document library (upload once, parse once), worker, per-file Generate embeddings with structure / sentence / fixed chunking, legacy import, library-based rebuild; Chroma server mode; worker systemd service | `ingestion/models.py` (+`Document`, `Job`), `ingestion/migrations/0002_document_library.py`, `ingestion/library.py` (new), `ingestion/chunking.py` (new), `ingestion/chunker.py`, `ingestion/vector_store.py`, `ingestion/library_views.py` (new), `ingestion/urls.py`, `ingestion/management/commands/{library_worker,library_import_legacy}.py` (new), `ingestion/management/commands/rebuild_rag_v2.py` (rewritten), `chat/rag_lab_views.py`, `neo_llm_api/settings.py`, `ingestion/tests/tests_library.py` (new), `ingestion/tests/tests_retry_rebuild.py`, `requirements.txt` (+pysbd), `README.md` | _not committed yet_ |

## Details

- **Models:** `Document` (file, `doc_key` = Chroma `asset_id`, sha256, parse status/page count, RAG status/strategy/params/chunk count, graph status placeholder) and `Job` (parse / embed; queued/running/done/failed/cancelled; progress, heartbeat, attempts).
- **`library.py`:** upload with SHA-256 de-duplication and unique doc keys; files + cached Docling output under `LIBRARY_BASE/<id>/`; queue (validate params, reuse an active job, atomic claim, heartbeat, stale re-queue after `LIBRARY_JOB_STALE_SECONDS`, cancel at checkpoints); parse once; embed = chunk → DGX embed in batches with progress → replace the document's chunks in `bsk_rag_v2` (metadata `document_id`, `chunk_strategy`, `chunk_variant`; ingest key includes the variant); remove embeddings; delete everywhere.
- **Chunking (`chunking.py`):** `structure` (existing chunker, now parametrized — defaults unchanged), `sentence` (pysbd; whole sentences up to max tokens, sentence overlap, never across sections or tables/lists; tables/lists whole), `fixed` (token windows); bounds/defaults per strategy served via `/api/rag/config/` → `chunking`. Short tables/lists are kept (the min-size filter only drops paragraph fragments).
- **API:** `/api/documents/` (list/upload), `/api/documents/<id>/` (detail/delete), `/api/documents/<id>/embed/`, `/api/documents/embed/` (bulk), `/api/documents/<id>/embeddings/` (remove from RAG), `/api/jobs/`, `/api/jobs/<id>/`, `/api/jobs/<id>/cancel/`. `DELETE /api/rag/docs/<doc_key>/` now removes a library document from RAG only. Legacy `/api/rag/ingest/` left unchanged (no longer used by the UI).
- **Worker:** `manage.py library_worker [--once]` — one job at a time, outside runserver.
- **Migration:** `library_import_legacy` created library documents for the 3 legacy uploads (chunks stamped with `document_id`), parse jobs cached their text.
- **Rebuild:** `rebuild_rag_v2` now re-embeds library documents with their stored strategy/params.

## Verification

- `GRAPH_TEST_URI=… manage.py test`: 117 passing (28 ingestion incl. 20 new library tests); `pytest ingestion/tests` 15.
- Real Attention PDF: structure 98 chunks (avg 130 tokens, paragraph-level), sentence 53 (≤300), fixed 35 (≤512); sentence 128/0 → 96, 128/2 → 113; pysbd handles "e.g.", "Fig. 3", "No. 5", "3.5 bar".
- Live: DB + Chroma backed up to the session scratchpad; legacy import → 3 documents; worker parsed them in ~1 min (15 / 6 / 116 pages); API re-embed of BSKLAB with sentence (8 → 26 chunks, retrieval OK), then back to structure (8). Corpus unchanged.
- Worker started in the background for UI testing (not yet a managed service).

## Notes

- Correction: sentence chunking produces *fewer* chunks than structure on this corpus (structure is paragraph-level), not 3–4× more as estimated in the plan.
- Bugs caught by tests: invalid params accepted when an embed job was already active (validation now first); tiny tables/lists dropped by the min-size filter.
- The structure-aware chunker still drops paragraphs/lists under 20 tokens (existing behavior, unchanged).

## Follow-up in the same branch: Chroma server mode + worker service

- **Problem (found by the user in the retrieval debugger):** with the worker/import writing to Chroma from other processes, the API's embedded `PersistentClient` kept a stale cache — right ids/vector scores, empty documents/metadata, rerank ≈ 0. Embedded Chroma isn't process-safe.
- **Fix:** Chroma runs as a server — `deploy/chroma/docker-compose.yml` → `bsklab-chroma` (`chromadb/chroma:1.5.5`, `127.0.0.1:8100`, existing data folder mounted, runs as uid/gid 1001, `restart: unless-stopped`, healthcheck; test profile `bsklab-chroma-test` on 8102). All access goes through `ingestion/chroma_client.py` (`HttpClient` when `CHROMA_HOST` is set, embedded only for tests): `VectorStore`, `chat/retrieval.py`, RAG Lab views, chat health/restart, `/api/rag/health/`, `rebuild_rag_v2`, historian summarize command. Legacy historian retrieval moved into `chat/legacy_retrieval.py` (no more GraphRAG `rag.retrieval` import); legacy `RAG_AUTO_INGEST` (GraphRAG embedded writer) is skipped in server mode. `neo_llm_api/test_runner.py` forces embedded mode during tests.
- **Validation before cutover:** server started on a *copy* of the data → both collections, cosine, v2 + legacy queries correct, file ownership preserved.
- **Cutover (no overlap of embedded and server access):** backups (DB + Chroma) → worker stopped → `.env` `CHROMA_HOST/PORT/UID/GID` → runserver reloaded (server mode) → verified no process held the folder → container started (healthy). Verified: `/api/rag/health/` (`http://127.0.0.1:8100`, 601 chunks), rag-chroma health/restart (601 document chunks, 963 historian summaries), decoder/encoder queries → Attention §3/§3.1 (rerank 0.9998).
- **Worker service:** `deploy/systemd/neo-library-worker.service` installed as systemd user service (enabled, lingering on, `PYTHONUNBUFFERED=1`, `Restart=on-failure`, SIGTERM + 120 s stop timeout; journal logs).
- **Live regression:** worker re-embedded BSKLAB as sentence (26 chunks) and back to structure (8); the API returned fresh chunks immediately with no server reload.
- **Tests:** 122 passing (5 new in `tests_chroma_client.py`, incl. a multi-process regression test: a subprocess writes through the Chroma server, the test process reads it immediately). `pysbd` added to `requirements.txt`.
- **Handoff:** frontend `claude/SERVICES_MAP.md` maps every service, port, container, data path and caveat.
