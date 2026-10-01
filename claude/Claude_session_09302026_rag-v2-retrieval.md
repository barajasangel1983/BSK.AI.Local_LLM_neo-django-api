# Claude Session Log — 09/30/2026 — RAG v2 retrieval (Phase 5, PR A)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/rag-v2-retrieval` (off `main` @ `d91be5f`, after PR #1 merge)
- **Plan/brief:** frontend repo `claude/Claude_session_09302026.md` (entry 3) and `claude/3_phase5_ui_task_brief.md`

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Chat + RAG Lab retrieval from `bsk_rag_v2` with DGX reranker, RAG budget, stored citations | `chat/retrieval.py` (new), `chat/views.py`, `chat/models.py`, `chat/serializers.py`, `chat/migrations/0005_message_sources.py`, `neo_llm_api/settings.py`, `chat/test_rag_v2.py` (new) | `f855850` — PR [#2](https://github.com/barajasangel1983/BSK.AI.Local_LLM_neo-django-api/pull/2) |

## Details

- **`chat/retrieval.py`** — `search_v2(query, top_n)`: DGX query embedding (`Embedder.embed_queries`, `query: ` prefix, reused from `ingestion/` without changes) → Chroma `bsk_rag_v2` vector search (`RAG_V2_CANDIDATES=20`) → DGX reranker (`DGX_RERANK_URL`/`DGX_RERANK_MODEL`, `top_n`). Reranker failure/timeout (`RAG_RERANK_TIMEOUT=10`s) → vector order (`reranker="fallback"`). Read-only `get_collection` (never creates/alters the collection). Similarity computed from the collection's actual distance space.
- **Chat (`chat_view`)** — `use_rag` defaults to `True`. Documents from v2 (`RAG_CHAT_TOP_N=5`); factory questions keep plc_historian summaries from legacy `bsk_rag` first (5) + v2 (3). Chunks below `RAG_MIN_RERANK_SCORE=0.1` dropped. `build_rag_context()` adds chunks in rank order within `RAG_CONTEXT_SHARE=0.5` of the model budget (12k chars DGX/Grok, 3k Ollama); top chunk truncated if it alone doesn't fit. Historian or v2 failure doesn't fail the chat.
- **Citations** — new `Message.sources` JSONField (migration `0005`), returned by the message serializer: `{source, asset_id, section_path, page_start, page_end, snippet, score, vector_score, rerank_score}`. Text footer kept (now "doc — section, p.N") until the UI renders `sources` (PR B/C).
- **`/api/rag/query/`** — now v2 + reranker; `top_k` clamped to 1..`RAG_QUERY_MAX_TOP_K=10`; returns `vector_score`, `rerank_score`, `score` (rerank if available, higher = better), `section_path`, `asset_id`, pages, `reranker`, `candidates`, `latency_ms`. Old fields `id/text/source/document_path/score` kept for the current UI. Failure → 502 JSON.
- **Logging** — `rag conversation=… model=… retrieved=… used=… reranker=ok|fallback|error rag_chars=<used>/<budget> latency_ms=…` and `rag_query top_k=… candidates=… returned=… reranker=…`.
- **Settings** — `CHROMA_DIR` (same default as before; `VectorStore` already reads it), `RAG_V2_CANDIDATES`, `RAG_CHAT_TOP_N`, `RAG_QUERY_MAX_TOP_K`, `RAG_CONTEXT_SHARE`, `RAG_RERANK_TIMEOUT`, `RAG_MIN_RERANK_SCORE` (all env-overridable).

## Verification

- `python manage.py test chat`: 29 passing (18 new: search/rerank/fallback/L2 conversion/missing+empty collection; context budget/truncation/labels; chat default RAG + stored citations, Ollama budget, min rerank score, RAG off, retrieval failure, factory/historian path; `/rag/query/` clamp/fields/400/502).
- Live `/rag/query/` "How does multi-head attention work?": reranker ok, 20 candidates, relevant §3.2.x chunks, vector 0.47–0.58 / rerank 0.96–1.0; `top_k=50` → clamped to 10.
- Live chat (RAG default): DGX 5 citations (3.2k/12k chars), Ollama 4 citations (1.9k/3k chars), answers cite the paper; factory question → 5 historian summaries, unrelated v2 chunks filtered out. Test conversations deleted.
- Migration `0005` applied to the live `db.sqlite3` (additive column); backup taken first in the session scratchpad.

## Findings (not fixed — in `ingestion/`, out of scope)

- **`bsk_rag_v2` uses L2 distance, not cosine:** `VectorStore` passes `metadata={"hnsw:space": "cosine"}`, which Chroma 1.5.5 ignores (collection `configuration_json` shows `space: l2`). Ranking is unaffected for unit-length embeddings; retrieval converts L2 → cosine-equivalent similarity. Fixing it means recreating the collection with `configuration={"hnsw": {"space": "cosine"}}` and re-ingesting.
- **`page_start` is 1 for all 98 chunks** — page metadata from the chunker/Docling mapping looks broken; citations show "p.1".
- Raw query vs raw passage embeddings score higher (0.34) than `query:`/`passage:` prefixed (0.22) on a sample — worth confirming the Nemotron embed prompt format; reranking makes this low impact.
- Ingest idempotency returns the existing job even when it failed (from the plan).

## Incident

- The new `sources` model field went live via runserver auto-reload a few seconds before `migrate` ran; conversation endpoints could have errored briefly (`no such column`). Migrated immediately; lesson: create + apply migrations before saving model changes in the live checkout.
