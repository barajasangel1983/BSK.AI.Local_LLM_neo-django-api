# Claude Session Log — 10/04/2026 — P11b MCP tools for BSKLAB EDGE's viewers

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p11b-mcp-viewers` (off `main` @ `28f7efc`)
- **Plan:** frontend repo `claude/P11_bsklab_edge_plan.md` (P11b); the client side is in `BSK.AI_bsklab-edge`, branch `feat/p11b-viewers`
- **Why:** EDGE's RAG and Graph tabs are view-only and may read only through MCP, so the Studio has to offer the library and the graph as tools.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | Service functions: `list_documents`, `get_document` (described figures), `get_document_page` (page text with figure descriptions), `figure_image` (JPEG), `get_graph` (curated layer around an asset) | `context_service/service.py` |
| 2 | MCP tools `list_documents`, `get_document`, `get_document_page`, `get_figure_image` (returns MCP image content), `get_graph`: 14 tools in total | `context_service/mcp_server.py`, `context_service/views.py` |
| 3 | `mcp_smoke` also checks the five new tools | `context_service/management/commands/mcp_smoke.py` |
| 4 | Tests | `context_service/tests/tests_viewers.py` |

## Behaviour

- **Read-only, approved content only.** Only figures with status "done" are listed or served (skipped and pending ones are not). `get_graph` always uses the curated layer; internal properties (`evidence_ids`, timestamps) are removed. Depth is 1–3, at most 600 nodes.
- **Page text** is the parsed Markdown split at page breaks, with figure descriptions in place and image placeholders removed.
- **Figure images** are the same crops the vision model saw, returned as `image/jpeg`.
- No migration. The MCP service was restarted to load the tools (it does not reload code); the worker and the API were not touched.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **301 OK** (1 skipped; 5 new).
- **Live, `mcp_smoke` over the Tailscale address:** 14 tools listed; `get_graph` 68 nodes / 117 links; `list_documents` 6; `get_document` (BX80 Manual, 10 described figures); `get_document_page`; `get_figure_image` image/jpeg 55,653 bytes; all PASS.
- **Commit:** — (waiting for Angel)
