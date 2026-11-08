# Claude Session Log — 11/08/2026 — P24-3 (backend): EDGE answers through the engine

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p24-3-edge-answers` (off `main` @ `e12b0d1`)
- **Plan:** frontend `claude/P24_studio_edge_alignment_plan.md`, Revision 2, P24-3. Paired with EDGE
  `feat/p24-3-edge-engine`. Built overnight on Angel's go; **not merged before Angel validates**.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | `answer_events(...)`: the engine for a client with its own conversations — the same `_answer_events` steps; nothing saved in the Studio; `done` = {reply, sources}; its prompt, picture reading, `asked_before` | `chat/views.py` |
| 2 | `POST /answer` on the MCP server: the engine streamed as server-sent events, behind the same token (BearerAuth); the engine runs in one worker thread | `context_service/mcp_server.py` |
| 3 | The historian citation also carries `latest_ts`, `state`, `running_ts` (EDGE shows them apart) | `chat/asset_context.py` |
| 4 | Tests: the model gets exactly what the Studio chat sends; own prompt, picture, history; bad requests; the route's token, stream, errors | `chat/test_stream.py`, `context_service/tests/tests_mcp.py` |

## Verification

- Suite 826 OK.
- Preview (`~/backups/p24-3-preview/`): MCP from this worktree on :8012, EDGE from its worktree on :8093 (a copy of
  EDGE's data). Safety set asked through EDGE's chat API (as the tablet does), scored with chat_eval's scorer:
  **engine 9/10** (the Studio chat: 9/10 on the 8B); **EDGE's old path 3/10**. Engine ~17 s per answer, old ~6 s.
