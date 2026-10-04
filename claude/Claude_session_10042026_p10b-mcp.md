# Claude Session Log — 10/04/2026 — P10b Context MCP server

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p10b-mcp` (stacked on `feat/p10a-context-service`, PR #27, not merged yet)
- **Plan:** frontend repo `claude/P10_context_service_mcp_plan.md`; client contract `claude/MCP_client_brief.md`; paired with frontend `feat/p10b-mcp-ui`
- **Why:** BSKLAB EDGE, Claude and other agents get the Studio's knowledge through MCP. Angel wants where the server is exposed to be a Studio setting, and everything inside Tailscale.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | MCP server process: Streamable HTTP (stateless, JSON), 9 read-only tools wrapping `context_service.service`, token sign-in, `GET /health`; it follows the exposure setting without a restart | `context_service/mcp_server.py`, `context_service/management/commands/mcp_server.py` |
| 2 | Tokens (hash only, shown once, last used, revoke) and exposure (Off / This machine / Tailscale) | `context_service/models.py`, `context_service/mcp_access.py`, `context_service/migrations/0001_mcp_access.py` |
| 3 | Settings API: `GET / PUT /api/context/mcp/`, `POST /api/context/mcp/tokens/`, `DELETE …/tokens/<id>/` | `context_service/views.py`, `context_service/urls.py` |
| 4 | Health entry "Context MCP server" (`idle` when set to Off) | `chat/views.py` |
| 5 | `manage.py mcp_smoke [--url] [--ask]`: a real MCP client with a temporary token calls every tool | `context_service/management/commands/mcp_smoke.py` |
| 6 | Service unit `neo-context-mcp` (enabled at boot, `Restart=always`) | `deploy/systemd/neo-context-mcp.service` |
| 7 | Settings `MCP_PORT` (8002), `MCP_TAILSCALE_HOST`, `MCP_STUDIO_API`; dependencies `mcp==1.30.0`, `uvicorn` | `neo_llm_api/settings.py`, `requirements.txt` |

## Behaviour

- **Tools:** `list_assets`, `resolve_entity`, `get_entity_context`, `get_sources`, `search_documents`, `assemble_context`, `ask`, `get_operational_state`, `get_system_health`. Same JSON as the REST endpoints.
- **Exposure:** the process binds port 8002 on `127.0.0.1` and, for "Tailscale", on the Tailscale address itself (no tunnel service). It re-reads the setting every 3 s and opens or closes listeners. It never binds all interfaces. LAN / WLAN and Internet are listed in the setting as not available.
- **Sign-in:** every request except `/health` needs `Authorization: Bearer <token>`; otherwise 401. Tokens are `bsk_…`; only the SHA-256 is stored.
- **Records:** each tool call is logged (`mcp tool=… client=<token name>`) and saved as a `ModelCall` with purpose `mcp`, model `mcp:<tool>`.
- **SDK version:** `mcp` 2.x renamed `FastMCP` and changed its API; the server uses the 1.x line (1.30.0, pinned). The plan's note that the SDK was already installed was wrong.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **296 OK** (1 skipped; 8 new in `context_service/tests/tests_mcp.py`).
- Migration `context_service.0001` applied on the live DB (two new tables; nothing existing changed).
- **Live:** service installed and running; exposure set to **Tailscale**.
  - 127.0.0.1:8002 and 100.92.170.72:8002 answer; the public address refuses.
  - No token → 401.
  - `mcp_smoke` over the Tailscale address: health, 401 check, 9 tools listed, and 8 tool calls, all PASS (the packet in 0.9 s).
  - Switching the setting to Off / This machine / Tailscale opens and closes the listeners within seconds.
  - Forced kill of the service → back in under 10 s.
- Not tried: Claude Desktop on a PC (the brief gives the expected setup); `ask` through MCP in the smoke run (it works through REST).

## Notes

- The worker and the API were not restarted for this (new tables only; the API reloads by itself). The MCP service does not reload code: `systemctl --user restart neo-context-mcp` after deploys.
- **Commit:** — (waiting for Angel)
