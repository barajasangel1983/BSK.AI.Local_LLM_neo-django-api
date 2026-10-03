# Claude Session Log — 10/03/2026 — BSK GPU orchestrator client + Docling through it

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/gpu-orchestrator` (off `main` @ `ec99bf2`, after the Analytics PR #16 merged)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/gpu-orchestrator` (frontend log entry 29)
- **Contract:** frontend repo `claude/VLM_service_brief.md`. v1 contract and Neo's answers; v1.2 image transport (the Hub crops and sends base64) confirmed by Angel.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | `gpu` app: `orchestrator.py` with `status`, `activate`, `use(service)` (Hub-wide file lock, 2 s polling, 409 back-off, one retry on `error`, 5 s connect timeout, Docling direct fallback), `health_summary` | `gpu/*`, `neo_llm_api/settings.py` | — |
| 2 | Docling parse goes through `gpu.use("docling")`; GPU errors become `DoclingError("Docling unavailable: …")` (the parse job fails with a clear message) | `ingestion/docling_client.py` | — |
| 3 | Health: GPU orchestrator / Docling / VLM entries; new status `idle` (stopped by design), counted as up for uptime | `chat/views.py` | — |
| 4 | Settings: `GPU_ORCHESTRATOR_ENABLED` (false), `GPU_ORCHESTRATOR_URL`, `GPU_ACTIVATE_TIMEOUT` 90, `GPU_LOCK_PATH`, `GPU_LOCK_WAIT` 1800, `VLM_URL`, `VLM_MODEL`, `VLM_TIMEOUT` 90 (VLM ones for the next phase) | `neo_llm_api/settings.py` | — |
| 5 | Tests, README | `gpu/tests/tests_orchestrator.py`, `README.md` | — |

## Verification

- `manage.py test` with the test Neo4j: **187 passing** (2 skipped). `gpu` adds 14 tests:
  - ready immediately
  - starting → polled every 2 s
  - 409 back-off
  - error retried once, then unavailable
  - 422 / unknown service / unreachable
  - activation timeout
  - activation recorded in usage
  - switch off → no calls
  - unreachable → direct Docling only if it answers, never for the VLM
  - busy while another holder has the lock, never switching under it
  - lock released on failure
  - parse activates Docling before converting
  - Health page statuses with the switch on and off
- **Live** (BSK orchestrator not built yet):
  - switch off: Docling parse OK in 2.1 s
  - switch on: the orchestrator didn't answer and Docling was called directly, OK in 7.2 s (22 s before the 5 s connect timeout)
  - Health page: orchestrator and VLM Idle, Docling Online

## Next

- Neo builds the orchestrator and the VLM container on BSK. The Hub tests against them, then sets `GPU_ORCHESTRATOR_ENABLED=true` (and restarts the API and worker) **before** Neo flips Docling to `restart: no`.
- VLM phase: chat model (DEFER rule, 1 image per message, ≤ 1280 px), then Describe figures (Docling JSON page / bbox in points, BOTTOMLEFT; crop on the VPS; base64).
