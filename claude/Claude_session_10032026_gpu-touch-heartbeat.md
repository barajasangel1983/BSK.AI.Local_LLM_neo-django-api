# Claude Session Log — 10/03/2026 — GPU idle-timer heartbeat

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/gpu-touch-heartbeat` (off `main` @ `b80ce12`)
- **Why:** Neo confirmed (orchestrator `core.py:202`) that `POST /gpu/activate` on the already-active service is a no-op and does **not** reset the 30-min idle timer. Only a real transition or `POST /gpu/touch` resets it. The Hub never touched, so after Neo's Docling cutover a long job could lose its GPU service.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | `touch()` (failures logged, never raised); `use()` runs a heartbeat while it holds the GPU (every `GPU_TOUCH_INTERVAL`, default 300 s) and touches once at the end. No heartbeat when the orchestrator is off or unreachable (direct-Docling fallback). | `gpu/orchestrator.py`, `neo_llm_api/settings.py` | — |
| 2 | Tests: heartbeat during the block + final touch; touch failures don't break the job; parse order is activate → convert → touch | `gpu/tests/tests_orchestrator.py` | — |

## Verification

- `manage.py test gpu ingestion chat`: 116 OK (2 skipped).
- Read-only probes: :5003 healthz/status OK (docling active), :5001 OK, :5002 timed out (firewall pending per Neo).
- Live smoke parse with the flag on: **not run**. The tool permission check blocked it as a production action; it's left for Angel.

## Incident

- 22:30:12 UTC: the OpenClaw gateway restarted and runserver (in its process group, see `SERVICES_MAP.md`) died with it. Found at about 22:38 during this work; restarted detached (`setsid nohup … runserver 0.0.0.0:8000 >> /tmp/django.log`). Ping OK, no jobs lost (none active). The worker was unaffected.
