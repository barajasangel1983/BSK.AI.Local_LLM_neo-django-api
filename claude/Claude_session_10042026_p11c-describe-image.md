# Claude Session Log — 10/04/2026 — P11c MCP tool describe_image

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p11c-describe-image` (off `main` @ `3bd4374`)
- **Plan:** frontend repo `claude/P11_bsklab_edge_plan.md` (P11c); the client side is in `BSK.AI_bsklab-edge`, branch `feat/p11c-chat-options`
- **Why:** BSKLAB EDGE's chat accepts a picture, but its model can't read images, and only this backend may use the vision model on BSK (one GPU; the contract with Neo). So the Studio reads the picture for the client.

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | `describe_image(data, question)`: downscale to JPEG ≤ 1280 px, ask the VLM through the GPU lock for what the picture shows, its legible text and anything abnormal; nothing is stored | `context_service/service.py` |
| 2 | MCP tool `describe_image(image_base64, question?)`; request bodies up to 12 MB; 15 tools in total | `context_service/mcp_server.py`, `context_service/views.py` |
| 3 | `mcp_smoke --image` (uses the BSK GPU, so it is optional) | `context_service/management/commands/mcp_smoke.py` |
| 4 | Tests | `context_service/tests/tests_viewers.py` |

## Behaviour

- The picture is processed in memory; it is not written to the library, Chroma or disk.
- Chat lock wait (120 s). GPU busy → tool error starting `gpu_busy:`; BSK asleep or the VLM failing → `gpu_unavailable:`. The client turns these into "try later".
- Recorded for Analytics as model `bsk-qwen3-vl-4b`, purpose `mcp-image`, plus the `mcp:describe_image` call.
- A line where the model repeats the user's question is dropped from the reading.
- This is the first tool that uses a model on the client's behalf besides `ask`. It still changes nothing in the Studio.

## Verification

- `manage.py test` with the test Neo4j and Chroma: **303 OK** (1 skipped; 2 new).
- Live: `mcp_smoke --image` read the text of a generated picture in 1.6 s (GPU already on the VLM). Through EDGE in a browser: a picture of an operator panel with an active DIE_PLUG alarm was read correctly (alarm, severity, three values, machine state).
- **Commit:** — (waiting for Angel)
