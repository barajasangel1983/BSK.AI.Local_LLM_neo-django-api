# Claude Session Log — 11/08/2026 — P22f: coverage view API

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p22f-coverage` (off `main` @ `a81c3a9`)
- **Plan:** frontend repo `claude/P22f_coverage_view_plan.md` (D1–D4 as recommended); paired with frontend `feat/p22f-coverage`
- **Why:** one screen for what the knowledge base knows about an asset, part by part and facet by facet.

## Changes

| # | Change | Files |
|---|--------|-------|
| F1 | `grid(asset)`: rows = the asset's parts (tree order); per facet approved facts (graph nodes the part owns through itself, its model, its family), pending triples (anchored on the part, model or family, or with the part as an end), status covered / pending / none / na / no_manual | `context_graph/coverage.py` |
| F2 | Not applicable: defaults by `component_type` (mechanical parts and assemblies: no Controls, Setup; motors, contactors: no Setup; only measuring parts have Quality), a person's override per part and facet | `context_graph/coverage.py`, `context_graph/models.py`, `migrations/0016_p22f_coverage_overrides.py` |
| F4 | `cell(...)`: the facts with part / model / family, pending by manual, the part's manuals | `context_graph/coverage.py` |
| — | `GET /api/graph/coverage/?asset=`, `GET coverage/cell/?asset=&component=&facet=`, `POST coverage/override/` | `context_graph/views.py`, `urls.py` |
| F5 | Tests (6) | `context_graph/tests/tests_coverage.py` |

## Verification

- Suite 805 OK (47 skipped). Live EXTR01 grid read: cutter drive Safety 65, main gearbox Installation none, Controls n/a.
- After merge: `migrate` (one table), touch `settings.py`, worker restarted only with 0 active jobs.
