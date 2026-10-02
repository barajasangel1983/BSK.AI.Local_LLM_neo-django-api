# Claude Session Log — 10/02/2026 — Context Graph P1 (graph core)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/context-graph-p1` (off `main` @ `71cbffd`, after P0 PR #8 merged)
- **Plan:** frontend repo `claude/Claude_session_09302026.md` (entries 15–17)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Versioned global schema registry, canonical IDs, schema-validated Cypher repository, services, read API, EXTR01 seed, management commands, test Neo4j | `context_graph/{schema,ids,models,registry,repository,services,views,urls}.py`, `context_graph/migrations/0001_initial.py`, `context_graph/schemas/industrial_v1.yaml`, `context_graph/seeds/extr01.yaml`, `context_graph/management/commands/{graph_init,graph_seed,graph_reset}.py`, `context_graph/tests/{tests_core,tests_integration}.py`, `deploy/neo4j/docker-compose.yml`, `requirements.txt`, `README.md` | `be6f62f` — PR [#9](https://github.com/barajasangel1983/BSK.AI.Local_LLM_neo-django-api/pull/9) |

## Details

- **Schema (data, not code):** default "Industrial" schema in `schemas/industrial_v1.yaml` — the 12 entity types and the relationship list from the brief, plus two additions: `Asset -MONITORED_BY-> Signal` (asset-level KPIs such as OEE/throughput) and `Component -CONNECTED_TO-> Component` (property `kind`; answers "how are components connected"). `schema.py` validates definitions (PascalCase labels, UPPER_SNAKE relationship types, reserved `Entity`/`Lab`, endpoints exist, no ID-prefix collisions) and checks every write (`check_label`, `check_relationship`). Labels/relationship types are interpolated into Cypher only after these checks.
- **Registry:** `GraphSchemaVersion` (SQLite, append-only versions, exactly one active via a partial unique constraint); `registry.active_schema()` / `save_version()`. Migration applied on the dev DB.
- **IDs:** `bsk:<kebab-type>:<key>` (`ids.py`), case preserved (EXTR01, DIE_PLUG).
- **Neo4j model:** every curated node has `:Entity` + one schema label, unique `id` (constraint `entity_id`), full-text index on name/description, name index per label; `created_at`/`updated_at`; `source` = provenance.
- **Repository / services:** MERGE-based idempotent upserts (ID must match its label; existing node can't change type; relationships must be allowed), node/neighbors, search, counts, subgraph for the visualizer; `get_asset_context` (hierarchy, component tree with signals/limits/OPC nodes/alarms, connections, asset signals, alarms with procedures, procedures, documents), `alarm_procedures`, seed planning that validates the whole file before writing.
- **API (`/api/graph/`):** `schema/`, `assets/`, `assets/<id>/context/`, `nodes/?type=&q=&limit=&offset=`, `nodes/<id>/`, `alarms/<id>/procedures/`, `data/?root=&depth=&types=&limit=`; errors → 503 (graph unavailable) / 404 (unknown node) / 400 (schema/input).
- **EXTR01 seed** (87 nodes, 136 relationships), from the historian: Plant/Area/Line, asset, 9 components (Barrel Zones 1–3 under Screw & Barrel), material-flow + mechanical connections, 30 signals mapped to `plc_1_historian.extruder_samples` columns, 16 normal-range limits (P5–P95 while RUNNING, field `basis`), the 20 alarm codes that occur (severity, component associations), 8 draft procedures. Generated with a scratch script; the YAML is the editable artifact.
- **Commands:** `graph_init`, `graph_seed <file> [--dry-run]`, `graph_reset --yes`.
- **Test Neo4j:** compose profile `test` → `bsklab-neo4j-test` on `127.0.0.1:7689`, tmpfs data; integration tests wipe it before each test.

## Verification

- Live dev graph: `graph_init` (schema v1, 14 constraint/index statements); seed loaded twice → still 87 nodes / 136 relationships (idempotent).
- API answers: asset + location (BSKLAB Demo Plant > Extrusion > Extrusion Line 1), component tree, Die Head signals with limits (die pressure 84.3–99.2 bar) and 5 alarms, DIE_PLUG → "Die plug / screen block clearing", material-flow chain, search, subgraph (Die Head depth 1: 12 nodes / 23 links), schema counts.
- Tests: `GRAPH_TEST_URI=bolt://127.0.0.1:7689 manage.py test context_graph chat ingestion` → 96 passing (1 skipped: opt-in P0 test vs dev Neo4j); 6 integration tests against the test Neo4j; `pytest ingestion/tests` 15 passing.

## Notes

- Fixed during testing: the OperatingLimit field `source` (derivation) clashed with provenance `source` → renamed to `basis`.
- Neo4j Python driver: a relationship returned alone has no node properties → endpoint ids are returned explicitly.
