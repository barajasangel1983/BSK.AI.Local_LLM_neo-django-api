# BSK.AI.Local_LLM_neo-django-api
## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in keys/passwords
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver 0.0.0.0:8000
```

Tests: `.venv/bin/python manage.py test` (Django tests) and `.venv/bin/python -m pytest ingestion/tests` (pure unit tests).

## Context Graph (Neo4j)

The Context Graph (entities/relationships of industrial assets, with references
back to Chroma document evidence) lives in the `context_graph` app and a
dedicated Neo4j instance:

```bash
docker compose -f deploy/neo4j/docker-compose.yml --env-file .env up -d
curl http://127.0.0.1:8000/api/graph/health/
```

Set `GRAPH_ENABLED=true` and `NEO4J_*` in `.env`; with `GRAPH_ENABLED=false` the
rest of the app runs without Neo4j. Neo4j Browser: http://127.0.0.1:7474
(localhost only). Integration test against the real instance:
`GRAPH_INTEGRATION=1 .venv/bin/python manage.py test context_graph`.

### Schema, seed and API (P1)

```bash
.venv/bin/python manage.py migrate context_graph
.venv/bin/python manage.py graph_init                                   # schema v1 + constraints/indexes
.venv/bin/python manage.py graph_seed context_graph/seeds/extr01.yaml   # EXTR01 (add --dry-run to validate only)
.venv/bin/python manage.py graph_reset --yes                            # dev: delete the curated graph
```

- Schema: `context_graph/schemas/industrial_v1.yaml` (default), stored as versions in `GraphSchemaVersion`; the active version validates every write.
- Canonical IDs: `bsk:<type>:<key>`, e.g. `bsk:asset:EXTR01`, `bsk:component:EXTR01/die-head`.
- Read API under `/api/graph/`: `schema/`, `assets/`, `assets/<id>/context/`, `nodes/?type=&q=`, `nodes/<id>/`, `alarms/<id>/procedures/`, `data/?root=&depth=&types=&limit=`.

Integration tests use a throwaway Neo4j (never the real graph):

```bash
docker compose -f deploy/neo4j/docker-compose.yml --env-file .env --profile test up -d neo4j-test
GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
```

## Document library (shared by RAG Lab and GraphLab)

Uploads go to a shared library (`/api/documents/`); each file is parsed once by Docling and cached under
`LIBRARY_BASE/<id>/`. RAG is opt-in per file: **Generate embeddings** (`POST /api/documents/embed/`) chunks with a
chosen strategy — `structure` (default), `sentence` (pysbd, max 300 tokens, 1-sentence overlap) or `fixed`
(512/64 tokens) — and writes to `bsk_rag_v2`, replacing that document's previous chunks.

Long steps run in a separate worker (so runserver reloads can't kill them):

```bash
.venv/bin/python manage.py library_worker          # run forever (or --once to drain the queue)
.venv/bin/python manage.py library_import_legacy   # one-off: bring old pipeline documents into the library
.venv/bin/python manage.py rebuild_rag_v2          # re-embed all library documents with their strategies
```

## Chroma server and the worker service

Chroma runs as a server (`deploy/chroma/docker-compose.yml`, `127.0.0.1:8100`, data folder mounted as-is);
every process connects over HTTP through `ingestion/chroma_client.py` (`CHROMA_HOST`/`CHROMA_PORT`). Embedded
Chroma isn't process-safe — with the worker writing, the API would read stale/empty data. Tests force embedded
mode on temp dirs (`neo_llm_api/test_runner.py`).

The worker runs as a systemd user service (`deploy/systemd/neo-library-worker.service`; install steps in the file).
Restart it after deploys: `systemctl --user restart neo-library-worker`.

See the frontend repo's `claude/SERVICES_MAP.md` for every service, port and container this project uses.
