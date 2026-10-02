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
