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
