# BSK.AI.Local_LLM_neo-django-api
## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in keys/passwords
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver 0.0.0.0:8000
```

On the VPS the API runs as a systemd user service (`deploy/systemd/neo-llm-api.service`; install steps in the file): the same `runserver` on `127.0.0.1:8000`, started at boot and restarted after a crash. `neo-llm-api-tunnel.service` makes it reachable on the Tailscale address only; never bind it to `0.0.0.0` (the VPS has a public address). It still reloads itself on code changes; `systemctl --user restart neo-llm-api` is only needed for a new `.env` value.

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
- Schema editor API (GraphLab): `POST schema/validate/` (validate + diff, no writes), `POST schema/` (save a new active
  version), `GET schema/versions/`, `POST schema/versions/<n>/activate/`, `GET schema/export/?fmt=yaml|json&version=<n>`.
  Body: `{"text": "...", "format": "yaml"|"json", "note": "..."}` or `{"definition": {...}}`. Adding types/relationships/pairs
  is always allowed; removing or renaming one the graph still uses returns **409** with the conflicts.

Integration tests use a throwaway Neo4j (never the real graph):

```bash
docker compose -f deploy/neo4j/docker-compose.yml --env-file .env --profile test up -d neo4j-test
GRAPH_TEST_URI=bolt://127.0.0.1:7689 .venv/bin/python manage.py test context_graph
```

### Triple extraction and review (P4)

GraphLab **Generate triples** runs on the library worker (`POST /api/graph/extract/` with `document_ids`, `mode`
`schema` | `freeform` | `both`, optional `chunking` (default fixed 800/100 tokens), `presets` and `asset_id` scope).
Each window goes to the DGX Qwen model (prompted JSON in `json_object` mode, strict validation, one retry on malformed
output, one retry on a DGX timeout); a failed window is noted on the document and doesn't fail the run.

- **Staged, not written:** results land in `CandidateTriple` (pending) with evidence (page, section, window text),
  model, prompt preset + version and schema version. Schema-mode triples are checked against the active schema
  (`issue` explains a disallowed pair) and matched to existing entities (`existing: true`) or given proposed ids.
  Document/DocumentSection are not extracted — they are written as evidence on approval. A re-run replaces pending
  triples of the modes it ran; approved/rejected ones are kept and not staged again.
- **Review API:** `GET triples/?document=&status=&mode=&issues=1&q=`, `PATCH triples/<id>/` (pending/rejected only; ids
  re-resolved when names/types change), `POST triples/approve/ | reject/ | delete/` `{ids}`, `POST triples/<id>/promote/`.
- **Approve:** schema triples → curated graph (new entities `source: "text"`; existing ones keep their names), the edge
  records its supporting `triple_ids` (+ provenance when text created it), plus Document → DocumentSection →
  DESCRIBES evidence and Asset DOCUMENTED_BY Document when scoped. Free-form triples → the **lab layer**
  (`:Lab` nodes, `LAB_RELATION {predicate}`), never mixed with `:Entity`. **Promote** maps a lab triple onto the schema
  and moves it to the curated graph.
- **Delete** removes what a triple wrote: its support on each edge (an edge goes only when no triple supports it and
  text created it — seeded edges are never deleted), then orphaned text-created entities, sections and documents.
  Deleting a library document does the same for its approved triples.
- **Prompt presets:** `GET/POST presets/`, `PATCH/DELETE presets/<id>/`; a template change bumps the version.
  Placeholders: `{entity_types}`, `{relationships}`, `{document}`, `{section}`, `{text}` (without `{text}` the window is
  sent as the user message). Built-in defaults can be edited but not deleted.
- **Visualizer:** `data/?layer=curated|lab|both&doc=<doc_key>`. `GET lab/` returns the lab layer's size and source documents.
- **UI config:** `GET extract/config/` — DGX model, modes, default/available window strategies, extractable schema types and pairs.
- Settings: `GRAPH_EXTRACT_TIMEOUT` (120 s per call), `GRAPH_EXTRACT_MAX_TOKENS` (2000).

### Evidence and aliases (P8a)

Every place a fact was found is an `Evidence` row: the source kind (text, structured, later vision / human / OPC), the document and pages / section and window, or the data file and row, a region, the extractor, model and prompt, and a ≤ 1,500-char excerpt.

- **Staging:** a triple links to *all* its occurrences (`CandidateTriple.evidence`); `occurrences` is their count.
- **Approving:**
  - copies the evidence IDs onto the Neo4j nodes and edges it writes (`evidence_ids`)
  - adds a differing name as an **alias** of an existing node, never a rename. An alias several approved triples use is shared and goes only with its last user.
- **Deleting** removes exactly those again.
- **API:** `GET /api/graph/evidence/<id>/`; the triples list includes `sources`.
- **After migrating:** run `manage.py graph_backfill_evidence` once (idempotent) to link already-approved triples.

### Identity resolution and vocabulary (P8b)

`context_graph/identity.py` resolves a name to an entity in one order for every source (extraction, imports, edits, promote, later the VLM):

1. explicit id
2. same asset + engineering **tag** (`M101`, `VFD-101`, `DIE_PLUG`, historian columns)
3. tag + type, any scope
4. exact name or **alias**
5. fuzzy similarity: only a **possible** match with up to 3 candidates. A different number ("Zone 1" vs "Zone 2") never scores high.
6. otherwise **new**

Each triple end records `match` (`id`, `tag`, `alias`, `name`, `possible` or `new`) and its `candidates`. Approving is blocked until a possible match is resolved: `PATCH triples/<id>/` with `{subject_id}` (pick) or `{subject_match: "new"}`.

**Schema vocabulary** (optional; v1 still valid):
- `subtypes` per entity type. A known subtype goes to the node's `subtype` property.
- `aliases` per relationship. A synonym becomes the schema relationship only when the pair is allowed.
- Both are listed in extraction prompts and in `extract/config/`. Vocabulary changes count as a schema change.

**Schema v2** (DRIVES / PROTECTS / CONTROLS, subtypes, aliases): `context_graph/schemas/industrial_v2.yaml`. Saved and active since 2026-10-03; version 1 can be re-activated.

Units in imports are normalized (`context_graph/units.py`).

### Traceability (P8c)

`context_graph/provenance.py` answers "why is this in the graph?":

- `GET /api/graph/nodes/<id>/sources/` returns `origin` (e.g. `Seed (extr01-v1)`, `Document extraction`, `Structured import`), `aliases`, `subtype`, `tag` and `sources`. `sources` is every evidence record behind the node: its own `evidence_ids` plus those of approved triples naming it. Each record has a short `label` ("manual.pdf p.4", "bom.csv row 2"), the file names and the approved `triples` that used it. Lab nodes (`lab:…`) collect the evidence of their `LAB_RELATION` triples.
- `GET /api/graph/edges/sources/?from=&type=&to=` returns the same for one curated relationship (its `evidence_ids` plus those of its `triple_ids`). `?triple_id=` is for a lab relationship.
- Asset chat: the graph citation carries `fact_sources`, one label per fact in `facts`, either the first evidence or the origin.

### Structured data import (P6)

GraphLab **Import** turns CSV / Excel files (tag lists, alarm lists, BOMs) into staged triples by column mapping — no
LLM. Data files are kept apart from the document library (`GRAPH_DATA_BASE/<id>/`, model `DataFile`).

- **Files:** `GET/POST datafiles/` (multipart `files`; `.csv/.tsv/.txt/.xlsx/.xlsm`, deduplicated by content),
  `GET/PATCH/DELETE datafiles/<id>/` (PATCH `{sheet, header_row}` re-reads the columns; detail returns sample rows and
  template suggestions; DELETE takes approved triples back out of the graph).
- **Mapping:** `{"entities": [...], "relationships": [...]}` — an entity has a `type`, an `id_column` (key), optional
  `name_column` / `name_template` (`"{Description} normal range"`), `id_suffix`, `properties` (property ← column) and
  `skip_if_empty`; `{"scope": true}` is the selected asset. A relationship links two entities of the row
  (`unless: <entity>` = only when that entity is absent). See `context_graph/structured.py`.
- **Templates** (auto-filled from the header names): tag list (Signal MONITORED_BY Component/Asset, HAS_LIMIT
  OperatingLimit), alarm list (Asset HAS_ALARM, Component ASSOCIATED_WITH), BOM (parent/child HAS_COMPONENT).
  Saved mappings: `GET/POST mappings/`, `DELETE mappings/<id>/`.
- **Preview / stage:** `POST datafiles/<id>/preview/` and `.../stage/` with `{mapping, asset_id}`. Preview returns
  mapping errors, stats (rows, triples, new / existing entities, skipped rows), row problems and the first triples;
  nothing is saved. Stage replaces the file's pending triples (reviewed ones are kept). Limit:
  `GRAPH_IMPORT_MAX_ROWS` (5000) rows, `GRAPH_IMPORT_MAX_UPLOAD_BYTES` (20 MB).
- **Ids and matching:** `bsk:<type>:<asset key>/<key>[/<suffix>]`; an entity is *existing* only on an exact id, key or
  name match (no fuzzy matching for structured data).
- **Review:** the same triples API with `?data_file=<id>` (`&new=1`: triples that add an entity; `&fields=ids`: all
  matching ids). Approved triples use `source: "structured"` with file + row provenance and no document evidence
  nodes. New entities get the row's properties; an existing entity only gets properties it doesn't have yet, and
  those are removed again if the triple is deleted.
- Samples for EXTR01 (derived from the seed, plus a few new rows): `context_graph/samples/`.

### Asset-scoped chat (P7)

`POST /api/chat/` accepts `asset_id` (`bsk:asset:...`; `""` clears it, absent keeps the conversation's scope, stored on
`Conversation.asset_id`). For a scoped conversation the system prompt carries the asset's **fact sheet**
(`chat/asset_context.py`) — location, component tree, signals with units and normal ranges, alarms with components and
procedures, procedures, connections, linked documents — built from the curated graph (never the lab layer), independent
of `use_rag`. Each signal line includes the **last RUNNING historian value** (`plc_1_historian`, via the signal's
`historian_ref`) compared with its range; the latest sample's timestamp and machine state are stated. If the sheet exceeds
`ASSET_CONTEXT_SHARE` (0.3) of the model's budget, lines matching the question are kept first.

The legacy `plc_historian` shift summaries (old `bsk_rag`) are not added to asset-scoped chats. With RAG on, documents linked to the asset (DOCUMENTED_BY) are searched first (`search_v2(..., doc_keys=...)`), then the
whole library; excerpts get `ASSET_RAG_SHARE` (0.25). Sources: `kind: "graph"` ([G], with the facts used),
`"historian"` ([H]) and `"document"` ([1], [2], …). Unknown asset → 400; graph down when choosing → 503, mid-conversation →
the answer says the facts are unavailable. DGX chat timeout: `DGX_CHAT_TIMEOUT` (180 s).

## BSK GPU orchestrator (Docling + VLM)

The BSK desktop has one 8 GB GPU that runs **either** Docling **or** the VLM (Qwen3-VL-4B). Its orchestrator (`:5003`) starts the requested service and stops the other, mid-request included. The contract is in the frontend repo's `claude/VLM_service_brief.md` (v1.2).

`gpu.orchestrator.use(service)` makes the Hub the only, well-behaved client:
- **Lock:** a Hub-wide file lock (`GPU_LOCK_PATH`, shared by the API and the library worker) for the whole call. Interactive requests wait ("GPU busy") rather than interrupting a parse.
- **Activation:** `POST /gpu/activate`; while the service is starting, poll `/gpu/status` every 2 s. A 409 backs off; `health: "error"` is retried once, then BSK counts as unavailable.
- **Switch:** `GPU_ORCHESTRATOR_ENABLED` (default `false`). Until BSK's orchestrator is live, Docling is called directly as before. When it's enabled but the orchestrator doesn't answer, Docling is still called directly if its `/health` answers (5 s connect timeout).
- **Recording:** activations are recorded in Analytics (purpose `gpu`).
- **Health page:** GPU orchestrator, Docling and VLM entries. `idle` = stopped by design, started on demand; the VLM is never probed directly.

**Deploy order with the BSK side:**
1. Ship this, switch off.
2. BSK builds the orchestrator and VLM.
3. Set `GPU_ORCHESTRATOR_ENABLED=true` and restart the API and worker.
4. Only then does BSK switch Docling to `restart: no`.

## Usage analytics

Every call to a model or AI service is recorded in `usage.ModelCall` by `usage.recorder.track(...)`:

- **Purposes:** chat, compare, regenerate, title, extract, embed, rerank, parse.
- **Fields:** prompt / completion tokens from the provider's usage (estimated and flagged when absent), latency, ok / error / timeout.
- Recording never changes the call.

`GET /api/usage/summary/?days=7|14|30|90&purpose=` returns totals, per model, per purpose, a daily series, and conversations per model. The chat accepts `purpose: "compare" | "regenerate"` to label calls.

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
