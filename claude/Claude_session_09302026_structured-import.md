# Claude Session Log — 10/02/2026 — P6 Structured data import (backend)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/structured-import` (off `main` @ `d9096b6`, after P5 PR #13 merged)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/structured-import` (frontend log entry 24)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Models: `DataFile`, `ImportMapping`; `CandidateTriple` gets `data_file` (document now nullable), `source`, `row_number`, `subject_props` / `object_props`, `applied_props` | `context_graph/models.py`, `context_graph/migrations/0003_structured_import.py` | — |
| 2 | Import engine: CSV (delimiter sniffing, utf-8 / latin-1) and XLSX (sheet, header row) reading, templates with header-name suggestions, mapping validation against the schema, rows → triples, staging | `context_graph/structured.py`, `neo_llm_api/settings.py`, `requirements.txt` (+`openpyxl`) | — |
| 3 | Review/commit for structured triples: `source: "structured"` provenance, node properties, missing-only properties on existing entities (removed again on delete), no document evidence nodes | `context_graph/triples.py`, `context_graph/extraction.py` (`EntityIndex.match_exact`) | — |
| 4 | API: data files, preview, stage, saved mappings; triples `?data_file=`, `?new=1`, `?fields=ids` | `context_graph/import_views.py`, `context_graph/extraction_views.py`, `context_graph/urls.py` | — |
| 5 | EXTR01 sample files (from the seed + new rows), tests, README | `context_graph/samples/*.csv`, `context_graph/tests/tests_structured.py`, `README.md` | — |

## Design notes

- **Separate from the document library** (user decision): data files have their own model, folder and list.
- **Mapping model:** entities (type, key column, name column/template, id suffix, properties, skip-if-empty, or "the selected asset") + relationships between entities of the same row, with `unless` for fallbacks (signal under its component, or under the asset when the component cell is empty; top-level BOM items under the asset).
- **Names and properties are merged per entity across the file** before staging, so a BOM parent that first appears only as a key gets its name from its own row.
- **Exact matching only** for structured data (id, key or normalized name). Fuzzy matching would confuse "Barrel Zone 1" with "Barrel Zone 2".
- **Existing entities:** names and existing properties are never overwritten; only missing properties are added, recorded in `applied_props` and removed if the triple is deleted.
- **Edges:** same support model as P4 (`triple_ids` per edge; seeded edges are never deleted or re-attributed).
- **Synchronous:** up to `GRAPH_IMPORT_MAX_ROWS` (5000) rows per file in the request; no worker job.

## Verification

- `manage.py test` with the test Neo4j: **157 passing** (2 skipped: real-instance checks). New `tests_structured.py` (11): upload/columns/dedupe, template suggestions for the three samples, mapping validation, tag-list preview and staging (limits, coercion, asset fallback, re-stage keeps reviewed), BOM name merging and row errors, Excel sheet/header row, row limit, saved mappings and delete; integration: tag list against the seed (6 new / 56 existing entities, approve, seed provenance untouched, delete restores the graph), BOM missing-only properties added and removed, alarm list.
- **Live** (migration applied; DB backup `data/backups/db.sqlite3.pre-structured-import.bak`): the three sample files uploaded through the UI; the tag list staged — 51 triples, 6 new entities, 56 already in the graph. Nothing approved; the live graph is untouched.

## Open points

- The same new component named differently in two files staged before either is approved (tag list "Gearbox" by name, BOM "gearbox" by key) gets two proposed ids. Approving one file first makes the other match it; importing the BOM first is the natural order.
