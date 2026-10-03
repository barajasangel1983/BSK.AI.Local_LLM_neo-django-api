# Claude Session Log — 10/03/2026 — P8a Evidence records and entity aliases

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p8a-evidence` (off `main` @ `d8b6897`)
- **Pairs with:** frontend `BSK.AI_neo-llm-hub` branch `feat/p8a-evidence` (frontend log entry 32)
- **Plan:** frontend repo `claude/P8_canonical_context_plan.md` (all recommendations approved by Angel)

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | `Evidence` model; `CandidateTriple.evidence` (M2M) + `applied_aliases` | `context_graph/models.py`, migration `0004_evidence` | — |
| 2 | Data migration: one Evidence per existing triple from its current fields | migration `0005_backfill_evidence` | — |
| 3 | Staging keeps every occurrence: one Evidence per extraction window / import row; orphan evidence removed when pending triples are replaced or deleted | `context_graph/extraction.py`, `context_graph/structured.py`, `context_graph/evidence.py` | — |
| 4 | Approve: evidence IDs on nodes and edges (curated and lab); differing names become aliases (shared ownership, never renames). Delete removes exactly those. Promote keeps the evidence. | `context_graph/triples.py` | — |
| 5 | API: `sources` in the triples list, `GET /api/graph/evidence/<id>/` | `context_graph/extraction_views.py`, `context_graph/urls.py` | — |
| 6 | `manage.py graph_backfill_evidence` (idempotent) for triples approved before P8a | `context_graph/management/commands/graph_backfill_evidence.py` | — |
| 7 | Tests, README | `context_graph/tests/tests_evidence.py`, `README.md` | — |

## Verification

- `manage.py test` with the test Neo4j: **196 passing** (2 skipped). New `tests_evidence.py` (6):
  - every window kept as evidence, with the API and the evidence endpoint
  - re-run and delete leave no orphan evidence
  - import rows become evidence
  - the migration backfill is idempotent
  - integration: aliases and evidence IDs added and removed exactly; the seeded edge survives
  - integration: the backfill command restores the graph links idempotently
- **Bug found by the integration test and fixed:** two triples naming a node the same way. Only the first recorded the alias, so after both were deleted a stray alias remained. Fix: shared alias ownership, with the same rule in approval and backfill.
- **Live database:**
  - dry run on a copy first: 512 triples → 512 evidence rows, none missing
  - then the live DB: backup `data/backups/db.sqlite3.pre-p8a-evidence.bak`, migrated, worker restarted
  - `graph_backfill_evidence`: 408 approved (lab) triples → evidence IDs on 408 edges; a second run changed nothing
  - API: triple 171 → source evidence 52 (text, p.1, "Default free-form v1")

## Notes

- The old evidence fields on `CandidateTriple` (`page_start`, `evidence_text`, …) stay one more release for older clients; they hold the first occurrence.
- No curated text triples were approved yet, so no aliases were added on the live graph; the first will come with schema-mode extraction of the extruder manual.
