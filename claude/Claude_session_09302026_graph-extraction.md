# Claude Session Log — 10/02/2026 — P4 Graph extraction backend ("Generate triples")

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/graph-extraction` (off `main` @ `2068d77`, after P3 PR #11 merged)
- **Frontend:** no code changes in P4 (the Extract UI is P5); log entry 22 in the frontend session log.

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Models: `PromptPreset` (editable, versioned), `CandidateTriple` (staged triple + evidence + provenance); `Job.Kind.EXTRACT`; `Document.graph_error`, `graph_updated_at` | `context_graph/models.py`, `ingestion/models.py`, `context_graph/migrations/0002_extraction.py`, `ingestion/migrations/0003_extraction.py` | — |
| 2 | Extraction pipeline on the library worker: windows → DGX Qwen per mode → strict parse → schema checks → entity matching → de-dup → staged | `context_graph/extraction.py`, `ingestion/library.py`, `neo_llm_api/settings.py` | — |
| 3 | Review/commit: edit, approve (curated / lab layer), reject, delete with graph cleanup, promote lab → curated | `context_graph/triples.py` | — |
| 4 | API: presets, extract, triples (+ bulk approve/reject/delete, promote), `data/?layer=&doc=`; graph counts in document JSON; document delete uncommits approved triples | `context_graph/extraction_views.py`, `context_graph/urls.py`, `context_graph/views.py`, `context_graph/services.py`, `context_graph/repository.py`, `ingestion/library_views.py` | — |
| 5 | Neo4j outages (`ServiceUnavailable`/`SessionExpired`) mapped to `GraphUnavailable` → 503 everywhere | `context_graph/driver.py` | — |
| 6 | Tests + README | `context_graph/tests/tests_extraction.py`, `context_graph/tests/tests_core.py`, `README.md` | — |

## Design notes

- **LLM:** DGX `qwen38-27b-fp8` only. Constrained decoding (`json_schema` / `guided_json`) was unusable on the DGX (timeouts, invalid output), so: prompted JSON, `response_format: json_object`, `temperature 0`, thinking off, strict validation (incomplete items skipped), one retry on malformed output. The model reports confidence 1.0 almost always — not a usable signal.
- **Windows:** default `fixed` 800/100 tokens (Custom: any library chunking strategy/params, validated by `ingestion.chunking.resolve_params`). Parsed first if needed.
- **Prompt presets:** built-in "Default schema-guided" / "Default free-form" (created on first use, editable, not deletable); placeholders `{entity_types} {relationships} {document} {section} {text}` substituted literally (no `str.format`, so JSON braces in templates are safe). A template change bumps `version`; each triple records preset name + version + schema version.
- **Schema mode:** the prompt lists only extractable types/pairs — Document/DocumentSection are excluded (live run 1 showed the model producing document-metadata triples) and dropped if returned. Disallowed pairs are staged with an `issue` (reviewer can fix types and approve). Names are matched to existing entities (exact normalized name, id key, token Jaccard ≥ 0.6); otherwise ids are proposed in the asset scope (`bsk:component:EXTR01/<name>`) or document scope.
- **Re-runs** replace pending triples of the modes that ran; approved/rejected triples are kept and not re-staged.
- **Commit (curated):** new entities get `source: "text"`, `created_by_triple`; existing ones are not renamed. Edges are MERGEd per (from, type, to) and carry `triple_ids` (all supporting triples); provenance (doc, pages, model, preset, confidence) is written only on edges text created — seeded edges keep their provenance. Evidence: `Document` → `HAS_SECTION` → `DocumentSection` → `DESCRIBES` → subject/object (where the schema allows), `Asset DOCUMENTED_BY Document` when an asset scope is set.
- **Commit (lab):** `(:Lab {id: "lab:<doc_key>:<slug>", name, type})-[:LAB_RELATION {predicate, triple_id, …}]->(:Lab)`; no `:Entity` label, so curated queries never see it.
- **Delete:** removes the triple's support from edges; deletes an edge only if no triple supports it and `source = "text"`; then orphaned text-created entities, sections without DESCRIBES, documents without sections; lab nodes without relationships. Approved triples can't be edited (delete, or promote a lab triple).
- **Worker safety:** heartbeat before every attempt (an attempt is ≤ 2 calls × 120 s, under the 300 s stale threshold — otherwise `requeue_stale` could re-queue a running job); one retry on DGX timeout/connection errors.

## Verification

- `manage.py test` with the test Neo4j: **145 passing** (2 skipped: real-instance checks). New `tests_extraction.py` (18): parsing/think-stripping/clamping, literal placeholder rendering, malformed retry, timeout retry, params validation, staging/de-dup/issues/proposed ids, evidence types neither prompted nor staged, re-run semantics, edit re-resolution, presets CRUD/versioning/default fallback, full API review flow, 503 on approve without graph; integration: provenance + evidence + cleanup on delete, seed edge never deleted when restated by two triples, invalid triple reported not written, lab layer + `layer=lab` data + promote (matched the seeded Die Head) + document delete uncommits.
- **Live** (worker restarted on the new code; real graph untouched — nothing approved):
  - Run 1 on `BSKLAB-COPILOTV0-01-PDF` (6 pages, pitch deck), mode `both`, scope `bsk:asset:EXTR01`: 25 windows × 2 = 50 calls, **22 min**, 118 triples staged (115 free-form, 3 schema), 7 calls timed out at 120 s.
  - Diagnosis: the DGX is shared (`vllm:num_requests_running` = 2 with nothing of ours in flight); the same 7-token request took 137 s, then 11 s. Not prompt size or generation length. → added the timeout retry + heartbeats. The 3 schema triples were document metadata → excluded evidence types.
  - Run 2 (schema only, same doc): **57 s**, 0 failures, 1 triple (`Extruder 01` MONITORED_BY `18 %` — junk to reject; expected for a slide deck), the 115 pending free-form triples kept.
  - The deck isn't a machine manual; real quality validation needs the extruder manual/SOP.

## Open points

- "Extruder 01" doesn't match `Extruder EXTR01` (Jaccard 1/3) → proposed `bsk:asset:Extruder-01`; fixable in review (P5 UI). Kept matching conservative on purpose.
- Extraction time depends on DGX load (≈ 2–10 s per window when idle, minutes when busy).
- The 116 pending triples on the BSKLAB doc are left for review in the P5 UI (or delete them).
