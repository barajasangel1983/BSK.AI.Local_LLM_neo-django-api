# Claude Session Log — 10/27–11/07/2026 — P27 automated review (P27a/b/c)

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p27-assisted-review` (off `main` @ `2a5e8a7`)
- **Plans:** frontend repo `claude/P27_assisted_review_plan.md`, `claude/P27c_apply_and_undo_plan.md`; session log entries 226–237 in `claude/Claude_session_09302026.md`. Paired with frontend `feat/p27c-doubtful-review`.
- **Why:** reviewing 1,200+ Safety facts one by one "will take forever" (Angel, 2026-11-06). Code checks and the Lab Ring's own model decide; automatic approvals must be right ≥ 95 % on the answer key (R1); doubtful facts go to Angel (R2); every batch can be undone (R4).

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | Review assistant: code checks (exact quote, repeats, hazard follow-up), 3-vote judge, tiers auto / doubtful / reject / leave with a reason | `context_graph/review_assist.py` |
| 2 | `manage.py review_replay` (read-only replay on decided or pending facts, `--safety-pending`, `--resume`) | `context_graph/management/commands/review_replay.py` |
| 3 | `review_tier`, `review_reason`, `review_batch` on `CandidateTriple`; `ReviewBatch` model (before/after per edited triple) | `context_graph/models.py`, `context_graph/migrations/0015_p27c_review_batches.py` |
| 4 | Apply and undo: `manage.py review_apply` (report + sheet, one manual per batch, stale rows skipped), `manage.py review_batch --undo` | `context_graph/review_batches.py`, `context_graph/management/commands/review_apply.py`, `context_graph/management/commands/review_batch.py` |
| 5 | `approve(automatic=True)` skips only the one-at-a-time rule; graph fixes found in rehearsal (section cleanup scope, existing evidence links kept, family documentation link removed on undo) | `context_graph/triples.py` |
| 6 | `review_tier` filter in the triples API | `context_graph/extraction_views.py` |
| 7 | Tests | `context_graph/tests/tests_review_assist.py`, `context_graph/tests/tests_review_batches.py` |

## Verification

- Answer key: automatic approvals right **98.3 %** (R1 ≥ 95 %), recall 85.5 %.
- Rehearsal on a scratch Neo4j (copy of the live graph): all 19 manuals → 627 approved, 321 rejected, 3 doubtful; undo of all 19 leaves the graph identical (hash of every node, link and property).
- Tests: 790 of 792 (the 2 failures also fail on `main`).
- **Nothing applied to the live DB or graph.** P27d (first real batch) needs Angel's go; run `review_replay --resume` on the live DB first and export `LIBRARY_BASE` when running from a worktree.
