# Claude Session Log — 11/08/2026 — P24-2 (part 1): EDGE's features in the one answer engine

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `feat/p24-2-engine-features` (off `main` @ `60ccd58`)
- **Plan:** frontend repo `claude/P24_studio_edge_alignment_plan.md`, Revision 2, P24-2. Angel "go" on: the question
  agent, "nothing found", "which machine?" and the Context Service's ask in the engine (this part); pictures EDGE's way
  and one prompt with the standard editable part in Studio Settings (next part).

## Changes

| # | Change | Files |
|---|--------|-------|
| 1 | The question agent ported from EDGE (search topics, "did you mean", one question back, at most once, never blocks); the answer's model; names from the Context Service | `chat/question_agent.py` |
| 2 | `answer_context(search_queries=)`: the question as asked plus the agent's topics, each searched the same way, merged in turn; the closest documents when nothing passes; found counts | `chat/views.py` |
| 3 | `_answer_events`: one generator for both endpoints — no machine chosen → the one named, or `needs_choice`; agent; context; nothing found → where it looked (no model call); the model streams; saved; `status` events | `chat/views.py` |
| 4 | `POST /api/chat/` runs the same generator (409 `needs_choice`) | `chat/views.py` |
| 5 | Context Service `ask` answers through the engine (its own prompt removed) | `context_service/service.py` |
| 6 | `CHAT_QUESTION_AGENT` (default off; on in `.env`) | `neo_llm_api/settings.py` |
| 7 | Tests | `chat/test_stream.py`, `context_service/tests/tests_service.py` |

## Verification

- Suite 816 OK. Live (worktree, 8B, agent on): "G12OC" → "Did you mean G120C?" with no model call (1.4 s);
  "how do I install it?" → one question back (3.3 s); a two-topic question: the 8B kept one search (3.4 s for the agent).
- Evals with the agent on (8B): safety 10/10; hard 17 ✓ (19 before; the 3 that changed read the same: H20, H21 wording, U1 still abstains but names another part's torque); 20.8 s per answer vs 13.2 s.
