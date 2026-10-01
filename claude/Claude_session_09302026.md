# Claude Session Log — 09/30/2026

- **Repo:** `BSK.AI.Local_LLM_neo-django-api`
- **Branch:** `main`
- **Starting commit:** `38f3c1c` — fix: chunker handles base64 images, empty section_path, and large tables

## Changes

| # | Change | Files | Commit |
|---|--------|-------|--------|
| 1 | Created session log | `claude/Claude_session_09302026.md` | same commit as this log entry (branch `chore/gitignore-data-gitkeep`) |
| 2 | Fixed `data/` ignore rule so the `.gitkeep` exceptions take effect (`data/` → `data/**` + `!data/**/`) | `.gitignore`, `data/incoming/.gitkeep`, `data/processed/.gitkeep` (now visible to git, untracked) | same commit as this log entry (branch `chore/gitignore-data-gitkeep`) |

## Notes

### Repo state check (start of session)

- Working tree clean; `main` 0 ahead / 0 behind `origin/main` after a fresh fetch.
- No other local branches, stashes, extra worktrees, submodules, or in-progress merge/rebase.
- Untracked-by-design (ignored): `.env`, `.venv/`, `db.sqlite3`, `data/`, `:memory:/`, Python caches.

### `.gitignore` `data/` fix (change 2)

- Problem: `data/` ignored the whole directory, and git does not descend into an ignored directory, so `!data/incoming/.gitkeep` and `!data/processed/.gitkeep` never applied.
- Fix: `data/**` ignores everything under `data/`, `!data/**/` re-includes the directories so git descends into them, and the two `.gitkeep` exceptions now match.
- Verified with `git check-ignore -v`: both `.gitkeep` files are un-ignored; the PDFs in `data/incoming/` and `data/failed/`, the JSONL in `data/processed/`, and files in any new subdirectory remain ignored.

### Open items

- `data/failed/` has no `.gitkeep`, so it will not exist on a fresh clone (only `incoming/` and `processed/` are kept).
