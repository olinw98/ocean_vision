---
name: fathomfollow-session-end
description: >-
  End a FathomFollow session cleanly: run tests, update Current State and diary,
  check baselines, and prepare a commit on the build machine. Use when wrapping
  up, signing off, ending session, or before switching machines on ocean_vision.
---

# FathomFollow Session End

Close the loop so the next session (or machine) can resume without guesswork.

## When to use

- User says they are done for now, signing off, ending session, or switching machines
- After completing or partially completing an implementation step
- Before `git commit` on the **build machine**

## Steps (in order)

1. **Run tests**
   ```bash
   pytest -q --tb=no
   ```
   If failing, fix or log a diary entry for the fix before proceeding.

2. **Run status + diary check**
   ```bash
   ff-status --check-diary
   ```
   Resolve any warnings before ending.

3. **Update living sections** in `implementation-spec.md`:
   - **Current State** — refresh Last updated, Last completed step, Test suite count, Active blockers, Next action
   - **Open Judgment Calls** — add new rows or strike through resolved items

4. **Append diary entry** if the session changed implementation state (use `<!-- DIARY_ENTRY -->` format from `agent-prompt.md`).

5. **Update `docs/baselines.json`** if any integration measurement changed (firing rate, mAP, ablation targets).

6. **Commit guidance** (do not commit unless user asks):
   - Include: `src/`, `tests/`, `implementation-spec.md`, `docs/baselines.json`, config, fixtures in git
   - Exclude: `data/`, `runs/`, `.venv/`
   - Suggest a one-line commit message focused on *why*

7. **Remind push** if work happened on the build machine and origin should stay in sync.

## Output format

```markdown
# Session wrap-up

## Done this session
- [bullet list]

## Spec updates
- Current State: updated / needs update
- Diary entry: appended / none
- Baselines: updated / unchanged

## Git
- Suggested commit message: "..."
- Reminder: push so other machine can pull

## Next session (build machine)
[One sentence from Current State Next action]
```

## Rules

- Do not commit or push unless explicitly asked.
- Do not edit `implementation-plan.md`.
- On a **PM-only machine** (no GPU artifacts), skip artifact-dependent steps; still update spec if workflow docs changed.
- See `docs/workflow.md` for multi-machine roles.
