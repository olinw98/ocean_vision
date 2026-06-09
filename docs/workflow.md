# FathomFollow — Project Workflow

How to run the project across machines, sessions, and agents without losing state.

## Two-machine split

| Role | Typical machine | What you do here | Needs `data/` + `runs/`? |
|------|-----------------|------------------|--------------------------|
| **Build** | GPU workstation | Implementation, training, integration gates, diary entries | Yes |
| **PM / docs** | Laptop or secondary | Workflow, specs, resume planning, test-only validation | No |

**Git carries:** source code, tests, fixtures, `implementation-spec.md`, `docs/baselines.json`, config YAMLs.

**Git ignores:** `data/`, `runs/`, `.venv/`, downloaded weights (`yolo11n.pt`). Metrics that matter for the plan live in committed files (`docs/baselines.json`, diary entries) — not in gitignored train outputs alone.

If you switch machines mid-phase, **pull first**, then either re-run the integration step or copy gitignored artifacts manually. Never assume `runs/detect/train-2/weights/best.pt` exists after a fresh clone.

## Session rituals

### Start (build machine)

1. `git pull`
2. Activate venv: `source .venv/bin/activate` (mac/linux) or `.venv\Scripts\activate` (Windows)
3. `ff-status` — confirms pytest, diary snapshot, local artifacts
4. Read `implementation-spec.md` → **Current State** and the latest diary `**next:**`
5. Tell the agent: *"Resume FathomFollow"* (triggers the resume skill) or paste the recommended next step

### End (build machine)

1. `pytest -q` — must be green before you leave
2. Append a diary entry if you completed or partially completed a step
3. Update **Current State** in place (timestamp, step, blockers, next action)
4. Update **Open Judgment Calls** if you made or resolved a `[JUDGMENT CALL]`
5. If you recorded new metrics, update `docs/baselines.json`
6. `git add` + commit: code, spec, baselines — **not** `data/` or `runs/`
7. `git push` so the other machine sees diary + baselines

Use the **fathomfollow-session-end** skill in Cursor to walk through steps 1–6.

### PM-only session (this machine)

Safe to work without GPU artifacts. Focus on: agent prompt, skills, rules, spec hygiene, `ff-status` improvements. Run `pytest` to validate code changes; skip download/train steps.

## What agents should read

| File | When |
|------|------|
| `agent-prompt.md` | Every implementation session — build contract |
| `implementation-spec.md` → Current State | Every session start |
| `implementation-plan.md` | Phase acceptance criteria only — **do not edit** |
| `docs/baselines.json` | Before/after integration measurements |
| `docs/workflow.md` | Multi-machine or session-boundary questions |

## Diary contract (quick reference)

- **Complete** — tests pass *and* manual gate done (if any)
- **Partial** — code/tests done; manual gate or measurement pending
- **Blocked** — external dependency missing; log `[BLOCKED]` + fallback

Living sections (**Current State**, **Open Judgment Calls**) are updated in place. The Implementation Log is append-only.

Run `ff-status --check-diary` to flag a stale or missing Current State block.

## Integration measurements → git

When you finish a manual gate (baseline, train, ablation):

1. Write outputs under `runs/` or `data/` (local, gitignored)
2. Copy the numbers that matter into `docs/baselines.json` and/or the diary entry
3. Commit `docs/baselines.json` + `implementation-spec.md` with the code that produced them

Example: Step 1.3 firing rate lives in `docs/baselines.json` → `ablation_target_firing_rate`, not only in `runs/pre_gs_baseline_batho.json`.

## Blocker escalation

If a blocker survives the session, the agent must surface:

```
⛔ UNRESOLVED BLOCKER: [description]
Needs: [what the user must provide]
Fallback in use: [substitute]
```

And list it under **Current State** → Active blockers. HoloOcean and GS conda are standing blockers until resolved on the build machine — see `docs/holoocean_install.md` and `docs/gs_setup.md`.

## Quick commands

```bash
ff-status                  # human-readable project snapshot
ff-status --json           # machine-readable (for agents)
ff-status --check-diary    # warn if Current State is missing/stale
pytest -q                  # full suite (61 tests)
```
