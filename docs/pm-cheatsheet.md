# FathomFollow — PM cheat sheet

One-page reference for you (human PM). Agents use `agent-prompt.md`, `docs/workflow.md`, and the **fathomfollow-daily-workflow** Cursor skill.

## Which machine?

| Machine | Role | You do |
|---------|------|--------|
| **Windows GPU** | Build | Train, integrate, diary entries, push git |
| **Mac** | PM / code | Spec, workflow, tests, planning — no `data/` required |

## Session start (5 min)

1. `git pull`
2. Activate venv (Windows: `.venv\Scripts\activate` · Mac: `source .venv/bin/activate`)
3. `ff-status`
4. Open `implementation-spec.md` → **Current State**
5. **PM chat:** *"Resume FathomFollow"* (recap) · **Builder chat:** paste **Next action** as the task

**Which chat?** PM = plan/review/explain · Builder Agent = implement/train/fix

Cursor skill with full steps: **fathomfollow-daily-workflow**

## Session end (build machine)

1. `pytest -q` — green
2. `ff-status --check-diary` — fix warnings
3. Update **Current State** + **Open Judgment Calls** if needed
4. Append diary entry if step changed
5. Update `docs/baselines.json` if a number changed
6. Commit: code, spec, baselines — **not** `data/` or `runs/`
7. `git push`

Cursor: use **fathomfollow-session-end** skill.

## When to intervene (you, not the agent)

| Situation | Your call |
|-----------|-----------|
| `[JUDGMENT CALL]` in diary | Approve, reject, or redirect |
| `[BLOCKED]` HoloOcean / GS conda | Install, accept fallback, or pause |
| Step marked **Complete** without a manual gate | Push back — should be **Partial** |
| Agent wants to edit `implementation-plan.md` | No — plan is frozen |
| Null GS ablation result | Accept — honest result is success for the experiment |

## What git carries vs local

| In git (shared) | Local only (GPU machine) |
|-----------------|--------------------------|
| Code, tests, fixtures | `data/` datasets |
| `implementation-spec.md` | `runs/` weights, train outputs |
| `docs/baselines.json` | `.venv/` |

## Phase at a glance

| Phase | Plain status | Next gate |
|-------|--------------|-----------|
| 0 | Complete (HoloOcean smoke on build machine) | — |
| 1 | Complete on GPU | — |
| 1.5 | Complete — real GS ablation recorded regression | Optional larger render batch |
| 2 | Complete — drift margin **1.29 m** within dropout (holoocean_smoke) | — |
| 3–4 | Complete — hero run 68% retention, 1.29 m drift margin | Optional `--live` hero |

## Key numbers (committed)

- Taxon: **Bathochordaeus**
- Pre-GS HoloOcean firing_rate: **0.58** (train-2); proxy **2.12**
- GS real ablation: regression on both fixtures (see `docs/baselines.json`)
- Tests: run `pytest -q` (expect 73+ passing)

## Quick commands

```bash
ff-status
ff-status --check-diary
ff-status --json
pytest -q
ff-drift-gate --fixture fixtures/sim/holoocean_smoke.npz --scenario config/scenario_holoocean.yaml
```

## Repo map

| File | Purpose |
|------|---------|
| `implementation-plan.md` | Scope (read-only for agents) |
| `agent-prompt.md` | Agent contract |
| `implementation-spec.md` | Progress diary + Current State |
| `docs/workflow.md` | Multi-machine detail |
| `docs/notion-setup.md` | How to use Notion alongside git |
