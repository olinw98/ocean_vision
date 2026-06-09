---
name: fathomfollow-resume
description: >-
  Resumes FathomFollow work after a break by reading the spec diary, baselines,
  and live repo state, then summarizing progress, blockers, and next steps.
  Use when the user says they are back, resuming, catching up, wants a reminder,
  review, recap, status update, where we left off, or what to do next on
  ocean_vision / FathomFollow.
---

# FathomFollow Session Resume

Orient the user (and yourself) before coding. **Read live state** — never quote hardcoded metrics from memory.

## Read first (in order)

1. `agent-prompt.md` — build contract, TDD, diary rules
2. `implementation-spec.md` — **Implementation Log** (latest `DIARY_ENTRY` blocks) + Revised Blueprint blockers
3. `docs/baselines.json` — recorded metrics and ablation targets
4. `implementation-plan.md` — **read only** for phase names/acceptance; **do not edit**

Optional if relevant: `docs/holoocean_install.md`, `docs/gs_setup.md`, `README.md`

## Verify live state (run commands)

```bash
cd <project-root>
git pull                    # if user may have pushed from another machine
git status -sb
git log --oneline -5
ff-status                   # pytest + diary + artifacts (preferred)
# or: pytest -q --tb=no
```

Activate venv first if needed: `source .venv/bin/activate` (mac/linux) or `.venv\Scripts\activate` (Windows).

**Multi-machine:** See `docs/workflow.md`. Gitignored `data/` and `runs/` may be MISSING on a PM-only machine — that is expected. Trust committed `docs/baselines.json` + diary over missing local weights.

| Path | What it proves |
|------|----------------|
| `data/fathomnet_raw/Bathochordaeus/images/` | Phase 1.1 download |
| `data/fathomnet_batho/manifest.json` | Reorganized YOLO splits |
| `data/fathomnet_batho/metrics.json` | Phase 1.2 train metrics |
| `runs/detect/train-2/weights/best.pt` | Latest detector weights (local) |
| `data/nav_model/velocity_estimator.pt` | Nav model (local) |
| `fixtures/sim/fathomnet_proxy.npz` | Step 1.3 proxy fixture |

`ff-status` reports all of the above. For manual counts: `find data/... -type f | wc -l` (mac/linux) or `(Get-ChildItem <dir> -File).Count` (Windows).

## Classify each implementation-plan phase

For Phases 0, 1, 1.5, 2, 3, 4 — mark each as:

- **Complete** — diary says complete + tests/artifacts support it
- **Partial** — code exists but manual gate or measurement pending
- **Blocked** — documented blocker (HoloOcean, GS conda, etc.)
- **Not started**

Known standing blockers (confirm still true; do not assume):

- **HoloOcean**: Epic EULA + clone; PyPI client incompatible with Python 3.13 → see `docs/holoocean_install.md`
- **GS / Phase 1.5**: `water_splatting` conda env not set up → see `docs/gs_setup.md`

## Output format (use every time)

```markdown
# FathomFollow — Session Resume

## Executive summary
[2–3 sentences: biggest win since last session + where we are on the plan + top blocker]

## Phase status
| Phase | Status | Notes |
|-------|--------|-------|
| 0 — Scaffolding | … | … |
| 1 — Perception | … | … |
| 1.5 — GS augmentation | … | … |
| 2 — Navigation | … | … |
| 3 — Follow control | … | … |
| 4 — Integration | … | … |

## Key numbers (from disk)
| Metric | Value | Source |
|--------|-------|--------|
| Tests | … | pytest |
| Latest commit | … | git log |
| Selected taxon | … | baselines.json / diary |
| mAP50 / mAP50-95 | … | data/fathomnet_batho/metrics.json |
| Pre-GS ablation target | … | docs/baselines.json `ablation_target_firing_rate` |

## Git state
- Branch: …
- Clean / uncommitted: …
- Synced with origin: …

## Local artifacts (not in git)
- [ ] Bathochordaeus images: N
- [ ] Detector weights: path or MISSING
- [ ] Nav checkpoint: path or MISSING

## Blockers
- [List only verified blockers with doc links]

## Recommended next step (one task)
[Single concrete task for this session + why it unblocks the plan]

## Optional: paste-ready builder prompt
[Only if user may want to implement next — one bounded Agent prompt]
```

## Behavior rules

- **Do not implement** unless the user asks to start building.
- **Do not edit** `implementation-plan.md`.
- **Do not commit or push** unless explicitly asked.
- Explain in plain English; user may be new to Cursor ("vibecoder").
- If `git pull` may be needed (user merged elsewhere), suggest `git pull` before summarizing.
- If diary lags behind code, say so and trust git + baselines over stale diary lines.

## Phase-order "what's next" logic

Pick the **first incomplete unblocked** step:

1. Phase 0.3 HoloOcean smoke (if unblocked)
2. Phase 1 gaps (Bathochordaeus retrain, HoloOcean baseline refresh)
3. Phase 1.5: GS env → `ff-gs train` → `ff-gs render` → merge → retrain → ablation vs `ablation_target_firing_rate`
4. Phase 2 drift margin on held-out trajectories
5. Phase 3 target mimic in live sim
6. Phase 4 consolidated `ff-run` + `ff-eval` report

If conda/GPU/HoloOcean unavailable, recommend the highest-value **offline** step (RecordedGSRenderer render batch, proxy baseline refresh, workflow/diary hygiene on a PM machine).

## Related

- Session end: **fathomfollow-session-end** skill
- Multi-machine roles: `docs/workflow.md`

## Trigger phrases

Activates on: "I'm back", "resume", "catch me up", "remind me", "review progress", "what have we done", "what's next", "where did we leave off", "session resume".
