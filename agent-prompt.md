# Agent Instruction Book: FathomFollow

**Target Agent:** Cursor (Claude Sonnet, Agent Mode) — adapt file-reference syntax if using Claude Code or another agent.
**Blueprint file:** `implementation-plan.md` (rev. 2)
**Spec diary file:** `implementation-spec.md` *(you will create and maintain this)*
**Date issued:** 2026-06-08

---

## How to Use This Book

Read this document in full before doing anything. Then execute the four chapters in order. Do not skip ahead. Do not write any implementation code until Chapter 2 is complete.

This book is your contract. If you hit ambiguity not resolved by the blueprint or this book, make the most conservative reasonable decision and log it in the spec diary with a `[JUDGMENT CALL]` tag for developer review.

Three project-specific standing rules:
1. **The simulator is not a test dependency.** All logic must be testable via `RecordedSimEnv` against recorded fixtures. Never write a unit test that requires launching HoloOcean/Unreal.
2. **The Gaussian splat is not a test dependency.** GS reconstruction and rendering are GPU-bound and slow; unit tests use `RecordedGSRenderer` against a small pre-rendered fixture. Training a splat or running the real renderer is a manual integration step, logged in the diary, never in the pytest suite.
3. **Never feed ground truth into a controller or estimator.** `gt_pose` / `gt_target_pose` exist only for the evaluation harness. Reading ground truth outside `src/fathomfollow/eval/` is a correctness bug — stop and log `[JUDGMENT CALL]`.

Architectural reminder that governs the whole build: **HoloOcean owns physics and sensors; the Gaussian splat owns appearance only.** In v1 the splat is used offline to generate training imagery and never renders inside the live control loop. Keep them behind their interfaces (`SimEnv`, `GSRenderer`) so the in-loop renderer remains a clean future drop-in.

---

## Resuming Mid-Project

If `implementation-spec.md` already exists, this is a resumption session. Before doing anything else:

1. `git pull` if you may have pushed from another machine (see `docs/workflow.md`).
2. Read the `## Current State` section at the top of `implementation-spec.md` to understand where the last session ended.
3. Run `ff-status` or `pytest` and confirm the suite is green. If tests are failing, treat fixing them as the first task before resuming planned work — log the fix as a diary entry.
4. Review the `## Open Judgment Calls` section and note any that require user sign-off before proceeding.
5. Read only the diary entries since the last `status: Complete` step to reorient; you do not need to re-read all prior entries.
6. Continue from the `**next:**` field of the most recent diary entry.

Do not start new implementation work until steps 1–6 are satisfied. At session end, follow `docs/workflow.md` and update **Current State** (or use the **fathomfollow-session-end** skill).

---

## The Spec Diary: `implementation-spec.md`

Create `implementation-spec.md` at the start of Chapter 1 and append throughout. Never overwrite prior content; only append — except for the two living sections below, which must be updated in place each session.

```
# Implementation Spec: FathomFollow
## Current State              ← UPDATED IN PLACE each session (see format below)
## Open Judgment Calls        ← UPDATED IN PLACE (see format below)
## Critical Review            ← Chapter 1
## Revised Blueprint          ← Chapter 2
## Implementation Log         ← Chapter 3 (append per step)
## Final Spec                 ← Chapter 4
```

### Living section: `## Current State`

Replace this section's content at the start and end of every session. Keep it short — it is an orientation snapshot for the next agent.

```markdown
## Current State

**Last updated:** {ISO-8601 timestamp}
**Last completed step:** {step number and name}
**Test suite:** {N}/{N} passing | last run: {timestamp}
**Active blockers:** {list of [BLOCKED] items, or "None"}
**Next action:** {one sentence — what to do first in the next session}
```

### Living section: `## Open Judgment Calls`

Append a row when a `[JUDGMENT CALL]` is made. Strike through and add a resolution note when a developer signs off or the situation resolves. This section is never deleted.

```markdown
## Open Judgment Calls

| Step | Timestamp | Decision | Status |
|------|-----------|----------|--------|
| {N}  | {ISO-8601} | {one-line summary} | Open / Resolved: {note} |
```

### Diary entry format (Implementation Log)

Wrap every entry in the tags exactly; do not rename fields — they are a machine-parseable contract.

```markdown
<!-- DIARY_ENTRY -->
### [{TIMESTAMP}] Step {N} — {Step Name}

**project:** FathomFollow
**step:** {N}
**phase:** {Phase Name}
**status:** Complete | Partial | Blocked
**files_touched:** comma-separated list
**tests_written:** comma-separated list
**tests_passing:** N/N
**summary:** 1–3 sentences of what actually happened
**tdd_cycle:** RED — [failing test] | GREEN — [passing impl] | REFACTOR — [cleanup or "none"]
**deviations:** None | [what changed and why]
**judgment_calls:** None | [decision + rationale, tag [JUDGMENT CALL]]
**blockers:** None | [describe, tag [BLOCKED]]
**next:** next step, or "Proceeding to Chapter 4"
<!-- /DIARY_ENTRY -->
```

Use real ISO-8601 timestamps. Write honest notes for your future self.

### Manual integration gates

Some steps include a manual integration gate (a real run that cannot be automated in CI). A step with a pending gate is marked `Partial`, not `Complete`. **You may continue to the next step past a `Partial`, but you must:**
- Record the gate as an active blocker in `## Current State`
- Note in the next step's diary entry that it depends on the pending gate
- Never mark a step `Complete` until both tests and the manual gate are logged

---

## Blocked Item Protocol

When you hit a `[BLOCKED]` condition:

1. Log it in the diary entry with a `[BLOCKED]` tag and a one-sentence description.
2. Add it to `## Current State` → Active blockers.
3. Define the minimum fallback that lets work continue without the blocked dependency (e.g. a stub, a proxy fixture, a synthetic substitute). Implement the fallback and log it as a `[JUDGMENT CALL]`.
4. **Timebox the block:** if the blocked dependency cannot be resolved within the current session, surface it to the user explicitly before ending the session:
   ```
   ⛔ UNRESOLVED BLOCKER: [description]
   Needs: [what the user must provide or decide]
   Fallback in use: [what is standing in]
   ```
5. Do not let a blocker silently carry across multiple sessions without surfacing it.

---

## Chapter 1 — Critical Review

**Objective:** Read the blueprint, find its weaknesses, write them down before building.

1. Read `implementation-plan.md` completely.
2. Create `implementation-spec.md`; add the header, `## Current State` (initial), `## Open Judgment Calls` (empty table), and `## Critical Review`.
3. **Run the dependency smoke sprint** before writing the critique: attempt to `import` or install each major dependency (HoloOcean, `ultralytics`, `fathomnet`, the chosen GS library). Record what actually installs vs. what fails. This takes five minutes and prevents CUDA/version surprises from surfacing phase-deep into the build.
4. Write the critique covering each, specifically:

**Assumption Audit** — List every implicit assumption, marked SAFE / RISKY / UNKNOWN. Give special attention to: (a) FathomNet-trained detector firing usably on rendered sim frames; (b) HoloOcean's DVL/sensor API names and shapes; (c) that a textured mimic resembles the taxon enough to be detected; (d) sim throughput for Phases 2–4; (e) **that GS-rendered imagery actually transfers to HoloOcean's look better than FathomNet alone — this is a hypothesis, not a given, and Phase 1.5's ablation is the test of it**; (f) **that the chosen GS source dataset ships usable camera poses and that the target organism can be represented in it.**

**Gap Analysis** — What's underspecified? Specifically: how the target mimic asset is created; the exact dropout rule; how body-frame velocity ground truth is derived from pose logs; nav↔controller coupling at Phase 4; **the GS label strategy (compositing a target organism into rendered scenes vs. annotating an existing one), and how GS camera paths are chosen to cover useful viewpoints.**

**Tech Stack Assessment** — Verify current install paths and compatibility for HoloOcean, Ultralytics + torch CUDA, fathomnet-py, and **the chosen underwater-GS implementation (WaterSplatting default / SeaSplat alt) — confirm it builds on this GPU/CUDA, and confirm the SeaThru-NeRF (or chosen) source dataset is obtainable with poses.** Flag version/CUDA conflicts before they bite. Cross-reference with the dependency smoke sprint results from step 3.

**Interface & Contract Risks** — Is `Command` sufficient for visual servoing and a future RL controller? Is `SimObservation` complete for both loops? **Is `GSRenderer` sufficient to later swap an offline render batch for an in-loop renderer without touching callers?** Where do the seams leak?

**Scope & Complexity Check** — Which phase is most likely to balloon? (Plan's bet: Phase 1.5 GS reconstruction setup and Phase 4 integration.) Is anything labeled simple that isn't? **Confirm GS stays offline in v1; flag any creep toward in-loop rendering.**

5. End with a **Severity Summary**:
```
BLOCKERS (must resolve before building): [list or "none"]
WARNINGS (should address in revision): [list]
NOTES (low-risk observations): [list]
```
6. Do not proceed until the Critical Review is fully written.

---

## Chapter 2 — Blueprint Revision

**Objective:** Resolve what the review found; produce the blueprint you'll actually build from.

1. Add `## Revised Blueprint`.
2. For each BLOCKER and WARNING write: **Issue / Resolution / Changed: X → Y / Rationale.**
3. If a BLOCKER needs user input, write it up and **stop** (follow the Blocked Item Protocol above):
```
⛔ UNRESOLVED BLOCKER: [description]
Needs: [information/decision required]
```
Likely candidates to surface here: (a) the target taxon (Open Question 1) — verify the FathomNet concept and box count, blocks Phase 1; (b) **the GS source scene and whether the target organism appears in it or must be composited (Open Question 2) — blocks Phase 1.5; verify the dataset is downloadable with poses before committing.**
4. Produce a **Revised Architecture Summary** — the condensed final design, ground truth from here on.
5. Do not proceed until all BLOCKERs are resolved.

---

## Chapter 3 — Implementation

**Objective:** Build the revised blueprint phase by phase, strict TDD (red → green → refactor), documenting each step.

Per step: write failing tests first → confirm red (log the failure) → minimum implementation → confirm green → refactor → re-run → write the diary entry. A step is complete only when tests pass and the diary entry is written.

### Discipline rules
- **Run `pytest` at the start of every session** to confirm the inherited state before touching anything.
- TDD is non-negotiable; tests precede implementation every step.
- Never modify a test to make it pass; if a test is wrong, surface `[TEST QUESTION]`.
- Red must precede green.
- One step at a time; no speculative work from later phases.
- Log genuine blockers with `[BLOCKED]` and follow the Blocked Item Protocol.
- `implementation-spec.md` body is append-only; `## Current State` and `## Open Judgment Calls` are updated in place.
- Update `## Current State` at the end of every session, even if no step was completed.

### The steps

**Phase 0 — Environment & Scaffolding**

- **Step 0.1 — Repo skeleton & config.** Tests: each package module imports; pydantic scenario/training/render config models reject malformed YAML and accept valid samples. Build: layout per plan §7, `pyproject.toml` with pinned deps, config models. Files: `pyproject.toml`, `src/fathomfollow/config/*`, `tests/test_config.py`. Acceptance: config round-trips; imports succeed.
- **Step 0.2 — `SimEnv` protocol + `RecordedSimEnv`.** Tests: replay a checked-in fixture, yield well-formed `SimObservation`s (shapes/dtypes/flags); protocol methods honored. Build: `SimEnv`, `SimObservation`, `RecordedSimEnv`, a synthetic fixture. Files: `src/fathomfollow/sim/{base,recorded}.py`, `fixtures/`, `tests/test_recorded_sim.py`. Acceptance: tests pass, no Unreal.
- **Step 0.3 — `HoloOceanSimEnv` smoke (integration).** Tests: unit-test the raw-sensor→`SimObservation` mapping from a mock (no live sim). Build: `HoloOceanSimEnv`; manual `scripts/smoke_sim.py` that steps real HoloOcean and records a fixture. Run once; log results incl. actual HoloOcean version + sensor API. Files: `src/fathomfollow/sim/holoocean_env.py`, `scripts/smoke_sim.py`, `tests/test_obs_mapping.py`. Acceptance: mapping tests pass; manual smoke logged. *Manual gate — step may be Partial until HoloOcean is installed.*

**Phase 1 — Perception (baseline)**

- **Step 1.1 — FathomNet data pipeline.** Tests: box→YOLO conversion correct on hand-built samples (normalization, class mapping); split-by-hash deterministic; manifest written. Build: `ff-data prepare` (fathomnet-py, download cache, offline test mode). Files: `src/fathomfollow/data/*`, `tests/test_data_pipeline.py`. Acceptance: conversion/split tests pass offline; small live download yields valid `data.yaml` + manifest (logged).
- **Step 1.2 — Detector wrapper + training.** Tests: wrapper returns well-formed `DetectionRecord`s on a fixture image; confidence filtering works; metrics parser writes `metrics.json`. Build: `Detector` around Ultralytics, `ff-train-detector`. Files: `src/fathomfollow/perception/detector.py`, `cli.py`, `tests/test_detector.py`. Acceptance: wrapper tests pass; short training records mAP (set the recorded floor).
- **Step 1.3 — Detector on sim frames (pre-GS baseline).** Tests: sim-frame inference path produces well-formed detections on a recorded sim fixture. Build: glue to run the detector over `RecordedSimEnv` frames + firing-rate report. Files: `src/fathomfollow/perception/sim_infer.py`, `tests/test_sim_infer.py`. Acceptance: tests pass; **sim-frame firing rate recorded in the diary as the pre-GS baseline — Phase 1.5 must beat this number.**

**Phase 1.5 — Gaussian-Splatting Augmentation**

- **Step 1.5.1 — `GSRenderer` interface + `RecordedGSRenderer`.** Tests: `RecordedGSRenderer.render(pose, turbidity)` returns a well-formed RGB array from a small pre-rendered fixture; interface methods honored; turbidity outside [0,1] rejected. Build: `GSRenderer` Protocol, `RecordedGSRenderer`, a tiny fixture of pre-rendered frames. Files: `src/fathomfollow/gs/{base,recorded}.py`, `fixtures/gs/`, `tests/test_gs_renderer.py`. Acceptance: tests pass, no GPU/GS train needed. *(Build this first so everything downstream is testable without a real splat.)*
- **Step 1.5.2 — GS reconstruction (integration).** Tests: `GSScene` manifest writer validates and round-trips; the source-dataset loader parses poses correctly on a fixture. Build: `WaterSplattingGSRenderer` (or chosen lib) implementing `GSRenderer`; `ff-gs train` to reconstruct a scene from the posed source dataset. Run the real reconstruction manually; log `train_psnr` and a visual check. Files: `src/fathomfollow/gs/watersplatting.py`, `cli.py`, `tests/test_gs_manifest.py`. Acceptance: manifest/loader tests pass; manual reconstruction reaches a recorded render-quality floor (logged). If the GS library or dataset won't install/download, follow the Blocked Item Protocol immediately. *Manual gate — step may be Partial until GS env is set up.*
- **Step 1.5.3 — Turbidity-swept labeled render pipeline.** Tests: given a camera path + turbidity list, the render driver produces one labeled frame per (pose, turbidity) with YOLO labels; **the same pose at higher turbidity is deterministically different from lower turbidity (and identical given the same seed/params)**; `GSRenderManifest` written; labels are well-formed. Use `RecordedGSRenderer` so tests need no GPU. Build: `ff-gs render` with `--label-strategy`, camera-path config, label writer. Files: `src/fathomfollow/gs/render_pipeline.py`, `src/fathomfollow/gs/labeling.py`, `cli.py`, `tests/test_gs_render_pipeline.py`. Acceptance: tests pass on the recorded renderer; a manual real render batch produces a labeled dataset (logged).
- **Step 1.5.4 — Merge + retrain + ablation (the payoff).** Tests: `ff-data merge` combines FathomNet + GS sources with provenance tags and fresh deterministic splits; no leakage across splits; combined `data.yaml` valid. Build: `ff-data merge`; retrain the detector on the combined set. Files: `src/fathomfollow/data/merge.py`, `cli.py`, `tests/test_merge.py`. Acceptance: merge tests pass; **retrained detector's sim-frame firing rate is recorded and compared to the Step 1.3 baseline. Improvement → log the margin. No improvement → log it honestly as a null result with a short analysis (turbidity range? scene mismatch? label strategy?).** Do not tune until it looks good and hide the path; the ablation is a result either way.

**Phase 2 — Navigation**

- **Step 2.1 — Dropout injector + trajectory logger.** Tests: injector flips `dvl_valid` per the configured rule deterministically; logger writes replayable trajectories with body-frame velocity ground truth from consecutive poses. Build: dropout injector wrapping any `SimEnv`; trajectory recorder. Files: `src/fathomfollow/nav/dropout.py`, `src/fathomfollow/nav/trajectories.py`, `tests/test_dropout.py`, `tests/test_trajectories.py`. Acceptance: tests pass on fixtures.
- **Step 2.2 — Dead-reckoning baseline.** Tests: integrating a known constant velocity yields the analytic path; with valid DVL, baseline tracks GT within tolerance on a fixture. Build: `DeadReckoning` + optional `filterpy` EKF. Files: `src/fathomfollow/nav/deadreckon.py`, `tests/test_deadreckon.py`. Acceptance: tests pass; baseline drift on a dropout fixture recorded as the number to beat.
- **Step 2.3 — DriftGuard estimator.** Tests: forward-pass output shape correct; tiny overfit set drives loss down; `estimate()` handles `dvl=None`. Build: GRU/TCN `VelocityEstimator`, `ff-train-nav`, training loop. Files: `src/fathomfollow/nav/estimator.py`, `cli.py`, `tests/test_estimator.py`. Acceptance: tests pass; held-out drift within dropout windows beats the Step 2.2 baseline by a recorded margin.

**Phase 3 — Tracking & Follow Control**

- **Step 3.1 — Tracker wrapper.** Tests: stable IDs across a synthetic sequence; track survives a gap up to `max_gap` then drops; single active-target selection deterministic. Build: `Tracker` around ByteTrack/`supervision`. Files: `src/fathomfollow/perception/tracker.py`, `tests/test_tracker.py`. Acceptance: tests pass.
- **Step 3.2 — Visual-servoing controller.** Tests: centroid right-of-center → positive yaw; bbox over target band → reduced/negative forward vel; vertical error → vertical command; `active=None` → safe default. Deterministic, no sim. Build: PID `FollowController` emitting `Command`. Files: `src/fathomfollow/control/visual_servo.py`, `tests/test_controller.py`. Acceptance: tests pass on synthetic tracks.
- **Step 3.3 — Target mimic + follow integration (integration).** Tests: scenario config validates; mimic trajectory generator deterministic. Build: scripted target mimic, wire detector→tracker→controller→sim. Manual in-sim run; log target-in-frame fraction. Files: `src/fathomfollow/sim/target.py`, `config/scenario.yaml`, `tests/test_scenario_config.py`. Acceptance: config tests pass; manual run's in-frame fraction recorded. *Manual gate — step may be Partial until HoloOcean is installed.*

**Phase 4 — Integration & Evaluation**

- **Step 4.1 — Evaluation harness.** Tests: drift metrics on a known log match hand-computed values; dropout-window slicing correct; tracking-retention correct on a synthetic log; domain-split mAP table correct; GS ablation comparison renders. Build: `eval/metrics.py`, `eval/report.py`, `ff-eval` with `--ablate-gs`. Files: those + `tests/test_metrics.py`. Acceptance: metric tests pass; `report.md` generates from a recorded run, including the GS firing-rate ablation.
- **Step 4.2 — End-to-end `ff-run`.** Tests: orchestrator driven by `RecordedSimEnv` produces a complete run log with both loops populated. Build: `ff-run` closing both loops with dropout active; nav→controller coupling per Open Question (default parallel-eval; document if closed). Files: `cli.py`, `src/fathomfollow/run.py`, `tests/test_run_orchestration.py`. Acceptance: orchestration test passes on recorded env; one manual live end-to-end run produces a consolidated `report.md` showing drift-within-dropout (learned vs baseline) while the target is actively followed. Log results; proceed to Chapter 4.

*(Stretch, only if base metrics are met — each its own TDD step with diary entries: (a) online GS rendering implementing `GSRenderer` against the live loop, co-registered to HoloOcean's camera pose/intrinsics, with the moving target handled by dynamic-GS or a composited target splat; (b) RL follow controller behind the `FollowController` interface.)*

---

## Chapter 4 — Spec Finalization

**Objective:** Leave a complete, accurate record of what was built.

1. Run `ff-eval --run runs/<final> --baseline dead_reckoning --ablate-gs` and confirm the output satisfies the success criterion: the AUV kept the target in frame for ≥ the recorded floor fraction of steps, and drift-within-dropout under the learned estimator is lower than the dead-reckoning baseline. Record the output verbatim in the Final Spec. If the criterion is not met, document the gap honestly — do not omit or soften the numbers.
2. Add `## Final Spec`.
3. Write:
- **Actual File Structure** — real tree.
- **Actual Dependencies** — pinned versions exactly from `pyproject.toml` (note resolved `holoocean`, `torch`+CUDA, `ultralytics`, `fathomnet`, the GS implementation, and the GS source dataset used).
- **Actual API / Interface Contracts** — real CLI signatures and the `SimEnv`/`GSRenderer`/`Detector`/`Tracker`/`FollowController`/`VelocityEstimator` signatures as implemented.
- **Deviation Log** — table of every difference from the Revised Blueprint with reasons; or "None".
- **Known Issues / Future Work** — domain-gap state and the GS ablation outcome, GS turbidity-model and pose-source caveats, dropout-model realism, in-loop GS rendering and RL left for v2.
4. One-paragraph closing note: what was built, whether it met the success criterion (followed target + lower drift-within-dropout than baseline), whether GS augmentation moved the sim-frame firing rate, and what a second pass would change.

---

## Closing

When Chapter 4 is complete, `implementation-spec.md` is the canonical record of FathomFollow. Build carefully. Keep the simulator and the splat out of the test suite. Never let ground truth leak into the estimators. Report the GS ablation honestly whichever way it falls. Document honestly.
