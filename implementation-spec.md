# Implementation Spec: FathomFollow

## Current State

**Last updated:** 2026-06-09T03:45:00Z
**Last completed step:** 1.3 — Pre-GS baseline (Bathochordaeus weights)
**Test suite:** 61/61 passing | last run: 2026-06-09 (GPU build machine)
**Active blockers:** HoloOcean (Epic EULA + GitHub install on Python 3.11; PyPI client fails on 3.13); `water_splatting` conda env not set up (Phase 1.5)
**Next action:** Phase 1.5 — set up GS conda env, run `ff-gs train` + `ff-gs render`, merge with FathomNet, retrain detector, ablation vs `ablation_target_firing_rate` 2.12 in `docs/baselines.json`

## Open Judgment Calls

| Step | Timestamp | Decision | Status |
|------|-----------|----------|--------|
| 0.3 | 2026-06-08T13:00:00Z | Accept synthetic `smoke.npz` until HoloOcean EULA install completed on target machine | Open |
| 1.5.2 | 2026-06-08T16:00:00Z | Stub render returns solid-color array when GS subprocess unavailable | Open |
| 3.1 | 2026-06-08T19:00:00Z | SimpleTracker meets v1 test contract; upgrade to ByteTrack when real detection sequences available | Open |
| 1.1 | 2026-06-08T23:30:00Z | Default auto-prepare to YOLO format; COCO path retained but known broken for some taxa in fathomnet-py 1.10 | Open |
| 1.2 | 2026-06-09T00:30:00Z | Use Benthocodon for first live train to unblock pipeline; swap to Bathochordaeus before final ablation | Resolved: Bathochordaeus retrain complete 2026-06-09T03:30:00Z |
| 0.3 / 1.3 | 2026-06-09T01:30:00Z | Proxy fixture uses real FathomNet RGB + synthetic IMU/DVL; honest interim baseline, not a HoloOcean substitute | Open |

## Critical Review

### Assumption Audit

| Assumption | Verdict | Notes |
|------------|---------|-------|
| FathomNet detector fires on HoloOcean frames | RISKY | Phase 1.3 baseline + Phase 1.5 ablation measure this |
| HoloOcean DVL/IMU API names and shapes | RISKY | DVL global-frame 3-vel; IMU up to 18-D — map in HoloOceanSimEnv |
| Textured mimic resembles taxon enough | UNKNOWN | Start with proxy mesh; iterate visually |
| Sim throughput for Phases 2–4 | RISKY | Headless for nav; rendered only for perception |
| GS imagery transfers better than FathomNet alone | UNKNOWN (hypothesis) | Step 1.5.4 ablation; null result valid |
| SeaThru-NeRF ships usable poses | SAFE (with nuance) | COLMAP sparse/0 + poses_bounds.npy bundled |
| Target organism appears in GS scene | RISKY | Composited-target likely; decide at Step 1.5.2 |
| Single Python env for HoloOcean + WaterSplatting + YOLO | RISKY | Dual-env: main 3.11 + water_splatting conda 3.8 |

### Gap Analysis

1. Taxon: auto-select at Phase 1 via `fathomnet-generate --count` (single class v1).
2. Target mimic: secondary agent on deterministic spline; proxy texture first.
3. DVL dropout: `dvl_valid = altitude > alt_min AND |pitch|,|roll| < tilt_max` + scripted windows in scenario YAML.
4. Body-frame velocity GT: finite-difference global displacement rotated into body frame via quaternion.
5. Nav↔controller: parallel-eval in v1; no nav feedback into controller.
6. GS labeling: strategy enum deferred to Step 1.5.2.
7. FathomNet→YOLO: custom COCO converter required.

### Tech Stack Assessment

- HoloOcean: GitHub install after EULA; pin version in Phase 0.3 smoke.
- Dual-environment: main fathomfollow venv (3.11); water_splatting conda (3.8) for GS subprocess.
- SeaSplat fallback if WaterSplatting install fails.

### Interface & Contract Risks

- `Command`: forward_vel, yaw_rate, vertical_vel sufficient for v1 hover AUV.
- `SimObservation.dvl`: body-frame 3-vec after HoloOcean mapping.
- `SimObservation.imu`: 6-D slice [accel_xyz, gyro_xyz] from full IMU vector.
- GT isolation: `gt_*` only in eval/ and trajectory generation.

### Scope & Complexity Check

Phase 1.5 and Phase 4 most likely to balloon. GS stays offline in v1.

### Severity Summary

```
BLOCKERS (must resolve before building): none for Phase 0
WARNINGS (should address in revision): dual Python env; DVL global→body; composited labeling; HoloOcean API verify; parallel-eval nav
NOTES: fathomnet-generate --count before taxon; record ablation numbers; headless sim for nav
```

## Revised Blueprint

### Issue: Single Python env for GS + main stack
**Resolution:** Dual env; `WaterSplattingGSRenderer` invokes subprocess or consumes pre-rendered checkpoints.
**Changed:** monolithic env → main 3.11 + gs conda 3.8
**Rationale:** WaterSplatting pins torch 2.1.2+cu118, Python 3.8.

### Issue: DVL body-frame assumption
**Resolution:** Rotate global DVL velocity to body in `HoloOceanSimEnv.map_observation`.
**Changed:** raw HoloOcean DVL → body-frame 3-vec in SimObservation
**Rationale:** DriftGuard trains on body-frame velocity.

### Issue: IMU dimensionality
**Resolution:** Extract 6-D [accel, gyro] slice; log full vector in fixtures.
**Changed:** imu(6) documented as slice, not raw HoloOcean vector
**Rationale:** Estimator input size matches plan.

### Issue: GS label strategy undecided
**Resolution:** Defer to Step 1.5.2; build `LabelStrategy` enum now.
**Changed:** TBD → composited_target default at integration gate
**Rationale:** SeaThru-NeRF scenes lack most taxa.

### Issue: Nav↔controller coupling
**Resolution:** parallel-eval in v1.
**Changed:** ambiguous → nav runs on sensor stream; controller actuates sim only
**Rationale:** Simpler integration; eval compares drift in same run windows.

### Issue: Target taxon unknown
**Resolution:** Auto-select by annotation count at Phase 1.1.
**Changed:** user pick → count gate on candidate concepts
**Rationale:** Easiest to train; single class v1.

### Revised Architecture Summary

Two loops (perception→control, navigation) share HoloOcean via `SimEnv`. Offline GS augments detector training only. Taxon auto-selected. GT eval-only. Tests use RecordedSimEnv + RecordedGSRenderer.

## Implementation Log

<!-- DIARY_ENTRY -->
### [2026-06-08T00:00:00Z] Step 0 — Plan adoption

**project:** FathomFollow
**step:** 0
**phase:** Planning
**status:** Complete
**files_touched:** implementation-spec.md
**tests_written:** none
**tests_passing:** N/A
**summary:** Critical review and revised blueprint written from coding plan. Beginning Phase 0 implementation.
**tdd_cycle:** RED — n/a | GREEN — n/a | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 0.1 repo skeleton and config
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T12:00:00Z] Step 0.1 — Repo skeleton & config

**project:** FathomFollow
**step:** 0.1
**phase:** Phase 0
**status:** Complete
**files_touched:** pyproject.toml, src/fathomfollow/config/*, tests/test_config.py
**tests_written:** tests/test_config.py
**tests_passing:** 6/6
**summary:** Package layout, pydantic config models (Scenario, Render, Training), YAML round-trip validation. CLI entry points registered.
**tdd_cycle:** RED — import/config validation tests | GREEN — config models + loaders | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 0.2 RecordedSimEnv
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T12:30:00Z] Step 0.2 — SimEnv protocol + RecordedSimEnv

**project:** FathomFollow
**step:** 0.2
**phase:** Phase 0
**status:** Complete
**files_touched:** src/fathomfollow/sim/{base,recorded}.py, tests/test_recorded_sim.py, tests/conftest.py
**tests_written:** tests/test_recorded_sim.py
**tests_passing:** 2/2
**summary:** SimObservation/SimEnv/Command protocols; RecordedSimEnv replays NPZ fixtures; write_sim_fixture helper for synthetic data.
**tdd_cycle:** RED — replay shape/dtype tests | GREEN — RecordedSimEnv | REFACTOR — none
**deviations:** Fixtures generated at test time via tmp_path rather than checked-in NPZ (addressed in Step 0.3 follow-up).
**judgment_calls:** None
**blockers:** None
**next:** Step 0.3 HoloOcean mapping
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T13:00:00Z] Step 0.3 — HoloOceanSimEnv smoke (integration)

**project:** FathomFollow
**step:** 0.3
**phase:** Phase 0
**status:** Partial
**files_touched:** src/fathomfollow/sim/holoocean_env.py, scripts/smoke_sim.py, tests/test_obs_mapping.py
**tests_written:** tests/test_obs_mapping.py
**tests_passing:** 3/3
**summary:** map_holoocean_state unit-tested with mock dict; IMU 18-D→6-D slice; DVL global→body rotation. HoloOcean not installed on dev machine — synthetic fixture written to fixtures/sim/smoke.npz instead of live smoke.
**tdd_cycle:** RED — mapping tests | GREEN — holoocean_env mappers + HoloOceanSimEnv | REFACTOR — none
**deviations:** Live HoloOcean smoke deferred; mapping tests use constructed state dict.
**judgment_calls:** [JUDGMENT CALL] Accept synthetic smoke.npz until HoloOcean EULA install completed on target machine.
**blockers:** HoloOcean package not in venv (ModuleNotFoundError on import).
**next:** Step 1.1 FathomNet data pipeline
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T14:00:00Z] Step 1.1 — FathomNet data pipeline

**project:** FathomFollow
**step:** 1.1
**phase:** Phase 1
**status:** Complete
**files_touched:** src/fathomfollow/data/{pipeline,__init__}.py, tests/test_data_pipeline.py, cli.py
**tests_written:** tests/test_data_pipeline.py
**tests_passing:** 3/3
**summary:** COCO→YOLO conversion, deterministic hash splits, manifest writer. ff-data prepare accepts offline COCO JSON.
**tdd_cycle:** RED — bbox/split/conversion tests | GREEN — pipeline.py | REFACTOR — none
**deviations:** Live fathomnet-generate download not yet run; offline COCO path only.
**judgment_calls:** None
**blockers:** None
**next:** Step 1.2 Detector wrapper
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T14:30:00Z] Step 1.2 — Detector wrapper + training

**project:** FathomFollow
**step:** 1.2
**phase:** Phase 1
**status:** Complete
**files_touched:** src/fathomfollow/perception/{detector,types}.py, tests/test_detector.py, tests/test_detector_metrics.py, cli.py
**tests_written:** tests/test_detector.py, tests/test_detector_metrics.py
**tests_passing:** 4/4
**summary:** YoloDetector wrapper + MockDetector for tests. parse_training_metrics extracts Ultralytics results dict; ff-train-detector writes real metrics.json keys.
**tdd_cycle:** RED — test_detector_metrics (2026-06-08 session 2) | GREEN — parse_training_metrics + metrics_from_train_results | REFACTOR — wired CLI
**deviations:** Short training run with recorded mAP floor not yet executed manually.
**judgment_calls:** None
**blockers:** None
**next:** Step 1.3 sim-frame baseline
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T15:00:00Z] Step 1.3 — Detector on sim frames (pre-GS baseline)

**project:** FathomFollow
**step:** 1.3
**phase:** Phase 1
**status:** Partial
**files_touched:** src/fathomfollow/perception/sim_infer.py, tests/test_sim_infer.py
**tests_written:** tests/test_sim_infer.py
**tests_passing:** 1/1
**summary:** run_sim_inference computes firing rate over RecordedSimEnv. Test uses MockDetector; pre-GS baseline number on real sim frames not yet recorded.
**tdd_cycle:** RED — firing rate test | GREEN — sim_infer.py | REFACTOR — none
**deviations:** Baseline firing rate diary entry pending manual run with trained YOLO weights.
**judgment_calls:** None
**blockers:** None
**next:** Step 1.5.1 GSRenderer interface
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T15:30:00Z] Step 1.5.1 — GSRenderer + RecordedGSRenderer

**project:** FathomFollow
**step:** 1.5.1
**phase:** Phase 1.5
**status:** Complete
**files_touched:** src/fathomfollow/gs/{base,recorded}.py, tests/test_gs_renderer.py, fixtures/gs/recorded/
**tests_written:** tests/test_gs_renderer.py
**tests_passing:** 3/3
**summary:** GSRenderer protocol, RecordedGSRenderer with turbidity tinting, write_gs_fixture helper. Checked-in GS fixture at fixtures/gs/recorded/.
**tdd_cycle:** RED — render shape/turbidity tests | GREEN — recorded.py | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 1.5.2 GS reconstruction
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T16:00:00Z] Step 1.5.2 — GS reconstruction (integration)

**project:** FathomFollow
**step:** 1.5.2
**phase:** Phase 1.5
**status:** Partial
**files_touched:** src/fathomfollow/gs/watersplatting.py, tests/test_gs_manifest.py, cli.py
**tests_written:** tests/test_gs_manifest.py
**tests_passing:** 2/2
**summary:** GSScene manifest round-trip; colmap pose loader on fixture. WaterSplattingGSRenderer adapter shells to conda but render/train are placeholders until water_splatting env exists.
**tdd_cycle:** RED — manifest/loader tests | GREEN — watersplatting adapter stub | REFACTOR — none
**deviations:** Real ns-train reconstruction not run; train_psnr logged as 0.0 placeholder.
**judgment_calls:** [JUDGMENT CALL] Stub render returns solid-color array when subprocess unavailable.
**blockers:** [BLOCKED] water_splatting conda env not set up on dev machine.
**next:** Step 1.5.3 render pipeline
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T16:30:00Z] Step 1.5.3 — Turbidity-swept labeled render pipeline

**project:** FathomFollow
**step:** 1.5.3
**phase:** Phase 1.5
**status:** Complete
**files_touched:** src/fathomfollow/gs/{render_pipeline,labeling}.py, tests/test_gs_render_pipeline.py, config/cam_path.yaml
**tests_written:** tests/test_gs_render_pipeline.py
**tests_passing:** 2/2
**summary:** ff-gs render driver produces labeled frames per (pose, turbidity); turbidity determinism verified on RecordedGSRenderer.
**tdd_cycle:** RED — label + turbidity tests | GREEN — render_pipeline.py | REFACTOR — none
**deviations:** Manual real render batch not yet logged.
**judgment_calls:** None
**blockers:** None
**next:** Step 1.5.4 merge + ablation
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T17:00:00Z] Step 1.5.4 — Merge + retrain + ablation

**project:** FathomFollow
**step:** 1.5.4
**phase:** Phase 1.5
**status:** Partial
**files_touched:** src/fathomfollow/data/merge.py, tests/test_merge.py, cli.py
**tests_written:** tests/test_merge.py
**tests_passing:** 2/2
**summary:** ff-data merge combines sources with provenance tags and fresh deterministic splits. Retrain + sim-frame ablation not yet run.
**tdd_cycle:** RED — merge leakage tests | GREEN — merge.py | REFACTOR — none
**deviations:** Ablation numbers pending Phase 1.3 baseline + GS render batch.
**judgment_calls:** None
**blockers:** None
**next:** Step 2.1 dropout + trajectories
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T17:30:00Z] Step 2.1 — Dropout injector + trajectory logger

**project:** FathomFollow
**step:** 2.1
**phase:** Phase 2
**status:** Complete
**files_touched:** src/fathomfollow/nav/{dropout,trajectories}.py, tests/test_dropout.py, tests/test_trajectories.py
**tests_written:** tests/test_dropout.py, tests/test_trajectories.py
**tests_passing:** 3/3
**summary:** DropoutSimEnv wraps SimEnv with forced windows; log_trajectory writes body-frame velocity GT from pose finite differences.
**tdd_cycle:** RED — dropout + velocity tests | GREEN — dropout.py, trajectories.py | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 2.2 dead reckoning
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T18:00:00Z] Step 2.2 — Dead-reckoning baseline

**project:** FathomFollow
**step:** 2.2
**phase:** Phase 2
**status:** Complete
**files_touched:** src/fathomfollow/nav/deadreckon.py, tests/test_deadreckon.py
**tests_written:** tests/test_deadreckon.py
**tests_passing:** 2/2
**summary:** DeadReckoning integrates body velocity to global position; constant-velocity analytic path test passes.
**tdd_cycle:** RED — integration tests | GREEN — deadreckon.py | REFACTOR — none
**deviations:** Baseline drift-within-dropout number on fixture not yet recorded in diary.
**judgment_calls:** None
**blockers:** None
**next:** Step 2.3 DriftGuard
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T18:30:00Z] Step 2.3 — DriftGuard estimator

**project:** FathomFollow
**step:** 2.3
**phase:** Phase 2
**status:** Complete
**files_touched:** src/fathomfollow/nav/{estimator,training}.py, tests/test_estimator.py, tests/test_nav_training.py, config/nav_train.yaml, cli.py
**tests_written:** tests/test_estimator.py, tests/test_nav_training.py
**tests_passing:** 5/5
**summary:** VelocityGRU + VelocityEstimator.estimate(); training loop via train_nav_estimator saves checkpoint + metrics.json. ff-train-nav wired to config YAML.
**tdd_cycle:** RED — test_nav_training (session 2) | GREEN — nav/training.py + CLI | REFACTOR — run logs include gt_pos
**deviations:** Held-out drift margin vs dead reckoning not yet recorded manually.
**judgment_calls:** None
**blockers:** None
**next:** Step 3.1 tracker
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T19:00:00Z] Step 3.1 — Tracker wrapper

**project:** FathomFollow
**step:** 3.1
**phase:** Phase 3
**status:** Complete
**files_touched:** src/fathomfollow/perception/tracker.py, tests/test_tracker.py
**tests_written:** tests/test_tracker.py
**tests_passing:** 3/3
**summary:** SimpleTracker with stable IDs, max_gap drop, deterministic active selection. ByteTrack deferred — supervision dep present but minimal tracker sufficient for v1 tests.
**tdd_cycle:** RED — track lifecycle tests | GREEN — tracker.py | REFACTOR — none
**deviations:** SimpleTracker instead of ByteTrack wrapper per plan.
**judgment_calls:** [JUDGMENT CALL] SimpleTracker meets v1 test contract; upgrade to ByteTrack when real detection sequences available.
**blockers:** None
**next:** Step 3.2 visual servo
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T19:30:00Z] Step 3.2 — Visual-servoing controller

**project:** FathomFollow
**step:** 3.2
**phase:** Phase 3
**status:** Complete
**files_touched:** src/fathomfollow/control/visual_servo.py, tests/test_controller.py
**tests_written:** tests/test_controller.py
**tests_passing:** 3/3
**summary:** PID FollowController: yaw from centroid error, forward from bbox size, safe default when active=None.
**tdd_cycle:** RED — synthetic track tests | GREEN — visual_servo.py | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 3.3 target mimic integration
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T20:00:00Z] Step 3.3 — Target mimic + follow integration

**project:** FathomFollow
**step:** 3.3
**phase:** Phase 3
**status:** Partial
**files_touched:** src/fathomfollow/sim/target.py, config/scenario.yaml, tests/test_scenario_config.py
**tests_written:** tests/test_scenario_config.py
**tests_passing:** 2/2
**summary:** Deterministic circle/spline target_position_at(); scenario YAML validates. HoloOcean mimic agent wiring and in-frame fraction not yet logged.
**tdd_cycle:** RED — config + trajectory tests | GREEN — target.py | REFACTOR — none
**deviations:** Target not yet spawned in live sim.
**judgment_calls:** None
**blockers:** HoloOcean not installed.
**next:** Step 4.1 evaluation harness
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T20:30:00Z] Step 4.1 — Evaluation harness

**project:** FathomFollow
**step:** 4.1
**phase:** Phase 4
**status:** Complete
**files_touched:** src/fathomfollow/eval/{metrics,report,run_eval}.py, tests/test_metrics.py, tests/test_eval_from_run.py, cli.py
**tests_written:** tests/test_metrics.py, tests/test_eval_from_run.py
**tests_passing:** 5/5
**summary:** Drift/retention/ablation metrics; evaluate_run reads nav_log + ctrl_log and writes report.md; ff-eval uses real log data instead of hardcoded placeholders.
**tdd_cycle:** RED — test_eval_from_run (session 2) | GREEN — run_eval.py + run.py gt_pos/target_in_frame fields | REFACTOR — CLI
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 4.2 ff-run orchestration
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T21:00:00Z] Step 4.2 — End-to-end ff-run

**project:** FathomFollow
**step:** 4.2
**phase:** Phase 4
**status:** Partial
**files_touched:** src/fathomfollow/run.py, tests/test_run_orchestration.py, cli.py, fixtures/sim/smoke.npz
**tests_written:** tests/test_run_orchestration.py
**tests_passing:** 1/1 (52/52 total suite)
**summary:** run_orchestration closes perception→control and nav loops on RecordedSimEnv; writes nav_log.json and ctrl_log.json. Uses MockDetector; manual live end-to-end + consolidated report pending.
**tdd_cycle:** RED — orchestration log tests | GREEN — run.py | REFACTOR — eval wiring
**deviations:** MockDetector in loop until trained weights available.
**judgment_calls:** None
**blockers:** None
**next:** Manual integration gates (HoloOcean, FathomNet download, GS train, ablation)
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T22:00:00Z] Step 4.2+ — Stub fixes (session 2)

**project:** FathomFollow
**step:** 4.2+
**phase:** Phase 4 / cross-cutting
**status:** Complete
**files_touched:** eval/run_eval.py, nav/training.py, perception/detector.py, run.py, cli.py, config/{nav_train,detector_train}.yaml, fixtures/
**tests_written:** tests/test_eval_from_run.py, tests/test_detector_metrics.py, tests/test_nav_training.py
**tests_passing:** 52/52
**summary:** Closed gaps left by initial scaffold: eval-from-run, detector metrics parsing, nav training loop, checked-in fixtures, training config YAMLs.
**tdd_cycle:** RED — 6 new tests (import errors) | GREEN — implementations | REFACTOR — none
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** HoloOcean install + Phase 1 manual baseline
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T23:30:00Z] Step 1.1 — FathomNet live integration

**project:** FathomFollow
**step:** 1.1
**phase:** Phase 1
**status:** Partial
**files_touched:** src/fathomfollow/data/fathomnet.py, tests/test_fathomnet.py, cli.py
**tests_written:** tests/test_fathomnet.py
**tests_passing:** 59/59
**summary:** ff-data count auto-selects Bathochordaeus (2017 boxes) over Benthocodon (662) and Granelledone (0). ff-data auto-prepare added with YOLO format default. COCO export fails on fathomnet-py pydantic validation for Bathochordaeus (negative bbox area); YOLO export works. Benthocodon YOLO dataset downloaded to data/fathomnet_raw/Benthocodon (390 images).
**tdd_cycle:** RED — test_fathomnet.py | GREEN — fathomnet.py + CLI count/auto-prepare | REFACTOR — yolo fallback
**deviations:** Used Benthocodon for first live download due to COCO bug on Bathochordaeus; selected taxon remains Bathochordaeus per count gate.
**judgment_calls:** [JUDGMENT CALL] Default auto-prepare to YOLO format; COCO path retained but known broken for some taxa in fathomnet-py 1.10.
**blockers:** Bathochordaeus full YOLO download (~2000 images) not yet run; HoloOcean still not installed.
**next:** Step 1.2 short detector train on Benthocodon; Step 1.3 sim baseline with real weights
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-08T23:45:00Z] Step 2.3 / 4.2 — Dual nav + trajectory pipeline

**project:** FathomFollow
**step:** 2.3+
**phase:** Phase 2 / 4
**status:** Complete
**files_touched:** src/fathomfollow/integration.py, run.py, scripts/integration_prep.py, tests/test_integration.py, data/trajectories/, data/nav_model/
**tests_written:** tests/test_integration.py
**tests_passing:** 59/59
**summary:** run_orchestration logs baseline_pos (DVL-only DR) alongside learned est_pos. generate_trajectories_from_sim + integration_prep.py produce nav training NPZ. ff-train-nav ran successfully on synthetic trajectories (checkpoint at data/nav_model/velocity_estimator.pt).
**tdd_cycle:** RED — test_integration.py | GREEN — integration.py + run.py dual nav | REFACTOR — none
**deviations:** Pre-GS baseline (Step 1.3) recorded with MockDetector on synthetic fixture (firing_rate=1.0); real baseline pending HoloOcean + trained detector.
**judgment_calls:** None
**blockers:** None
**next:** Bathochordaeus YOLO download + ff-train-detector; HoloOcean smoke
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T00:30:00Z] Step 1.2 — Detector training (live)

**project:** FathomFollow
**step:** 1.2
**phase:** Phase 1
**status:** Complete
**files_touched:** config/detector_train.yaml, data/fathomnet/, runs/detect/train/
**tests_written:** n/a (manual integration)
**tests_passing:** 60/60 unit tests
**summary:** 3-epoch YOLO11n train on Benthocodon (390 imgs, dev proxy for Bathochordaeus). metrics.json: mAP50=0.764, mAP50-95=0.540. Weights at runs/detect/train/weights/best.pt. CPU-only train (~2.5 min).
**tdd_cycle:** GREEN — metrics_from_train_results wrote real values | REFACTOR — none
**deviations:** Trained on Benthocodon reorganized split while Bathochordaeus selected by count gate; full Bathochordaeus download pending.
**judgment_calls:** [JUDGMENT CALL] Use Benthocodon for first live train to unblock pipeline; swap to Bathochordaeus before final ablation.
**blockers:** None
**next:** Step 1.3 sim baseline with trained weights
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T00:45:00Z] Step 1.3 — Pre-GS sim baseline (live)

**project:** FathomFollow
**step:** 1.3
**phase:** Phase 1
**status:** Complete
**files_touched:** runs/pre_gs_baseline_yolo.json, scripts/integration_prep.py
**tests_written:** n/a (manual integration)
**tests_passing:** 60/60
**summary:** YoloDetector(best.pt) on synthetic fixtures/sim/smoke.npz: firing_rate=0.0 (0/50 frames). Documents domain gap — detector trained on real FathomNet imagery does not fire on random synthetic sim pixels. Phase 1.5 ablation target: beat 0.0 on HoloOcean or GS-augmented training.
**tdd_cycle:** n/a — measurement step
**deviations:** Baseline recorded on synthetic fixture; HoloOcean fixture pending install.
**judgment_calls:** None
**blockers:** HoloOcean not installed for real sim-frame baseline.
**next:** Phase 1.5 GS reconstruction OR HoloOcean smoke
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T01:30:00Z] Step 0.3 — HoloOcean install attempt

**project:** FathomFollow
**step:** 0.3
**phase:** Phase 0
**status:** Blocked
**files_touched:** docs/holoocean_install.md, scripts/smoke_sim.py, src/fathomfollow/sim/recorder.py, tests/test_sim_recorder.py
**tests_written:** tests/test_sim_recorder.py
**tests_passing:** 61/61
**summary:** HoloOcean pip install failed: legacy BYU-PCCL/holodeck URL obsolete; PyPI holoocean==0.5.8 incompatible with Python 3.13 (pywin32<=228). Documented correct byu-holoocean/HoloOcean path. Added record_sim_fixture + build_fathomnet_proxy_fixture as interim Step 1.3 path.
**tdd_cycle:** RED — test_sim_recorder | GREEN — sim/recorder.py | REFACTOR — smoke_sim.py
**deviations:** FathomNet-image proxy fixture substitutes for HoloOcean until Epic-linked clone on Py3.11.
**judgment_calls:** [JUDGMENT CALL] Proxy fixture uses real FathomNet RGB + synthetic IMU/DVL; honest interim baseline, not a HoloOcean substitute.
**blockers:** [BLOCKED] HoloOcean client on Python 3.13 / Epic EULA access.
**next:** Bathochordaeus download; Step 1.3 proxy baseline
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T01:45:00Z] Step 1.3 — Pre-GS baseline (FathomNet proxy)

**project:** FathomFollow
**step:** 1.3
**phase:** Phase 1
**status:** Complete
**files_touched:** fixtures/sim/fathomnet_proxy.npz, docs/baselines.json, runs/pre_gs_baseline_fathomnet_proxy.json, scripts/integration_prep.py
**tests_written:** n/a
**tests_passing:** 61/61
**summary:** YoloDetector(best.pt) on fathomnet_proxy.npz (50 real FathomNet frames): firing_rate=0.18 (9/50). Recorded in docs/baselines.json as ablation_target_firing_rate. Synthetic smoke baseline remains 0.0.
**tdd_cycle:** n/a — measurement
**deviations:** Proxy not HoloOcean; pending live sim fixture.
**judgment_calls:** None
**blockers:** None
**next:** Bathochordaeus dataset + retrain; Phase 1.5 GS
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T02:00:00Z] Step 1.1 — Bathochordaeus download (in progress)

**project:** FathomFollow
**step:** 1.1
**phase:** Phase 1
**status:** Partial
**files_touched:** data/fathomnet_raw/Bathochordaeus/
**tests_written:** n/a
**tests_passing:** 61/61
**summary:** fathomnet-generate YOLO download started for Bathochordaeus (~1350 images, ETA ~30 min). Benthocodon dev set already reorganized at data/fathomnet/.
**tdd_cycle:** n/a
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** reorganize + retrain on Bathochordaeus; Phase 1.5.2 GS env setup
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T03:00:00Z] Step 1.1 — Bathochordaeus download complete

**project:** FathomFollow
**step:** 1.1
**phase:** Phase 1
**status:** Complete
**files_touched:** data/fathomnet_raw/Bathochordaeus/, data/fathomnet_batho/, config/detector_train.yaml
**tests_written:** n/a
**tests_passing:** 61/61
**summary:** Resumed fathomnet-generate YOLO download; 1350 images complete. reorganize_yolo_flat → data/fathomnet_batho (train=1083, val=132, test=135).
**tdd_cycle:** n/a
**deviations:** None
**judgment_calls:** None
**blockers:** None
**next:** Step 1.2 retrain on Bathochordaeus
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T03:30:00Z] Step 1.2 — Bathochordaeus detector training

**project:** FathomFollow
**step:** 1.2
**phase:** Phase 1
**status:** Complete
**files_touched:** config/detector_train.yaml, data/fathomnet_batho/metrics.json, runs/detect/train-2/
**tests_written:** n/a
**tests_passing:** 61/61
**summary:** 5-epoch YOLO11n on Bathochordaeus (CPU). metrics.json: mAP50=0.644, mAP50-95=0.415. Weights: runs/detect/train-2/weights/best.pt.
**tdd_cycle:** GREEN — metrics_from_train_results | REFACTOR — none
**deviations:** CPU train ~13 min; GPU would be faster.
**judgment_calls:** None
**blockers:** None
**next:** Step 1.3 baseline refresh
<!-- /DIARY_ENTRY -->

<!-- DIARY_ENTRY -->
### [2026-06-09T03:45:00Z] Step 1.3 — Pre-GS baseline (Bathochordaeus weights)

**project:** FathomFollow
**step:** 1.3
**phase:** Phase 1
**status:** Complete
**files_touched:** docs/baselines.json, runs/pre_gs_baseline_batho.json
**tests_written:** n/a
**tests_passing:** 61/61
**summary:** YoloDetector(train-2/best.pt) on fathomnet_proxy.npz: firing_rate=2.12 (106 dets / 50 frames). Updated ablation_target_firing_rate in docs/baselines.json. Prior Benthocodon dev baseline was 0.18.
**tdd_cycle:** n/a — measurement
**deviations:** Proxy fixture still; HoloOcean pending Epic/Py3.11 install.
**judgment_calls:** None
**blockers:** None
**next:** Phase 1.5 GS render + merge + ablation vs 2.12
<!-- /DIARY_ENTRY -->

## Final Spec

*(To be completed in Chapter 4.)*
