# FathomFollow metrics glossary

Authoritative recorded values: [`baselines.json`](baselines.json). This page defines terms so reports and demos use consistent, honest language.

## Detector / sim transfer

### `firing_rate`

Total detections divided by number of frames (`n_detections / n_frames`). Can exceed 1.0 when multiple boxes appear per frame.

Reported separately per fixture (`fathomnet_proxy`, `holoocean_smoke`, live runs). Pre-GS Bathochordaeus baseline on live HoloOcean: **0.58** (58 dets / 100 frames, train-2).

### `ablation_target_firing_rate`

Reference pre-GS firing rate on `fathomnet_proxy` (50 frames): **2.12**. Used for GS merge/ablation comparison. Live HoloOcean baseline (0.58) is a separate sim-domain reference — use both when interpreting ablation.

### mAP50 / mAP50-95

Ultralytics validation metrics on held-out YOLO splits. Bathochordaeus train-2: mAP50 **0.644**, mAP50-95 **0.415**. Val mAP alone does not imply sim transfer (see GS ablation regression).

## Navigation / drift

### `drift_within_dropout`

Mean position error (meters) integrated during **DVL-invalid** timesteps only (forced time windows + altitude gate; tilt gate pending FB-008).

### `margin_dropout`

`drift_baseline_within_dropout − drift_learned_within_dropout`. Phase 2/4 acceptance uses learned **lower** than baseline.

### `nav_attitude`

Post-fix (2026-07-04): gyro-integrated `AttitudeIntegrator` with mission-start quaternion seed. Neither learned nor baseline dead reckoning reads per-step GT orientation.

## Tracking (v1 — being refined in FB-014)

### `tracking_retention` / `active_track_coverage`

Fraction of control steps where `SimpleTracker` had an **active** track (detection-linked ID). **Not** the same as ground-truth target visible in frame.

Pre-FB-002 live hero: **79%** over ~100 steps (~10 s), PierHarbor-HoveringCamera, **no spawned target mimic** — retention reflects detector+tracker on scene content, not follow of a scripted organism.

### `gt_in_frame_fraction` (FB-003+)

Eval-only: fraction of frames where projected GT bbox of the target mimic lies in the image. Requires spawned mimic (FB-002) and eval harness projection.

## Coupling

### `parallel-eval`

v1 default: `run_orchestration` runs perception→control on commands; nav estimator runs on the same sensor stream but **does not** feed the controller. Drift and retention are compared in the same run windows — not a nav-steered follow loop.

## Dropout

### Forced windows

Scenario YAML `dropout.forced_windows`: DVL marked invalid regardless of altitude.

### Altitude gate

`dvl_valid` false when altitude below `dropout.alt_min`.

### Tilt gate (pending FB-008)

`tilt_max_deg` in YAML not yet wired due to HoloOcean quaternion/Euler mapping issue on recorded fixtures.

## Hero run labels

| Label | Meaning |
|-------|---------|
| fixture hero | `ff-run` on `holoocean_smoke.npz` replay |
| live hero | `ff-run --live` on Unreal PierHarbor |
| drift gate | nav-only acceptance (`ff-drift-gate`); may use MockDetector unless FB-006 |

## GS-specific

### `train_psnr` / eval PSNR

WaterSplatting reconstruction quality on SeaThru-NeRF scene — appearance fit, not detector sim transfer.

### `verdict: regression`

Post-GS detector firing rate on sim fixtures **lower** than pre-GS at same weights comparison (train-4 vs train-2).
