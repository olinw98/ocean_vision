# Implementation Plan: FathomFollow

**Date:** 2026-06-08 (rev. 2 — Gaussian-splatting visual layer added)
**Stack:** Python 3.11 · PyTorch 2.x (CUDA) · HoloOcean · Ultralytics YOLO · fathomnet-py · underwater 3DGS (WaterSplatting / SeaSplat) · pytest
**Target Context:** Local research project, single GPU workstation. Runs as Python scripts/CLI + offline evaluation. Simulation-only; no physical vehicle.
**Author:** Claude (coding-blueprint skill)

---

## 1. Executive Summary

FathomFollow is a simulation-based system in which an autonomous underwater vehicle (AUV) visually detects and follows a marine organism while maintaining a robust position estimate through Doppler Velocity Log (DVL) dropout. It fuses two learning components: a perception model fine-tuned on real FathomNet imagery that detects and tracks a target organism, and a learned dead-reckoning estimator ("DriftGuard") that predicts body-frame velocity from inertial data to bridge gaps when the DVL is unavailable. The link between the two is causal: aggressive follow maneuvers are exactly the conditions that degrade DVL bottom-lock, so the tracking task generates the navigation stress the estimator must absorb. To close the sim-to-real appearance gap that defeats most underwater vision work, a Gaussian-splatting visual layer reconstructs photorealistic underwater scenes from real footage and renders training imagery with a controllable turbidity dial, so the detector learns on realistic ocean rather than the simulator's synthetic look. The user is a developer/researcher building this with an AI coding agent on a local GPU. Success means a closed-loop run in HoloOcean where the AUV keeps a moving target in frame while position drift during DVL-dropout windows is measurably lower than a classical dead-reckoning baseline.

---

## 2. Core Requirements

### Functional Requirements
1. The system shall download and prepare a labeled object-detection dataset from FathomNet for a chosen target taxon, in YOLO format with train/val/test splits.
2. The system shall train an object detector on that dataset and report detection metrics (mAP@50, mAP@50-95) on a held-out test split.
3. The system shall run the trained detector on a HoloOcean RGB camera stream and emit per-frame detections (class, bounding box, confidence).
4. The system shall associate detections across frames into stable tracks with persistent IDs, tolerating short detection gaps.
5. The system shall produce vehicle motion commands from the active track that keep the target centered in the image and within a desired apparent-size band (visual servoing).
6. The system shall simulate an AUV in HoloOcean with RGB camera, IMU, DVL, and ground-truth pose sensors.
7. The system shall inject DVL dropout as a function of vehicle state (e.g., altitude and attitude thresholds) and/or scripted windows.
8. The system shall train a learned velocity estimator that predicts body-frame velocity from IMU history (and DVL when present), and integrate it into a dead-reckoning pose estimate.
9. The system shall compute navigation drift (position error vs. ground truth) globally and specifically within dropout windows, comparing the learned estimator against a classical dead-reckoning baseline.
10. The system shall run an end-to-end scenario combining detection, tracking, follow control, DVL dropout, and navigation estimation, and emit a consolidated evaluation report.
11. The system shall reconstruct an underwater scene as an underwater-adapted 3D Gaussian splat from a posed real-image source, and render novel views from it.
12. The system shall render a photorealistic, labeled detection dataset from the splat across a swept turbidity range, and shall measure the change in detector firing rate on HoloOcean frames when this augmented data is added to training.

### Non-Functional Requirements
- Detector inference shall run at ≥ 10 FPS on the target GPU at the sim camera resolution (so the control loop is not perception-bound).
- The navigation estimator shall be lightweight enough to run faster than real time (inference < 5 ms per step on GPU; ideally CPU-capable).
- All training, rendering, and evaluation runs shall be reproducible: fixed seeds, pinned dependency versions, config files checked into the repo. GS renders shall be deterministic given a seed + camera path + turbidity setting.
- The codebase shall be testable without a running simulator and without a trained splat: sim-dependent code sits behind an interface with recorded fixtures; GS-dependent code behind an interface with a small pre-rendered fixture, so unit tests run in CI without HoloOcean or a GPU-bound GS train.
- Detection metrics shall be reported separately on real (FathomNet), GS-rendered, and simulated (HoloOcean) frames, never conflated.

### Out of Scope
- Reinforcement-learning follow control (stretch only; baseline is classical visual servoing).
- Online (in-the-loop) Gaussian-splat rendering of the camera feed — v1 uses GS for offline dataset augmentation only; in-loop GS rendering co-registered to the sim is a Phase 4 stretch.
- Real hardware deployment, real acoustic modems, or real sensor I/O.
- Multi-target tracking (single active target only in v1).
- 3D bathymetric mapping / SLAM (this is the navigation-via-dead-reckoning project, not the neural-sonar project).
- Guaranteed sim-to-real transfer. GS augmentation is expected to *narrow* the appearance gap; the project measures the gap, it does not claim to eliminate it.

---

## 3. System Architecture

### Overview

Two loops share the simulator. The **perception→control loop** turns camera frames into motion commands. The **navigation loop** turns inertial/DVL data into a pose estimate. A separate **offline GS augmentation pipeline** runs before training to produce photorealistic labeled imagery; it does not run in the live loop in v1. Ground truth from the sim is used only for evaluation, never as a controller input.

```
   OFFLINE (pre-training):
   [posed real underwater images] --> [underwater GS train] --> [GS renderer + turbidity sweep]
                                                                        |
                                                                        v
                                                          [GS-augmented detection dataset]
                                                                        |
   [FathomNet dataset] -----------------------------------------------> + --> [Detector training]

   ONLINE (closed loop):
                         HoloOcean Simulator
        ┌──────────────────────────────────────────────────┐
        │  AUV agent + target mimic + DVL-dropout injector   │
        └───┬───────────────┬───────────────┬───────────────┘
            │ RGB frame      │ IMU / DVL     │ ground-truth pose (eval only)
            v                v               
      [Detector] --> [Tracker] --> [Follow Controller] --> motion cmd --> HoloOcean

      [IMU/DVL] --> [DriftGuard estimator] --> [Dead-reckoning / EKF] --> pose estimate
                                                                        |
                                                                        v
                                                              [Evaluation Harness]
```

### Component Breakdown

**Detector** — per-frame detection of the target taxon. RGB image → list of (class, bbox, confidence). Ultralytics YOLO fine-tuned from COCO weights on FathomNet plus GS-rendered data. Reported on real, GS, and sim frames separately.

**Tracker** — cross-frame association into stable IDs; smooths through missed detections. ByteTrack via Ultralytics; configurable max-gap.

**Follow Controller** — keeps the active target centered and at desired range. Active track → `Command`. Classical visual servoing (PID on centroid + bbox-size error); RL is a stretch replacement behind the same interface.

**DriftGuard (navigation estimator)** — predicts body-frame velocity to bridge DVL dropout. IMU window (+ DVL when valid) → velocity. Small GRU/TCN, trained supervised on sim trajectories with ground-truth velocity.

**Navigation Fusion** — integrates velocity into a pose estimate. Pure dead-reckoning baseline (naive on purpose) plus optional `filterpy` EKF.

**Sim Harness** — configures the HoloOcean scenario, steps the sim, exposes sensors behind a stable interface, injects DVL dropout, logs ground truth. Wrapped behind a `SimEnv` interface with a `RecordedSimEnv` fixture for testing.

**GS Augmentation Pipeline** *(new)* — reconstructs an underwater scene as an underwater-adapted Gaussian splat from posed real images, then renders labeled detection frames over a turbidity sweep.
- Inputs/Outputs: posed image source → trained `.splat`/checkpoint → rendered RGB frames + bounding-box labels (the target organism composited or annotated) in YOLO format.
- Key decisions: use a source dataset that *ships camera poses* (SeaThru-NeRF underwater set) for v1 to sidestep underwater Structure-from-Motion; default library WaterSplatting (real-time, explicit medium model enabling turbidity removal/insertion), SeaSplat as alternative. Behind a `GSRenderer` interface with a small pre-rendered fixture for tests. Offline only in v1.

**Data Pipeline** — FathomNet download → YOLO dataset → augmentation (classical + GS-rendered) → splits. fathomnet-py; deterministic split by image hash; GS frames merged as an additional labeled source with provenance tags.

**Evaluation Harness** — computes all metrics; produces the consolidated report (drift-over-time, drift-within-dropout, tracking retention, mAP table split by domain, sim-frame firing-rate delta with vs. without GS augmentation).

---

## 4. Data Model

**DetectionRecord** — one detection. `frame_id:int`, `class_id:int`, `bbox:tuple` (xywh, normalized), `confidence:float`. In-memory; serialized to per-run JSONL.

**Track** — `track_id:int`, `last_bbox`, `last_seen_frame:int`, `state`, `history`. In-memory.

**SimObservation** — `t`, `rgb`, `imu(6)`, `dvl(3)`, `dvl_valid:bool`, `gt_pose`, `gt_target_pose`. Logged to parquet/NPZ for replay and tests.

**NavEstimate** — `t`, `position(3)`, `orientation`, `velocity_body(3)`, `source` (dvl/estimator/blended). Logged.

**DatasetManifest** — `taxa:list[str]`, `n_images`, `split_counts`, `fathomnet_snapshot_date`, `sources:list[str]` (e.g. `["fathomnet","gs:scene_a"]`), `data_yaml_path`. JSON beside the dataset.

**GSScene** *(new)* — describes a reconstructed splat. `scene_id`, `source_dataset`, `library` (watersplatting/seasplat), `n_gaussians`, `pose_source` (provided/colmap), `checkpoint_path`, `train_psnr`. JSON beside the checkpoint.

**GSRenderManifest** *(new)* — describes a render batch. `scene_id`, `camera_path_id`, `turbidity_values:list[float]`, `n_frames`, `label_strategy` (composited-target / annotated-region), `out_dir`. JSON beside the rendered frames.

Storage: datasets, splat checkpoints, and run logs on local disk under `data/`, `models/`, `runs/`. No database.

---

## 5. API / Interface Contracts

### External Interfaces (CLI)

```
ff-data prepare --taxa "<concept>" [--limit N] [--out data/<name>]
    Query FathomNet, download images + boxes, write YOLO dataset + manifest.

ff-gs train --source <dataset_path> [--library watersplatting] --out models/gs/<scene>
    Reconstruct an underwater scene as a GS; write checkpoint + GSScene manifest.

ff-gs render --scene models/gs/<scene> --camera-path config/cam_path.yaml \
             --turbidity 0.0,0.3,0.6 --label-strategy composited-target \
             --out data/gs_<scene>
    Render labeled frames over a turbidity sweep; write GSRenderManifest + YOLO labels.

ff-data merge --sources data/<name>,data/gs_<scene> --out data/<combined>
    Merge labeled sources into one dataset with provenance tags + fresh splits.

ff-train-detector --data data/<combined>/data.yaml --epochs N --model yolo11s.pt
ff-train-nav --trajectories runs/<traj_set> --epochs N --arch gru
ff-run --scenario config/scenario.yaml --detector models/<w>.pt --nav models/<w>.pt [--render]
ff-eval --run runs/<id> [--baseline dead_reckoning] [--ablate-gs]
    --ablate-gs reports detector sim-frame firing rate with vs. without GS-augmented training.
```

### Internal Interfaces

```python
class SimEnv(Protocol):
    def reset(self) -> SimObservation: ...
    def step(self, command: Command) -> SimObservation: ...
    def close(self) -> None: ...

class GSRenderer(Protocol):                 # new
    def load(self, checkpoint: str) -> None: ...
    def render(self, pose: Pose, turbidity: float) -> np.ndarray: ...
    # RecordedGSRenderer replays a pre-rendered fixture for tests.

class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[DetectionRecord]: ...

class Tracker(Protocol):
    def update(self, dets: list[DetectionRecord]) -> list[Track]: ...

class FollowController(Protocol):
    def command(self, active: Track | None, img_shape: tuple[int,int]) -> Command: ...

class VelocityEstimator(Protocol):
    def estimate(self, imu_window: np.ndarray, dvl: np.ndarray | None) -> np.ndarray: ...
```

`Command` (forward_vel, yaw_rate, vertical_vel) remains the single contract between control and sim, so the controller can be swapped without touching the sim layer. `GSRenderer` is the seam that keeps GS offline-only in v1 and makes an in-loop renderer a drop-in stretch.

---

## 6. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11 | Matches HoloOcean, fathomnet-py, PyTorch, GS ecosystems. |
| Simulator (physics + sensors) | HoloOcean (GitHub release, accept Unreal EULA) | Open-source, Python interface, ships DVL/IMU/sonar/camera. Stays the physics/dynamics source of truth. |
| DL framework | PyTorch 2.x + CUDA | Local GPU. |
| Detector | Ultralytics YOLO (v11) | COCO-pretrained, fast fine-tune, built-in ByteTrack. |
| Tracking | ByteTrack (via Ultralytics) | Avoids hand-rolled association. |
| Detection data | fathomnet-py | Native client for FathomNet REST API. |
| Underwater GS (visual layer) | WaterSplatting (default) / SeaSplat (alt) | Underwater-adapted 3DGS with explicit medium model → real-time render + turbidity control. Used **offline** for dataset augmentation. |
| GS source data | SeaThru-NeRF underwater dataset (default) | Ships camera poses, sidestepping underwater SfM/COLMAP for v1. BlueCoral3D / Submerged3D / own ROV footage as alternatives. |
| Nav model | PyTorch GRU / TCN | Body-frame velocity regression. |
| Nav fusion | filterpy (EKF), NumPy integrator | Baseline + optional filter. |
| RL (stretch) | Stable-Baselines3 + sb3-contrib | Phase 4 stretch only. |
| Testing | pytest | Fixtures for RecordedSimEnv and RecordedGSRenderer. |
| Config | pydantic + YAML | Typed scenario/training/render configs. |
| Experiment logging | TensorBoard + CSV (W&B optional) | Reproducible local logs. |
| Build / packaging | uv or pip + venv, pyproject.toml | Pinned deps. |

Version pins to fix at first install and record in the Final Spec: `torch`, `ultralytics`, `holoocean`, `fathomnet`, the chosen GS implementation, `filterpy`, `numpy`, `pydantic`.

---

## 7. File / Project Structure

```
fathomfollow/
├── src/fathomfollow/
│   ├── data/            # FathomNet download, YOLO conversion, merge, splits
│   ├── gs/              # GSRenderer protocol, WaterSplatting wrapper, render+turbidity, labeling
│   ├── perception/      # detector wrapper, tracker wrapper, sim inference
│   ├── control/         # visual-servoing controller; rl/ (stretch)
│   ├── nav/             # velocity estimator, dead-reckoning, EKF, dropout injector, trajectories
│   ├── sim/             # SimEnv protocol, HoloOceanSimEnv, RecordedSimEnv, target mimic
│   ├── eval/            # metrics, plots, report generation
│   ├── config/          # pydantic models for scenarios, training, render
│   └── cli.py           # ff-data / ff-gs / ff-train-* / ff-run / ff-eval
├── tests/               # pytest; recorded fixtures, no live sim or GS train required
├── config/              # scenario.yaml, cam_path.yaml, training configs
├── data/                # (gitignored) datasets + manifests
├── models/              # (gitignored) detector + nav weights, gs/ checkpoints
├── runs/                # (gitignored) run logs + reports
├── fixtures/            # recorded sim observations + small pre-rendered GS frames
├── pyproject.toml
└── README.md
```

---

## 8. Implementation Phases

### Phase 0 — Environment & Scaffolding
- Goal: repo skeleton and a HoloOcean smoke test.
- Deliverables: project layout, pinned deps, `SimEnv` protocol + `HoloOceanSimEnv` logging one frame of each sensor, `RecordedSimEnv` replay.
- Acceptance: a script steps HoloOcean N ticks and writes a valid observation log; `RecordedSimEnv` replays it; tests pass without launching Unreal.

### Phase 1 — Perception (baseline)
- Goal: a detector trained on FathomNet that runs on sim frames, with the domain gap measured.
- Deliverables: `ff-data prepare`, `ff-train-detector`, detector wrapper, metrics.json on FathomNet test, plus measured firing rate on HoloOcean frames.
- Acceptance: mAP@50 on FathomNet test exceeds a recorded floor; detector returns well-formed detections on sim frames; **sim-frame firing rate recorded as the pre-GS baseline** (this is the number Phase 1.5 must improve).

### Phase 1.5 — Gaussian-Splatting Augmentation *(new)*
- Goal: narrow the FathomNet→sim appearance gap with photorealistic GS-rendered training data.
- Deliverables: `ff-gs train` (reconstruct an underwater scene from a posed source), `ff-gs render` (turbidity-swept labeled frames), `ff-data merge`, a retrained detector, and an ablation.
- Acceptance: GS reconstruction reaches a recorded render-quality floor (PSNR/visual check); the render pipeline emits well-formed YOLO-labeled frames whose turbidity varies deterministically with the setting; **the detector retrained on FathomNet + GS data shows a higher sim-frame firing rate than the Phase 1 baseline by a recorded margin** (the payoff). If it does not improve, that null result is recorded and analyzed, not hidden.

### Phase 2 — Navigation
- Goal: learned velocity estimator that beats naive dead-reckoning through DVL dropout.
- Deliverables: dropout injector, trajectory logger, `ff-train-nav`, dead-reckoning baseline, optional EKF, drift evaluation.
- Acceptance: on held-out trajectories with injected dropouts, mean position drift within dropout windows is lower than the baseline by a recorded margin.

### Phase 3 — Tracking & Follow Control
- Goal: in-sim visual following of a moving target mimic.
- Deliverables: target mimic in scenario, tracker wrapper, visual-servoing controller, `Command` wired into sim.
- Acceptance: in a scripted-target scenario, the controller keeps the target in frame for ≥ a recorded fraction of steps; controller unit tests pass against synthetic tracks.

### Phase 4 — Integration & Evaluation (+ stretch)
- Goal: end-to-end closed loop + consolidated report.
- Deliverables: `ff-run` (full loop, dropout active), `ff-eval` consolidated report with all metric families, domain-split mAP, and the GS ablation.
- Acceptance: a single run produces the report; drift-within-dropout under the learned estimator is lower than baseline during the same run while the target is actively followed.
- Stretch (each its own TDD step): (a) **online GS rendering** co-registered to HoloOcean's camera frame/intrinsics, replacing the sim camera feed with GS-rendered observations while HoloOcean keeps physics — moving target handled via a dynamic-GS or composited-target approach (ReaDy-Go style); (b) RL follow controller behind the existing interface.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Domain gap: FathomNet-trained detector doesn't fire on rendered sim frames | High | High | **Phase 1.5 GS augmentation is the primary mitigation**: train the detector on photorealistic GS-rendered ocean with swept turbidity. Also: textured target mimic, heavy underwater augmentation, optional fine-tune on a few labeled sim frames. Report real/GS/sim mAP separately; treat residual gap as a measured result. |
| GS source data lacks camera poses; underwater SfM/COLMAP fails in turbid/low-texture/refractive water | Medium | Medium | Default to a source dataset that *ships poses* (SeaThru-NeRF underwater) for v1; only attempt self-reconstruction of own footage as a later extension. |
| Underwater GS assumes homogeneous water per scene; turbidity dial may be coarse | Medium | Low | Accept per-scene homogeneity in v1; sweep discrete turbidity levels; document as a limitation; spatially varying media is future work. |
| GS train/render competes with detector/sim for the single GPU | Medium | Medium | GS is offline and one-time per scene; render batches to disk, then free the GPU for training; cache renders. |
| HoloOcean install/EULA/GPU/headless-render friction | Medium | High | Phase 0 smoke test; document EULA + drivers; run sim and training as separate processes; `RecordedSimEnv` keeps most work sim-free. |
| DVL-dropout model unrealistic | Medium | Medium | Model dropout from altitude/attitude/velocity + scripted windows; document assumptions; frame estimator as bridging arbitrary gaps. |
| Scope creep into RL and/or in-loop GS rendering | Medium | High | Both are explicit Phase 4 stretch behind fixed interfaces (`Command`, `GSRenderer`); v1 is classical control + offline GS. |
| HoloOcean / GS library API differs from assumptions here | Medium | Low | Phase 0 and Phase 1.5 verify actual APIs against current docs before building on them; wrap behind `SimEnv` / `GSRenderer` so changes localize. |

---

## 10. Open Questions

- **Q:** Which target taxon, and does a FathomNet concept have enough boxes? **Impact:** dataset query, class set, mimic look, and what the GS labels depict. **Owner:** user / research.
- **Q:** Which GS source scene/dataset, and does the target organism actually appear in it (for composited vs. annotated labeling)? **Impact:** the `ff-gs render` label strategy and how realistic the augmentation is. **Owner:** user / research. **Default:** SeaThru-NeRF underwater scene with a composited target-organism splat/sprite.
- **Q:** WaterSplatting vs. SeaSplat (vs. another current underwater-3DGS impl)? **Impact:** the `gs/` wrapper and install. **Owner:** agent/research. **Default:** WaterSplatting for real-time render + explicit medium model.
- **Q:** Cursor or Claude Code as the implementing agent? **Impact:** file-reference syntax in the agent book. **Owner:** user.
- **Q:** Does the nav estimate feed the controller (full closed loop) or run parallel-eval in v1? **Impact:** loop coupling. **Owner:** agent. **Default:** parallel-eval first, closed-loop at Phase 4.
- **Q:** Headless vs rendered sim during nav data collection? **Impact:** throughput. **Owner:** agent. **Default:** headless for nav trajectories; rendered only where camera frames are needed.

---

## 11. Existing Constraints

N/A — greenfield project. External dependencies the build must respect: the HoloOcean Unreal EULA acceptance and GitHub-only install (not PyPI); the FathomNet REST API and its availability via fathomnet-py; the chosen underwater-GS implementation's license and data-format expectations; the availability of a posed underwater source dataset for reconstruction; and a single local GPU as the compute ceiling, shared between GS reconstruction, detector training, and simulation.
