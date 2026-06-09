# Implementation Spec: FathomFollow

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

## Final Spec

*(To be completed in Chapter 4.)*
