# FathomFollow (ocean_vision)

> **Sim-first research platform** for underwater perception (FathomNet YOLO) and DVL-dropout navigation in HoloOcean — **not** a field-deployable AUV product in v1.

FathomFollow closes a perception→control loop in simulation while evaluating navigation drift in parallel. Offline Gaussian splatting augments detector training only; it does not render inside the live control loop.

Canonical build record: [`implementation-spec.md`](implementation-spec.md) (Final Spec + diary). Metrics: [`docs/baselines.json`](docs/baselines.json).

**Positioning & honest claims:** [`docs/positioning.md`](docs/positioning.md) · **Metric definitions:** [`docs/metrics-glossary.md`](docs/metrics-glossary.md)

## Quick start (tiered)

### Tier 0 — Unit tests (no HoloOcean, no GPU weights)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

Expect **90/90** passing on Python 3.11.

### Tier 1 — Fixture replay (recorded HoloOcean NPZ, local weights)

Requires gitignored artifacts (`runs/detect/train-2/`, `data/nav_model/`). See [`docs/workflow.md`](docs/workflow.md).

```powershell
ff-run --fixture fixtures/sim/holoocean_smoke.npz --scenario config/scenario_holoocean.yaml `
  --detector runs/detect/train-2/weights/best.pt `
  --nav-checkpoint data/nav_model/velocity_estimator.pt `
  --out runs/hero_fixture
ff-eval --run runs/hero_fixture
```

### Tier 2 — Live HoloOcean (Epic EULA + Python 3.11)

See [`docs/holoocean_install.md`](docs/holoocean_install.md). HoloOcean is **not** a pytest dependency.

```powershell
ff-run --live --scenario config/scenario_holoocean.yaml `
  --detector runs/detect/train-2/weights/best.pt `
  --nav-checkpoint data/nav_model/velocity_estimator.pt `
  --out runs/hero_live
```

Record a fixture: `python scripts/smoke_sim.py --live --scenario config/scenario_holoocean.yaml --out fixtures/sim/holoocean_smoke.npz --frames 100`

### Tier 3 — GS augmentation (optional, separate conda env)

See [`docs/gs_setup.md`](docs/gs_setup.md). Real GS train/render is a manual integration gate, not part of pytest.

## CLI overview

| Command | Purpose |
|---------|---------|
| `ff-status` | Project snapshot (tests, diary, local artifacts) |
| `ff-data` | FathomNet → YOLO, merge GS renders |
| `ff-train-detector` / `ff-train-nav` | Train YOLO / DriftGuard |
| `ff-run` | Perception closed loop + parallel nav eval |
| `ff-drift-gate` | Phase 2 nav acceptance on fixture replay |
| `ff-eval` | Report from run logs |
| `ff-gs` | WaterSplatting subprocess (offline) |

## Non-goals (v1)

- Field deployment or hardware integration
- Nav→controller closed loop (parallel-eval only)
- In-loop GS rendering during control
- Guaranteed sim-to-real transfer (GS ablation at 9 frames was a measured **null/regression**)

Architecture and build contract: [`agent-prompt.md`](agent-prompt.md), [`implementation-plan.md`](implementation-plan.md) (read-only reference).
