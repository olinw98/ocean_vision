# Reproduce the hero run

Step-by-step guide to reproduce the **fixture hero** (`holoocean_smoke` replay) and optionally the **full retrain chain**. Canonical artifact paths are pinned in code and [`artifacts.json`](artifacts.json) — retraining must write the same paths every time (not accidental `train-3`, `train-4`, …).

**Reference metrics:** [`baselines.json`](baselines.json) → `phase_4_hero_run` (fixture) and `phase_4_live_hero_run` (live HoloOcean).

## Canonical artifact paths

| Artifact | Path | Registry id |
|----------|------|-------------|
| Bathochordaeus YOLO (train-2) | `runs/detect/train-2/weights/best.pt` | `detector_weights` |
| DriftGuard nav checkpoint | `data/nav_model/velocity_estimator.pt` | `nav_checkpoint` |

**Verify:** `ff-status` → **Registry artifacts (hero)** should show `sha256 ok` after install.

**Training pins (FB-011):**

- `config/detector_train.yaml` sets `project: runs/detect` and `run_name: train-2` → `ff-train-detector` always writes the canonical detector path.
- `config/nav_train_holoocean.yaml` sets `checkpoint_dir: data/nav_model` → `ff-train-nav` always writes `velocity_estimator.pt` there.

Do **not** rely on Ultralytics auto-increment (`train`, `train-2`, `train-3`) without these config fields.

---

## Fast path (~5 min, no GPU training)

For a fresh clone on any machine with Python 3.11. Uses bundled weights in `fixtures/artifacts/hero/` (committed in git).

```powershell
python -m venv .venv
.venv\Scripts\activate          # mac/linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # expect 115+ passing

ff-fetch hero                     # copies + SHA-256 verify → canonical paths
ff-status                         # registry artifacts: sha256 ok

ff-run --fixture fixtures/sim/holoocean_smoke.npz `
  --scenario config/scenario_holoocean.yaml `
  --detector runs/detect/train-2/weights/best.pt `
  --nav-checkpoint data/nav_model/velocity_estimator.pt `
  --out runs/hero_repro

ff-eval --run runs/hero_repro
```

**Compare to baselines** (`phase_4_hero_run`):

| Metric | Baseline (fixture hero) |
|--------|-------------------------|
| `tracking_retention` | **0.68** (active-track) |
| `margin_dropout` | **1.38 m** |
| `coupling_mode` | `parallel-eval` |

Detection quality (GT projection) requires spawned mimic in fixture; see `gt_in_frame_fraction` in report when mimic GT varies.

---

## Full path (GPU build machine, retrains weights)

Recomputes weights from data. Requires gitignored `data/fathomnet_batho/` (or run data prep). GPU recommended for detector train.

### 1. Environment

Same as fast path: venv, `pip install -e ".[dev]"`, `pytest -q`.

### 2. FathomNet YOLO dataset (if missing)

```powershell
ff-data auto-prepare --out data/fathomnet_batho
# or use existing data/fathomnet_batho/ from build machine
```

### 3. Train detector → canonical `train-2`

```powershell
ff-train-detector --config config/detector_train.yaml
# writes runs/detect/train-2/weights/best.pt (exist_ok=True, overwrites prior train-2)
```

Confirm stdout includes `"weights": "runs/detect/train-2/weights/best.pt"`.

### 4. Nav trajectories + DriftGuard train

```powershell
python scripts/integration_prep.py trajectories `
  --fixture fixtures/sim/holoocean_smoke.npz `
  --out data/trajectories_holoocean `
  --n-steps 200

ff-train-nav --config config/nav_train_holoocean.yaml
# writes data/nav_model/velocity_estimator.pt
```

### 5. Hero run + eval

Same commands as **fast path** step `ff-run` / `ff-eval` (skip `ff-fetch hero` if training succeeded).

### 6. Optional live gate

Requires HoloOcean install ([`holoocean_install.md`](holoocean_install.md)):

```powershell
ff-run --live --scenario config/scenario_holoocean.yaml `
  --detector runs/detect/train-2/weights/best.pt `
  --nav-checkpoint data/nav_model/velocity_estimator.pt `
  --out runs/hero_live_repro
ff-eval --run runs/hero_live_repro
```

Compare to `phase_4_live_hero_run` in baselines (~**79%** active-track retention, **1.27 m** margin within dropout).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Missing artifact` on `ff-run` | `ff-fetch hero` |
| `sha256 mismatch` after fetch | `ff-fetch hero --force` or restore `fixtures/artifacts/hero/` |
| Weights at `runs/detect/train-3/` | Re-run with `config/detector_train.yaml` (`run_name: train-2`) or `ff-fetch hero` |
| Drift-gate fails without detector | Add `--allow-mock-detector` (nav-only) or `--detector runs/detect/train-2/weights/best.pt` |

---

## Related docs

- [`workflow.md`](workflow.md) — multi-machine git vs local artifacts
- [`metrics-glossary.md`](metrics-glossary.md) — honest metric names
- [`artifacts.json`](artifacts.json) — SHA-256 registry for `ff-fetch hero`
