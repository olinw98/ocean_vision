# HoloOcean install (Step 0.3)

HoloOcean requires Epic Games ↔ GitHub account linking and Unreal EULA acceptance.

## Standard install

1. Link GitHub to Epic: https://www.unrealengine.com/en-US/ue-on-github
2. Clone (requires access): `git clone https://github.com/byu-holoocean/HoloOcean.git holoocean`
3. Install client: `cd holoocean/client && pip install .`
4. Download Ocean worlds: `python -c "import holoocean; holoocean.install('Ocean')"`
5. Smoke + record fixture:
   ```bash
   python scripts/smoke_sim.py --live --out fixtures/sim/holoocean_smoke.npz --frames 100
   ```

## Known issues on this project

- **Python 3.13**: PyPI `holoocean==0.5.8` pins `pywin32<=228`, which is unavailable on Python 3.13. Use Python 3.11 venv or clone the current GitHub client.
- **Legacy plan URL** (`BYU-PCCL/holodeck`) is obsolete; use `byu-holoocean/HoloOcean`.

## Proxy fixture (when HoloOcean blocked)

Use real FathomNet images as RGB stand-in for sim-frame detector baseline:

```bash
python scripts/smoke_sim.py --fathomnet-proxy data/fathomnet/images/train \
  --out fixtures/sim/fathomnet_proxy.npz --frames 50
python scripts/integration_prep.py baseline \
  --fixture fixtures/sim/fathomnet_proxy.npz \
  --out runs/pre_gs_baseline_fathomnet_proxy.json
```

Pass trained weights via editing `integration_prep.py` or use the Python API with `YoloDetector("runs/detect/train/weights/best.pt")`.
