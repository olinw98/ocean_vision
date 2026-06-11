# HoloOcean install (Step 0.3)

HoloOcean requires Epic Games ↔ GitHub account linking and Unreal EULA acceptance.

## Standard install

1. Link GitHub to Epic: https://www.unrealengine.com/en-US/ue-on-github
2. Clone (requires access): `git clone https://github.com/byu-holoocean/HoloOcean.git holoocean`
3. Install client: `cd holoocean/client && pip install .`
4. Download Ocean worlds: `python -c "import holoocean; holoocean.install('Ocean')"`
5. Smoke + record fixture (use a **camera** scenario — `PierHarbor-Hovering` has no RGB in HoloOcean 2.3):
   ```bash
   python scripts/smoke_sim.py --live --out fixtures/sim/holoocean_smoke.npz --frames 100
   ```
   Default scenario in `config/scenario.yaml` is `PierHarbor-HoveringCamera` (`LeftCamera` / `RightCamera`, 512×512 RGBA).

## Project venv (Windows GPU machine)

Recreate `.venv` on **Python 3.11** before installing HoloOcean (Python 3.13 breaks pywin32 / holoocean):

```powershell
cd ocean_vision
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
pip install ..\holoocean\client
python -c "import holoocean; holoocean.install('Ocean')"
```

Verified: `holoocean==2.3.0` from `byu-holoocean/HoloOcean` client (Epic ↔ GitHub linked).

## Known issues on this project

- **Python 3.13**: PyPI `holoocean==0.5.8` pins `pywin32<=228`, which is unavailable on Python 3.13. Use Python 3.11 venv + GitHub client (see above).
- **Legacy plan URL** (`BYU-PCCL/holodeck`) is obsolete; use `byu-holoocean/HoloOcean` or `byu-holoocean-mirror/HoloOcean` after Epic linking.
- **No camera on default Hovering scenarios**: `PierHarbor-Hovering`, `Dam-Hovering`, etc. ship IMU/DVL only. Use `PierHarbor-HoveringCamera` (or `Dam-HoveringCamera`) for perception smoke tests.

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
