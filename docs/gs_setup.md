# Gaussian Splatting setup (Step 1.5.2)

WaterSplatting requires a separate conda environment (Python 3.8, torch 2.1.2+cu118).

## Create env

```bash
conda create -n water_splatting python=3.8 -y
conda activate water_splatting
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# Follow WaterSplatting / nerfstudio install for your CUDA version
```

## Train scene

```bash
ff-gs train --source /path/to/seathru-nerf/scene --out models/gs/scene01
```

Log `train_psnr` from `scene.json` in `implementation-spec.md`.

## Render labeled batch

```bash
ff-gs render --config config/render.yaml
```

Tests use `RecordedGSRenderer` only; real renders are manual integration.

## Fallback

If WaterSplatting fails to install, log `[BLOCKED]` in the diary and try SeaSplat per revised blueprint.
