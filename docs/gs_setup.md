# Gaussian Splatting setup (Step 1.5.2)

WaterSplatting requires a separate conda environment (Python 3.8, torch 2.1.2+cu118).

## Windows notes (this machine)

Anaconda is at `%USERPROFILE%\anaconda3` but may not be on PATH in Cursor terminals.
Use the full path or add `anaconda3`, `anaconda3\Scripts`, `anaconda3\Library\bin` to user PATH, then restart Cursor.

**Main project GPU (YOLO / ff-train-detector):** uses `.venv` with CUDA PyTorch — not conda.
Reinstall if needed: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

**GS env status (2026-06-10, VS 2022 setup complete):**
- `water_splatting` conda env: created
- `torch 2.1.2+cu118`: installed, `cuda.is_available() == True` on RTX 4060
- **CUDA dev via conda (minimal, not full toolkit):** `cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`, `cuda-nvrtc-dev`, `libcusparse-dev`, `libcublas-dev`, `libcusolver-dev` — provides `nvcc`, `nvrtc.h`, `cusparse.h`, etc. Set `CUDA_HOME=%USERPROFILE%\anaconda3\envs\water_splatting` before pip builds.
- **Full `cuda-toolkit` conda:** still fails on Windows path-length (`cuda-nvvp` extract)
- **MSVC:** VS **2022** Build Tools required (`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`). VS 2026/18.x is too new for CUDA 11.8 nvcc host check.
- **Repos cloned (sibling of ocean_vision):** `../tiny-cuda-nn` (patched setup.py), `../water-splatting`
- **`tinycudann`:** installed (VS 2022 vcvars + Windows Kits `rc.exe` on PATH + STL mismatch define)
- **`nerfstudio==1.1.4`:** installed (preinstall `pywinpty==2.0.13` + `pywin32==306` wheels before pip)
- **`water-splatting` pip install -e .:** installed (same vcvars + `NVCC_FLAGS=-allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`)
- **`ns-install-cli`:** run once; set `HOME=%USERPROFILE%` if it exits early on Windows

## Create env

```powershell
# If conda not on PATH:
$conda = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
& $conda create -n water_splatting python=3.8 -y
& $conda run -n water_splatting pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

```bash
conda create -n water_splatting python=3.8 -y
conda activate water_splatting
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Remaining GS install (after MSVC Build Tools)

**Critical:** CUDA 11.8 + WaterSplatting need **Visual Studio 2022** Build Tools (MSVC v143), not VS 2026/18.x. Install side-by-side:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Minimal CUDA compiler headers (if full `cuda-toolkit` conda fails on path length):

```powershell
$conda = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
& $conda install -n water_splatting -c "nvidia/label/cuda-11.8.0" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-nvrtc-dev libcusparse-dev libcublas-dev libcusolver-dev -y
```

Build `tiny-cuda-nn` from patched local clone (use **VS 2022** `vcvars64.bat`, not VS 18). Patch `bindings/torch/setup.py` to add `-allow-unsupported-compiler` and `/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` (MSVC 14.44 + CUDA 11.8). Ensure `tinycudann_bindings/` exists before pip install. Add Windows Kits `rc.exe` to PATH:

```powershell
$conda = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
$cudaHome = "$env:USERPROFILE\anaconda3\envs\water_splatting"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$winkit = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
$src = "$env:USERPROFILE\Desktop\Projects\tiny-cuda-nn\bindings\torch"
New-Item -ItemType Directory -Force -Path "$src\tinycudann_bindings" | Out-Null
cmd /c "`"$vcvars`" && set DISTUTILS_USE_SDK=1&& set CUDA_HOME=$cudaHome&& set PATH=$winkit;$cudaHome\bin;$cudaHome\Scripts;%PATH%&& cd /d `"$src`" && `"$cudaHome\python.exe`" -m pip install -e ."
```

Then nerfstudio + water-splatting (use a `.bat` file or single `cmd /c` session so `cl.exe` stays on PATH for CUDA extensions):

```powershell
& $conda run -n water_splatting pip install pywin32==306 pywinpty==2.0.13
& $conda run -n water_splatting pip install nerfstudio==1.1.4
& $conda run -n water_splatting cmd /c "set HOME=%USERPROFILE%&& ns-install-cli"
```

`water-splatting` must compile CUDA inside the same vcvars session (PowerShell `conda run` alone loses `cl.exe`):

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CUDA_HOME=%USERPROFILE%\anaconda3\envs\water_splatting
set NVCC_FLAGS=-allow-unsupported-compiler -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%CUDA_HOME%\bin;%CUDA_HOME%\Scripts;%PATH%
cd /d "%USERPROFILE%\Desktop\Projects\water-splatting"
"%CUDA_HOME%\python.exe" -m pip install --no-use-pep517 -e .
```

Verify:

```powershell
& $conda run -n water_splatting python -c "import tinycudann, nerfstudio, water_splatting, torch; print(torch.cuda.is_available())"
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
