"""Record observations from any SimEnv into an NPZ fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fathomfollow.sim.base import Command, SimEnv


def build_fathomnet_proxy_fixture(
    image_dir: Path,
    out_path: Path,
    n_frames: int | None = None,
    seed: int = 0,
) -> int:
    """Build a sim fixture using real FathomNet images as RGB (HoloOcean proxy for Step 1.3)."""
    images = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no images under {image_dir}")

    rng = np.random.default_rng(seed)
    n = n_frames or min(len(images), 100)
    t = np.arange(n, dtype=np.float64) * 0.1
    rgb_list: list[np.ndarray] = []
    for i in range(n):
        img = Image.open(images[i % len(images)]).convert("RGB")
        arr = np.asarray(img.resize((640, 480)), dtype=np.uint8)
        rgb_list.append(arr)

    imu = rng.normal(size=(n, 6)).astype(np.float32)
    dvl = rng.normal(size=(n, 3)).astype(np.float32)
    dvl_valid = np.ones(n, dtype=bool)
    dvl_valid[max(1, n // 10) : max(2, n // 5)] = False
    gt_pose = np.zeros((n, 7), dtype=np.float64)
    gt_pose[:, 6] = 1.0
    gt_pose[:, 0] = t * 0.05
    gt_target_pose = gt_pose.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        t=t,
        rgb=np.stack(rgb_list),
        imu=imu,
        dvl=dvl,
        dvl_valid=dvl_valid,
        gt_pose=gt_pose,
        gt_target_pose=gt_target_pose,
    )
    return n


def record_sim_fixture(
    env: SimEnv,
    path: Path,
    n_frames: int,
    dt: float = 0.1,
) -> None:
    obs = env.reset()
    frames_t: list[float] = [obs.t]
    frames_rgb: list[np.ndarray] = [obs.rgb]
    frames_imu: list[np.ndarray] = [obs.imu]
    frames_dvl: list[np.ndarray] = [obs.dvl]
    frames_valid: list[bool] = [obs.dvl_valid]
    frames_gt: list[np.ndarray] = [obs.gt_pose]
    frames_tgt: list[np.ndarray] = [obs.gt_target_pose]

    cmd = Command(0.0, 0.0, 0.0)
    for _ in range(n_frames - 1):
        obs = env.step(cmd)
        frames_t.append(obs.t if obs.t > frames_t[-1] else frames_t[-1] + dt)
        frames_rgb.append(obs.rgb)
        frames_imu.append(obs.imu)
        frames_dvl.append(obs.dvl)
        frames_valid.append(obs.dvl_valid)
        frames_gt.append(obs.gt_pose)
        frames_tgt.append(obs.gt_target_pose)

    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=np.asarray(frames_t, dtype=np.float64),
        rgb=np.stack(frames_rgb),
        imu=np.stack(frames_imu),
        dvl=np.stack(frames_dvl),
        dvl_valid=np.asarray(frames_valid, dtype=bool),
        gt_pose=np.stack(frames_gt),
        gt_target_pose=np.stack(frames_tgt),
    )
