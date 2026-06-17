from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fathomfollow.config.models import NavTrainingConfig
from fathomfollow.nav.estimator import VelocityGRU


def write_trajectory_npz(
    path: Path,
    imu: np.ndarray,
    dvl: np.ndarray,
    dvl_valid: np.ndarray,
    gt_velocity_body: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        imu=imu.astype(np.float32),
        dvl=dvl.astype(np.float32),
        dvl_valid=dvl_valid.astype(bool),
        gt_velocity_body=gt_velocity_body.astype(np.float32),
    )


def build_training_windows(
    imu: np.ndarray,
    dvl: np.ndarray,
    dvl_valid: np.ndarray,
    gt_velocity_body: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(imu)
    if n < window_size:
        return np.zeros((0, window_size, 9), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(window_size - 1, n):
        start = end - window_size + 1
        imu_win = imu[start : end + 1]
        dvl_win = np.zeros((window_size, 3), dtype=np.float32)
        for i, idx in enumerate(range(start, end + 1)):
            if dvl_valid[idx]:
                dvl_win[i] = dvl[idx]
        xs.append(np.concatenate([imu_win, dvl_win], axis=1))
        ys.append(gt_velocity_body[end])

    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def _load_trajectory_windows(traj_dir: Path, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for path in sorted(traj_dir.glob("*.npz")):
        data = np.load(path)
        x, y = build_training_windows(
            data["imu"],
            data["dvl"],
            data["dvl_valid"],
            data["gt_velocity_body"],
            window_size,
        )
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        raise ValueError(f"no training windows from {traj_dir}")
    return np.concatenate(all_x), np.concatenate(all_y)


def train_nav_estimator(cfg: NavTrainingConfig, out_dir: Path) -> Path:
    torch.manual_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np, y_np = _load_trajectory_windows(cfg.trajectories_dir, cfg.window_size)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    model = VelocityGRU(input_size=9, hidden_size=cfg.hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    final_loss = 0.0
    for _ in range(cfg.epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    ckpt_path = out_dir / "velocity_estimator.pt"
    torch.save(model.state_dict(), ckpt_path)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "final_loss": final_loss,
                "epochs": cfg.epochs,
                "hidden_size": cfg.hidden_size,
                "window_size": cfg.window_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ckpt_path
