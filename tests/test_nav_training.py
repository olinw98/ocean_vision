from pathlib import Path

import numpy as np

from fathomfollow.config.models import NavTrainingConfig
from fathomfollow.nav.training import build_training_windows, train_nav_estimator, write_trajectory_npz


def test_build_training_windows_shape() -> None:
    n = 30
    imu = np.random.randn(n, 6).astype(np.float32)
    dvl = np.random.randn(n, 3).astype(np.float32)
    dvl_valid = np.array([True, False, True] * 10, dtype=bool)
    vel = np.random.randn(n, 3).astype(np.float32)

    x, y = build_training_windows(imu, dvl, dvl_valid, vel, window_size=10)

    assert x.ndim == 3
    assert x.shape[1] == 10
    assert x.shape[2] == 9
    assert y.shape[1] == 3
    assert len(x) == n - 10 + 1


def test_train_nav_estimator_saves_checkpoint(tmp_path: Path) -> None:
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir()
    n = 40
    write_trajectory_npz(
        traj_dir / "traj_0.npz",
        imu=np.random.randn(n, 6).astype(np.float32),
        dvl=np.random.randn(n, 3).astype(np.float32),
        dvl_valid=np.ones(n, dtype=bool),
        gt_velocity_body=np.random.randn(n, 3).astype(np.float32),
    )

    cfg = NavTrainingConfig(
        trajectories_dir=traj_dir,
        epochs=5,
        window_size=10,
        hidden_size=16,
        lr=1e-2,
    )
    ckpt = train_nav_estimator(cfg, out_dir=tmp_path / "nav_ckpt")

    assert ckpt.exists()
    metrics = (tmp_path / "nav_ckpt" / "metrics.json").read_text(encoding="utf-8")
    assert "final_loss" in metrics
