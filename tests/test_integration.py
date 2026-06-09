from pathlib import Path

import numpy as np

from fathomfollow.integration import generate_trajectories_from_sim, run_dual_nav_step
from fathomfollow.nav.deadreckon import DeadReckoning
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture


def test_run_dual_nav_baseline_worse_in_dropout() -> None:
    """Baseline (DVL-only) should stop updating velocity when DVL drops."""
    dr_learned = DeadReckoning()
    dr_baseline = DeadReckoning()
    estimator = VelocityEstimator()
    imu_history: list[np.ndarray] = []

    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    dr_learned.reset(initial)
    dr_baseline.reset(initial)

    from fathomfollow.sim.base import SimObservation

    obs_valid = SimObservation(
        t=0.0,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        imu=np.zeros(6, dtype=np.float32),
        dvl=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dvl_valid=True,
        gt_pose=initial,
        gt_target_pose=initial,
    )
    run_dual_nav_step(obs_valid, dr_learned, dr_baseline, estimator, imu_history)

    obs_dropout = SimObservation(
        t=0.1,
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        imu=np.zeros(6, dtype=np.float32),
        dvl=np.zeros(3, dtype=np.float32),
        dvl_valid=False,
        gt_pose=initial,
        gt_target_pose=initial,
    )
    _, baseline_pos = run_dual_nav_step(
        obs_dropout, dr_learned, dr_baseline, estimator, imu_history
    )
    assert baseline_pos[0] > 0.0


def test_generate_trajectories_from_sim(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=30)
    paths = generate_trajectories_from_sim(fixture, tmp_path / "traj", n_steps=20)
    assert len(paths) == 1
    data = np.load(paths[0])
    assert data["imu"].shape[0] == 20
    assert data["gt_velocity_body"].shape == (20, 3)
