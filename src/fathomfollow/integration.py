from __future__ import annotations

from pathlib import Path

import numpy as np

from fathomfollow.nav.attitude import AttitudeIntegrator
from fathomfollow.nav.deadreckon import DeadReckoning
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.nav.training import write_trajectory_npz
from fathomfollow.nav.trajectories import log_trajectory
from fathomfollow.perception.detector import MockDetector, YoloDetector
from fathomfollow.perception.sim_infer import SimInferReport, run_sim_inference
from fathomfollow.sim.base import SimEnv
from fathomfollow.sim.recorded import RecordedSimEnv


def record_sim_baseline(
    env: SimEnv,
    detector: YoloDetector | MockDetector,
    max_steps: int = 100,
    conf_threshold: float = 0.25,
) -> SimInferReport:
    """Run detector over sim frames and return firing-rate report (Step 1.3)."""
    return run_sim_inference(env, detector, max_steps=max_steps, conf_threshold=conf_threshold)


def generate_trajectories_from_sim(
    fixture: Path,
    out_dir: Path,
    n_steps: int = 200,
    seed: int = 0,
) -> list[Path]:
    """Log body-frame velocity GT from a recorded sim fixture for nav training."""
    env = RecordedSimEnv(fixture)
    frames = log_trajectory(env, n_steps)
    env.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    imu = np.stack([f.imu for f in frames])
    dvl = np.stack([f.dvl if f.dvl is not None else np.zeros(3) for f in frames])
    dvl_valid = np.array([f.dvl_valid for f in frames], dtype=bool)
    vel = np.stack([f.gt_velocity_body for f in frames])
    path = out_dir / f"traj_{seed}.npz"
    write_trajectory_npz(path, imu, dvl, dvl_valid, vel)
    return [path]


def run_dual_nav_step(
    obs,
    dr_learned: DeadReckoning,
    dr_baseline: DeadReckoning,
    estimator: VelocityEstimator,
    imu_history: list[np.ndarray],
    attitude: AttitudeIntegrator,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Step learned (estimator-assisted) and baseline (DVL-only) dead reckoning."""
    imu_history.append(obs.imu.copy())
    win = np.stack(imu_history[-estimator.window_size :])
    est_vel = estimator.estimate(win, obs.dvl if obs.dvl_valid else None)

    quat = attitude.step(obs.imu[3:6], dt)

    learned_pos = dr_learned.step(
        obs.dvl if obs.dvl_valid else None,
        est_vel,
        obs.dvl_valid,
        dt,
        quat,
    )
    baseline_pos = dr_baseline.step(
        obs.dvl if obs.dvl_valid else None,
        None,
        obs.dvl_valid,
        dt,
        quat,
    )
    return learned_pos, baseline_pos
