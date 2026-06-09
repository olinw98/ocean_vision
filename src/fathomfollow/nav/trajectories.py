from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fathomfollow.sim.base import Command, SimEnv


@dataclass
class TrajectoryFrame:
    t: float
    imu: np.ndarray
    dvl: np.ndarray | None
    dvl_valid: bool
    gt_pose: np.ndarray
    gt_velocity_body: np.ndarray


def body_velocity_from_poses(
    pose_prev: np.ndarray, pose_curr: np.ndarray, dt: float
) -> np.ndarray:
    """Finite-difference global displacement rotated into body frame."""
    from fathomfollow.sim.holoocean_env import global_to_body_velocity

    pos_prev, quat = pose_prev[:3], pose_prev[3:7]
    pos_curr = pose_curr[:3]
    vel_global = (pos_curr - pos_prev) / dt
    return global_to_body_velocity(vel_global, quat)


def log_trajectory(env: SimEnv, n_steps: int) -> list[TrajectoryFrame]:
    frames: list[TrajectoryFrame] = []
    obs = env.reset()
    prev_pose = obs.gt_pose
    prev_t = obs.t
    cmd = Command(0.0, 0.0, 0.0)

    vel_body = np.zeros(3, dtype=np.float32)
    frames.append(
        TrajectoryFrame(
            t=obs.t,
            imu=obs.imu.copy(),
            dvl=obs.dvl.copy() if obs.dvl_valid else None,
            dvl_valid=obs.dvl_valid,
            gt_pose=obs.gt_pose.copy(),
            gt_velocity_body=vel_body,
        )
    )

    for _ in range(n_steps - 1):
        obs = env.step(cmd)
        dt = max(obs.t - prev_t, 1e-6)
        vel_body = body_velocity_from_poses(prev_pose, obs.gt_pose, dt)
        frames.append(
            TrajectoryFrame(
                t=obs.t,
                imu=obs.imu.copy(),
                dvl=obs.dvl.copy() if obs.dvl_valid else None,
                dvl_valid=obs.dvl_valid,
                gt_pose=obs.gt_pose.copy(),
                gt_velocity_body=vel_body,
            )
        )
        prev_pose = obs.gt_pose
        prev_t = obs.t
    return frames
