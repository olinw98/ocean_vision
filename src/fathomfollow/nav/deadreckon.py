from __future__ import annotations

import numpy as np


class DeadReckoning:
    """Naive dead-reckoning: integrate velocity into position."""

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity_body = np.zeros(3, dtype=np.float32)

    def reset(self, initial_pose: np.ndarray | None = None) -> None:
        if initial_pose is not None:
            self.position = initial_pose[:3].copy()
        else:
            self.position = np.zeros(3, dtype=np.float64)
        self.velocity_body = np.zeros(3, dtype=np.float32)

    def step(
        self,
        dvl: np.ndarray | None,
        estimator_vel: np.ndarray | None,
        dvl_valid: bool,
        dt: float,
        orientation: np.ndarray,
    ) -> np.ndarray:
        from fathomfollow.sim.holoocean_env import quat_to_rotation_matrix

        if dvl_valid and dvl is not None:
            self.velocity_body = dvl.astype(np.float32)
        elif estimator_vel is not None:
            self.velocity_body = estimator_vel.astype(np.float32)

        r = quat_to_rotation_matrix(orientation)
        vel_global = r @ self.velocity_body
        self.position = self.position + vel_global * dt
        return self.position.copy()
