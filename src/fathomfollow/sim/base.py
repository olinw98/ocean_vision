from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Command:
    forward_vel: float
    yaw_rate: float
    vertical_vel: float


@dataclass(frozen=True)
class SimObservation:
    t: float
    rgb: np.ndarray
    imu: np.ndarray  # shape (6,) accel_xyz + gyro_xyz
    dvl: np.ndarray  # shape (3,) body-frame velocity m/s
    dvl_valid: bool
    gt_pose: np.ndarray  # shape (7,) position xyz + quat xyzw
    gt_target_pose: np.ndarray  # shape (7,)


@runtime_checkable
class SimEnv(Protocol):
    def reset(self) -> SimObservation: ...

    def step(self, command: Command) -> SimObservation: ...

    def close(self) -> None: ...
