from __future__ import annotations

import math

import numpy as np

from fathomfollow.config.models import TargetMimicConfig


def target_position_at(t: float, config: TargetMimicConfig) -> np.ndarray:
    if config.trajectory == "circle":
        angle = t * config.speed / max(config.radius, 0.1)
        x = config.radius * math.cos(angle)
        y = config.radius * math.sin(angle)
        return np.array([x, y, config.depth], dtype=np.float64)
    return np.array([t * config.speed, 0.0, config.depth], dtype=np.float64)


def target_pose_at(t: float, config: TargetMimicConfig) -> np.ndarray:
    """Ground-truth target pose (xyz + quat xyzw) for eval harness."""
    position = target_position_at(t, config)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.concatenate([position, quat])
