from __future__ import annotations

import numpy as np


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / n
    if q[3] < 0:
        q = -q
    return q


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


class AttitudeIntegrator:
    """Integrate body-frame gyro into orientation (no per-step ground truth)."""

    def __init__(self) -> None:
        self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def reset(self, initial_quat: np.ndarray | None = None) -> None:
        if initial_quat is not None:
            self._quat = _normalize_quat(initial_quat)
        else:
            self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    def step(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        gyro = np.asarray(gyro, dtype=np.float64).ravel()[:3]
        angle = float(np.linalg.norm(gyro) * dt)
        if angle < 1e-10:
            return self._quat.copy()
        axis = gyro / np.linalg.norm(gyro)
        half = angle * 0.5
        sh = np.sin(half)
        dq = np.array([axis[0] * sh, axis[1] * sh, axis[2] * sh, np.cos(half)], dtype=np.float64)
        self._quat = _normalize_quat(_quat_multiply(self._quat, dq))
        return self._quat.copy()
