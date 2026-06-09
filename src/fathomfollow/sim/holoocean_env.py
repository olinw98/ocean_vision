from __future__ import annotations

import math
from typing import Any

import numpy as np

from fathomfollow.sim.base import SimObservation


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion xyzw to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def global_to_body_velocity(vel_global: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    r = quat_to_rotation_matrix(quat_xyzw)
    return (r.T @ vel_global).astype(np.float32)


def extract_imu_6(raw_imu: np.ndarray) -> np.ndarray:
    """Extract accel_xyz + gyro_xyz from HoloOcean IMU vector."""
    arr = np.asarray(raw_imu, dtype=np.float32).ravel()
    if arr.size >= 18:
        accel = arr[0:3]
        gyro = arr[12:15]
        return np.concatenate([accel, gyro]).astype(np.float32)
    if arr.size >= 6:
        return arr[:6].astype(np.float32)
    raise ValueError(f"IMU vector too short: {arr.size}")


def map_holoocean_state(state: dict[str, Any], t: float) -> SimObservation:
    """Map raw HoloOcean state dict to SimObservation."""
    rgb_key = next((k for k in state if "Camera" in k or "RGB" in k), None)
    if rgb_key is None:
        raise KeyError("no RGB camera key in HoloOcean state")
    rgb = np.asarray(state[rgb_key], dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    imu_raw = np.asarray(state["IMUSensor"], dtype=np.float32)
    imu = extract_imu_6(imu_raw)

    dvl_raw = np.asarray(state["DVLSensor"], dtype=np.float64).ravel()
    vel_global = dvl_raw[:3]

    pose_key = next((k for k in state if "Pose" in k), "PoseSensor")
    pose_raw = np.asarray(state.get(pose_key, np.zeros(13)), dtype=np.float64).ravel()
    if pose_raw.size >= 13:
        position = pose_raw[9:12]
        rpy = pose_raw[15:18] if pose_raw.size >= 18 else pose_raw[3:6]
        quat = rpy_to_quat(rpy)
    else:
        position = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])

    gt_pose = np.concatenate([position, quat])
    dvl_body = global_to_body_velocity(vel_global, quat)

    target_pose = np.asarray(state.get("TargetPoseSensor", gt_pose), dtype=np.float64)
    if target_pose.size != 7:
        target_pose = gt_pose.copy()

    dvl_valid = bool(np.linalg.norm(vel_global) < 50.0)

    return SimObservation(
        t=t,
        rgb=rgb,
        imu=imu,
        dvl=dvl_body,
        dvl_valid=dvl_valid,
        gt_pose=gt_pose.astype(np.float64),
        gt_target_pose=target_pose.astype(np.float64),
    )


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float64)


class HoloOceanSimEnv:
    """Live HoloOcean environment (requires holoocean installed)."""

    def __init__(self, scenario: str = "PierHarbor-Hovering") -> None:
        import holoocean

        self._env = holoocean.make(scenario)
        self._t = 0.0
        self._dt = 0.033

    def reset(self) -> SimObservation:
        self._t = 0.0
        self._env.reset()
        return self._blank_obs()

    def step(self, command: "Command") -> SimObservation:
        from fathomfollow.sim.base import Command

        del Command
        thruster = self._command_to_thrusters(command)
        state = self._env.step(thruster)
        self._t += self._dt
        return map_holoocean_state(state, self._t)

    def close(self) -> None:
        self._env.__exit__(None, None, None)

    def _command_to_thrusters(self, command) -> np.ndarray:
        base = 10.0
        fwd = base + command.forward_vel * 5.0
        yaw = command.yaw_rate * 2.0
        vert = command.vertical_vel * 3.0
        return np.array([fwd + yaw, fwd - yaw, fwd + vert, fwd - vert, 0, 0, 0, 0])

    def _blank_obs(self) -> SimObservation:
        state = {}
        for key in ("FrontCamera", "RGBCamera", "IMUSensor", "DVLSensor", "PoseSensor"):
            try:
                state = self._env.step(np.zeros(8))
                break
            except Exception:
                continue
        return map_holoocean_state(state, self._t)
