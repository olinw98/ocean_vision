from __future__ import annotations

import math
from typing import Any

import numpy as np

from fathomfollow.config.models import TargetMimicConfig
from fathomfollow.sim.base import SimObservation
from fathomfollow.sim.target import target_pose_at, target_position_at

MAIN_AGENT_NAME = "auv0"
TARGET_MIMIC_AGENT_NAME = "target_mimic0"


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


def select_agent_state(state: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Extract one agent's sensor dict from HoloOcean multi-agent state."""
    if agent_name in state and isinstance(state[agent_name], dict):
        nested = state[agent_name]
        if any(isinstance(v, np.ndarray) for v in nested.values()):
            return nested
    if agent_name == MAIN_AGENT_NAME:
        flat = flatten_holoocean_state(state)
        if find_rgb_key(flat) is not None or "IMUSensor" in flat:
            return flat
    raise KeyError(f"agent {agent_name!r} not found in HoloOcean state")


def flatten_holoocean_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize HoloOcean 2.x state to a flat sensor-name dict."""
    if not state:
        return state
    if MAIN_AGENT_NAME in state and isinstance(state[MAIN_AGENT_NAME], dict):
        return state[MAIN_AGENT_NAME]
    agent_states = [
        v for v in state.values() if isinstance(v, dict) and any(isinstance(x, np.ndarray) for x in v.values())
    ]
    if agent_states:
        return agent_states[0]
    return state


def find_rgb_key(state: dict[str, Any]) -> str | None:
    preferred = ("LeftCamera", "FrontCamera", "RGBCamera", "RightCamera")
    for name in preferred:
        if name in state:
            return name
    return next((k for k in state if "Camera" in k or "RGB" in k.upper()), None)


def map_holoocean_state(
    state: dict[str, Any],
    t: float,
    *,
    main_agent: str = MAIN_AGENT_NAME,
    target_config: TargetMimicConfig | None = None,
) -> SimObservation:
    """Map raw HoloOcean state dict to SimObservation.

    When ``target_config`` is set, ``gt_target_pose`` is the scripted trajectory
    from ``target_pose_at(t, config)`` (eval harness contract). The spawned
    SphereAgent is teleported to match; logged GT does not read LocationSensor.
    """
    agent_state = select_agent_state(state, main_agent)
    rgb_key = find_rgb_key(agent_state)
    if rgb_key is None:
        keys = sorted(k for k in agent_state if k != "t")
        raise KeyError(
            f"no RGB camera key in HoloOcean state (keys={keys}). "
            "Use a camera-equipped scenario such as PierHarbor-HoveringCamera."
        )
    rgb = np.asarray(agent_state[rgb_key], dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    imu_raw = np.asarray(agent_state["IMUSensor"], dtype=np.float32)
    imu = extract_imu_6(imu_raw)

    dvl_raw = np.asarray(agent_state["DVLSensor"], dtype=np.float64).ravel()
    vel_global = dvl_raw[:3]

    pose_key = next((k for k in agent_state if "Pose" in k), "PoseSensor")
    pose_raw = np.asarray(agent_state.get(pose_key, np.zeros(13)), dtype=np.float64).ravel()
    if pose_raw.size >= 13:
        position = pose_raw[9:12]
        rpy = pose_raw[15:18] if pose_raw.size >= 18 else pose_raw[3:6]
        quat = rpy_to_quat(rpy)
    else:
        position = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])

    gt_pose = np.concatenate([position, quat])
    dvl_body = global_to_body_velocity(vel_global, quat)

    if target_config is not None:
        gt_target_pose = target_pose_at(t, target_config)
    else:
        target_pose = np.asarray(agent_state.get("TargetPoseSensor", gt_pose), dtype=np.float64)
        if target_pose.size != 7:
            target_pose = gt_pose.copy()
        gt_target_pose = target_pose

    dvl_valid = bool(np.linalg.norm(vel_global) < 50.0)

    return SimObservation(
        t=t,
        rgb=rgb,
        imu=imu,
        dvl=dvl_body,
        dvl_valid=dvl_valid,
        gt_pose=gt_pose.astype(np.float64),
        gt_target_pose=gt_target_pose.astype(np.float64),
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

    def __init__(
        self,
        scenario: str = "PierHarbor-HoveringCamera",
        target_config: TargetMimicConfig | None = None,
    ) -> None:
        import holoocean

        self._env = holoocean.make(scenario)
        self._target_config = target_config
        self._t = 0.0
        self._dt = 0.033
        self._last_rgb: np.ndarray | None = None
        self._target_spawned = False

    def reset(self) -> SimObservation:
        self._t = 0.0
        self._last_rgb = None
        self._env.reset()
        # HoloOcean may retain or drop dynamically added agents across reset().
        self._target_spawned = TARGET_MIMIC_AGENT_NAME in self._env.agents
        if self._target_config is not None and not self._target_spawned:
            self._spawn_target_mimic()
        return self._initial_obs()

    def step(self, command: "Command") -> SimObservation:
        from fathomfollow.sim.base import Command

        del Command
        thruster = self._command_to_thrusters(command)
        state = self._env.step(thruster)
        self._t += self._dt
        self._update_target_mimic()
        return self._observe(state)

    def _observe(self, state: dict[str, Any]) -> SimObservation:
        if find_rgb_key(select_agent_state(state, MAIN_AGENT_NAME)) is None and self._last_rgb is not None:
            state = {
                **state,
                MAIN_AGENT_NAME: {
                    **select_agent_state(state, MAIN_AGENT_NAME),
                    "LeftCamera": self._last_rgb,
                },
            }
        obs = map_holoocean_state(state, self._t, target_config=self._target_config)
        self._last_rgb = obs.rgb.copy()
        return obs

    def close(self) -> None:
        self._env.__exit__(None, None, None)

    def _command_to_thrusters(self, command) -> np.ndarray:
        base = 10.0
        fwd = base + command.forward_vel * 5.0
        yaw = command.yaw_rate * 2.0
        vert = command.vertical_vel * 3.0
        return np.array([fwd + yaw, fwd - yaw, fwd + vert, fwd - vert, 0, 0, 0, 0])

    def _initial_obs(self) -> SimObservation:
        """First observation after reset (HoloOcean returns sensors on first tick)."""
        state = self._env.step(np.zeros(8))
        self._update_target_mimic()
        return self._observe(state)

    def _spawn_target_mimic(self) -> None:
        if self._target_config is None or self._target_spawned:
            return
        if TARGET_MIMIC_AGENT_NAME in self._env.agents:
            self._target_spawned = True
            return
        from holoocean.agents import AgentDefinition
        from holoocean.sensors import SensorDefinition

        loc = target_position_at(0.0, self._target_config).tolist()
        agent_def = AgentDefinition(
            agent_name=TARGET_MIMIC_AGENT_NAME,
            agent_type="SphereAgent",
            sensors=[
                SensorDefinition(
                    TARGET_MIMIC_AGENT_NAME,
                    "SphereAgent",
                    "LocationSensor",
                    "LocationSensor",
                )
            ],
            starting_loc=loc,
            starting_rot=(0, 0, 0),
        )
        self._env.add_agent(agent_def)
        self._target_spawned = True

    def _update_target_mimic(self) -> None:
        if self._target_config is None:
            return
        if not self._target_spawned:
            self._spawn_target_mimic()
        if TARGET_MIMIC_AGENT_NAME in self._env.agents:
            loc = target_position_at(self._t, self._target_config).tolist()
            self._env.agents[TARGET_MIMIC_AGENT_NAME].teleport(loc)
