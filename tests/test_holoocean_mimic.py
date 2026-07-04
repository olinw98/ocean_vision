"""HoloOcean target mimic wiring (mock holoocean — no Unreal in pytest)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fathomfollow.config.models import TargetMimicConfig
from fathomfollow.sim.base import Command
from fathomfollow.sim.holoocean_env import (
    MAIN_AGENT_NAME,
    TARGET_MIMIC_AGENT_NAME,
    map_holoocean_state,
    select_agent_state,
)
from fathomfollow.sim.target import target_pose_at, target_position_at


def _auv_state_dict() -> dict:
    pose = np.zeros(18, dtype=np.float64)
    pose[9:12] = [10.0, 20.0, -5.0]
    pose[15:18] = [0.0, 0.0, 0.0]
    return {
        "LeftCamera": np.zeros((64, 64, 3), dtype=np.uint8),
        "IMUSensor": np.zeros(18, dtype=np.float32),
        "DVLSensor": np.array([0.1, 0.0, 0.0, 0, 0, 0, 0], dtype=np.float64),
        "PoseSensor": pose,
    }


def test_select_agent_state_multi_agent() -> None:
    state = {MAIN_AGENT_NAME: {"LeftCamera": np.zeros(1)}, "other": {"x": 1}}
    flat = select_agent_state(state, MAIN_AGENT_NAME)
    assert "LeftCamera" in flat


def test_select_agent_state_legacy_flat_state() -> None:
    flat = {
        "LeftCamera": np.zeros((2, 2, 3), dtype=np.uint8),
        "IMUSensor": np.zeros(6, dtype=np.float32),
    }
    out = select_agent_state(flat, MAIN_AGENT_NAME)
    assert out is flat


def test_map_holoocean_state_without_target_config_falls_back_to_auv() -> None:
    state = _auv_state_dict()
    obs = map_holoocean_state(state, t=0.0, target_config=None)
    np.testing.assert_allclose(obs.gt_target_pose, obs.gt_pose)


def test_map_holoocean_state_target_pose_differs_from_auv() -> None:
    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5, depth=-10.0)
    t = 2.0
    state = {
        MAIN_AGENT_NAME: _auv_state_dict(),
        TARGET_MIMIC_AGENT_NAME: {
            "LocationSensor": target_position_at(t, cfg),
        },
    }
    obs = map_holoocean_state(state, t=t, target_config=cfg)
    np.testing.assert_allclose(obs.gt_target_pose, target_pose_at(t, cfg))
    assert not np.allclose(obs.gt_pose[:3], obs.gt_target_pose[:3], atol=0.5)


@patch("holoocean.make")
def test_holoocean_sim_env_spawns_and_teleports_target(mock_make: MagicMock) -> None:
    holoocean = pytest.importorskip("holoocean", reason="holoocean client needed for import path")
    del holoocean

    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5, depth=-10.0)
    mock_env = MagicMock()
    mock_target_agent = MagicMock()
    mock_env.agents = {}

    def _add_agent(agent_def) -> None:
        mock_env.agents[agent_def.name] = mock_target_agent

    mock_env.add_agent.side_effect = _add_agent
    mock_env.step.return_value = {MAIN_AGENT_NAME: _auv_state_dict()}
    mock_make.return_value = mock_env

    from fathomfollow.sim.holoocean_env import HoloOceanSimEnv

    env = HoloOceanSimEnv(scenario="PierHarbor-HoveringCamera", target_config=cfg)
    obs = env.reset()
    assert mock_env.add_agent.call_count == 1
    assert not np.allclose(obs.gt_pose[:3], obs.gt_target_pose[:3], atol=0.5)

    env.step(Command(0.0, 0.0, 0.0))
    assert mock_target_agent.teleport.called
    teleport_loc = mock_target_agent.teleport.call_args[0][0]
    expected = target_position_at(env._t, cfg).tolist()
    np.testing.assert_allclose(teleport_loc, expected, rtol=1e-5)

    env.close()


@patch("holoocean.make")
def test_holoocean_sim_env_reset_does_not_double_add_agent(mock_make: MagicMock) -> None:
    pytest.importorskip("holoocean")

    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5, depth=-10.0)
    mock_env = MagicMock()
    mock_target_agent = MagicMock()
    mock_env.agents = {}

    def _add_agent(agent_def) -> None:
        mock_env.agents[agent_def.name] = mock_target_agent

    mock_env.add_agent.side_effect = _add_agent
    mock_env.step.return_value = {MAIN_AGENT_NAME: _auv_state_dict()}
    mock_make.return_value = mock_env

    from fathomfollow.sim.holoocean_env import HoloOceanSimEnv

    env = HoloOceanSimEnv(scenario="PierHarbor-HoveringCamera", target_config=cfg)
    env.reset()
    assert mock_env.add_agent.call_count == 1

    env.reset()
    assert mock_env.add_agent.call_count == 1
    assert mock_target_agent.teleport.call_count >= 2

    env.close()
