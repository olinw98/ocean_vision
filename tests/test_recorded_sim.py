from pathlib import Path

import numpy as np
import pytest

from fathomfollow.sim.base import Command
from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture


def test_recorded_sim_replay(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=10)
    env = RecordedSimEnv(fixture)
    obs = env.reset()
    assert obs.rgb.shape == (480, 640, 3)
    assert obs.imu.shape == (6,)
    assert obs.dvl.shape == (3,)
    assert isinstance(obs.dvl_valid, bool)

    cmd = Command(0.1, 0.0, 0.0)
    for _ in range(9):
        obs = env.step(cmd)
    assert obs.t == pytest.approx(0.9, abs=0.01)
    env.close()


def test_protocol_honored(tmp_path: Path) -> None:
    from fathomfollow.sim.base import SimEnv

    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture)
    env = RecordedSimEnv(fixture)
    assert isinstance(env, SimEnv)
