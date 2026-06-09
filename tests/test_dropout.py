from pathlib import Path

import numpy as np

from fathomfollow.config.models import DropoutConfig
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture


def test_dropout_forced_window(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture)
    env = RecordedSimEnv(fixture)
    cfg = DropoutConfig(alt_min=0.0, forced_windows=[(0.2, 0.5)])
    wrapped = DropoutSimEnv(env, cfg)
    obs = wrapped.reset()
    from fathomfollow.sim.base import Command

    cmd = Command(0, 0, 0)
    saw_invalid = False
    for _ in range(10):
        if not obs.dvl_valid:
            saw_invalid = True
            np.testing.assert_array_equal(obs.dvl, np.zeros(3))
        obs = wrapped.step(cmd)
    assert saw_invalid
