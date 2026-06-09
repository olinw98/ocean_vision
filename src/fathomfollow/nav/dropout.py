from __future__ import annotations

from dataclasses import replace

import numpy as np

from fathomfollow.config.models import DropoutConfig
from fathomfollow.sim.base import Command, SimEnv, SimObservation


class DropoutSimEnv:
    """Wraps SimEnv and flips dvl_valid per dropout rules."""

    def __init__(self, env: SimEnv, config: DropoutConfig, seed: int = 0) -> None:
        self._env = env
        self._config = config
        self._rng = np.random.default_rng(seed)

    def reset(self) -> SimObservation:
        return self._apply_dropout(self._env.reset())

    def step(self, command: Command) -> SimObservation:
        return self._apply_dropout(self._env.step(command))

    def close(self) -> None:
        self._env.close()

    def _apply_dropout(self, obs: SimObservation) -> SimObservation:
        alt = -obs.gt_pose[2]
        valid = alt > self._config.alt_min
        for start, end in self._config.forced_windows:
            if start <= obs.t <= end:
                valid = False
        if not valid:
            obs = replace(obs, dvl_valid=False, dvl=np.zeros(3, dtype=np.float32))
        return obs
