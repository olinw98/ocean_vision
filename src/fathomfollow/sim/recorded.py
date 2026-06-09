from __future__ import annotations

from pathlib import Path

import numpy as np

from fathomfollow.sim.base import Command, SimEnv, SimObservation


class RecordedSimEnv:
    """Replay logged observations from an NPZ fixture."""

    def __init__(self, fixture_path: Path | str) -> None:
        self._path = Path(fixture_path)
        data = np.load(self._path, allow_pickle=False)
        self._t = data["t"]
        self._rgb = data["rgb"]
        self._imu = data["imu"]
        self._dvl = data["dvl"]
        self._dvl_valid = data["dvl_valid"]
        self._gt_pose = data["gt_pose"]
        self._gt_target_pose = data["gt_target_pose"]
        self._idx = 0
        self._done = False

    def reset(self) -> SimObservation:
        self._idx = 0
        self._done = False
        return self._obs_at(0)

    def step(self, command: Command) -> SimObservation:
        del command
        if self._idx >= len(self._t) - 1:
            self._done = True
            return self._obs_at(len(self._t) - 1)
        self._idx += 1
        return self._obs_at(self._idx)

    def close(self) -> None:
        pass

    @property
    def done(self) -> bool:
        return self._done

    def _obs_at(self, idx: int) -> SimObservation:
        return SimObservation(
            t=float(self._t[idx]),
            rgb=self._rgb[idx].copy(),
            imu=self._imu[idx].copy(),
            dvl=self._dvl[idx].copy(),
            dvl_valid=bool(self._dvl_valid[idx]),
            gt_pose=self._gt_pose[idx].copy(),
            gt_target_pose=self._gt_target_pose[idx].copy(),
        )


def write_sim_fixture(path: Path, n_frames: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    t = np.arange(n_frames, dtype=np.float64) * 0.1
    rgb = rng.integers(0, 255, size=(n_frames, h, w, 3), dtype=np.uint8)
    imu = rng.normal(size=(n_frames, 6)).astype(np.float32)
    dvl = rng.normal(size=(n_frames, 3)).astype(np.float32)
    dvl_valid = np.ones(n_frames, dtype=bool)
    dvl_valid[3:5] = False
    gt_pose = np.zeros((n_frames, 7), dtype=np.float64)
    gt_pose[:, 6] = 1.0
    gt_pose[:, 0] = t * 0.1
    gt_target_pose = gt_pose.copy()
    gt_target_pose[:, 0] += 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=t,
        rgb=rgb,
        imu=imu,
        dvl=dvl,
        dvl_valid=dvl_valid,
        gt_pose=gt_pose,
        gt_target_pose=gt_target_pose,
    )
