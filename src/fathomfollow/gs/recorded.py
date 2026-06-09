from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fathomfollow.gs.base import GSRenderer, Pose


@dataclass
class GSScene:
    scene_id: str
    source_dataset: str
    library: str
    n_gaussians: int
    pose_source: str
    checkpoint_path: str
    train_psnr: float

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GSScene:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class GSRenderManifest:
    scene_id: str
    camera_path_id: str
    turbidity_values: list[float]
    n_frames: int
    label_strategy: str
    out_dir: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class RecordedGSRenderer:
    """Replays pre-rendered RGB frames from fixture for tests."""

    def __init__(self, fixture_dir: Path | str) -> None:
        self._dir = Path(fixture_dir)
        meta_path = self._dir / "manifest.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self._meta = {}
        self._frames = sorted(self._dir.glob("frame_*.npy"))
        self._idx = 0
        self._seed = 0

    def load(self, checkpoint: str) -> None:
        self._checkpoint = checkpoint

    def render(self, pose: Pose, turbidity: float) -> np.ndarray:
        if not 0.0 <= turbidity <= 1.0:
            raise ValueError(f"turbidity {turbidity} must be in [0, 1]")
        if not self._frames:
            h, w = 480, 640
            base = np.full((h, w, 3), int(80 + turbidity * 100), dtype=np.uint8)
            return base
        frame = np.load(self._frames[self._idx % len(self._frames)])
        self._idx += 1
        tinted = frame.astype(np.float32)
        tinted[:, :, 2] = np.clip(tinted[:, :, 2] * (1.0 - turbidity * 0.5), 0, 255)
        return tinted.astype(np.uint8)


def write_gs_fixture(path: Path, n_frames: int = 3, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        np.save(path / f"frame_{i:04d}.npy", img)
    manifest = {"n_frames": n_frames, "seed": seed}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
