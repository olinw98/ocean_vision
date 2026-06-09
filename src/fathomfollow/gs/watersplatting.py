from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fathomfollow.config.models import CameraPathConfig, LabelStrategy, RenderConfig, load_yaml_model
from fathomfollow.gs.base import GSRenderer, Pose
from fathomfollow.gs.recorded import GSRenderManifest, GSScene


def load_colmap_poses_fixture(data: dict) -> list[Pose]:
    poses = []
    for entry in data.get("poses", []):
        poses.append(
            Pose(
                position=tuple(entry["position"]),
                orientation=tuple(entry["orientation"]),
            )
        )
    return poses


class WaterSplattingGSRenderer:
    """Adapter for WaterSplatting; subprocess in separate conda env for train/render."""

    def __init__(self, conda_env: str = "water_splatting") -> None:
        self._conda_env = conda_env
        self._checkpoint: str | None = None

    def load(self, checkpoint: str) -> None:
        self._checkpoint = checkpoint

    def render(self, pose: Pose, turbidity: float) -> np.ndarray:
        if self._checkpoint is None:
            raise RuntimeError("call load() before render()")
        # Placeholder: real impl shells out to water_splatting env
        h, w = 480, 640
        return np.full((h, w, 3), int(100 - turbidity * 50), dtype=np.uint8)

    def train_subprocess(self, source: Path, out: Path) -> GSScene:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "conda",
            "run",
            "-n",
            self._conda_env,
            "ns-train",
            "water-splatting",
            "--data",
            str(source),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        scene = GSScene(
            scene_id=out.name,
            source_dataset=str(source),
            library="watersplatting",
            n_gaussians=0,
            pose_source="colmap",
            checkpoint_path=str(out / "checkpoint"),
            train_psnr=0.0,
        )
        scene.write(out / "scene.json")
        return scene
