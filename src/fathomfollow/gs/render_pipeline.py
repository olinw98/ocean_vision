from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from fathomfollow.config.models import LabelStrategy, RenderConfig
from fathomfollow.gs.base import GSRenderer, Pose
from fathomfollow.gs.recorded import GSRenderManifest


def write_yolo_label(path: Path, class_id: int, bbox: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n"
    path.write_text(line, encoding="utf-8")


def composite_target_bbox(strategy: LabelStrategy) -> tuple[float, float, float, float]:
    if strategy == LabelStrategy.COMPOSITED_TARGET:
        return (0.5, 0.5, 0.15, 0.15)
    return (0.4, 0.4, 0.3, 0.3)


def render_labeled_batch(
    renderer: GSRenderer,
    poses: list[Pose],
    turbidity_values: list[float],
    out_dir: Path,
    label_strategy: LabelStrategy,
    scene_id: str,
    camera_path_id: str,
    seed: int = 42,
) -> GSRenderManifest:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    bbox = composite_target_bbox(label_strategy)
    n = 0
    for ti, turb in enumerate(turbidity_values):
        for pi, pose in enumerate(poses):
            rgb = renderer.render(pose, turb)
            if ti > 0 and pi == 0:
                prev = np.load(out_dir / f"frame_t{turbidity_values[ti-1]:.2f}_p000.npy")
                assert not np.array_equal(rgb, prev) or turb == turbidity_values[ti - 1]
            name = f"frame_t{turb:.2f}_p{pi:03d}"
            np.save(out_dir / f"{name}.npy", rgb)
            Image.fromarray(rgb).save(img_dir / f"{name}.jpg")
            write_yolo_label(lbl_dir / f"{name}.txt", 0, bbox)
            n += 1
    manifest = GSRenderManifest(
        scene_id=scene_id,
        camera_path_id=camera_path_id,
        turbidity_values=turbidity_values,
        n_frames=n,
        label_strategy=label_strategy.value,
        out_dir=str(out_dir),
    )
    manifest.write(out_dir / "render_manifest.json")
    return manifest
