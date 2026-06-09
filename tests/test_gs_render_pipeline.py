from pathlib import Path

import numpy as np

from fathomfollow.config.models import LabelStrategy
from fathomfollow.gs.base import Pose
from fathomfollow.gs.recorded import RecordedGSRenderer, write_gs_fixture
from fathomfollow.gs.render_pipeline import render_labeled_batch


def test_render_pipeline_labels(tmp_path: Path) -> None:
    fixture = tmp_path / "gs_fix"
    write_gs_fixture(fixture, n_frames=2)
    renderer = RecordedGSRenderer(fixture)
    renderer.load(str(fixture))
    poses = [Pose((0, 0, -5), (0, 0, 0, 1)), Pose((1, 0, -5), (0, 0, 0, 1))]
    out = tmp_path / "render_out"
    manifest = render_labeled_batch(
        renderer,
        poses,
        [0.0, 0.5],
        out,
        LabelStrategy.COMPOSITED_TARGET,
        scene_id="s1",
        camera_path_id="p1",
        seed=42,
    )
    assert manifest.n_frames == 4
    labels = list((out / "labels").glob("*.txt"))
    assert len(labels) == 4
    for lbl in labels:
        parts = lbl.read_text(encoding="utf-8").strip().split()
        assert len(parts) == 5
    assert (out / "render_manifest.json").exists()


def test_turbidity_determinism(tmp_path: Path) -> None:
    write_gs_fixture(tmp_path / "gs", n_frames=1)
    renderer = RecordedGSRenderer(tmp_path / "gs")
    renderer.load("x")
    pose = Pose((0, 0, -5), (0, 0, 0, 1))
    a = renderer.render(pose, 0.3)
    renderer._idx = 0
    b = renderer.render(pose, 0.3)
    np.testing.assert_array_equal(a, b)
