from pathlib import Path

import numpy as np
import pytest

from fathomfollow.gs.base import Pose
from fathomfollow.gs.recorded import RecordedGSRenderer, write_gs_fixture


def test_recorded_gs_renderer(tmp_path: Path) -> None:
    fixture = tmp_path / "gs"
    write_gs_fixture(fixture, n_frames=3)
    renderer = RecordedGSRenderer(fixture)
    renderer.load(str(fixture))
    pose = Pose((0, 0, -5), (0, 0, 0, 1))
    rgb = renderer.render(pose, turbidity=0.3)
    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8


def test_turbidity_out_of_range(tmp_path: Path) -> None:
    renderer = RecordedGSRenderer(tmp_path)
    renderer.load("dummy")
    with pytest.raises(ValueError):
        renderer.render(Pose((0, 0, 0), (0, 0, 0, 1)), turbidity=1.5)


def test_turbidity_changes_output(tmp_path: Path) -> None:
    write_gs_fixture(tmp_path, n_frames=1)
    renderer = RecordedGSRenderer(tmp_path)
    renderer.load(str(tmp_path))
    pose = Pose((0, 0, -5), (0, 0, 0, 1))
    low = renderer.render(pose, 0.0)
    renderer._idx = 0
    high = renderer.render(pose, 0.8)
    assert not np.array_equal(low, high)
