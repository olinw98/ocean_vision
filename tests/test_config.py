from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fathomfollow.config.models import (
    CameraPathConfig,
    DetectorTrainingConfig,
    NavTrainingConfig,
    RenderConfig,
    ScenarioConfig,
    dump_yaml_model,
    load_yaml_model,
)


def test_package_imports() -> None:
    import fathomfollow
    import fathomfollow.config
    import fathomfollow.control
    import fathomfollow.data
    import fathomfollow.eval
    import fathomfollow.gs
    import fathomfollow.nav
    import fathomfollow.perception
    import fathomfollow.sim

    assert fathomfollow.__version__ == "0.1.0"


def test_scenario_config_roundtrip(tmp_path: Path) -> None:
    cfg = ScenarioConfig(name="test")
    path = tmp_path / "scenario.yaml"
    dump_yaml_model(path, cfg)
    loaded = load_yaml_model(path, ScenarioConfig)
    assert loaded.name == "test"
    assert loaded.max_steps == 1000


def test_scenario_config_rejects_bad_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("max_steps: not_a_number\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_yaml_model(path, ScenarioConfig)


def test_render_config_turbidity_bounds() -> None:
    with pytest.raises(Exception):
        RenderConfig(
            scene_checkpoint=Path("models/gs/x"),
            camera_path=Path("config/cam_path.yaml"),
            turbidity_values=[1.5],
            out_dir=Path("data/out"),
        )


def test_camera_path_requires_poses() -> None:
    with pytest.raises(Exception):
        CameraPathConfig(path_id="empty", poses=[])


def test_training_configs(tmp_path: Path) -> None:
    det = DetectorTrainingConfig(data_yaml=tmp_path / "data.yaml")
    nav = NavTrainingConfig(trajectories_dir=tmp_path)
    assert det.epochs == 10
    assert nav.arch == "gru"
