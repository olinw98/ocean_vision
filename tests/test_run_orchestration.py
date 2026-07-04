import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fathomfollow.config.models import ScenarioConfig
from fathomfollow.run import run_orchestration
from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture


def test_run_orchestration(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=10)
    env = RecordedSimEnv(fixture)
    scenario = ScenarioConfig(max_steps=10)
    out = tmp_path / "run"
    result = run_orchestration(env, scenario, out)
    assert (out / "nav_log.json").exists()
    assert (out / "ctrl_log.json").exists()
    assert len(result["est_positions"]) >= 1
    assert len(result["in_frame"]) >= 1


def test_run_orchestration_uses_yolo_when_weights_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=5)
    env = RecordedSimEnv(fixture)
    scenario = ScenarioConfig(max_steps=5)
    out = tmp_path / "run"
    fake = MagicMock()
    fake.detect.return_value = []
    monkeypatch.setattr(
        "fathomfollow.run.YoloDetector",
        lambda weights, class_id=None: fake,
    )
    run_orchestration(env, scenario, out, detector_weights=Path("models/fake.pt"))
    assert fake.detect.called


def test_run_orchestration_writes_run_metadata(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=2)
    env = RecordedSimEnv(fixture)
    scenario = ScenarioConfig(max_steps=2)
    out = tmp_path / "run"
    run_orchestration(env, scenario, out)
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["coupling_mode"] == "parallel-eval"
    assert metadata["scenario"] == scenario.name
    assert metadata["camera_width"] == scenario.camera_width
    assert metadata["camera_height"] == scenario.camera_height


def test_run_orchestration_logs_detections(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=3)
    env = RecordedSimEnv(fixture)
    scenario = ScenarioConfig(max_steps=3)
    out = tmp_path / "run"
    run_orchestration(env, scenario, out)
    ctrl = json.loads((out / "ctrl_log.json").read_text(encoding="utf-8"))
    assert len(ctrl) == 3
    for entry in ctrl:
        assert "t" in entry
        assert "cmd" in entry
        assert "n_dets" in entry
        assert "dets" in entry
        assert isinstance(entry["dets"], list)
        assert "gt_pose" in entry
        assert "gt_target_pose" in entry
        assert "track_active" in entry
        assert entry["n_dets"] == len(entry["dets"])


def test_run_orchestration_passes_target_class_id_to_yolo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=2)
    env = RecordedSimEnv(fixture)
    scenario = ScenarioConfig(max_steps=2, target_class_id=0)
    out = tmp_path / "run"
    captured: dict = {}

    def fake_yolo(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        fake = MagicMock()
        fake.detect.return_value = []
        return fake

    monkeypatch.setattr("fathomfollow.run.YoloDetector", fake_yolo)
    run_orchestration(env, scenario, out, detector_weights=Path("models/fake.pt"))
    assert captured.get("class_id") == 0


def test_parallel_eval_ctrl_invariant_to_nav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=5)
    scenario = ScenarioConfig(max_steps=5)
    out_baseline = tmp_path / "run_baseline"
    out_noisy = tmp_path / "run_noisy"

    run_orchestration(RecordedSimEnv(fixture), scenario, out_baseline, nav_checkpoint=None)

    def noisy_dual_nav(*args, **kwargs):
        del args, kwargs
        garbage = np.array([999.0, 999.0, 999.0], dtype=np.float64)
        return garbage, garbage

    monkeypatch.setattr("fathomfollow.run.run_dual_nav_step", noisy_dual_nav)
    run_orchestration(RecordedSimEnv(fixture), scenario, out_noisy, nav_checkpoint=None)

    ctrl_baseline = json.loads((out_baseline / "ctrl_log.json").read_text(encoding="utf-8"))
    ctrl_noisy = json.loads((out_noisy / "ctrl_log.json").read_text(encoding="utf-8"))
    nav_baseline = json.loads((out_baseline / "nav_log.json").read_text(encoding="utf-8"))
    nav_noisy = json.loads((out_noisy / "nav_log.json").read_text(encoding="utf-8"))

    assert nav_baseline != nav_noisy
    assert len(ctrl_baseline) == len(ctrl_noisy)
    for base, noisy in zip(ctrl_baseline, ctrl_noisy):
        assert base["cmd"] == noisy["cmd"]
        assert base["track_active"] == noisy["track_active"]
        assert base["n_dets"] == noisy["n_dets"]


def test_run_preflight_missing_detector_weights(tmp_path: Path) -> None:
    from fathomfollow.artifacts import preflight_run_artifacts

    missing = tmp_path / "runs/detect/train-2/weights/best.pt"
    with pytest.raises(FileNotFoundError, match="ff-fetch hero"):
        preflight_run_artifacts(detector_weights=missing, root=tmp_path)
