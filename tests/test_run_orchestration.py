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
        lambda weights: fake,
    )
    run_orchestration(env, scenario, out, detector_weights=Path("models/fake.pt"))
    assert fake.detect.called
