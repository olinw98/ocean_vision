from pathlib import Path
from unittest.mock import MagicMock

import json

import numpy as np
import pytest
import torch

from fathomfollow.config.models import NavTrainingConfig, ScenarioConfig, load_yaml_model
from fathomfollow.eval.metrics import DriftMetrics
from fathomfollow.nav.drift_gate import run_drift_gate
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.nav.training import train_nav_estimator, write_trajectory_npz
from fathomfollow.run import run_orchestration
from fathomfollow.sim.base import Command
from fathomfollow.sim.recorded import RecordedSimEnv

HOLO_FIXTURE = Path("fixtures/sim/holoocean_smoke.npz")
HOLO_SCENARIO = Path("config/scenario_holoocean.yaml")


def test_holoocean_scenario_forces_dropout() -> None:
    scenario = load_yaml_model(HOLO_SCENARIO, ScenarioConfig)
    env = RecordedSimEnv(HOLO_FIXTURE)
    wrapped = DropoutSimEnv(env, scenario.dropout)
    obs = wrapped.reset()
    dropout_steps = 0
    for _ in range(scenario.max_steps):
        if not obs.dvl_valid:
            dropout_steps += 1
        if env.done:
            break
        obs = wrapped.step(Command(0.0, 0.0, 0.0))
    wrapped.close()
    assert dropout_steps > 0


def test_drift_gate_records_dropout_drift(tmp_path: Path) -> None:
    scenario = load_yaml_model(HOLO_SCENARIO, ScenarioConfig)
    out = tmp_path / "gate"
    result = run_drift_gate(
        HOLO_FIXTURE,
        scenario,
        out,
        allow_mock_detector=True,
    )
    assert result.n_dropout_steps > 0
    assert result.drift_baseline.drift_within_dropout > 0.0
    assert (out / "drift_gate.json").exists()
    assert (out / "nav_log.json").exists()


def test_drift_gate_requires_detector_or_flag(tmp_path: Path) -> None:
    scenario = load_yaml_model(HOLO_SCENARIO, ScenarioConfig)
    out = tmp_path / "gate"
    with pytest.raises(ValueError, match="--detector|--allow-mock-detector"):
        run_drift_gate(HOLO_FIXTURE, scenario, out)


def test_drift_gate_records_detector_context_with_mock_flag(tmp_path: Path) -> None:
    scenario = load_yaml_model(HOLO_SCENARIO, ScenarioConfig)
    out = tmp_path / "gate"
    result = run_drift_gate(
        HOLO_FIXTURE,
        scenario,
        out,
        allow_mock_detector=True,
    )
    payload = json.loads((out / "drift_gate.json").read_text(encoding="utf-8"))
    assert payload["detector_context"] == result.detector_context
    assert "MockDetector" in result.detector_context


def test_drift_gate_passes_detector_weights_to_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_yaml_model(HOLO_SCENARIO, ScenarioConfig)
    out = tmp_path / "gate"
    captured: dict = {}

    def fake_orchestration(env, scen, out_dir, nav_checkpoint=None, detector_weights=None):
        captured["detector_weights"] = detector_weights
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "nav_log.json").write_text(
            json.dumps([{"dvl_valid": True}]), encoding="utf-8"
        )
        (out_dir / "ctrl_log.json").write_text(
            json.dumps([{"target_in_frame": False}]), encoding="utf-8"
        )
        return {"est_positions": [], "gt_positions": [], "dropout_mask": [], "in_frame": []}

    fake_eval = MagicMock()
    fake_eval.drift_learned = DriftMetrics(0.5, 0.5, 0.5)
    fake_eval.drift_baseline = DriftMetrics(1.0, 1.0, 1.0)
    fake_eval.tracking_retention = 0.5
    fake_eval.coupling_mode = "parallel-eval"

    monkeypatch.setattr("fathomfollow.nav.drift_gate.run_orchestration", fake_orchestration)
    monkeypatch.setattr("fathomfollow.nav.drift_gate.evaluate_run", lambda *a, **k: fake_eval)

    weights = tmp_path / "best.pt"
    weights.write_text("fake", encoding="utf-8")
    result = run_drift_gate(HOLO_FIXTURE, scenario, out, detector_weights=weights)
    assert captured["detector_weights"] == weights
    assert result.detector_context == str(weights)


def test_estimator_load_checkpoint(tmp_path: Path) -> None:
    traj = tmp_path / "traj.npz"
    n = 30
    imu = np.random.randn(n, 6).astype(np.float32)
    dvl = np.tile(np.array([0.5, 0.0, 0.0], dtype=np.float32), (n, 1))
    dvl_valid = np.ones(n, dtype=bool)
    vel = np.tile(np.array([0.5, 0.0, 0.0], dtype=np.float32), (n, 1))
    write_trajectory_npz(traj, imu, dvl, dvl_valid, vel)

    cfg = NavTrainingConfig(
        trajectories_dir=tmp_path,
        epochs=5,
        window_size=10,
        hidden_size=32,
    )
    ckpt = train_nav_estimator(cfg, tmp_path / "nav_model")

    est = VelocityEstimator(window_size=10, hidden_size=32)
    imu_win = imu[:10]
    before = est.estimate(imu_win, dvl=dvl[0])
    est.load(ckpt)
    after = est.estimate(imu_win, dvl=dvl[0])
    assert not np.allclose(before, after)


def test_run_orchestration_loads_nav_checkpoint(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    n = 40
    imu = np.random.randn(n, 6).astype(np.float32)
    dvl = np.tile(np.array([0.3, 0.0, 0.0], dtype=np.float32), (n, 1))
    dvl_valid = np.ones(n, dtype=bool)
    vel = np.tile(np.array([0.3, 0.0, 0.0], dtype=np.float32), (n, 1))
    write_trajectory_npz(fixture, imu, dvl, dvl_valid, vel)

    cfg = NavTrainingConfig(
        trajectories_dir=tmp_path,
        epochs=8,
        window_size=10,
        hidden_size=32,
    )
    ckpt = train_nav_estimator(cfg, tmp_path / "nav_model")

    # Build a minimal replay fixture from trajectory arrays for RecordedSimEnv.
    t = np.arange(n, dtype=np.float64) * 0.1
    rgb = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    gt_pose = np.zeros((n, 7), dtype=np.float64)
    gt_pose[:, 6] = 1.0
    gt_pose[:, 0] = t * 0.3
    gt_target_pose = gt_pose.copy()
    np.savez(
        fixture,
        t=t,
        rgb=rgb,
        imu=imu,
        dvl=dvl,
        dvl_valid=dvl_valid,
        gt_pose=gt_pose,
        gt_target_pose=gt_target_pose,
    )

    scenario = ScenarioConfig(
        max_steps=n,
        dropout={"alt_min": 0.0, "tilt_max_deg": 25.0, "forced_windows": [[0.5, 2.5]]},
    )
    env = RecordedSimEnv(fixture)

    out_untrained = tmp_path / "run_untrained"
    run_orchestration(env, scenario, out_untrained, nav_checkpoint=None)
    env2 = RecordedSimEnv(fixture)
    out_trained = tmp_path / "run_trained"
    run_orchestration(env2, scenario, out_trained, nav_checkpoint=ckpt)

    untrained = np.array(
        [e["est_pos"] for e in json.loads((out_untrained / "nav_log.json").read_text())]
    )
    trained = np.array(
        [e["est_pos"] for e in json.loads((out_trained / "nav_log.json").read_text())]
    )
    assert not np.allclose(untrained, trained)
