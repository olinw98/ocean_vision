from pathlib import Path

from fathomfollow.perception.detector import MockDetector
from fathomfollow.perception.sim_infer import run_sim_inference
from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture


def test_sim_infer_firing_rate(tmp_path: Path) -> None:
    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=10)
    env = RecordedSimEnv(fixture)
    report = run_sim_inference(env, MockDetector(), max_steps=10)
    assert report.n_frames == 10
    assert 0.0 <= report.firing_rate <= 1.0
    assert report.firing_rate == 1.0  # bright synthetic frames
