import numpy as np

from fathomfollow.nav.trajectories import body_velocity_from_poses


def test_body_velocity_constant_forward() -> None:
    pose0 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    pose1 = np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    vel = body_velocity_from_poses(pose0, pose1, dt=1.0)
    np.testing.assert_allclose(vel, [1, 0, 0], atol=1e-5)


def test_log_trajectory(tmp_path) -> None:
    from fathomfollow.nav.trajectories import log_trajectory
    from fathomfollow.sim.recorded import RecordedSimEnv, write_sim_fixture

    fixture = tmp_path / "sim.npz"
    write_sim_fixture(fixture, n_frames=5)
    env = RecordedSimEnv(fixture)
    frames = log_trajectory(env, n_steps=5)
    assert len(frames) == 5
    assert frames[0].gt_velocity_body.shape == (3,)
