import numpy as np

from fathomfollow.sim.holoocean_env import (
    extract_imu_6,
    global_to_body_velocity,
    map_holoocean_state,
)


def test_extract_imu_6_from_18d() -> None:
    raw = np.zeros(18, dtype=np.float32)
    raw[0:3] = [1, 2, 3]
    raw[12:15] = [0.1, 0.2, 0.3]
    imu = extract_imu_6(raw)
    assert imu.shape == (6,)
    np.testing.assert_allclose(imu[:3], [1, 2, 3])
    np.testing.assert_allclose(imu[3:], [0.1, 0.2, 0.3])


def test_global_to_body_velocity_identity() -> None:
    vel = np.array([1.0, 0.0, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    body = global_to_body_velocity(vel, quat)
    np.testing.assert_allclose(body, vel, atol=1e-5)


def test_map_holoocean_state_mock() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    state = {
        "FrontCamera": rgb,
        "IMUSensor": np.zeros(18, dtype=np.float32),
        "DVLSensor": np.array([1.0, 0.0, 0.0, 10, 10, 10, 10], dtype=np.float64),
        "PoseSensor": np.zeros(18, dtype=np.float64),
    }
    state["PoseSensor"][9:12] = [0, 0, -5]
    state["PoseSensor"][15:18] = [0, 0, 0]
    obs = map_holoocean_state(state, t=0.5)
    assert obs.rgb.shape == (480, 640, 3)
    assert obs.imu.shape == (6,)
    assert obs.dvl.shape == (3,)
    assert obs.t == 0.5
