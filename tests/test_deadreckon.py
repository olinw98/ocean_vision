import numpy as np

from fathomfollow.nav.deadreckon import DeadReckoning


def test_constant_velocity_integration() -> None:
    dr = DeadReckoning()
    dr.reset()
    quat = np.array([0, 0, 0, 1], dtype=np.float64)
    dvl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pos = dr.step(dvl, None, True, 1.0, quat)
    np.testing.assert_allclose(pos, [1, 0, 0], atol=1e-5)


def test_dvl_valid_tracks_gt_on_fixture() -> None:
    dr = DeadReckoning()
    gt = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    dr.reset(gt)
    quat = gt[3:7]
    dvl = np.array([0.1, 0, 0], dtype=np.float32)
    for _ in range(10):
        dr.step(dvl, None, True, 0.1, quat)
    assert np.linalg.norm(dr.position) > 0
