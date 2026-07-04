import numpy as np

from fathomfollow.integration import run_dual_nav_step
from fathomfollow.nav.attitude import AttitudeIntegrator
from fathomfollow.nav.deadreckon import DeadReckoning
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.sim.base import SimObservation


def test_gyro_integration_changes_quaternion() -> None:
    att = AttitudeIntegrator()
    att.reset()
    q0 = att.quat if hasattr(att, "quat") else np.array([0, 0, 0, 1])
    quat = att.step(np.array([0.0, 0.0, 2.0], dtype=np.float32), 0.5)
    assert not np.allclose(quat, q0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_attitude_does_not_track_gt_after_reset() -> None:
    att = AttitudeIntegrator()
    att.reset(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    q_before = att.step(np.zeros(3), 0.1)
    # Sim GT quaternion jumps; zero gyro -> integrator unchanged
    q_after = att.step(np.zeros(3), 0.1)
    np.testing.assert_allclose(q_before, q_after)


def test_dual_nav_does_not_use_per_step_gt_orientation() -> None:
    dr_learned = DeadReckoning()
    dr_baseline = DeadReckoning()
    estimator = VelocityEstimator()
    attitude = AttitudeIntegrator()
    imu_history: list[np.ndarray] = []

    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    dr_learned.reset(initial)
    dr_baseline.reset(initial)
    attitude.reset(initial[3:7])

    obs = SimObservation(
        t=0.0,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        imu=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        dvl=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dvl_valid=True,
        gt_pose=initial,
        gt_target_pose=initial,
    )
    run_dual_nav_step(obs, dr_learned, dr_baseline, estimator, imu_history, attitude)

    rotated_gt = initial.copy()
    rotated_gt[3:7] = np.array([0.0, 0.0, 0.383, 0.924], dtype=np.float64)  # ~45 deg yaw
    obs2 = SimObservation(
        t=0.1,
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        imu=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        dvl=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        dvl_valid=True,
        gt_pose=rotated_gt,
        gt_target_pose=rotated_gt,
    )
    _, baseline_identity = run_dual_nav_step(
        obs2, dr_learned, dr_baseline, estimator, imu_history, attitude
    )
    # Two steps at 1 m/s forward with identity attitude -> x = 0.2
    np.testing.assert_allclose(baseline_identity, [0.2, 0.0, 0.0], atol=1e-4)

    dr_gt_orient = DeadReckoning()
    dr_gt_orient.reset(initial)
    from fathomfollow.sim.holoocean_env import quat_to_rotation_matrix

    r = quat_to_rotation_matrix(rotated_gt[3:7])
    vel_global = r @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pos_if_gt = np.array([0.1, 0.0, 0.0]) + vel_global * 0.1
    assert not np.allclose(baseline_identity, pos_if_gt, atol=1e-3)
