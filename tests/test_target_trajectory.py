import numpy as np

from fathomfollow.config.models import TargetMimicConfig
from fathomfollow.sim.target import target_position_at


def test_target_trajectory_deterministic() -> None:
    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5)
    p0 = target_position_at(0.0, cfg)
    p1 = target_position_at(1.0, cfg)
    p0b = target_position_at(0.0, cfg)
    np.testing.assert_allclose(p0, p0b)
    assert not np.allclose(p0, p1)


def test_target_circle_stays_at_configured_depth() -> None:
    cfg = TargetMimicConfig(trajectory="circle", radius=3.0, speed=0.2, depth=-12.0)
    for t in (0.0, 1.5, 10.0):
        pos = target_position_at(t, cfg)
        assert pos[2] == -12.0


def test_target_linear_trajectory_along_x() -> None:
    cfg = TargetMimicConfig(trajectory="spline", speed=0.25, depth=-8.0)
    p0 = target_position_at(0.0, cfg)
    p4 = target_position_at(4.0, cfg)
    np.testing.assert_allclose(p0, [0.0, 0.0, -8.0])
    np.testing.assert_allclose(p4, [1.0, 0.0, -8.0])
