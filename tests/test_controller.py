from fathomfollow.control.visual_servo import FollowController
from fathomfollow.perception.types import Track


def test_centroid_right_positive_yaw() -> None:
    ctrl = FollowController()
    track = Track(1, (0.7, 0.5, 0.2, 0.2), 0)
    cmd = ctrl.command(track, (480, 640))
    assert cmd.yaw_rate > 0


def test_bbox_large_reduces_forward() -> None:
    ctrl = FollowController()
    track = Track(1, (0.5, 0.5, 0.5, 0.5), 0)
    cmd = ctrl.command(track, (480, 640))
    assert cmd.forward_vel <= 0


def test_active_none_safe_default() -> None:
    ctrl = FollowController()
    cmd = ctrl.command(None, (480, 640))
    assert cmd.forward_vel == 0.0
    assert cmd.yaw_rate == 0.0
    assert cmd.vertical_vel == 0.0
