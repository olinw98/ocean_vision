from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fathomfollow.perception.types import Track
from fathomfollow.sim.base import Command


@dataclass
class PIDGains:
    yaw_kp: float = 1.0
    forward_kp: float = 0.5
    vertical_kp: float = 0.8
    target_bbox_area: float = 0.04


class FollowController:
    def __init__(self, gains: PIDGains | None = None) -> None:
        self._gains = gains or PIDGains()

    def command(self, active: Track | None, img_shape: tuple[int, int]) -> Command:
        if active is None:
            return Command(forward_vel=0.0, yaw_rate=0.0, vertical_vel=0.0)

        h, w = img_shape
        xc, yc, bw, bh = active.last_bbox
        cx_err = xc - 0.5
        area = bw * bh
        area_err = area - self._gains.target_bbox_area
        cy_err = yc - 0.5

        yaw_rate = self._gains.yaw_kp * cx_err
        forward_vel = -self._gains.forward_kp * area_err
        vertical_vel = -self._gains.vertical_kp * cy_err
        return Command(
            forward_vel=float(np.clip(forward_vel, -1.0, 1.0)),
            yaw_rate=float(np.clip(yaw_rate, -1.0, 1.0)),
            vertical_vel=float(np.clip(vertical_vel, -1.0, 1.0)),
        )
