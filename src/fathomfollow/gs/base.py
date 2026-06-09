from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # xyzw


@runtime_checkable
class GSRenderer(Protocol):
    def load(self, checkpoint: str) -> None: ...

    def render(self, pose: Pose, turbidity: float) -> np.ndarray: ...
