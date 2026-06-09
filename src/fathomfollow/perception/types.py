from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectionRecord:
    frame_id: int
    class_id: int
    bbox: tuple[float, float, float, float]  # xywh normalized
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Track:
    track_id: int
    last_bbox: tuple[float, float, float, float]
    last_seen_frame: int
    state: str = "active"
    history: list[tuple[float, float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = [self.last_bbox]
