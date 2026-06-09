from __future__ import annotations

from dataclasses import dataclass

from fathomfollow.perception.detector import MockDetector, YoloDetector
from fathomfollow.perception.types import DetectionRecord
from fathomfollow.sim.base import SimEnv


@dataclass
class SimInferReport:
    n_frames: int
    n_detections: int
    firing_rate: float


def run_sim_inference(
    env: SimEnv,
    detector: YoloDetector | MockDetector,
    max_steps: int = 100,
    conf_threshold: float = 0.25,
) -> SimInferReport:
    obs = env.reset()
    n_det = 0
    steps = 0
    from fathomfollow.sim.base import Command

    cmd = Command(0.0, 0.0, 0.0)
    for _ in range(max_steps):
        dets = detector.detect(obs.rgb, frame_id=steps)
        n_det += sum(1 for d in dets if d.confidence >= conf_threshold)
        steps += 1
        obs = env.step(cmd)
    return SimInferReport(
        n_frames=steps,
        n_detections=n_det,
        firing_rate=n_det / steps if steps else 0.0,
    )
