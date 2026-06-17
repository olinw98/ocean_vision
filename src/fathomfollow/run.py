from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fathomfollow.config.models import ScenarioConfig, load_yaml_model
from fathomfollow.control.visual_servo import FollowController
from fathomfollow.integration import run_dual_nav_step
from fathomfollow.nav.deadreckon import DeadReckoning
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.perception.detector import MockDetector
from fathomfollow.perception.tracker import SimpleTracker
from fathomfollow.sim.base import Command
from fathomfollow.sim.recorded import RecordedSimEnv


def run_orchestration(
    env: RecordedSimEnv,
    scenario: ScenarioConfig,
    out_dir: Path,
    nav_checkpoint: Path | None = None,
) -> dict:
    detector = MockDetector()
    tracker = SimpleTracker()
    controller = FollowController()
    dr = DeadReckoning()
    dr_baseline = DeadReckoning()
    estimator = VelocityEstimator()
    if nav_checkpoint is not None:
        estimator.load(nav_checkpoint)
    wrapped = DropoutSimEnv(env, scenario.dropout)

    obs = wrapped.reset()
    dr.reset(obs.gt_pose)
    dr_baseline.reset(obs.gt_pose)
    imu_history: list[np.ndarray] = []
    est_positions: list[np.ndarray] = []
    gt_positions: list[np.ndarray] = []
    dropout_mask: list[bool] = []
    in_frame: list[bool] = []
    nav_log: list[dict] = []
    ctrl_log: list[dict] = []

    cmd = Command(0.0, 0.0, 0.0)
    for step in range(scenario.max_steps):
        est_pos, baseline_pos = run_dual_nav_step(
            obs, dr, dr_baseline, estimator, imu_history
        )
        dets = detector.detect(obs.rgb, frame_id=step)
        tracks = tracker.update(dets)
        active = tracker.select_active(tracks)
        cmd = controller.command(active, obs.rgb.shape[:2])
        in_frame.append(active is not None)

        est_positions.append(est_pos)
        gt_positions.append(obs.gt_pose[:3].copy())
        dropout_mask.append(not obs.dvl_valid)

        nav_log.append(
            {
                "t": obs.t,
                "est_pos": est_pos.tolist(),
                "baseline_pos": baseline_pos.tolist(),
                "gt_pos": obs.gt_pose[:3].tolist(),
                "dvl_valid": obs.dvl_valid,
            }
        )
        ctrl_log.append(
            {
                "t": obs.t,
                "cmd": asdict(cmd),
                "n_dets": len(dets),
                "target_in_frame": active is not None,
            }
        )
        if hasattr(env, "done") and env.done:
            break
        obs = wrapped.step(cmd)

    wrapped.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nav_log.json").write_text(json.dumps(nav_log, indent=2), encoding="utf-8")
    (out_dir / "ctrl_log.json").write_text(json.dumps(ctrl_log, indent=2), encoding="utf-8")
    return {
        "est_positions": est_positions,
        "gt_positions": gt_positions,
        "dropout_mask": dropout_mask,
        "in_frame": in_frame,
        "nav_log_path": str(out_dir / "nav_log.json"),
        "ctrl_log_path": str(out_dir / "ctrl_log.json"),
    }
