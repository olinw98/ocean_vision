from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fathomfollow.config.models import ScenarioConfig, load_yaml_model
from fathomfollow.control.visual_servo import FollowController
from fathomfollow.nav.deadreckon import DeadReckoning
from fathomfollow.nav.dropout import DropoutSimEnv
from fathomfollow.nav.estimator import VelocityEstimator
from fathomfollow.perception.detector import MockDetector
from fathomfollow.perception.sim_infer import run_sim_inference
from fathomfollow.perception.tracker import SimpleTracker
from fathomfollow.sim.base import Command
from fathomfollow.sim.recorded import RecordedSimEnv


def run_orchestration(
    env: RecordedSimEnv,
    scenario: ScenarioConfig,
    out_dir: Path,
) -> dict:
    detector = MockDetector()
    tracker = SimpleTracker()
    controller = FollowController()
    dr = DeadReckoning()
    estimator = VelocityEstimator()
    wrapped = DropoutSimEnv(env, scenario.dropout)

    obs = wrapped.reset()
    dr.reset(obs.gt_pose)
    imu_history: list[np.ndarray] = []
    est_positions: list[np.ndarray] = []
    gt_positions: list[np.ndarray] = []
    dropout_mask: list[bool] = []
    in_frame: list[bool] = []
    nav_log: list[dict] = []
    ctrl_log: list[dict] = []

    cmd = Command(0.0, 0.0, 0.0)
    for step in range(scenario.max_steps):
        dets = detector.detect(obs.rgb, frame_id=step)
        tracks = tracker.update(dets)
        active = tracker.select_active(tracks)
        cmd = controller.command(active, obs.rgb.shape[:2])
        in_frame.append(active is not None)

        imu_history.append(obs.imu.copy())
        win = np.stack(imu_history[-estimator.window_size :])
        est_vel = estimator.estimate(win, obs.dvl if obs.dvl_valid else None)
        est_pos = dr.step(
            obs.dvl if obs.dvl_valid else None,
            est_vel,
            obs.dvl_valid,
            0.1,
            obs.gt_pose[3:7],
        )
        est_positions.append(est_pos)
        gt_positions.append(obs.gt_pose[:3].copy())
        dropout_mask.append(not obs.dvl_valid)

        nav_log.append({"t": obs.t, "est_pos": est_pos.tolist(), "dvl_valid": obs.dvl_valid})
        ctrl_log.append(
            {
                "t": obs.t,
                "cmd": asdict(cmd),
                "n_dets": len(dets),
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
