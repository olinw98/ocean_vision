import json
from pathlib import Path

import numpy as np
import pytest

from fathomfollow.eval.run_eval import evaluate_run


def test_evaluate_run_computes_drift_from_logs(tmp_path: Path) -> None:
    nav = [
        {"t": 0.0, "est_pos": [0.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": True},
        {"t": 0.1, "est_pos": [1.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": False},
        {"t": 0.2, "est_pos": [2.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": False},
    ]
    ctrl = [
        {"t": 0.0, "target_in_frame": True},
        {"t": 0.1, "target_in_frame": True},
        {"t": 0.2, "target_in_frame": False},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "nav_log.json").write_text(json.dumps(nav), encoding="utf-8")
    (run_dir / "ctrl_log.json").write_text(json.dumps(ctrl), encoding="utf-8")

    result = evaluate_run(run_dir)

    assert result.drift_learned.mean_drift == 1.0
    assert result.drift_learned.drift_within_dropout == 1.5
    assert result.tracking_retention == 2 / 3


def test_evaluate_run_writes_report(tmp_path: Path) -> None:
    nav = [
        {"t": 0.0, "est_pos": [0.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": True},
    ]
    ctrl = [{"t": 0.0, "target_in_frame": True}]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "nav_log.json").write_text(json.dumps(nav), encoding="utf-8")
    (run_dir / "ctrl_log.json").write_text(json.dumps(ctrl), encoding="utf-8")

    result = evaluate_run(run_dir, report_path=run_dir / "report.md")

    assert (run_dir / "report.md").exists()
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Navigation Drift" in text
    assert result.tracking_retention == 1.0


def test_report_includes_coupling_mode(tmp_path: Path) -> None:
    nav = [
        {"t": 0.0, "est_pos": [0.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": True},
    ]
    ctrl = [{"t": 0.0, "target_in_frame": True}]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "nav_log.json").write_text(json.dumps(nav), encoding="utf-8")
    (run_dir / "ctrl_log.json").write_text(json.dumps(ctrl), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"coupling_mode": "parallel-eval"}),
        encoding="utf-8",
    )

    evaluate_run(run_dir, report_path=run_dir / "report.md")

    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Architecture" in text
    assert "parallel-eval" in text
    assert "nav does not steer" in text.lower() or "does not steer the controller" in text.lower()


def test_evaluate_run_uses_camera_shape_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nav = [
        {"t": 0.0, "est_pos": [0.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": True},
    ]
    ctrl = [{"t": 0.0, "track_active": True}]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "nav_log.json").write_text(json.dumps(nav), encoding="utf-8")
    (run_dir / "ctrl_log.json").write_text(json.dumps(ctrl), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"camera_height": 512, "camera_width": 512}),
        encoding="utf-8",
    )
    captured: dict = {}

    def capture_shape(ctrl_log, image_shape):
        captured["shape"] = image_shape
        return None

    monkeypatch.setattr(
        "fathomfollow.eval.run_eval.compute_detection_quality",
        capture_shape,
    )
    evaluate_run(run_dir)
    assert captured["shape"] == (512, 512)


def test_track_active_and_gt_in_frame_can_diverge(tmp_path: Path) -> None:
    from fathomfollow.eval.bbox_gt import compute_detection_quality

    gt_pose = [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    gt_target = [3.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    ctrl = [
        {
            "t": 0.0,
            "track_active": False,
            "gt_pose": gt_pose,
            "gt_target_pose": gt_target,
            "dets": [],
        }
    ]
    metrics = compute_detection_quality(ctrl, (480, 640))
    assert metrics is not None
    assert metrics.gt_in_frame_fraction == 1.0
