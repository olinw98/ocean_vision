import json
from pathlib import Path

import numpy as np

from fathomfollow.eval.bbox_gt import (
    DetectionQualityMetrics,
    compute_detection_quality,
    iou_xyxy,
    project_target_bbox_xyxy,
)
from fathomfollow.eval.run_eval import evaluate_run


def test_project_target_bbox_ahead_of_camera() -> None:
    gt_pose = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0])
    gt_target = np.array([3.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0])
    bbox = project_target_bbox_xyxy(gt_pose, gt_target, (480, 640))
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 < 320 < x2
    assert y1 < 240 < y2


def test_project_target_bbox_behind_camera_returns_none() -> None:
    gt_pose = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0])
    gt_target = np.array([-2.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0])
    assert project_target_bbox_xyxy(gt_pose, gt_target, (480, 640)) is None


def test_iou_identical_boxes() -> None:
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou_xyxy(box, box) == 1.0


def test_compute_detection_quality_tp_and_gt_in_frame() -> None:
    gt_pose = [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    gt_target = [3.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    gt_bbox = project_target_bbox_xyxy(
        np.asarray(gt_pose), np.asarray(gt_target), (480, 640)
    )
    assert gt_bbox is not None
    x1, y1, x2, y2 = gt_bbox
    cx = ((x1 + x2) / 2) / 640
    cy = ((y1 + y2) / 2) / 480
    bw = (x2 - x1) / 640
    bh = (y2 - y1) / 480
    ctrl = [
        {
            "t": 0.0,
            "gt_pose": gt_pose,
            "gt_target_pose": gt_target,
            "dets": [{"bbox": [cx, cy, bw, bh], "class_id": 0, "confidence": 0.9}],
        }
    ]
    metrics = compute_detection_quality(ctrl, (480, 640))
    assert metrics is not None
    assert metrics.gt_in_frame_fraction == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.mean_iou > 0.9


def test_evaluate_run_includes_detection_quality_in_report(tmp_path: Path) -> None:
    nav = [
        {"t": 0.0, "est_pos": [0.0, 0.0, 0.0], "gt_pos": [0.0, 0.0, 0.0], "dvl_valid": True},
    ]
    gt_pose = [0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    gt_target = [3.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0]
    gt_bbox = project_target_bbox_xyxy(
        np.asarray(gt_pose), np.asarray(gt_target), (480, 640)
    )
    assert gt_bbox is not None
    x1, y1, x2, y2 = gt_bbox
    cx = ((x1 + x2) / 2) / 640
    cy = ((y1 + y2) / 2) / 480
    bw = (x2 - x1) / 640
    bh = (y2 - y1) / 480
    ctrl = [
        {
            "t": 0.0,
            "track_active": True,
            "gt_pose": gt_pose,
            "gt_target_pose": gt_target,
            "dets": [{"bbox": [cx, cy, bw, bh], "class_id": 0, "confidence": 0.9}],
        }
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "nav_log.json").write_text(json.dumps(nav), encoding="utf-8")
    (run_dir / "ctrl_log.json").write_text(json.dumps(ctrl), encoding="utf-8")

    result = evaluate_run(run_dir, report_path=run_dir / "report.md", image_shape=(480, 640))

    assert result.detection_quality is not None
    assert isinstance(result.detection_quality, DetectionQualityMetrics)
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Detection quality" in text
    assert "GT in-frame fraction" in text
