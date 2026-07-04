from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fathomfollow.eval.bbox_gt import DetectionQualityMetrics, compute_detection_quality
from fathomfollow.eval.metrics import DriftMetrics, compute_drift_metrics, tracking_retention
from fathomfollow.eval.report import generate_report


@dataclass
class EvalResult:
    drift_learned: DriftMetrics
    drift_baseline: DriftMetrics
    tracking_retention: float
    detection_quality: DetectionQualityMetrics | None = None


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_run(
    run_dir: Path,
    report_path: Path | None = None,
    image_shape: tuple[int, int] = (480, 640),
) -> EvalResult:
    nav = _load_json(run_dir / "nav_log.json")
    ctrl = _load_json(run_dir / "ctrl_log.json")

    est_positions = [np.asarray(entry["est_pos"], dtype=np.float64) for entry in nav]
    gt_positions = [np.asarray(entry["gt_pos"], dtype=np.float64) for entry in nav]
    dropout_mask = [not entry["dvl_valid"] for entry in nav]

    drift_learned = compute_drift_metrics(est_positions, gt_positions, dropout_mask)

    baseline_est = []
    for entry in nav:
        if entry.get("baseline_pos") is not None:
            baseline_est.append(np.asarray(entry["baseline_pos"], dtype=np.float64))
        else:
            baseline_est.append(np.asarray(entry["est_pos"], dtype=np.float64))
    drift_baseline = compute_drift_metrics(baseline_est, gt_positions, dropout_mask)

    in_frame = [bool(entry.get("target_in_frame", False)) for entry in ctrl]
    retention = tracking_retention(in_frame)
    detection_quality = compute_detection_quality(ctrl, image_shape)

    result = EvalResult(
        drift_learned=drift_learned,
        drift_baseline=drift_baseline,
        tracking_retention=retention,
        detection_quality=detection_quality,
    )

    if report_path is not None:
        generate_report(
            report_path,
            drift_baseline=result.drift_baseline,
            drift_learned=result.drift_learned,
            retention=result.tracking_retention,
            ablation=None,
            detection_quality=result.detection_quality,
        )

    return result
