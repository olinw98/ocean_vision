from __future__ import annotations

from pathlib import Path

from fathomfollow.eval.metrics import (
    DriftMetrics,
    compute_drift_metrics,
    gs_ablation_table,
    tracking_retention,
)


def generate_report(
    out_path: Path,
    drift_baseline: DriftMetrics,
    drift_learned: DriftMetrics,
    retention: float,
    ablation: dict | None = None,
    detection_quality: "DetectionQualityMetrics | None" = None,
) -> None:
    lines = [
        "# FathomFollow Evaluation Report",
        "",
        "## Navigation Drift",
        f"- Baseline mean drift: {drift_baseline.mean_drift:.4f} m",
        f"- Baseline drift within dropout: {drift_baseline.drift_within_dropout:.4f} m",
        f"- Learned mean drift: {drift_learned.mean_drift:.4f} m",
        f"- Learned drift within dropout: {drift_learned.drift_within_dropout:.4f} m",
        "",
        "## Tracking",
        f"- Target in-frame retention: {retention:.2%}",
        "",
    ]
    if detection_quality is not None:
        lines.extend(
            [
                "## Detection quality",
                f"- GT in-frame fraction: {detection_quality.gt_in_frame_fraction:.2%}",
                f"- Precision (IoU≥0.3 vs projected GT): {detection_quality.precision:.2%}",
                f"- Recall: {detection_quality.recall:.2%}",
                f"- Mean IoU (matches): {detection_quality.mean_iou:.4f}",
                "",
            ]
        )
    if ablation:
        lines.extend(
            [
                "## GS Ablation",
                f"- Baseline firing rate: {ablation['baseline_firing_rate']:.2%}",
                f"- Augmented firing rate: {ablation['augmented_firing_rate']:.2%}",
                f"- Delta: {ablation['delta']:+.2%}",
                f"- Improved: {ablation['improved']}",
                "",
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
