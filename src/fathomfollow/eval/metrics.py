from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftMetrics:
    mean_drift: float
    max_drift: float
    drift_within_dropout: float


def position_error(est: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(est[:3] - gt[:3]))


def compute_drift_metrics(
    est_positions: list[np.ndarray],
    gt_positions: list[np.ndarray],
    dropout_mask: list[bool],
) -> DriftMetrics:
    errors = [position_error(e, g) for e, g in zip(est_positions, gt_positions)]
    dropout_errors = [e for e, d in zip(errors, dropout_mask) if d]
    return DriftMetrics(
        mean_drift=float(np.mean(errors)) if errors else 0.0,
        max_drift=float(np.max(errors)) if errors else 0.0,
        drift_within_dropout=float(np.mean(dropout_errors)) if dropout_errors else 0.0,
    )


def tracking_retention(in_frame: list[bool]) -> float:
    return sum(in_frame) / len(in_frame) if in_frame else 0.0


def gs_ablation_table(baseline_rate: float, augmented_rate: float) -> dict:
    delta = augmented_rate - baseline_rate
    return {
        "baseline_firing_rate": baseline_rate,
        "augmented_firing_rate": augmented_rate,
        "delta": delta,
        "improved": delta > 0,
    }
