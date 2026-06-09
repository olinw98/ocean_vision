import numpy as np

from fathomfollow.eval.metrics import (
    compute_drift_metrics,
    gs_ablation_table,
    tracking_retention,
)


def test_drift_metrics_hand_computed() -> None:
    est = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])]
    gt = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
    mask = [False, True, True]
    m = compute_drift_metrics(est, gt, mask)
    assert m.mean_drift == 1.0
    assert m.drift_within_dropout == 1.5


def test_tracking_retention() -> None:
    assert tracking_retention([True, True, False, True]) == 0.75


def test_gs_ablation_table() -> None:
    import pytest

    t = gs_ablation_table(0.1, 0.3)
    assert t["improved"] is True
    assert t["delta"] == pytest.approx(0.2)
