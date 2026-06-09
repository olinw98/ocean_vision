import json
from pathlib import Path

import numpy as np

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
