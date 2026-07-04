from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fathomfollow.config.models import ScenarioConfig
from fathomfollow.eval.metrics import DriftMetrics
from fathomfollow.eval.run_eval import evaluate_run
from fathomfollow.run import run_orchestration
from fathomfollow.sim.recorded import RecordedSimEnv


@dataclass
class DriftGateResult:
    fixture: str
    scenario: str
    nav_checkpoint: str | None
    detector_context: str
    n_steps: int
    n_dropout_steps: int
    drift_learned: DriftMetrics
    drift_baseline: DriftMetrics
    learned_beats_baseline_dropout: bool
    margin_dropout: float
    tracking_retention: float
    coupling_mode: str = "parallel-eval"

    def to_dict(self) -> dict:
        return {
            "fixture": self.fixture,
            "scenario": self.scenario,
            "nav_checkpoint": self.nav_checkpoint,
            "detector_context": self.detector_context,
            "coupling_mode": self.coupling_mode,
            "n_steps": self.n_steps,
            "n_dropout_steps": self.n_dropout_steps,
            "drift_learned": asdict(self.drift_learned),
            "drift_baseline": asdict(self.drift_baseline),
            "learned_beats_baseline_dropout": self.learned_beats_baseline_dropout,
            "margin_dropout": self.margin_dropout,
            "tracking_retention": self.tracking_retention,
        }


def _detector_context(
    detector_weights: Path | None,
    allow_mock_detector: bool,
) -> str:
    if detector_weights is not None:
        return str(detector_weights)
    if allow_mock_detector:
        return "MockDetector (--allow-mock-detector)"
    raise ValueError(
        "drift gate requires --detector weights or --allow-mock-detector "
        "(MockDetector inflates tracking_retention without explicit opt-in)"
    )


def run_drift_gate(
    fixture: Path,
    scenario: ScenarioConfig,
    out_dir: Path,
    nav_checkpoint: Path | None = None,
    scenario_path: Path | None = None,
    detector_weights: Path | None = None,
    allow_mock_detector: bool = False,
) -> DriftGateResult:
    """Replay a recorded sim fixture with dropout and compare nav drift (Phase 2 gate)."""
    detector_context = _detector_context(detector_weights, allow_mock_detector)
    env = RecordedSimEnv(fixture)
    run_orchestration(
        env,
        scenario,
        out_dir,
        nav_checkpoint=nav_checkpoint,
        detector_weights=detector_weights,
    )
    eval_result = evaluate_run(out_dir, report_path=out_dir / "report.md")

    nav = json.loads((out_dir / "nav_log.json").read_text(encoding="utf-8"))
    n_dropout_steps = sum(1 for entry in nav if not entry["dvl_valid"])

    learned_dropout = eval_result.drift_learned.drift_within_dropout
    baseline_dropout = eval_result.drift_baseline.drift_within_dropout
    margin = baseline_dropout - learned_dropout

    result = DriftGateResult(
        fixture=str(fixture),
        scenario=str(scenario_path or scenario.name),
        nav_checkpoint=str(nav_checkpoint) if nav_checkpoint else None,
        detector_context=detector_context,
        n_steps=len(nav),
        n_dropout_steps=n_dropout_steps,
        drift_learned=eval_result.drift_learned,
        drift_baseline=eval_result.drift_baseline,
        learned_beats_baseline_dropout=learned_dropout < baseline_dropout,
        margin_dropout=margin,
        tracking_retention=eval_result.tracking_retention,
        coupling_mode=eval_result.coupling_mode,
    )
    (out_dir / "drift_gate.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return result
