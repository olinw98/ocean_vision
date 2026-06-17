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
    n_steps: int
    n_dropout_steps: int
    drift_learned: DriftMetrics
    drift_baseline: DriftMetrics
    learned_beats_baseline_dropout: bool
    margin_dropout: float
    tracking_retention: float

    def to_dict(self) -> dict:
        return {
            "fixture": self.fixture,
            "scenario": self.scenario,
            "nav_checkpoint": self.nav_checkpoint,
            "n_steps": self.n_steps,
            "n_dropout_steps": self.n_dropout_steps,
            "drift_learned": asdict(self.drift_learned),
            "drift_baseline": asdict(self.drift_baseline),
            "learned_beats_baseline_dropout": self.learned_beats_baseline_dropout,
            "margin_dropout": self.margin_dropout,
            "tracking_retention": self.tracking_retention,
        }


def run_drift_gate(
    fixture: Path,
    scenario: ScenarioConfig,
    out_dir: Path,
    nav_checkpoint: Path | None = None,
    scenario_path: Path | None = None,
) -> DriftGateResult:
    """Replay a recorded sim fixture with dropout and compare nav drift (Phase 2 gate)."""
    env = RecordedSimEnv(fixture)
    run_orchestration(env, scenario, out_dir, nav_checkpoint=nav_checkpoint)
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
        n_steps=len(nav),
        n_dropout_steps=n_dropout_steps,
        drift_learned=eval_result.drift_learned,
        drift_baseline=eval_result.drift_baseline,
        learned_beats_baseline_dropout=learned_dropout < baseline_dropout,
        margin_dropout=margin,
        tracking_retention=eval_result.tracking_retention,
    )
    (out_dir / "drift_gate.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return result
