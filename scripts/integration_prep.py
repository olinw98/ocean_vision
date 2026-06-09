"""Record sim-frame detector baseline (Step 1.3) and nav trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fathomfollow.integration import generate_trajectories_from_sim, record_sim_baseline
from fathomfollow.perception.detector import MockDetector, YoloDetector
from fathomfollow.sim.recorded import RecordedSimEnv


def _make_detector(weights: Path | None) -> MockDetector | YoloDetector:
    if weights is not None and weights.exists():
        return YoloDetector(weights)
    return MockDetector()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim baseline + trajectory generation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    baseline_p = sub.add_parser("baseline", help="Record pre-GS detector firing rate")
    baseline_p.add_argument("--fixture", type=Path, default=Path("fixtures/sim/smoke.npz"))
    baseline_p.add_argument("--out", type=Path, default=Path("runs/baseline.json"))
    baseline_p.add_argument("--max-steps", type=int, default=100)
    baseline_p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="YOLO weights path; uses MockDetector if omitted",
    )

    traj_p = sub.add_parser("trajectories", help="Generate nav training trajectories")
    traj_p.add_argument("--fixture", type=Path, default=Path("fixtures/sim/smoke.npz"))
    traj_p.add_argument("--out", type=Path, default=Path("data/trajectories"))
    traj_p.add_argument("--n-steps", type=int, default=200)

    args = parser.parse_args()
    if args.cmd == "baseline":
        env = RecordedSimEnv(args.fixture)
        detector = _make_detector(args.weights)
        report = record_sim_baseline(env, detector, max_steps=args.max_steps)
        env.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_frames": report.n_frames,
            "n_detections": report.n_detections,
            "firing_rate": report.firing_rate,
            "fixture": str(args.fixture),
            "detector": type(detector).__name__,
            "weights": str(args.weights) if args.weights else None,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    elif args.cmd == "trajectories":
        paths = generate_trajectories_from_sim(args.fixture, args.out, n_steps=args.n_steps)
        print(json.dumps({"n_trajectories": len(paths), "paths": [str(p) for p in paths]}))


if __name__ == "__main__":
    main()
