"""Generate synthetic sim fixture and optionally run HoloOcean smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fathomfollow.config.models import ScenarioConfig, load_yaml_model
from fathomfollow.sim.recorded import write_sim_fixture
from fathomfollow.sim.recorder import build_fathomnet_proxy_fixture, record_sim_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloOcean smoke test / fixture recorder")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/sim/smoke.npz"),
        help="Output fixture path",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--live", action="store_true", help="Run live HoloOcean (requires install)")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("config/scenario_holoocean.yaml"),
        help="Scenario YAML (target mimic + dropout)",
    )
    parser.add_argument(
        "--fathomnet-proxy",
        type=Path,
        default=None,
        help="Build fixture from FathomNet image dir (Step 1.3 when HoloOcean unavailable)",
    )
    args = parser.parse_args()

    if args.fathomnet_proxy is not None:
        n = build_fathomnet_proxy_fixture(args.fathomnet_proxy, args.out, n_frames=args.frames)
        print(json.dumps({"mode": "fathomnet_proxy", "frames": n, "out": str(args.out)}))
        return

    if args.live:
        try:
            from fathomfollow.sim.holoocean_env import HoloOceanSimEnv

            scenario = load_yaml_model(args.scenario, ScenarioConfig)
            env = HoloOceanSimEnv(
                scenario.holoocean_scenario,
                target_config=scenario.target,
            )
            record_sim_fixture(env, args.out, args.frames)
            np = __import__("numpy")
            obs = np.load(args.out)
            tgt = obs["gt_target_pose"][0]
            auv = obs["gt_pose"][0]
            gt_pose_differs = bool(np.max(np.abs(tgt[:3] - auv[:3])) > 0.5)
            payload = {
                "mode": "holoocean_live",
                "rgb_shape": list(obs["rgb"][0].shape),
                "out": str(args.out),
                "target_mimic": True,
                "gt_pose_differs": gt_pose_differs,
            }
            print(json.dumps(payload))
            if scenario.target is not None and not gt_pose_differs:
                print(
                    json.dumps({"error": "target mimic GT identical to AUV GT — spawn failed"}),
                    file=sys.stderr,
                )
                sys.exit(1)
        except ImportError as e:
            print(json.dumps({"error": str(e), "fallback": "synthetic"}))
            write_sim_fixture(args.out, n_frames=args.frames)
            print(f"Wrote synthetic fixture to {args.out}")
    else:
        write_sim_fixture(args.out, n_frames=args.frames)
        print(f"Wrote synthetic fixture to {args.out}")


if __name__ == "__main__":
    main()
