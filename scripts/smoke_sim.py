"""Generate synthetic sim fixture and optionally run HoloOcean smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from fathomfollow.sim.recorded import write_sim_fixture


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
    args = parser.parse_args()

    if args.live:
        try:
            from fathomfollow.sim.holoocean_env import HoloOceanSimEnv

            env = HoloOceanSimEnv()
            obs = env.reset()
            print(f"HoloOcean smoke OK: rgb={obs.rgb.shape}, imu={obs.imu.shape}, dvl={obs.dvl.shape}")
            env.close()
        except ImportError as e:
            print(f"HoloOcean not installed: {e}")
            print("Writing synthetic fixture instead.")
            write_sim_fixture(args.out, n_frames=args.frames)
    else:
        write_sim_fixture(args.out, n_frames=args.frames)
        print(f"Wrote synthetic fixture to {args.out}")


if __name__ == "__main__":
    main()
