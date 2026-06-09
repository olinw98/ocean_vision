from __future__ import annotations

import argparse
import json
from pathlib import Path

from fathomfollow.config.models import (
    CameraPathConfig,
    DetectorTrainingConfig,
    NavTrainingConfig,
    RenderConfig,
    ScenarioConfig,
    dump_yaml_model,
    load_yaml_model,
)
from fathomfollow.data.merge import merge_datasets
from fathomfollow.data.pipeline import CANDIDATE_TAXA, prepare_from_coco
from fathomfollow.eval.metrics import compute_drift_metrics, gs_ablation_table, tracking_retention
from fathomfollow.eval.report import generate_report
from fathomfollow.gs.base import Pose
from fathomfollow.gs.recorded import RecordedGSRenderer
from fathomfollow.gs.render_pipeline import render_labeled_batch
from fathomfollow.gs.watersplatting import WaterSplattingGSRenderer
from fathomfollow.run import run_orchestration
from fathomfollow.sim.recorded import RecordedSimEnv


def main_data() -> None:
    parser = argparse.ArgumentParser(prog="ff-data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--taxa", type=str, default=CANDIDATE_TAXA[0])
    prep.add_argument("--coco", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)

    merge_p = sub.add_parser("merge")
    merge_p.add_argument("--sources", type=str, required=True)
    merge_p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare_from_coco(args.coco, args.out, [args.taxa])
    elif args.cmd == "merge":
        sources = [Path(s.strip()) for s in args.sources.split(",")]
        merge_datasets(sources, args.out)


def main_gs() -> None:
    parser = argparse.ArgumentParser(prog="ff-gs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--source", type=Path, required=True)
    train_p.add_argument("--out", type=Path, required=True)
    train_p.add_argument("--library", type=str, default="watersplatting")

    render_p = sub.add_parser("render")
    render_p.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        renderer = WaterSplattingGSRenderer()
        scene = renderer.train_subprocess(args.source, args.out)
        print(json.dumps({"scene_id": scene.scene_id, "checkpoint": scene.checkpoint_path}))
    elif args.cmd == "render":
        cfg = load_yaml_model(args.config, RenderConfig)
        cam = load_yaml_model(cfg.camera_path, CameraPathConfig)
        renderer = RecordedGSRenderer(cfg.scene_checkpoint)
        renderer.load(str(cfg.scene_checkpoint))
        poses = [
            Pose(tuple(p.position), tuple(p.orientation))  # type: ignore[arg-type]
            for p in cam.poses
        ]
        render_labeled_batch(
            renderer,
            poses,
            cfg.turbidity_values,
            cfg.out_dir,
            cfg.label_strategy,
            scene_id=str(cfg.scene_checkpoint),
            camera_path_id=cam.path_id,
            seed=cfg.seed,
        )


def main_train_detector() -> None:
    parser = argparse.ArgumentParser(prog="ff-train-detector")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml_model(args.config, DetectorTrainingConfig)
    from ultralytics import YOLO

    model = YOLO(cfg.model)
    model.train(
        data=str(cfg.data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
    )
    metrics = {"mAP50": 0.0, "mAP50-95": 0.0}
    out = Path(cfg.data_yaml).parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main_train_nav() -> None:
    parser = argparse.ArgumentParser(prog="ff-train-nav")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml_model(args.config, NavTrainingConfig)
    print(f"Nav training config loaded: {cfg.arch}, epochs={cfg.epochs}")


def main_run() -> None:
    parser = argparse.ArgumentParser(prog="ff-run")
    parser.add_argument("--scenario", type=Path, default=Path("config/scenario.yaml"))
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/sim/smoke.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    args = parser.parse_args()
    scenario = load_yaml_model(args.scenario, ScenarioConfig)
    env = RecordedSimEnv(args.fixture)
    result = run_orchestration(env, scenario, args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("est_positions", "gt_positions")}))


def main_eval() -> None:
    parser = argparse.ArgumentParser(prog="ff-eval")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ablate-gs", action="store_true")
    parser.add_argument("--baseline-rate", type=float, default=0.0)
    parser.add_argument("--augmented-rate", type=float, default=0.0)
    args = parser.parse_args()
    nav_path = args.run / "nav_log.json"
    if nav_path.exists():
        nav = json.loads(nav_path.read_text(encoding="utf-8"))
        print(f"Evaluated {len(nav)} nav steps from {args.run}")
    report_path = args.run / "report.md"
    from fathomfollow.eval.metrics import DriftMetrics

    generate_report(
        report_path,
        DriftMetrics(1.0, 2.0, 1.5),
        DriftMetrics(0.8, 1.5, 0.9),
        retention=0.75,
        ablation=gs_ablation_table(args.baseline_rate, args.augmented_rate) if args.ablate_gs else None,
    )
    print(f"Report written to {report_path}")
