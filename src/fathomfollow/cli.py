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
from fathomfollow.data.fathomnet import (
    auto_select_and_prepare,
    run_fathomnet_count,
    select_taxon_by_count,
)
from fathomfollow.data.merge import merge_datasets
from fathomfollow.data.pipeline import CANDIDATE_TAXA, prepare_from_coco
from fathomfollow.eval.metrics import gs_ablation_table
from fathomfollow.eval.report import generate_report
from fathomfollow.nav.drift_gate import run_drift_gate
from fathomfollow.nav.training import train_nav_estimator
from fathomfollow.perception.detector import metrics_from_train_results
from fathomfollow.gs.base import GSRenderer, Pose
from fathomfollow.gs.recorded import GSScene, RecordedGSRenderer
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

    count_p = sub.add_parser("count")
    count_p.add_argument(
        "--taxa",
        type=str,
        default=",".join(CANDIDATE_TAXA),
        help="Comma-separated candidate taxa",
    )

    auto_p = sub.add_parser("auto-prepare")
    auto_p.add_argument("--out", type=Path, required=True)
    auto_p.add_argument(
        "--format",
        type=str,
        choices=["yolo", "coco"],
        default="yolo",
        help="FathomNet export format (yolo avoids COCO bbox bugs)",
    )
    auto_p.add_argument(
        "--taxa",
        type=str,
        default=",".join(CANDIDATE_TAXA),
        help="Comma-separated candidate taxa",
    )

    merge_p = sub.add_parser("merge")
    merge_p.add_argument("--sources", type=str, required=True)
    merge_p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare_from_coco(args.coco, args.out, [args.taxa])
    elif args.cmd == "count":
        taxa = [t.strip() for t in args.taxa.split(",")]
        counts = run_fathomnet_count(taxa)
        selected = select_taxon_by_count(counts, taxa)
        print(json.dumps({"counts": counts, "selected": selected}, indent=2))
    elif args.cmd == "auto-prepare":
        taxa = [t.strip() for t in args.taxa.split(",")]
        selected, manifest = auto_select_and_prepare(args.out, taxa, format=args.format)
        print(
            json.dumps(
                {
                    "selected_taxon": selected,
                    "n_images": manifest.n_images,
                    "data_yaml": manifest.data_yaml_path,
                },
                indent=2,
            )
        )
    elif args.cmd == "merge":
        sources = [Path(s.strip()) for s in args.sources.split(",")]
        merge_datasets(sources, args.out)


def _load_gs_renderer(scene_checkpoint: Path) -> GSRenderer:
    ckpt = Path(scene_checkpoint)
    scene_json = ckpt / "scene.json" if ckpt.is_dir() else ckpt.parent / "scene.json"
    if scene_json.is_file():
        scene = GSScene.load(scene_json)
        if scene.library == "watersplatting":
            renderer = WaterSplattingGSRenderer()
            renderer.load(str(ckpt))
            return renderer
    renderer = RecordedGSRenderer(ckpt)
    renderer.load(str(ckpt))
    return renderer


def main_gs() -> None:
    parser = argparse.ArgumentParser(prog="ff-gs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--source", type=Path, required=True)
    train_p.add_argument("--out", type=Path, required=True)
    train_p.add_argument("--library", type=str, default="watersplatting")
    train_p.add_argument("--max-iterations", type=int, default=3000)

    render_p = sub.add_parser("render")
    render_p.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        renderer = WaterSplattingGSRenderer()
        scene = renderer.train_subprocess(
            args.source, args.out, max_iterations=args.max_iterations
        )
        print(
            json.dumps(
                {
                    "scene_id": scene.scene_id,
                    "checkpoint": scene.checkpoint_path,
                    "train_psnr": scene.train_psnr,
                }
            )
        )
    elif args.cmd == "render":
        cfg = load_yaml_model(args.config, RenderConfig)
        cam = load_yaml_model(cfg.camera_path, CameraPathConfig)
        renderer = _load_gs_renderer(cfg.scene_checkpoint)
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
    results = model.train(
        data=str(cfg.data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
    )
    metrics = metrics_from_train_results(results)
    out = Path(cfg.data_yaml).parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main_train_nav() -> None:
    parser = argparse.ArgumentParser(prog="ff-train-nav")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml_model(args.config, NavTrainingConfig)
    out_dir = cfg.trajectories_dir.parent / "nav_model"
    ckpt = train_nav_estimator(cfg, out_dir)
    print(json.dumps({"checkpoint": str(ckpt), "out_dir": str(out_dir)}))


def main_run() -> None:
    parser = argparse.ArgumentParser(prog="ff-run")
    parser.add_argument("--scenario", type=Path, default=Path("config/scenario_holoocean.yaml"))
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/sim/holoocean_smoke.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    parser.add_argument(
        "--detector",
        type=Path,
        default=None,
        help="YOLO weights (e.g. runs/detect/train-2/weights/best.pt); MockDetector if omitted",
    )
    parser.add_argument(
        "--nav-checkpoint",
        type=Path,
        default=None,
        help="DriftGuard weights (e.g. data/nav_model/velocity_estimator.pt)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live HoloOcean instead of a recorded fixture",
    )
    args = parser.parse_args()
    scenario = load_yaml_model(args.scenario, ScenarioConfig)
    if args.live:
        from fathomfollow.sim.holoocean_env import HoloOceanSimEnv

        env = HoloOceanSimEnv(scenario.holoocean_scenario, target_config=scenario.target)
    else:
        env = RecordedSimEnv(args.fixture)
    result = run_orchestration(
        env,
        scenario,
        args.out,
        nav_checkpoint=args.nav_checkpoint,
        detector_weights=args.detector,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("est_positions", "gt_positions")}))


def main_drift_gate() -> None:
    parser = argparse.ArgumentParser(prog="ff-drift-gate")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/sim/holoocean_smoke.npz"),
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("config/scenario_holoocean.yaml"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/drift_gate"))
    parser.add_argument(
        "--nav-checkpoint",
        type=Path,
        default=None,
        help="DriftGuard weights; uses untrained estimator if omitted",
    )
    args = parser.parse_args()
    scenario = load_yaml_model(args.scenario, ScenarioConfig)
    result = run_drift_gate(
        args.fixture,
        scenario,
        args.out,
        nav_checkpoint=args.nav_checkpoint,
        scenario_path=args.scenario,
    )
    print(json.dumps(result.to_dict(), indent=2))


def main_eval() -> None:
    parser = argparse.ArgumentParser(prog="ff-eval")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ablate-gs", action="store_true")
    parser.add_argument("--baseline-rate", type=float, default=0.0)
    parser.add_argument("--augmented-rate", type=float, default=0.0)
    args = parser.parse_args()
    report_path = args.run / "report.md"
    from fathomfollow.eval.run_eval import evaluate_run

    result = evaluate_run(args.run, report_path=report_path)
    if args.ablate_gs:
        ablation = gs_ablation_table(args.baseline_rate, args.augmented_rate)
        generate_report(
            report_path,
            drift_baseline=result.drift_baseline,
            drift_learned=result.drift_learned,
            retention=result.tracking_retention,
            ablation=ablation,
            detection_quality=result.detection_quality,
        )
    payload = {
        "drift_learned_mean": result.drift_learned.mean_drift,
        "drift_within_dropout": result.drift_learned.drift_within_dropout,
        "tracking_retention": result.tracking_retention,
        "report": str(report_path),
    }
    if result.detection_quality is not None:
        payload["gt_in_frame_fraction"] = result.detection_quality.gt_in_frame_fraction
        payload["detection_precision"] = result.detection_quality.precision
        payload["detection_recall"] = result.detection_quality.recall
        payload["detection_mean_iou"] = result.detection_quality.mean_iou
    print(json.dumps(payload))
