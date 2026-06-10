from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from fathomfollow.gs.base import GSRenderer, Pose
from fathomfollow.gs.recorded import GSRenderManifest, GSScene


def load_colmap_poses_fixture(data: dict) -> list[Pose]:
    poses = []
    for entry in data.get("poses", []):
        poses.append(
            Pose(
                position=tuple(entry["position"]),
                orientation=tuple(entry["orientation"]),
            )
        )
    return poses


def detect_colmap_dataset_layout(source: Path) -> tuple[str, str]:
    """Return (images_subdir, colmap_path) relative to source."""
    for images_name in ("Images_wb", "images_wb", "images"):
        if (source / images_name).is_dir():
            if (source / "sparse" / "0").is_dir():
                return images_name, "sparse/0"
            if (source / "colmap" / "sparse" / "0").is_dir():
                return images_name, "colmap/sparse/0"
    raise ValueError(f"no COLMAP layout found under {source}")


def _conda_exe() -> Path:
    home = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", "")))
    return home / "anaconda3" / "Scripts" / "conda.exe"


def _ns_script(env: str, name: str) -> Path:
    home = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", "")))
    return home / "anaconda3" / "envs" / env / "Scripts" / f"{name}.exe"


def _gs_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = os.environ.get("USERPROFILE", env.get("HOME", ""))
    home = Path(env["HOME"])
    gs_env = home / "anaconda3" / "envs" / "water_splatting"
    gs_paths = [str(gs_env / "Scripts"), str(gs_env / "Library" / "bin")]
    env["PATH"] = os.pathsep.join(gs_paths) + os.pathsep + env.get("PATH", "")
    return env


def _find_latest_config(search_roots: list[Path]) -> Path | None:
    configs: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        configs.extend(root.rglob("config.yml"))
    if not configs:
        return None
    return max(configs, key=lambda p: p.stat().st_mtime)


def _read_train_psnr(config_path: Path) -> float:
    events = list(config_path.parent.glob("events.out.tfevents.*"))
    if not events:
        return 0.0
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        acc = EventAccumulator(str(config_path.parent), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        for tag in (
            "Eval Images Metrics Dict (all images)/psnr",
            "Eval Images Metrics/psnr",
            "Train Metrics Dict/psnr",
            "Eval Metrics Dict/psnr",
            "psnr",
        ):
            if tag in tags:
                events_list = acc.Scalars(tag)
                if events_list:
                    return float(events_list[-1].value)
    except Exception:
        pass
    return 0.0


def _ensure_downscaled_images(source: Path, images_path: str, factor: int) -> None:
    """Create Images_wb_N folders expected by nerfstudio colmap parser (no ffmpeg)."""
    if factor <= 1:
        return
    src_dir = source / images_path
    dst_dir = source / f"{images_path}_{factor}"
    if dst_dir.is_dir() and any(dst_dir.glob("*.png")):
        return
    from PIL import Image

    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(src_dir.glob("*.png")):
        out_path = dst_dir / img_path.name
        if out_path.is_file():
            continue
        with Image.open(img_path) as img:
            w, h = img.size
            new_w = max(1, int(np.floor(w / factor)))
            new_h = max(1, int(np.floor(h / factor)))
            resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized.save(out_path)


def _c2w_to_pose(c2w: np.ndarray) -> Pose:
    """Extract position + quaternion from a 3x4 or 4x4 camera-to-world matrix."""
    mat = np.asarray(c2w, dtype=np.float64)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    pos = tuple(mat[:3, 3].tolist())
    r = mat[:3, :3]
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return Pose(position=pos, orientation=(x, y, z, w))


def export_colmap_cameras_json(
    source: Path,
    dest: Path,
    downscale_factor: int = 2,
    conda_env: str = "water_splatting",
) -> None:
    """Export training camera intrinsics/extrinsics for aligned ns-render."""
    source = source.resolve()
    images_path, colmap_path = detect_colmap_dataset_layout(source)
    dest = dest.resolve()
    home = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", "")))
    py = home / "anaconda3" / "envs" / conda_env / "python.exe"
    script = f"""
import json, math
from pathlib import Path
import numpy as np
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig
cfg = ColmapDataParserConfig(
    data=Path({repr(str(source))}),
    images_path={repr(images_path)},
    colmap_path={repr(colmap_path)},
    downscale_factor={downscale_factor},
)
dp = ColmapDataParser(cfg)
parsed = dp.get_dataparser_outputs(split="train")
cams = parsed.cameras
entries = []
for i in range(len(cams)):
    c2w = cams.camera_to_worlds[i].cpu().numpy().tolist()
    h, w = int(cams.image_height[i]), int(cams.image_width[i])
    fx = float(cams.fx[i])
    fov = float(2 * math.atan(w / (2 * fx)) * 180 / math.pi)
    entries.append({{"c2w": c2w, "height": h, "width": w, "fov": fov}})
Path({repr(str(dest))}).write_text(json.dumps(entries, indent=2), encoding="utf-8")
"""
    subprocess.run([str(py)], input=script, text=True, check=True, env=_gs_subprocess_env())


def _apply_turbidity_tint(rgb: np.ndarray, turbidity: float) -> np.ndarray:
    tinted = rgb.astype(np.float32)
    tinted[:, :, 2] = np.clip(tinted[:, :, 2] * (1.0 - turbidity * 0.5), 0, 255)
    return tinted.astype(np.uint8)


class WaterSplattingGSRenderer:
    """Adapter for WaterSplatting; subprocess in separate conda env for train/render."""

    def __init__(self, conda_env: str = "water_splatting") -> None:
        self._conda_env = conda_env
        self._checkpoint: str | None = None
        self._config_path: Path | None = None
        self._render_cache: dict[tuple[tuple, float], np.ndarray] = {}
        self._colmap_cameras: list[dict] = []

    def load(self, checkpoint: str) -> None:
        self._checkpoint = checkpoint
        ckpt = Path(checkpoint)
        scene_json = ckpt / "scene.json" if ckpt.is_dir() else ckpt.parent / "scene.json"
        scene_dir = ckpt if ckpt.is_dir() else ckpt.parent
        self._colmap_cameras = []
        if scene_json.is_file():
            scene = GSScene.load(scene_json)
            config_candidate = Path(scene.checkpoint_path)
            if config_candidate.is_file():
                self._config_path = config_candidate
            elif (config_candidate / "config.yml").is_file():
                self._config_path = config_candidate / "config.yml"
            cam_json = scene_dir / "colmap_cameras.json"
            if cam_json.is_file():
                self._colmap_cameras = json.loads(cam_json.read_text(encoding="utf-8"))
        if self._config_path is None and ckpt.is_file() and ckpt.name == "config.yml":
            self._config_path = ckpt
        self._render_cache.clear()

    def _camera_spec_for_pose(self, pose: Pose) -> dict | None:
        if not self._colmap_cameras:
            return None
        target = np.array(pose.position, dtype=np.float64)
        best_idx = 0
        best_dist = float("inf")
        for i, spec in enumerate(self._colmap_cameras):
            c2w = np.array(spec["c2w"], dtype=np.float64)
            dist = float(np.linalg.norm(c2w[:3, 3] - target))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return self._colmap_cameras[best_idx]

    def render(self, pose: Pose, turbidity: float) -> np.ndarray:
        if self._checkpoint is None:
            raise RuntimeError("call load() before render()")
        if not 0.0 <= turbidity <= 1.0:
            raise ValueError(f"turbidity {turbidity} must be in [0, 1]")
        key = (pose.position, pose.orientation, turbidity)
        if key in self._render_cache:
            return self._render_cache[key]
        if self._config_path is None or not self._config_path.is_file():
            h, w = 480, 640
            rgb = np.full((h, w, 3), int(100 - turbidity * 50), dtype=np.uint8)
            self._render_cache[key] = rgb
            return rgb
        pose_key = (pose.position, pose.orientation)
        base = self._render_cache.get((pose_key, 0.0))
        if base is None:
            base = self._render_pose_subprocess(pose)
            self._render_cache[(pose_key, 0.0)] = base
        rgb = _apply_turbidity_tint(base, turbidity) if turbidity > 0 else base
        self._render_cache[key] = rgb
        return rgb

    def _render_pose_subprocess(self, pose: Pose) -> np.ndarray:
        assert self._config_path is not None
        out_dir = self._config_path.parent / "ff_render_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = self._camera_spec_for_pose(pose)
        if spec is not None:
            c2w = _c2w_to_4x4(spec["c2w"])
            height, width, fov = spec["height"], spec["width"], spec["fov"]
        else:
            c2w = _pose_to_camera_to_world(pose)
            height, width, fov = 480, 640, 60.0
        pose_tag = abs(hash((pose.position, pose.orientation))) % 1000000
        cam_path = out_dir / f"pose_cam_{pose_tag}.json"
        output_base = out_dir / f"frame_{pose_tag}"
        cam_path.write_text(
            json.dumps(
                {
                    "camera_type": "perspective",
                    "render_height": height,
                    "render_width": width,
                    "camera_path": [
                        {
                            "camera_to_world": c2w,
                            "fov": fov,
                        }
                    ],
                    "fps": 1,
                    "seconds": 1.0,
                    "is_cycle": False,
                    "smoothness_value": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        ns_render = _ns_script(self._conda_env, "ns-render")
        cmd = [
            str(ns_render),
            "camera-path",
            "--load-config",
            str(self._config_path),
            "--camera-path-filename",
            str(cam_path),
            "--output-path",
            str(output_base),
            "--image-format",
            "png",
            "--output-format",
            "images",
        ]
        subprocess.run(cmd, check=True, env=_gs_subprocess_env(), capture_output=True)
        pngs = sorted(output_base.rglob("*.png"))
        if not pngs:
            raise RuntimeError(f"ns-render produced no images under {output_base}")
        from PIL import Image

        return np.array(Image.open(pngs[0]).convert("RGB"))

    def train_subprocess(
        self,
        source: Path,
        out: Path,
        max_iterations: int = 3000,
        output_dir: Path | None = None,
    ) -> GSScene:
        source = source.resolve()
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        images_path, colmap_path = detect_colmap_dataset_layout(source)
        downscale_factor = 2
        _ensure_downscaled_images(source, images_path, downscale_factor)
        train_root = output_dir or out.parent / "train_outputs"
        train_root.mkdir(parents=True, exist_ok=True)

        ns_train = _ns_script(self._conda_env, "ns-train")
        cmd = [
            str(ns_train),
            "water-splatting",
            "--output-dir",
            str(train_root),
            "--experiment-name",
            out.name,
            "--vis",
            "tensorboard",
            "--max-num-iterations",
            str(max_iterations),
            "--steps-per-save",
            str(min(1000, max_iterations)),
            "--steps-per-eval-image",
            str(min(1000, max_iterations)),
            "colmap",
            "--data",
            str(source),
            "--images-path",
            images_path,
            "--colmap-path",
            colmap_path,
            "--downscale-factor",
            str(downscale_factor),
        ]
        subprocess.run(cmd, check=True, env=_gs_subprocess_env())

        config_path = _find_latest_config([train_root, Path("outputs")])
        if config_path is None:
            raise RuntimeError(f"no config.yml found under {train_root}")
        train_psnr = _read_train_psnr(config_path)
        scene = GSScene(
            scene_id=out.name,
            source_dataset=str(source),
            library="watersplatting",
            n_gaussians=0,
            pose_source="colmap",
            checkpoint_path=str(config_path),
            train_psnr=train_psnr,
        )
        scene.write(out / "scene.json")
        export_colmap_cameras_json(
            source, out / "colmap_cameras.json", downscale_factor, self._conda_env
        )
        return scene


def _c2w_to_4x4(c2w: list | np.ndarray) -> list[list[float]]:
    mat = np.asarray(c2w, dtype=np.float64)
    if mat.shape == (3, 4):
        mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
    return mat.tolist()


def _pose_to_camera_to_world(pose: Pose) -> list[list[float]]:
    """Build 4x4 camera-to-world from position + quaternion (x,y,z,w)."""
    x, y, z, w = pose.orientation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    tx, ty, tz = pose.position
    return [
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]
