"""Eval-only ground-truth bbox projection and detection quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fathomfollow.sim.holoocean_env import quat_to_rotation_matrix


@dataclass(frozen=True)
class DetectionQualityMetrics:
    gt_in_frame_fraction: float
    precision: float
    recall: float
    mean_iou: float
    n_frames: int


def world_to_body(point_world: np.ndarray, pose7: np.ndarray) -> np.ndarray:
    position = np.asarray(pose7[:3], dtype=np.float64)
    quat = np.asarray(pose7[3:7], dtype=np.float64)
    rot = quat_to_rotation_matrix(quat)
    return rot.T @ (np.asarray(point_world, dtype=np.float64) - position)


def project_target_bbox_xyxy(
    gt_pose: np.ndarray,
    gt_target_pose: np.ndarray,
    image_shape: tuple[int, int],
    *,
    target_radius_m: float = 0.5,
    fov_deg: float = 90.0,
) -> tuple[float, float, float, float] | None:
    """Project a spherical target into pixel xyxy bbox (eval-only pinhole model).

    Body frame convention (HoloOcean AUV): +x forward, +y right, +z down.
    Camera optical axis aligns with +x; depth is ``body[0]``.
    """
    target_world = np.asarray(gt_target_pose[:3], dtype=np.float64)
    body = world_to_body(target_world, gt_pose)
    x_fwd, y_right, z_down = body
    if x_fwd <= 0.1:
        return None

    height, width = image_shape
    fx = width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    u = fx * y_right / x_fwd + cx
    v = fy * z_down / x_fwd + cy
    r_px = fx * target_radius_m / x_fwd

    x1 = max(0.0, u - r_px)
    y1 = max(0.0, v - r_px)
    x2 = min(float(width), u + r_px)
    y2 = min(float(height), v + r_px)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_in_image(bbox_xyxy: tuple[float, float, float, float], image_shape: tuple[int, int]) -> bool:
    height, width = image_shape
    x1, y1, x2, y2 = bbox_xyxy
    return x2 > 0 and y2 > 0 and x1 < width and y1 < height


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _xywh_norm_to_xyxy(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    height, width = image_shape
    xc, yc, w, h = bbox
    bw, bh = w * width, h * height
    cx, cy = xc * width, yc * height
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def compute_detection_quality(
    ctrl_log: list[dict],
    image_shape: tuple[int, int],
    *,
    iou_threshold: float = 0.3,
    target_radius_m: float = 0.5,
    fov_deg: float = 90.0,
) -> DetectionQualityMetrics | None:
    """Aggregate GT projection + detection match metrics from control log entries."""
    if not ctrl_log:
        return None
    if not all("gt_pose" in e and "gt_target_pose" in e for e in ctrl_log):
        return None

    gt_in_frame = 0
    tp = fp = fn = 0
    ious: list[float] = []

    for entry in ctrl_log:
        gt_pose = np.asarray(entry["gt_pose"], dtype=np.float64)
        gt_target = np.asarray(entry["gt_target_pose"], dtype=np.float64)
        gt_bbox = project_target_bbox_xyxy(
            gt_pose,
            gt_target,
            image_shape,
            target_radius_m=target_radius_m,
            fov_deg=fov_deg,
        )
        gt_visible = gt_bbox is not None and bbox_in_image(gt_bbox, image_shape)
        if gt_visible:
            gt_in_frame += 1

        pred_boxes = [
            _xywh_norm_to_xyxy(tuple(d["bbox"]), image_shape)
            for d in entry.get("dets", [])
            if "bbox" in d
        ]

        if not gt_visible:
            fp += len(pred_boxes)
            continue

        if not pred_boxes:
            fn += 1
            continue

        best_iou = max(iou_xyxy(p, gt_bbox) for p in pred_boxes)
        if best_iou >= iou_threshold:
            tp += 1
            ious.append(best_iou)
            fp += len(pred_boxes) - 1
        else:
            fn += 1
            fp += len(pred_boxes)

    n = len(ctrl_log)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return DetectionQualityMetrics(
        gt_in_frame_fraction=gt_in_frame / n,
        precision=precision,
        recall=recall,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        n_frames=n,
    )
