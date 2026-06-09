from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fathomfollow.perception.types import DetectionRecord


def parse_training_metrics(raw: dict) -> dict:
    """Parse Ultralytics training metrics dict into a stable JSON-serializable form."""
    return {
        "mAP50": float(raw.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(raw.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(raw.get("metrics/precision(B)", 0.0)),
        "recall": float(raw.get("metrics/recall(B)", 0.0)),
    }


def metrics_from_train_results(results: object) -> dict:
    """Extract metrics from an Ultralytics train() return value."""
    if results is None:
        return parse_training_metrics({})
    raw = getattr(results, "results_dict", None)
    if raw is None and hasattr(results, "metrics"):
        metrics = results.metrics
        raw = metrics if isinstance(metrics, dict) else getattr(metrics, "results_dict", {})
    if not isinstance(raw, dict):
        raw = {}
    return parse_training_metrics(raw)


class YoloDetector:
    def __init__(self, weights: str | Path = "yolo11n.pt", conf_threshold: float = 0.25) -> None:
        self._weights = str(weights)
        self._conf_threshold = conf_threshold
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._weights)

    def detect(self, rgb: np.ndarray, frame_id: int = 0) -> list[DetectionRecord]:
        self._load()
        h, w = rgb.shape[:2]
        results = self._model.predict(rgb, verbose=False, conf=self._conf_threshold)
        dets: list[DetectionRecord] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                dets.append(
                    DetectionRecord(
                        frame_id=frame_id,
                        class_id=int(box.cls[0]),
                        bbox=(float(xc), float(yc), float(bw), float(bh)),
                        confidence=float(box.conf[0]),
                    )
                )
        return [d for d in dets if d.confidence >= self._conf_threshold]

    @staticmethod
    def parse_metrics(metrics_path: Path) -> dict:
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        return {}


class MockDetector:
    """Fixture detector for tests without weights."""

    def __init__(self, conf_threshold: float = 0.25) -> None:
        self._conf = conf_threshold

    def detect(self, rgb: np.ndarray, frame_id: int = 0) -> list[DetectionRecord]:
        if rgb.mean() > 100:
            return [
                DetectionRecord(
                    frame_id=frame_id,
                    class_id=0,
                    bbox=(0.5, 0.5, 0.2, 0.2),
                    confidence=0.9,
                )
            ]
        return []
