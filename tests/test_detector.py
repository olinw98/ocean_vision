from unittest.mock import MagicMock

import numpy as np

from fathomfollow.perception.detector import MockDetector, YoloDetector


def _fake_yolo_box(class_id: int, confidence: float, xyxy: list[float]) -> MagicMock:
    box = MagicMock()
    box.cls = [class_id]
    box.conf = [confidence]
    xyxy_tensor = MagicMock()
    xyxy_tensor.cpu.return_value.numpy.return_value = np.array(xyxy, dtype=np.float32)
    box.xyxy = [xyxy_tensor]
    return box


def _yolo_with_fake_boxes(boxes: list[MagicMock], class_id: int | None = None) -> YoloDetector:
    fake_result = MagicMock()
    fake_result.boxes = boxes
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]
    det = YoloDetector(weights="fake.pt", class_id=class_id)
    det._model = fake_model
    return det


def test_mock_detector_returns_records_on_bright_image() -> None:
    det = MockDetector(conf_threshold=0.25)
    bright = np.full((480, 640, 3), 150, dtype=np.uint8)
    dets = det.detect(bright, frame_id=0)
    assert len(dets) == 1
    assert dets[0].confidence >= 0.25
    assert len(dets[0].bbox) == 4


def test_mock_detector_filters_dark_image() -> None:
    det = MockDetector()
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    assert det.detect(dark) == []


def test_yolo_detector_filters_by_class_id() -> None:
    boxes = [
        _fake_yolo_box(0, 0.9, [100.0, 100.0, 200.0, 200.0]),
        _fake_yolo_box(1, 0.85, [300.0, 100.0, 400.0, 200.0]),
    ]
    det = _yolo_with_fake_boxes(boxes, class_id=0)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.detect(rgb, frame_id=7)
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].frame_id == 7


def test_yolo_detector_without_class_filter_returns_all_classes() -> None:
    boxes = [
        _fake_yolo_box(0, 0.9, [100.0, 100.0, 200.0, 200.0]),
        _fake_yolo_box(1, 0.85, [300.0, 100.0, 400.0, 200.0]),
    ]
    det = _yolo_with_fake_boxes(boxes)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert len(det.detect(rgb)) == 2
