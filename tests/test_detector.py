import numpy as np

from fathomfollow.perception.detector import MockDetector


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
