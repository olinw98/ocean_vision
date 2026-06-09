import json
from pathlib import Path

import pytest

from fathomfollow.data.pipeline import (
    coco_bbox_to_yolo,
    convert_coco_to_yolo,
    split_by_hash,
)


def test_coco_bbox_to_yolo_center() -> None:
    bbox = coco_bbox_to_yolo([100, 100, 200, 200], 1000, 1000)
    assert bbox == pytest.approx((0.2, 0.2, 0.2, 0.2), abs=1e-6)


def test_split_by_hash_deterministic() -> None:
    assert split_by_hash("img_a.jpg") == split_by_hash("img_a.jpg")
    splits = {split_by_hash(f"img_{i}.jpg") for i in range(100)}
    assert len(splits) > 1


def test_convert_coco_to_yolo(tmp_path: Path) -> None:
    coco = {
        "info": {"date_created": "2026-01-01"},
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "Bathochordaeus"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
    }
    coco_path = tmp_path / "coco.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    out = tmp_path / "yolo"
    manifest = convert_coco_to_yolo(coco, out, ["Bathochordaeus"])
    assert manifest.n_images >= 1
    assert (out / "data.yaml").exists()
    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) >= 1
    line = labels[0].read_text(encoding="utf-8").strip()
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "0"
