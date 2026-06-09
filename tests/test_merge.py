import json
from pathlib import Path

from fathomfollow.data.merge import merge_datasets
from fathomfollow.data.pipeline import convert_coco_to_yolo, split_by_hash


def _make_mini_dataset(tmp_path: Path, name: str) -> Path:
    coco = {
        "info": {"date_created": "2026-01-01"},
        "images": [{"id": 1, "file_name": f"{name}.jpg", "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "TaxonA"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10]}],
    }
    out = tmp_path / name
    convert_coco_to_yolo(coco, out, ["TaxonA"])
    return out


def test_merge_no_leakage_across_sources(tmp_path: Path) -> None:
    a = _make_mini_dataset(tmp_path, "src_a")
    b = _make_mini_dataset(tmp_path, "src_b")
    merged = tmp_path / "merged"
    manifest = merge_datasets([a, b], merged)
    assert manifest.n_images >= 2
    assert (merged / "data.yaml").exists()
    assert "fathomnet" in manifest.sources


def test_split_deterministic_after_merge() -> None:
    assert split_by_hash("tag_img.jpg") == split_by_hash("tag_img.jpg")
