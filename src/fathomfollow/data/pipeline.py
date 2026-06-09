from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class DatasetManifest:
    taxa: list[str]
    n_images: int
    split_counts: dict[str, int]
    fathomnet_snapshot_date: str
    sources: list[str]
    data_yaml_path: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


CANDIDATE_TAXA = ["Bathochordaeus", "Granelledone", "Benthocodon"]


def split_by_hash(image_id: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    h = int(hashlib.sha256(image_id.encode()).hexdigest(), 16) % 1000
    if h < int(train_ratio * 1000):
        return "train"
    if h < int((train_ratio + val_ratio) * 1000):
        return "val"
    return "test"


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return (xc, yc, w / img_w, h / img_h)


def convert_coco_to_yolo(coco: dict, out_dir: Path, class_names: list[str]) -> DatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_by_id = {img["id"]: img for img in coco["images"]}
    cat_to_class = {cat["id"]: class_names.index(cat["name"]) for cat in coco["categories"]}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for ann in coco["annotations"]:
        img = images_by_id[ann["image_id"]]
        split = split_by_hash(img["file_name"])
        split_counts[split] += 1
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        src = Path(coco.get("_image_root", ".")) / img["file_name"]
        dst = img_dir / Path(img["file_name"]).name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            Image.new("RGB", (int(img["width"]), int(img["height"])), color=(30, 60, 90)).save(
                dst
            )

        yolo_bbox = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
        class_id = cat_to_class[ann["category_id"]]
        label_path = lbl_dir / (dst.stem + ".txt")
        with label_path.open("a", encoding="utf-8") as f:
            f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n")

    data_yaml = out_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_names)}\nnames:\n{names_block}\n",
        encoding="utf-8",
    )

    manifest = DatasetManifest(
        taxa=class_names,
        n_images=sum(split_counts.values()),
        split_counts=split_counts,
        fathomnet_snapshot_date=coco.get("info", {}).get("date_created", "unknown"),
        sources=["fathomnet"],
        data_yaml_path=str(data_yaml),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest


def prepare_from_coco(coco_path: Path, out_dir: Path, class_names: list[str]) -> DatasetManifest:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    return convert_coco_to_yolo(coco, out_dir, class_names)
