from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from fathomfollow.data.pipeline import DatasetManifest, split_by_hash


def merge_datasets(sources: list[Path], out_dir: Path) -> DatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_sources: list[str] = []
    taxa: list[str] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    n_images = 0

    for src in sources:
        manifest = DatasetManifest.load(src / "manifest.json")
        all_sources.extend(manifest.sources)
        taxa = list(set(taxa + manifest.taxa))

        for split in ("train", "val", "test"):
            img_src = src / "images" / split
            lbl_src = src / "labels" / split
            if not img_src.exists():
                continue
            img_dst = out_dir / "images" / split
            lbl_dst = out_dir / "labels" / split
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)
            for img in img_src.glob("*"):
                tag = src.name
                new_name = f"{tag}_{img.name}"
                new_split = split_by_hash(new_name)
                img_dst_dir = out_dir / "images" / new_split
                lbl_dst_dir = out_dir / "labels" / new_split
                img_dst_dir.mkdir(parents=True, exist_ok=True)
                lbl_dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, img_dst_dir / new_name)
                lbl = lbl_src / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, lbl_dst_dir / f"{tag}_{img.stem}.txt")
                split_counts[new_split] += 1
                n_images += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(sorted(set(taxa))))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(set(taxa))}\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        taxa=sorted(set(taxa)),
        n_images=n_images,
        split_counts=split_counts,
        fathomnet_snapshot_date="merged",
        sources=all_sources,
        data_yaml_path=str(data_yaml),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest
