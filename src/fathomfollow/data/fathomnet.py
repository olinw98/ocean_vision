from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fathomfollow.data.pipeline import CANDIDATE_TAXA, DatasetManifest, prepare_from_coco, split_by_hash


def parse_count_output(text: str) -> dict[str, int]:
    if "No bounding boxes found" in text and "|" not in text:
        return {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if "|" not in line or "concept" in line.lower() or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2:
            continue
        name, raw_n = parts
        if not name:
            continue
        m = re.search(r"\d+", raw_n)
        if m:
            counts[name] = int(m.group())
    return counts


def select_taxon_by_count(counts: dict[str, int], candidates: list[str]) -> str:
    best = max((counts.get(t, 0) for t in candidates), default=0)
    if best <= 0:
        raise ValueError(f"no taxon with boxes among candidates: {candidates}")
    for taxon in candidates:
        if counts.get(taxon, 0) == best:
            return taxon
    raise ValueError(f"no taxon with boxes among candidates: {candidates}")


def _fathomnet_generate_bin() -> str:
    exe = shutil.which("fathomnet-generate")
    if exe:
        return exe
    return str(Path(sys.executable).parent / "fathomnet-generate")


def run_fathomnet_count(taxa: list[str]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for taxon in taxa:
        cmd = [_fathomnet_generate_bin(), "-c", taxon, "--count"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        combined.update(parse_count_output(result.stdout))
        if taxon not in combined:
            combined[taxon] = 0
    return combined


def download_fathomnet_coco(taxon: str, out_dir: Path, img_dir: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = img_dir or (out_dir / "images")
    images.mkdir(parents=True, exist_ok=True)
    cmd = [
        _fathomnet_generate_bin(),
        "-c",
        taxon,
        "-f",
        "coco",
        "-o",
        str(out_dir),
        "--img-download",
        str(images),
    ]
    subprocess.run(cmd, check=True)
    coco_path = out_dir / "annotations.json"
    if not coco_path.exists():
        candidates = list(out_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no COCO JSON in {out_dir}")
        coco_path = candidates[0]
    return coco_path


def download_fathomnet_yolo(taxon: str, out_dir: Path, img_dir: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = img_dir or (out_dir / "images")
    images.mkdir(parents=True, exist_ok=True)
    cmd = [
        _fathomnet_generate_bin(),
        "-c",
        taxon,
        "-f",
        "yolo",
        "-o",
        str(out_dir),
        "--img-download",
        str(images),
    ]
    subprocess.run(cmd, check=True)
    data_yaml = out_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = out_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"no data.yaml or dataset.yaml in {out_dir}")
    return data_yaml


def reorganize_yolo_flat(src_dir: Path, out_dir: Path, taxon: str) -> DatasetManifest:
    """Split flat fathomnet YOLO export into train/val/test folders."""
    images_src = src_dir / "images"
    labels_src = src_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for img in images_src.glob("*"):
        if not img.is_file():
            continue
        split = split_by_hash(img.name)
        split_counts[split] += 1
        img_dst = out_dir / "images" / split / img.name
        lbl_dst = out_dir / "labels" / split / f"{img.stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, img_dst)
        lbl = labels_src / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, lbl_dst)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames:\n  0: {taxon}\n",
        encoding="utf-8",
    )
    manifest = DatasetManifest(
        taxa=[taxon],
        n_images=sum(split_counts.values()),
        split_counts=split_counts,
        fathomnet_snapshot_date="live",
        sources=["fathomnet"],
        data_yaml_path=str(data_yaml),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest


def manifest_from_yolo_dir(out_dir: Path, taxon: str) -> DatasetManifest:
    split_counts = {
        split: len(list((out_dir / "images" / split).glob("*")))
        for split in ("train", "val", "test")
        if (out_dir / "images" / split).exists()
    }
    if not split_counts:
        flat_count = len(list((out_dir / "images").glob("*")))
        split_counts = {"train": flat_count}
    data_yaml = out_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml = out_dir / "dataset.yaml"
    manifest = DatasetManifest(
        taxa=[taxon],
        n_images=sum(split_counts.values()),
        split_counts=split_counts,
        fathomnet_snapshot_date="live",
        sources=["fathomnet"],
        data_yaml_path=str(data_yaml),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest


def auto_select_and_prepare(
    out_dir: Path,
    candidates: list[str] | None = None,
    cache_dir: Path | None = None,
    format: str = "yolo",
) -> tuple[str, DatasetManifest]:
    taxa = candidates or CANDIDATE_TAXA
    counts = run_fathomnet_count(taxa)
    selected = select_taxon_by_count(counts, taxa)
    raw_dir = cache_dir or (out_dir if format == "yolo" else out_dir.parent / "fathomnet_raw" / selected)

    if format == "yolo":
        download_fathomnet_yolo(selected, raw_dir)
        manifest = reorganize_yolo_flat(raw_dir, out_dir, selected)
    else:
        coco_path = download_fathomnet_coco(selected, raw_dir)
        manifest = prepare_from_coco(coco_path, out_dir, [selected])

    selection = {
        "selected_taxon": selected,
        "counts": counts,
        "format": format,
        "data_yaml": manifest.data_yaml_path,
    }
    (out_dir / "taxon_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return selected, manifest
