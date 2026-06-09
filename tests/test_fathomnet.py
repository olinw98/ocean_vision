import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fathomfollow.data.fathomnet import (
    parse_count_output,
    run_fathomnet_count,
    select_taxon_by_count,
)
from fathomfollow.data.fathomnet import reorganize_yolo_flat
from fathomfollow.data.pipeline import CANDIDATE_TAXA


def test_parse_count_output_single_taxon() -> None:
    text = "concept        |  # boxes\n---------------|---------\nBathochordaeus |     2017\n"
    assert parse_count_output(text) == {"Bathochordaeus": 2017}


def test_parse_count_output_no_boxes() -> None:
    assert parse_count_output("No bounding boxes found\n") == {}


def test_select_taxon_by_count_picks_max() -> None:
    counts = {"Bathochordaeus": 2017, "Benthocodon": 662, "Granelledone": 0}
    assert select_taxon_by_count(counts, CANDIDATE_TAXA) == "Bathochordaeus"


def test_select_taxon_by_count_raises_when_all_zero() -> None:
    with pytest.raises(ValueError, match="no taxon"):
        select_taxon_by_count({"Granelledone": 0}, CANDIDATE_TAXA)


def test_run_fathomnet_count_delegates_to_cli() -> None:
    fake_out = "concept | # boxes\nBathochordaeus | 10\n"
    with patch("fathomfollow.data.fathomnet.subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_out
        mock_run.return_value.returncode = 0
        counts = run_fathomnet_count(["Bathochordaeus"])
    assert counts["Bathochordaeus"] == 10
    args = mock_run.call_args[0][0]
    assert "fathomnet-generate" in args[0] or args[0].endswith("fathomnet-generate")


def test_reorganize_yolo_flat(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for i in range(5):
        name = f"img_{i}.jpg"
        (src / "images" / name).write_bytes(b"x")
        (src / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    out = tmp_path / "yolo"
    manifest = reorganize_yolo_flat(src, out, "Benthocodon")
    assert manifest.n_images == 5
    assert (out / "data.yaml").exists()
    assert any((out / "images" / split).exists() for split in ("train", "val", "test"))
