from pathlib import Path
from unittest.mock import MagicMock
import sys

import pytest

from fathomfollow.config.models import DetectorTrainingConfig, NavTrainingConfig, load_yaml_model


def test_detector_train_config_pins_canonical_run_name() -> None:
    cfg = load_yaml_model(Path("config/detector_train.yaml"), DetectorTrainingConfig)
    assert cfg.run_name == "train-2"
    assert cfg.project == Path("runs/detect")
    assert cfg.canonical_weights_path() == Path("runs/detect/train-2/weights/best.pt")


def test_nav_train_holoocean_pins_canonical_checkpoint_dir() -> None:
    cfg = load_yaml_model(Path("config/nav_train_holoocean.yaml"), NavTrainingConfig)
    assert cfg.checkpoint_dir == Path("data/nav_model")
    assert cfg.canonical_checkpoint_path() == Path("data/nav_model/velocity_estimator.pt")


def test_train_detector_passes_pinned_project_and_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from fathomfollow.cli import main_train_detector

    config_path = tmp_path / "detector_train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_yaml: {(tmp_path / 'data.yaml').as_posix()}",
                "epochs: 1",
                "project: runs/detect",
                "run_name: train-2",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text("path: .\n", encoding="utf-8")

    captured: dict = {}
    fake_results = MagicMock()
    fake_results.results_dict = {}

    def fake_train(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return fake_results

    fake_model = MagicMock()
    fake_model.train = fake_train
    monkeypatch.setattr("ultralytics.YOLO", lambda model: fake_model)
    monkeypatch.setattr(
        "sys.argv",
        ["ff-train-detector", "--config", str(config_path)],
    )

    main_train_detector()

    assert captured.get("project") == str(Path("runs/detect"))
    assert captured.get("name") == "train-2"
    assert captured.get("exist_ok") is True
