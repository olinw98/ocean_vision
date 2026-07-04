from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DropoutConfig(BaseModel):
    alt_min: float = Field(2.0, ge=0.0, description="Minimum altitude (m) for DVL lock")
    tilt_max_deg: float = Field(25.0, ge=0.0, le=90.0)
    forced_windows: list[tuple[float, float]] = Field(
        default_factory=list,
        description="(start_t, end_t) seconds where DVL is forced invalid",
    )


class TargetMimicConfig(BaseModel):
    trajectory: Literal["spline", "circle"] = "circle"
    radius: float = 5.0
    speed: float = 0.5
    depth: float = -10.0


class ScenarioConfig(BaseModel):
    name: str = "default"
    holoocean_scenario: str = "PierHarbor-HoveringCamera"
    max_steps: int = Field(1000, ge=1)
    camera_width: int = Field(640, ge=32)
    camera_height: int = Field(480, ge=32)
    dropout: DropoutConfig = Field(default_factory=DropoutConfig)
    target: TargetMimicConfig = Field(default_factory=TargetMimicConfig)
    target_class_id: int = Field(
        0,
        ge=0,
        description="YOLO class index for target taxon (Bathochordaeus=0 in single-class v1)",
    )
    seed: int = 42


class DetectorTrainingConfig(BaseModel):
    data_yaml: Path
    epochs: int = Field(10, ge=1)
    model: str = "yolo11n.pt"
    imgsz: int = Field(640, ge=32)
    batch: int = Field(8, ge=1)
    seed: int = 42


class NavTrainingConfig(BaseModel):
    trajectories_dir: Path
    epochs: int = Field(50, ge=1)
    arch: Literal["gru", "tcn"] = "gru"
    window_size: int = Field(20, ge=2)
    hidden_size: int = Field(64, ge=8)
    lr: float = Field(1e-3, gt=0)
    seed: int = 42


class PoseEntry(BaseModel):
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = Field(
        description="Quaternion x, y, z, w"
    )


class CameraPathConfig(BaseModel):
    path_id: str = "default"
    poses: list[PoseEntry] = Field(min_length=1)

    @field_validator("poses")
    @classmethod
    def at_least_one_pose(cls, v: list[PoseEntry]) -> list[PoseEntry]:
        if not v:
            raise ValueError("camera path requires at least one pose")
        return v


class LabelStrategy(str, Enum):
    COMPOSITED_TARGET = "composited-target"
    ANNOTATED_REGION = "annotated-region"


class RenderConfig(BaseModel):
    scene_checkpoint: Path
    camera_path: Path
    turbidity_values: list[float] = Field(default=[0.0, 0.3, 0.6])
    label_strategy: LabelStrategy = LabelStrategy.COMPOSITED_TARGET
    out_dir: Path
    seed: int = 42

    @field_validator("turbidity_values")
    @classmethod
    def turbidity_in_range(cls, v: list[float]) -> list[float]:
        for t in v:
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"turbidity {t} must be in [0, 1]")
        return v


def load_yaml_model(path: Path, model: type[BaseModel]) -> BaseModel:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"empty YAML: {path}")
    return model.model_validate(data)


def dump_yaml_model(path: Path, obj: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj.model_dump(mode="json"), f, sort_keys=False)
