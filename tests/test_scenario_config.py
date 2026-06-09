from pathlib import Path

import numpy as np

from fathomfollow.config.models import ScenarioConfig, load_yaml_model
from fathomfollow.sim.target import target_position_at


def test_scenario_config_from_repo() -> None:
    cfg = load_yaml_model(Path("config/scenario.yaml"), ScenarioConfig)
    assert cfg.holoocean_scenario == "PierHarbor-Hovering"


def test_target_trajectory_deterministic() -> None:
    from fathomfollow.config.models import TargetMimicConfig

    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5)
    p0 = target_position_at(0.0, cfg)
    p1 = target_position_at(1.0, cfg)
    p0b = target_position_at(0.0, cfg)
    np.testing.assert_allclose(p0, p0b)
    assert not np.allclose(p0, p1)
