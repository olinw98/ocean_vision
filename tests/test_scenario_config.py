from pathlib import Path

import numpy as np

from fathomfollow.config.models import ScenarioConfig, load_yaml_model
from fathomfollow.sim.target import target_position_at


def test_scenario_config_from_repo() -> None:
    cfg = load_yaml_model(Path("config/scenario.yaml"), ScenarioConfig)
    assert cfg.holoocean_scenario == "PierHarbor-HoveringCamera"


def test_scenario_holoocean_has_forced_dropout() -> None:
    cfg = load_yaml_model(Path("config/scenario_holoocean.yaml"), ScenarioConfig)
    assert cfg.name == "holoocean_smoke"
    assert len(cfg.dropout.forced_windows) >= 1


def test_target_trajectory_deterministic() -> None:
    from fathomfollow.config.models import TargetMimicConfig

    cfg = TargetMimicConfig(trajectory="circle", radius=5.0, speed=0.5)
    p0 = target_position_at(0.0, cfg)
    p1 = target_position_at(1.0, cfg)
    p0b = target_position_at(0.0, cfg)
    np.testing.assert_allclose(p0, p0b)
    assert not np.allclose(p0, p1)
