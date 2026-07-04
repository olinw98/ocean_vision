from pathlib import Path

import numpy as np

from fathomfollow.config.models import ScenarioConfig, load_yaml_model


def test_scenario_config_from_repo() -> None:
    cfg = load_yaml_model(Path("config/scenario.yaml"), ScenarioConfig)
    assert cfg.holoocean_scenario == "PierHarbor-HoveringCamera"


def test_scenario_holoocean_has_forced_dropout() -> None:
    cfg = load_yaml_model(Path("config/scenario_holoocean.yaml"), ScenarioConfig)
    assert cfg.name == "holoocean_smoke"
    assert len(cfg.dropout.forced_windows) >= 1
