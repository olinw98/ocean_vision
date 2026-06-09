"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest

from fathomfollow.sim.recorded import write_sim_fixture
from fathomfollow.gs.recorded import write_gs_fixture


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sim_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "sim.npz"
    write_sim_fixture(path)
    return path


@pytest.fixture
def gs_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "gs"
    write_gs_fixture(path)
    return path
