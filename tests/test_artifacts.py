import hashlib
import json
from pathlib import Path

import pytest

from fathomfollow.artifacts import (
    artifact_hint,
    fetch_bundle,
    load_registry,
    preflight_artifact,
    sha256_file,
    verify_artifact,
)


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_sha256_file(tmp_path: Path) -> None:
    payload = b"hero-artifact"
    path = tmp_path / "a.bin"
    digest = _write(path, payload)
    assert sha256_file(path) == digest


def test_fetch_hero_copies_and_verifies(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    src_dir = root / "fixtures" / "artifacts" / "hero"
    det_bytes = b"detector-weights"
    nav_bytes = b"nav-checkpoint"
    det_hash = _write(src_dir / "best.pt", det_bytes)
    nav_hash = _write(src_dir / "velocity_estimator.pt", nav_bytes)
    registry = {
        "bundles": {
            "hero": {
                "artifacts": [
                    {
                        "id": "detector_weights",
                        "dest": "runs/detect/train-2/weights/best.pt",
                        "source": "fixtures/artifacts/hero/best.pt",
                        "sha256": det_hash,
                    },
                    {
                        "id": "nav_checkpoint",
                        "dest": "data/nav_model/velocity_estimator.pt",
                        "source": "fixtures/artifacts/hero/velocity_estimator.pt",
                        "sha256": nav_hash,
                    },
                ]
            }
        }
    }
    registry_path = root / "docs" / "artifacts.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = fetch_bundle("hero", root=root, registry_path=registry_path)

    det_dest = root / "runs/detect/train-2/weights/best.pt"
    nav_dest = root / "data/nav_model/velocity_estimator.pt"
    assert det_dest.read_bytes() == det_bytes
    assert nav_dest.read_bytes() == nav_bytes
    assert result["detector_weights"] == str(det_dest.relative_to(root))
    assert result["nav_checkpoint"] == str(nav_dest.relative_to(root))


def test_fetch_fails_on_sha256_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    src = root / "fixtures" / "artifacts" / "hero" / "best.pt"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"wrong-bytes")
    registry_path = root / "docs/artifacts.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "bundles": {
                    "hero": {
                        "artifacts": [
                            {
                                "id": "detector_weights",
                                "dest": "runs/detect/train-2/weights/best.pt",
                                "source": "fixtures/artifacts/hero/best.pt",
                                "sha256": "0" * 64,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_bundle("hero", root=root, registry_path=registry_path)


def test_preflight_artifact_missing_exits_with_hint(tmp_path: Path) -> None:
    missing = tmp_path / "runs/detect/train-2/weights/best.pt"
    with pytest.raises(FileNotFoundError, match="ff-fetch hero"):
        preflight_artifact(missing, root=tmp_path, artifact_id="detector_weights")


def test_preflight_artifact_passes_when_present(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    path.write_bytes(b"ok")
    preflight_artifact(path, root=tmp_path)


def test_load_registry_from_repo() -> None:
    registry = load_registry()
    hero = registry["bundles"]["hero"]
    assert len(hero["artifacts"]) >= 2
    ids = {item["id"] for item in hero["artifacts"]}
    assert "detector_weights" in ids
    assert "nav_checkpoint" in ids


def test_verify_artifact_matches_registry(tmp_path: Path) -> None:
    content = b"canonical"
    path = tmp_path / "best.pt"
    digest = _write(path, content)
    entry = {"id": "detector_weights", "dest": "best.pt", "sha256": digest}
    assert verify_artifact(path, entry) is True


def test_artifact_hint_includes_fetch_command() -> None:
    assert "ff-fetch hero" in artifact_hint("detector_weights")
