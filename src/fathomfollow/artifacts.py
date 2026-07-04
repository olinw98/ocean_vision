"""Artifact registry, fetch, and preflight for reproducible hero runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "docs" / "artifacts.json"


def canonical_artifact_paths(root: Path | None = None) -> dict[str, Path]:
    """Resolve hero bundle dest paths from docs/artifacts.json."""
    base = root or ROOT
    registry = load_registry()
    return {
        entry["id"]: base / entry["dest"]
        for entry in registry["bundles"]["hero"]["artifacts"]
    }


def load_registry(registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path or REGISTRY_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hint(artifact_id: str | None = None) -> str:
    if artifact_id:
        return f"Run: ff-fetch hero  (provides {artifact_id})"
    return "Run: ff-fetch hero"


def verify_artifact(path: Path, entry: dict[str, Any]) -> bool:
    expected = entry.get("sha256")
    if not expected or not path.is_file():
        return False
    return sha256_file(path) == expected


def preflight_artifact(
    path: Path,
    root: Path | None = None,
    artifact_id: str | None = None,
) -> None:
    """Raise FileNotFoundError with ff-fetch hint when a required artifact is missing."""
    root = root or ROOT
    if path.is_file():
        return
    try:
        rel = path.relative_to(root)
        target = str(rel)
    except ValueError:
        target = str(path)
    raise FileNotFoundError(f"Missing artifact: {target}\n{artifact_hint(artifact_id)}")


def preflight_run_artifacts(
    detector_weights: Path | None = None,
    nav_checkpoint: Path | None = None,
    root: Path | None = None,
) -> None:
    if detector_weights is not None:
        preflight_artifact(detector_weights, root=root, artifact_id="detector_weights")
    if nav_checkpoint is not None:
        preflight_artifact(nav_checkpoint, root=root, artifact_id="nav_checkpoint")


def _bundle_entries(registry: dict[str, Any], bundle_name: str) -> list[dict[str, Any]]:
    bundles = registry.get("bundles", {})
    if bundle_name not in bundles:
        raise ValueError(f"unknown artifact bundle: {bundle_name}")
    artifacts = bundles[bundle_name].get("artifacts", [])
    if not artifacts:
        raise ValueError(f"artifact bundle {bundle_name!r} is empty")
    return artifacts


def fetch_bundle(
    bundle_name: str,
    root: Path | None = None,
    registry_path: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Copy bundled artifacts to canonical dest paths and verify SHA-256."""
    root = root or ROOT
    registry = load_registry(registry_path)
    fetched: dict[str, str] = {}
    for entry in _bundle_entries(registry, bundle_name):
        source = root / entry["source"]
        dest = root / entry["dest"]
        artifact_id = entry["id"]
        expected = entry["sha256"]

        if not source.is_file():
            raise FileNotFoundError(
                f"Bundled source missing: {source.relative_to(root)}\n"
                "Re-clone the repository or restore fixtures/artifacts/."
            )

        if dest.is_file() and not force:
            if sha256_file(dest) == expected:
                fetched[artifact_id] = str(dest.relative_to(root))
                continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        actual = sha256_file(dest)
        if actual != expected:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {artifact_id}: expected {expected}, got {actual}"
            )
        fetched[artifact_id] = str(dest.relative_to(root))
    return fetched


def registry_artifact_status(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Summarize hero registry artifacts for ff-status."""
    root = root or ROOT
    if not REGISTRY_PATH.is_file():
        return {}
    registry = load_registry()
    status: dict[str, dict[str, Any]] = {}
    for entry in _bundle_entries(registry, "hero"):
        dest = root / entry["dest"]
        info: dict[str, Any] = {
            "dest": entry["dest"],
            "sha256": entry["sha256"],
            "exists": dest.is_file(),
        }
        if dest.is_file():
            info["bytes"] = dest.stat().st_size
            info["sha256_ok"] = sha256_file(dest) == entry["sha256"]
        else:
            info["sha256_ok"] = False
        status[entry["id"]] = info
    return status
