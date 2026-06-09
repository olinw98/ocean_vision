"""Project snapshot for multi-machine workflow and agent resume."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "implementation-spec.md"
BASELINES_PATH = ROOT / "docs" / "baselines.json"

ARTIFACT_PATHS: dict[str, Path] = {
    "batho_raw_images": ROOT / "data" / "fathomnet_raw" / "Bathochordaeus" / "images",
    "batho_manifest": ROOT / "data" / "fathomnet_batho" / "manifest.json",
    "batho_metrics": ROOT / "data" / "fathomnet_batho" / "metrics.json",
    "detector_weights": ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt",
    "nav_checkpoint": ROOT / "data" / "nav_model" / "velocity_estimator.pt",
    "fathomnet_proxy_fixture": ROOT / "fixtures" / "sim" / "fathomnet_proxy.npz",
}


@dataclass
class DiaryEntry:
    timestamp: str
    step: str
    title: str
    status: str
    next_action: str
    blockers: str


@dataclass
class ProjectSnapshot:
    root: str
    generated_at: str
    git_branch: str | None = None
    git_clean: bool | None = None
    git_ahead: int | None = None
    git_behind: int | None = None
    pytest_passed: int | None = None
    pytest_failed: int | None = None
    pytest_skipped: bool = False
    current_state_present: bool = False
    current_state_last_updated: str | None = None
    current_state_next_action: str | None = None
    current_state_blockers: str | None = None
    latest_diary: DiaryEntry | None = None
    diary_entry_count: int = 0
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    baselines: dict[str, Any] | None = None
    diary_warnings: list[str] = field(default_factory=list)


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_sync() -> tuple[str | None, bool | None, int | None, int | None]:
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    clean = status == "" if status is not None else None
    upstream = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    ahead, behind = None, None
    if upstream:
        counts = _run_git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
        if counts:
            parts = counts.split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
    return branch, clean, ahead, behind


def _find_venv_pytest() -> Path | None:
    for rel in ((".venv", "bin", "pytest"), (".venv", "Scripts", "pytest.exe")):
        candidate = ROOT.joinpath(*rel)
        if candidate.exists():
            return candidate
    return None


def _run_pytest() -> tuple[int | None, int | None, bool]:
    venv_pytest = _find_venv_pytest()
    pytest_cmd = [str(venv_pytest)] if venv_pytest else [sys.executable, "-m", "pytest"]
    try:
        result = subprocess.run(
            [*pytest_cmd, "-q", "--tb=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None, True
    if result.returncode != 0 and "no tests ran" in (result.stdout + result.stderr).lower():
        return None, None, True
    match = re.search(r"(\d+) passed", result.stdout)
    failed_match = re.search(r"(\d+) failed", result.stdout)
    passed = int(match.group(1)) if match else (0 if result.returncode == 0 else None)
    failed = int(failed_match.group(1)) if failed_match else (0 if result.returncode == 0 else None)
    if result.returncode != 0 and passed is not None and failed is None:
        failed = 1
    return passed, failed, False


def _parse_field(block: str, name: str) -> str:
    match = re.search(rf"\*\*{re.escape(name)}:\*\*\s*(.+)", block)
    return match.group(1).strip() if match else ""


def _parse_diary_entries(text: str) -> list[DiaryEntry]:
    entries: list[DiaryEntry] = []
    for block in re.findall(r"<!-- DIARY_ENTRY -->(.*?)<!-- /DIARY_ENTRY -->", text, re.DOTALL):
        header = re.search(r"### \[([^\]]+)\] Step ([^\n]+)", block)
        if not header:
            continue
        entries.append(
            DiaryEntry(
                timestamp=header.group(1),
                step=header.group(2).split(" — ", 1)[0].strip(),
                title=header.group(2),
                status=_parse_field(block, "status"),
                next_action=_parse_field(block, "next"),
                blockers=_parse_field(block, "blockers"),
            )
        )
    return entries


def _parse_current_state(text: str) -> dict[str, str | None]:
    match = re.search(r"## Current State\s+(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if not match:
        return {"present": False}
    block = match.group(1)
    return {
        "present": True,
        "last_updated": _parse_field(block, "Last updated") or None,
        "next_action": _parse_field(block, "Next action") or None,
        "blockers": _parse_field(block, "Active blockers") or None,
    }


def _artifact_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path.relative_to(ROOT))}
    if path.is_dir():
        count = sum(1 for _ in path.rglob("*") if _.is_file())
        return {"exists": True, "path": str(path.relative_to(ROOT)), "file_count": count}
    return {"exists": True, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}


def _load_baselines() -> dict[str, Any] | None:
    if not BASELINES_PATH.exists():
        return None
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


def _diary_warnings(
    current: dict[str, Any],
    latest: DiaryEntry | None,
    check_diary: bool,
) -> list[str]:
    if not check_diary:
        return []
    warnings: list[str] = []
    if not current.get("present"):
        warnings.append("Missing ## Current State section in implementation-spec.md")
        return warnings
    last_updated = current.get("last_updated")
    if last_updated and latest:
        try:
            state_ts = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            diary_ts = datetime.fromisoformat(latest.timestamp.replace("Z", "+00:00"))
            if diary_ts > state_ts:
                warnings.append(
                    "Current State is older than the latest diary entry — update before ending session"
                )
        except ValueError:
            warnings.append("Could not parse Current State Last updated timestamp")
    if latest and current.get("next_action"):
        if latest.next_action and latest.next_action not in (current.get("next_action") or ""):
            if current.get("next_action") != latest.next_action:
                warnings.append(
                    "Current State Next action may diverge from latest diary **next:** field"
                )
    return warnings


def collect_snapshot(*, run_tests: bool = True, check_diary: bool = False) -> ProjectSnapshot:
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.exists() else ""
    entries = _parse_diary_entries(spec_text)
    latest = entries[-1] if entries else None
    current = _parse_current_state(spec_text)
    branch, clean, ahead, behind = _git_sync()

    passed, failed, skipped = (None, None, True)
    if run_tests:
        passed, failed, skipped = _run_pytest()

    warnings = _diary_warnings(current, latest, check_diary)

    return ProjectSnapshot(
        root=str(ROOT),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        git_branch=branch,
        git_clean=clean,
        git_ahead=ahead,
        git_behind=behind,
        pytest_passed=passed,
        pytest_failed=failed,
        pytest_skipped=skipped,
        current_state_present=bool(current.get("present")),
        current_state_last_updated=current.get("last_updated"),
        current_state_next_action=current.get("next_action"),
        current_state_blockers=current.get("blockers"),
        latest_diary=latest,
        diary_entry_count=len(entries),
        artifacts={name: _artifact_status(path) for name, path in ARTIFACT_PATHS.items()},
        baselines=_load_baselines(),
        diary_warnings=warnings,
    )


def format_text(snapshot: ProjectSnapshot) -> str:
    lines = [
        "FathomFollow project status",
        f"Generated: {snapshot.generated_at}",
        "",
        "## Git",
        f"Branch: {snapshot.git_branch or 'unknown'}",
        f"Clean: {snapshot.git_clean if snapshot.git_clean is not None else 'unknown'}",
    ]
    if snapshot.git_ahead is not None:
        lines.append(f"Ahead/behind upstream: {snapshot.git_ahead}/{snapshot.git_behind}")
    lines.extend(["", "## Tests"])
    if snapshot.pytest_skipped:
        lines.append("pytest: not run")
    elif snapshot.pytest_failed:
        lines.append(f"pytest: {snapshot.pytest_passed or 0} passed, {snapshot.pytest_failed} failed")
    else:
        lines.append(f"pytest: {snapshot.pytest_passed or 0} passed")

    lines.extend(["", "## Diary"])
    lines.append(f"Current State present: {snapshot.current_state_present}")
    if snapshot.current_state_last_updated:
        lines.append(f"Last updated: {snapshot.current_state_last_updated}")
    if snapshot.current_state_next_action:
        lines.append(f"Next action: {snapshot.current_state_next_action}")
    if snapshot.current_state_blockers:
        lines.append(f"Blockers: {snapshot.current_state_blockers}")
    if snapshot.latest_diary:
        d = snapshot.latest_diary
        lines.append(f"Latest diary: [{d.timestamp}] {d.title} ({d.status})")
        lines.append(f"Diary next: {d.next_action}")

    lines.extend(["", "## Local artifacts"])
    for name, info in snapshot.artifacts.items():
        if info["exists"]:
            extra = info.get("file_count") or info.get("bytes")
            suffix = f" ({extra} files)" if "file_count" in info else f" ({extra} bytes)" if extra else ""
            lines.append(f"  [x] {name}: {info['path']}{suffix}")
        else:
            lines.append(f"  [ ] {name}: MISSING")

    if snapshot.baselines:
        pre = snapshot.baselines.get("pre_gs_baseline", {})
        lines.extend(
            [
                "",
                "## Baselines (committed)",
                f"Taxon: {pre.get('selected_taxon', 'n/a')}",
                f"Ablation target firing_rate: {pre.get('ablation_target_firing_rate', 'n/a')}",
            ]
        )
        train = pre.get("train", {})
        if train:
            lines.append(f"mAP50: {train.get('mAP50', 'n/a')} | mAP50-95: {train.get('mAP50-95', 'n/a')}")

    if snapshot.diary_warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"  - {w}" for w in snapshot.diary_warnings)

    return "\n".join(lines)


def snapshot_to_json(snapshot: ProjectSnapshot) -> str:
    payload = asdict(snapshot)
    if snapshot.latest_diary:
        payload["latest_diary"] = asdict(snapshot.latest_diary)
    return json.dumps(payload, indent=2)


def main_status(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ff-status", description="FathomFollow project snapshot")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-pytest", action="store_true", help="Skip running pytest")
    parser.add_argument(
        "--check-diary",
        action="store_true",
        help="Warn if Current State is missing or stale vs latest diary",
    )
    args = parser.parse_args(argv)
    snapshot = collect_snapshot(run_tests=not args.no_pytest, check_diary=args.check_diary)
    if args.json:
        print(snapshot_to_json(snapshot))
    else:
        print(format_text(snapshot))
    if snapshot.diary_warnings:
        sys.exit(2)
