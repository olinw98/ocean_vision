from fathomfollow.project_status import (
    _find_venv_pytest,
    _parse_current_state,
    _parse_diary_entries,
    _diary_warnings,
)


SAMPLE_SPEC = """
## Current State

**Last updated:** 2026-06-09T03:45:00Z
**Next action:** Phase 1.5 GS work
**Active blockers:** None

<!-- DIARY_ENTRY -->
### [2026-06-09T04:00:00Z] Step 1.5.2 — GS reconstruction

**status:** Partial
**blockers:** [BLOCKED] water_splatting conda env
**next:** finish GS train
<!-- /DIARY_ENTRY -->
"""


def test_parse_diary_entries():
    entries = _parse_diary_entries(SAMPLE_SPEC)
    assert len(entries) == 1
    assert entries[0].step == "1.5.2"
    assert entries[0].status == "Partial"
    assert entries[0].next_action == "finish GS train"


def test_parse_current_state():
    state = _parse_current_state(SAMPLE_SPEC)
    assert state["present"] is True
    assert state["last_updated"] == "2026-06-09T03:45:00Z"
    assert state["next_action"] == "Phase 1.5 GS work"


def test_find_venv_pytest_prefers_existing_path(monkeypatch, tmp_path):
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    pytest_bin = bin_dir / "pytest"
    pytest_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("fathomfollow.project_status.ROOT", tmp_path)
    assert _find_venv_pytest() == pytest_bin


def test_diary_warnings_stale_current_state():
    current = _parse_current_state(SAMPLE_SPEC)
    latest = _parse_diary_entries(SAMPLE_SPEC)[0]
    warnings = _diary_warnings(current, latest, check_diary=True)
    assert any("older than the latest diary" in w for w in warnings)
