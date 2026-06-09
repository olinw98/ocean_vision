# ocean_vision (FathomFollow)

Simulation-based AUV visual follow with DVL-dropout navigation and offline Gaussian-splatting augmentation.

See `implementation-plan.md` and `agent-prompt.md` for architecture and build contract.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

HoloOcean requires a separate install from GitHub after accepting the Unreal EULA. See `docs/holoocean_install.md`. GS training uses a separate conda env (see `docs/gs_setup.md`).

Recorded baselines: `docs/baselines.json`. Multi-machine workflow: `docs/workflow.md`. PM cheat sheet: `docs/pm-cheatsheet.md`.
