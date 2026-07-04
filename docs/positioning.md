# FathomFollow positioning (v1)

**Audience:** researchers, collaborators, and anyone evaluating a demo or fork.

## What this is

FathomFollow is a **sim-first research platform** that integrates:

1. **Perception** — YOLO fine-tuned on FathomNet (Bathochordaeus v1) with optional offline GS-rendered augmentation.
2. **Control** — classical visual servoing on a tracked detection (SimpleTracker v1).
3. **Navigation** — DriftGuard velocity estimator vs dead-reckoning baseline under scripted DVL dropout.

HoloOcean provides physics and sensors. Gaussian splatting (WaterSplatting) provides **offline** appearance augmentation only.

## What this is not

- An off-the-shelf **AUV visual-follow product** ready for paid field deployment.
- A claim that GS **closes the sim-to-real gap** at current scale (see below).
- A fully coupled nav→control stack (v1 uses **parallel-eval**: nav logs drift alongside control; nav does not steer the controller).

Customer adversarial review (2026-07-04): **Conditional Pass** for R&D collaboration; **No-Go** for paid field deployment until P0 backlog items in `implementation-spec.md` → Builder Backlog are closed.

## Messaging (stop / start)

| Stop saying | Start saying |
|-------------|--------------|
| "AUV visual follow product" | "Sim-first research platform for perception + DVL-dropout nav" |
| "GS closes sim-to-real gap" | "GS ablation measured the gap; 9 frames insufficient — scaling is stretch" |
| "79% tracking retention" | "79% active-track coverage, ~10 s harbor demo; pre-FB-002 had no spawned target mimic" |
| "End-to-end closed loop" | "Perception closed loop + parallel nav eval (nav does not steer controller in v1)" |
| "Follow-induced DVL dropout" | "Scripted + altitude-gated dropout (`tilt_max_deg` pending FB-008)" |

## v1 success criterion (met)

From `implementation-plan.md` rev. 2: closed-loop run where the AUV keeps the target actively tracked while learned drift-within-dropout beats dead reckoning.

**Authoritative live gate** (post attitude fix): 79% active-track coverage, 1.27 m dropout drift margin — see `docs/baselines.json` → `phase_4_live_hero_run` and Final Spec in `implementation-spec.md`.

## GS augmentation — honest null result

Real WaterSplatting (IUI3-RedSea, 15k iter) merged **9 composited frames** with FathomNet Bathochordaeus:

| Fixture | Pre-GS firing rate | Post-GS (train-4) | Verdict |
|---------|-------------------|-------------------|---------|
| fathomnet_proxy | 2.12 | 0.46 | regression |
| holoocean_smoke | 0.58 | 0.00 | regression |

Validation mAP50 ticked up (+0.013) but **sim-domain firing rates fell**. Do not sell GS sim-transfer at this batch size; scaling (≥200 frames, domain-matched scenes) is stretch work (FB-018).

## Demo tiers

| Tier | Requires | Proves |
|------|----------|--------|
| pytest | venv only | Contracts, recorded fixtures, no Unreal |
| fixture hero | train-2 + nav checkpoint + `holoocean_smoke.npz` | Replay integration on recorded RGB |
| live hero | HoloOcean 2.3 + Epic worlds | Fresh Unreal RGB + sensors |
| GS train/render | `water_splatting` conda + SeaThru-NeRF | Offline augmentation pipeline |

## Related docs

- [`metrics-glossary.md`](metrics-glossary.md) — metric names and caveats
- [`baselines.json`](baselines.json) — committed numbers
- [`workflow.md`](workflow.md) — multi-machine roles
- [`holoocean_install.md`](holoocean_install.md) / [`gs_setup.md`](gs_setup.md)
