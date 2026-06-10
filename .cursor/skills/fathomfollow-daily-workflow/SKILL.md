---
name: fathomfollow-daily-workflow
description: >-
  Daily start/end ritual for FathomFollow on ocean_vision: git pull, venv,
  ff-status, which Cursor chat to use (PM vs builder), and session wrap-up.
  Use when the user asks how to work on the project, daily routine, starting
  or ending a session, what to run first, or beginner workflow steps.
---

# FathomFollow Daily Workflow

Playbook for **every session**. Pair with:
- **Start / catch-up:** `fathomfollow-resume` skill (deep status recap)
- **End / sign-off:** `fathomfollow-session-end` skill (diary + commit prep)
- **Human one-pager:** `docs/pm-cheatsheet.md`
- **Multi-machine:** `docs/workflow.md`

Project root: `C:\Users\olinw\Desktop\Projects\ocean_vision` (Windows GPU build machine).

## Which Cursor chat?

| You want… | Use this chat | Say… |
|-----------|---------------|------|
| Status, explain, review builder, "what's next" | **PM assistant** | "I'm back" / "Review the builder" |
| Implement, train, fix errors, run commands | **New Agent chat** | "Resume FathomFollow — [one task]" |
| End of day wrap-up | **Builder or PM** | "End session" (triggers session-end skill) |

**Rule:** PM decides → Builder executes → PM reviews. Avoid two builders editing the same files.

---

## Session start (~5 min)

Run in terminal (Windows):

```powershell
cd C:\Users\olinw\Desktop\Projects\ocean_vision
git pull
.venv\Scripts\activate
ff-status
```

Then:
1. Open `implementation-spec.md` → read **Current State** (not the whole diary)
2. Note **Next action** and **Active blockers**

**Passing signs:**
- `ff-status` shows pytest passed
- Git clean or you know what's uncommitted
- You know the one task for today

For a full recap after a break, say **"Resume FathomFollow"** (`fathomfollow-resume` skill).

**Builder kickoff** (paste into Agent chat):

```text
Resume FathomFollow. Read agent-prompt.md and implementation-spec.md Current State.
Run git pull, ff-status, pytest -q. Continue from Next action only.
Do not edit implementation-plan.md. Teach me briefly as you go.
```

---

## During the session

- One bounded task per session (e.g. "Phase 1.5 render batch only")
- Agent runs commands; you skim diffs in Source Control
- Require `pytest -q` before "done"
- Numbers that matter → `docs/baselines.json` + diary (not only `runs/`)

**You intervene when:**
- `[JUDGMENT CALL]` in diary needs your decision
- `[BLOCKED]` needs install (HoloOcean, GS conda, MSVC Build Tools)
- Agent tries to edit `implementation-plan.md` (never allowed)

---

## Session end (~5–10 min)

```powershell
pytest -q
ff-status --check-diary
```

Agent should (`fathomfollow-session-end` skill):
1. Update **Current State** + **Open Judgment Calls** in `implementation-spec.md`
2. Append diary entry if implementation changed
3. Update `docs/baselines.json` if metrics changed
4. Suggest commit message — **commit/push only if you ask**

**Commit includes:** `src/`, `tests/`, spec, `docs/baselines.json`, config, fixtures  
**Never commit:** `data/`, `runs/`, `.venv/`, `yolo11n.pt`

```powershell
git status
git add <files agent lists>
git commit -m "why-focused message"
git push
```

**Passing signs:**
- Tests green
- Current State reflects today's work and tomorrow's next action
- Pushed if you worked on the build machine

---

## Quick commands reference

| Command | When |
|---------|------|
| `ff-status` | Start of session |
| `ff-status --check-diary` | End of session |
| `pytest -q` | After changes / before leaving |
| `ff-train-detector --config config/detector_train.yaml` | Retrain (uses GPU if CUDA torch installed) |
| `conda activate water_splatting` | Phase 1.5 GS work only |

---

## Beginner troubleshooting

| Problem | Fix |
|---------|-----|
| `conda` not found | Add Anaconda to PATH or use `%USERPROFILE%\anaconda3\Scripts\conda.exe` — see `docs/gs_setup.md` |
| Training slow | Check `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` → should be `True` |
| Missing weights after clone | Normal — `runs/` is local; re-train or copy from backup |
| Confused after agent work | PM chat: "Review what the builder did" |

---

## Trigger phrases

"daily workflow", "how do I work on this project", "start session", "end session",
"what do I run first", "beginner steps", "session ritual", "signing off".
