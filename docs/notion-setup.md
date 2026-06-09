# Using Notion with FathomFollow

Notion is your **human dashboard**. Git stays the **agent source of truth**. Do not duplicate the full diary in Notion.

## Recommended Notion structure

Create a top-level page: **FathomFollow**. Under it, four child pages:

```
FathomFollow/
├── Overview          ← pitch, phase status, links
├── Decisions         ← judgment calls + blockers
├── My actions        ← human-only tasks
└── Agentic workflow  ← what you're learning about Cursor/agents
```

Optional fifth page: **Weekly sync log** (5-minute notes after each build session).

## What goes where

| Content | Git (canonical) | Notion (view) |
|---------|-----------------|---------------|
| Diary entries | `implementation-spec.md` | Summary bullets only |
| Metrics | `docs/baselines.json` | Dashboard numbers |
| Open decisions | Judgment Calls table | Table with priority / due |
| Your todos | — | Task list |
| Agent workflow learnings | `docs/workflow.md` | Experiment notes |

**Sync rule:** After a build session on Windows, spend ~5 minutes updating Notion from git — not the other way around.

## Page templates (copy into Notion)

### Overview

**Elevator pitch:** Simulated underwater robot follows a sea creature while staying oriented when its Doppler sensor cuts out; GS-rendered ocean imagery trains its eyes.

**Repo:** `github.com/olinw98/ocean_vision`

**Phase status**

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Scaffolding | Partial | HoloOcean pending |
| 1 Perception | Complete | Bathochordaeus, baseline 2.12 |
| 1.5 GS | Not started | Next on GPU |
| 2 Nav | Code done | Metrics pending |
| 3 Follow | Code done | Live sim pending |
| 4 Integration | Code done | Hero run pending |

**Success (v1):** Detector beats 2.12 after GS · DriftGuard beats dead reckoning in dropouts · Target stays in frame in sim · One honest report.

---

### Decisions

| Decision | Status | Owner | Notes |
|----------|--------|-------|-------|
| Proxy fixture until HoloOcean | Open | Agent | Interim baseline |
| GS library WaterSplatting vs SeaSplat | Open | You | After conda try |
| ByteTrack upgrade | Open | Later | When sim detections exist |
| YOLO not COCO for FathomNet | Accepted | — | COCO broken on some taxa |

Link: full table in repo `implementation-spec.md` → Open Judgment Calls.

---

### My actions

- [ ] Add `olin-wei-ai` as collaborator on GitHub repo
- [ ] Push `cursor/project-workflow-improvements`
- [ ] Set up `water_splatting` conda on Windows GPU
- [ ] HoloOcean Epic + Python 3.11 install
- [ ] Run Phase 1.5 ablation vs 2.12

---

### Agentic workflow

**Hypothesis:** Structured diary + skills + tests keep multi-session agent builds coherent.

**Working:** Resume skill, ff-status, TDD gate, honest Partial/Blocked.

**Friction:** Current State was missing early; push blocked across accounts; diary can lag code.

**Try next:** Session-end ritual on every GPU session; weekly Notion sync only.

---

## How to create pages in Notion

### Manual (simplest)

1. New page → paste a template section above.
2. Add a **linked database** for My actions (Status, Due, Phase).
3. Pin **Overview** to sidebar.

### With Cursor + Notion plugin

Ask Cursor: *"Create a Notion page under [parent] with the FathomFollow Overview template from docs/notion-setup.md"*

Available plugin skills: `create-page`, `create-task`, `knowledge-capture`, `search`.

Use **knowledge-capture** after a long chat to save a structured summary — point it at **Agentic workflow**, not the full diary.

### Weekly sync ritual (5 min)

1. On Windows after push: open `implementation-spec.md` → Current State.
2. Update Notion **Overview** phase row if changed.
3. Move **My actions** checkboxes.
4. Add one line to **Weekly sync log**: date, what shipped, next.

## What not to do

- Don't maintain two diaries — agents won't read Notion unless you use MCP every time.
- Don't mark Notion tasks "done" until git diary/baselines agree.
- Don't store secrets or `.env` in Notion.

## Cursor ↔ Notion workflow

| When | Tool |
|------|------|
| Start coding session | Git + resume skill (not Notion) |
| End build session | Session-end skill → git push → optional Notion sync |
| Explain project to someone | Notion Overview |
| Capture "what we learned about agents" | Notion Agentic workflow or knowledge-capture |
| Break spec into tasks | Notion task board (human priorities only) |

See also: `docs/pm-cheatsheet.md`, `docs/workflow.md`.
