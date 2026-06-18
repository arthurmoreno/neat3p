---
name: writing-specs
description: >
  Use when creating, updating, or organizing a design/research/implementation
  document ("spec") in the neat3p repo — e.g. the user says "write a spec",
  "add a research note", "plan this refactor", or references specs/. Defines the
  specs/ folder layout, file naming, header block, and the section structure of
  plan.md vs research.md as practiced in this repo.
---

# Writing specs in neat3p

This repo keeps design intent, investigations, and implementation plans as
version-controlled markdown under `specs/`. They are **committed** (unlike the
lifesim sibling repo, where `specs/` is gitignored) — so they are reviewable
history, not scratch. Match the existing pattern exactly.

## Folder layout

One folder per epic/ticket/topic, **date-prefixed**:

```
specs/<YYYYMMDD>-<kebab-topic>/
```

Examples already in the repo:
- `specs/20260613-module-reorganization/`
- `specs/20260615-benchmark-module-refactor/`
- `specs/20260616-3d-spatial-input-compression/`

The date is the **creation** date (when the folder was started), not the last
edit. Topic is short kebab-case describing the work, not the issue number.

## Files inside a folder

| File | Purpose |
|------|---------|
| `research.md` | Investigation, root-cause analysis, options compared, recommendation, how to validate. The *why* and the *what-are-the-choices*. |
| `plan.md` | The chosen approach: problem, proposed architecture, target layout, incremental migration steps, risks, open decisions. The *how*. |
| `research-<topic>.md` / `plan-<topic>.md` | Only when one folder genuinely needs multiple docs of the same type. |

Default to `research.md` and/or `plan.md`. A topic often has both: research
explores, plan commits.

## Required header block

Every doc opens with a one-line title (`# ...`) then a bullet header:

```markdown
# <Title — what this doc decides or investigates>

- **Repo:** `arthurmoreno/neat3p`
- **Tracking:** life-simulation #<issue> (under epic #45). <one line on the branch/checkpoint if relevant>
- **Status:** planned (not started) | in progress | done
- **Created:** <YYYY-MM-DD> · **Updated:** <YYYY-MM-DD>
```

When you revise a doc later, **bump `Updated:`** and, for a material change, add
a dated `> **Update (YYYY-MM-DD):** ...` callout near the top summarising what
shifted (see `20260615-benchmark-module-refactor/plan.md` for the pattern) —
don't silently rewrite history.

## Section structure

**research.md** typically flows:
1. The problem, in *this* codebase (anchor to real files/lines).
2. The honest framing (what's actually possible vs. wished-for).
3. Options, worst→best, each with its trade-off.
4. Recommendation (the synthesis).
5. How to validate — the experiment/metric that decides it.
6. Caveats & open questions. References/pointers.

**plan.md** typically flows:
1. Problem (and what is shared vs. genuinely specific — a table works well).
2. Proposed architecture (code blocks for the key APIs/dataclasses).
3. Target layout (a file tree of the end state).
4. Migration steps — **incremental and behaviour-preserving**; land new code
   side-by-side, parity-check, then delete the old.
5. Out of scope / already handled.
6. Risks & mitigations (table). 7. Open decisions (numbered, each with a "Lean: ..").

## House style

- **Anchor every claim to real code** — `file.py:line`, function names, actual
  numbers — not vibes. If you cite a line, verify it still exists.
- **Tables for trade-offs**, code blocks for proposed APIs, file trees for layout.
- **Be explicit about what's out of scope** and what's already done — a reader
  should never wonder whether a concern was missed or deliberately deferred.
- **Open decisions get a recommendation** ("Lean: X"), not just a question.
- Cross-reference sibling specs by path when a new doc revises or extends one.

## What does NOT go here

- Ephemeral task checklists, session notes, in-progress TODOs → conversation/tracker.
- The *why* behind a single landed change → git commit message / PR body.
- Non-obvious code invariants → a code comment.
