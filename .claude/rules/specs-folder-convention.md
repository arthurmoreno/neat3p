# Specs Folder Convention — neat3p

Adapted from the lifesim sibling repo's rule of the same name. For the detailed
authoring guide (header block, section structure, house style) see the
`writing-specs` skill in this repo (`.claude/skills/writing-specs/SKILL.md`) —
this rule is the short, always-on summary.

All design documents, research notes, and implementation plans for neat3p live
under `specs/`.

> **Key difference from the lifesim sibling repo: neat3p `specs/` is COMMITTED**
> — version-controlled, reviewable history, not scratch. (In lifesim, `specs/`
> is gitignored/local-only. Don't carry that habit across — here you *do* commit
> specs.)

---

## Folder naming

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

---

## File naming inside a folder

| File | Purpose |
|------|---------|
| `research.md` | Investigation findings, root-cause analysis, options compared, recommendation |
| `plan.md` | The chosen approach: problem, architecture, target layout, incremental steps, risks |
| `research-<topic>.md` / `plan-<topic>.md` | Only when a folder genuinely needs multiple docs of the same type |

Use `research.md` and `plan.md` as defaults. A topic often has both: research
explores, plan commits.

---

## What goes here vs elsewhere

- **`specs/`** — design intent, investigation results, architectural decisions (committed).
- **`.claude/rules/`** — persistent always-on workflow rules for Claude (like this file).
- **`.claude/skills/`** — invokable how-to guides (e.g. `writing-specs`).
- **Git commit messages / PR bodies** — the *why* behind a specific landed change.
- **Code comments** — only when a non-obvious invariant or workaround needs explanation.

Do **not** put ephemeral task notes, in-progress checklists, or session summaries
in `specs/`. Those belong in the conversation or a task tracker.
