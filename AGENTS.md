# AGENTS.md — Navigational Map for AI Agents

> This file serves as the **entry point** for any agent operating within this repository.
> Read only what you require, precisely when you require it (progressive disclosure).

---

## 1. Before Starting (Mandatory)

1. Execute `./init.sh` and verify that it terminates without errors. If it fails, **stop** and resolve the environment before altering any code.
2. Run `kin prime` (or inspect active Kindex context via MCP) to load active architectural constraints, memory boundaries, and directives from `.kin/`.
3. Read `progress/current.md` to comprehend the state left by the preceding session.
4. Read `feature_list.json`. Any new feature (`"sdd": true`) must strictly undergo **Spec Driven Development** — refer to `docs/specs.md` and §4 of this document.
5. Read `docs/specs.md` prior to interacting with any spec or feature marked `sdd: true`.

## 2. Repository Map

| File / Directory             | Contents                                                                    | When to read it |
|------------------------------|-----------------------------------------------------------------------------|---------------|
| `.kin/`                      | Kindex local knowledge graph (`index.json`, config, and memory nodes)       | Always, via `kin prime` / MCP |
| `feature_list.json`          | Task ledger detailing state (`pending` / `spec_ready` / `in_progress` / `done` / `blocked`) | Always, at inception |
| `progress/current.md`        | State of the active session                                                 | Always, at inception |
| `progress/history.md`        | Append-only chronological log of prior sessions                             | When historical context is requisite |
| `specs/<feature>/`           | `requirements.md` + `design.md` + `tasks.md` (SDD Spec Files)               | Prior to implementing any feature with `"sdd": true` |
| `docs/architecture.md`       | Defines what constitutes "quality work" within this project                 | Prior to implementation |
| `docs/conventions.md`        | Rules governing style, nomenclature, and coding conventions             | Prior to authoring code |
| `docs/specs.md`              | SDD Protocol: EARS-Agent notation, Kindex contract, human approval gate      | Prior to drafting or interpreting a spec |
| `docs/verification.md`       | Methodology to prove functional correctness (and mock LLM/BigQuery clients) | Prior to declaring a task as `done` |
| `CHECKPOINTS.md`             | Objective criteria denoting a "correct final state"                         | For rigorous self-evaluation |
| `.agents/`                   | Subagent prompt definitions (`leader`, `spec_author`, `implementer`, `reviewer`) | When orchestrating workflows |
| `src/`                       | Application source code (FastAPI, WebSockets, CLIAgentManager)              | For implementation |
| `tests/`                     | Automated unit test suites (completely mocked)                              | For verification |
| `tests_e2e/`                 | End-to-end integration tests (Playwright + Docker Compose)                  | For complete validation |

## 3. Hard Rules (Non-Negotiable)

- **Strictly one feature at a time.** Never conflate modifications from multiple tasks within the same session.
- **Never declare a task `done` without green tests.** Execute `./init.sh` and ensure the test suite passes at 100%.
- **Do not bypass the spec phase.** Every feature with `"sdd": true` must be processed by `spec_author` and secure human approval prior to touching code.
- **Do not bypass the human approval gate.** The leader halts the workflow at `spec_ready` and awaits explicit authorization.
- **Capture durable decisions & constraints in Kindex.** Never rely solely on transient context; record architectural choices with `kin add --type decision` and critical constraints with `kin add --type constraint`.
- **Preserve the `.kin/` project contract.** Stage `.kin/index.json` alongside code changes in Git; never delete or treat `.kin/` as disposable cache.
- **Document your actions** in `progress/current.md` synchronously as you work, not asynchronously at the end.
- **Leave the repository pristine** prior to session closure (refer to §5).
- **If you lack knowledge, consult `docs/` and Kindex** rather than fabricating a solution.

## 4. Workflow (SDD + Kindex)

```
pending → [spec_author] → spec_ready → ⏸ HUMAN → in_progress → [implementer (kin prime/add) → reviewer] → done (kin dream)
```

1. The leader detects the first `pending` feature flagged with `"sdd": true`.
2. The leader dispatches `spec_author`, which authors `specs/<name>/{requirements,design,tasks}.md` and advances the status to `spec_ready`.
3. **Pause.** The human reviews the spec in `specs/<name>/` and approves (or mandates revisions).
4. Once approved, the leader transitions the status to `in_progress` and dispatches `implementer`.
5. The implementer executes `kin prime` to align with active constraints, executes `tasks.md` sequentially marking them `[x]`, and registers any new technical choices via `kin add --type decision` or `kin add --type directive`.
6. The reviewer validates traceability `A<n>` ↔ test, confirms task completion, and verifies Kindex synchronization; it subsequently approves or rejects the implementation.
7. Upon approval, the implementer marks the feature `done`, executes `kin dream` and `kin index`, and appends the summary to `progress/history.md`.

## 5. Session Closure (Lifecycle)

Prior to terminating:

1. Execute `./init.sh` — verify a completely green state.
2. Consolidate Kindex knowledge graph: execute `kin dream` and `kin index`.
3. If the task is finalized: mark `status: "done"` in `feature_list.json`.
4. Append the summary from `progress/current.md` to the bottom of `progress/history.md`.
5. Purge `progress/current.md`, retaining only the bare template.
6. Do not leave behind temporary files, stray `print()` debug statements, or contextless TODOs.

## 6. Blockage Resolution

- Re-read the pertinent section within `docs/`.
- If a tool fails to yield the anticipated outcome, **do not fabricate a workaround**: document the blockage rigorously in `progress/current.md` and halt the session.
