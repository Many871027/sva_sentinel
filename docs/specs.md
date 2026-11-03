# Spec Driven Development (SDD)

> This project adheres to a Kiro-style workflow: requirements → design → tasks → code.
> Code is strictly never authored until the specification secures human approval.

## Structure

Each new feature (`"sdd": true` within `feature_list.json`) receives a dedicated
directory immediately upon departing the `pending` state:

```
specs/<feature-name>/
├── requirements.md   # WHAT is required (EARS notation)
├── design.md         # HOW it will be constructed (technical decisions)
└── tasks.md          # Concrete STEPS to execute
```

The `feature-name` corresponds identically to the `name` field in `feature_list.json`.

## Feature States

| State          | Definition                                                     |
|----------------|----------------------------------------------------------------|
| `pending`      | Spec absent. The `spec_author` initiates the action here.      |
| `spec_ready`   | Spec drafted. Awaiting human approval. Code remains untouched. |
| `in_progress`  | Spec approved. The `implementer` is actively executing.        |
| `done`         | Code is green, `reviewer` approved, session is closed.         |
| `blocked`      | Impeded. The rationale is documented in `progress/current.md`. |

## The Human Approval Gate

The automated pipeline pauses **once**: when the `spec_author` finalizes
its three files, it flags the feature as `spec_ready` and halts. The human
evaluates `specs/<feature>/` and responds with "approved" (or mandates revisions).

Only upon explicit approval does the `leader` transition `spec_ready → in_progress` and dispatch
the `implementer`.

```
pending → [spec_author] → spec_ready → ⏸ HUMAN → in_progress → [implementer → reviewer] → done
```

## requirements.md — Strict EARS

Requirements are authored utilizing **EARS** (Easy Approach to Requirements
Syntax). Each requirement is a numbered paragraph adhering to one of five
patterns:

| Pattern        | Template                                                    |
|----------------|-------------------------------------------------------------|
| **Ubiquitous** | `The system SHALL <action>.`                                |
| **Event-driven**| `WHEN <trigger>, the system SHALL <action>.`               |
| **State-driven**| `WHILE <state>, the system SHALL <action>.`                |
| **Optional**   | `WHERE <optional feature is present>, the system SHALL <action>.` |
| **Unwanted**   | `IF <unwanted trigger> THEN the system SHALL <action>.`     |

Hard rules:

- Every requirement possesses an immutable id: `R1`, `R2`, ...
- Every requirement MUST be verifiable by at least one concrete test.
- Never conflate multiple `SHALL` statements within a single requirement. If multiple exist, split them.
- Avoid weak verbs ("might", "can", "supports"). Exclusively utilize `SHALL` / `SHALL NOT`.

Example:

```markdown
## R1
WHEN the user executes `python -m src.cli recent`, the system SHALL
print up to 5 notes ordered by `created_at` descending.

## R2
IF the `--limit` flag receives a value <= 0 THEN the system SHALL
print an error message to stderr and terminate with an exit code != 0.
```

## design.md — Technical Decisions

This captures the architecture **prior** to any code modification:

- Which files are created / modified.
- Which new signatures emerge (functions, classes, commands).
- Which exceptions are repurposed or introduced.
- Which alternative was discarded and why (minimum of one required).

This is NOT first-principles engineering — lean heavily upon
`docs/architecture.md` and `docs/conventions.md`. The `design.md` file documents instances
where your feature approaches the boundary of these foundational rules.

## tasks.md — Executable Checklist

Sequential, discrete steps, each possessing a checkbox. Every task must reference at
least one `R<n>` that it fulfills.

Example:

```markdown
- [ ] T1 — Implement `cmd_recent` within `src/cli.py`. Covers: R1, R3.
- [ ] T2 — Register subparser `recent` utilizing flag `--limit`. Covers: R1, R2.
- [ ] T3 — Implement `test_recent_default_limit` within `tests/test_cli.py`. Covers: R1.
- [ ] T4 — Implement `test_recent_invalid_limit` within `tests/test_cli.py`. Covers: R2.
```

The `implementer` marks `[x]` for each task upon finalization. The `reviewer`
will reject the work if any `[ ]` remains unchecked without documented justification.

## Traceability (Hard Rule)

- Every test in `tests/` must map unequivocally to an `R<n>` from its specification.
- Every `R<n>` must possess at least one concrete test.
- The `reviewer` validates this mapping explicitly and rejects
  the implementation if traceability is deficient.

The `implementer` documents this mapping within `progress/impl_<name>.md`:

```markdown
## Traceability
- R1 → `test_recent_default_limit`
- R2 → `test_recent_invalid_limit`
- R3 → `test_recent_custom_limit`
```

## When SDD does NOT Apply

Features configured with `"sdd": false` or lacking the `sdd` key entirely (e.g., legacy features 1–6) DO NOT
possess a spec. SDD applies exclusively to forward-looking development.
