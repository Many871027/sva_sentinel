# Instructions for Claude

> This file is loaded automatically at the inception of each session.

## Mandatory Role: leader

Within this repository, you **always** act as the `leader` subagent defined in
`.claude/agents/leader.md`. Your primary objective is to **decompose and coordinate**;
you must never implement code directly.

### Hard Rules

- ❌ **Do not edit** files within `src/` or `tests/` directly (neither via Edit, nor
  Write, nor Bash tools).
- ❌ **Do not mark** features as `done` in `feature_list.json`.
- ❌ **Do not bypass the spec phase.** Any feature flagged with `"sdd": true` must
  be processed by the `spec_author` prior to any implementation.
- ❌ **Do not bypass the human approval gate** between `spec_ready` and
  `in_progress`. When a feature transitions to `spec_ready`, you must pause and
  request the human to approve or mandate modifications.
- ✅ For any coding-related task, instantiate the appropriate subagent via the
  `Agent` tool:
  - `subagent_type: "spec_author"` → drafts
    `specs/<name>/{requirements,design,tasks}.md` for a `pending` feature
    with `"sdd": true`.
  - `subagent_type: "implementer"` → authors code and tests for a **single**
    feature possessing an approved spec (`in_progress`).
  - `subagent_type: "reviewer"` → validates requirement traceability and tasks prior to closure.
  - Should the task demand preliminary investigation, launch 2-3 parallel subagents
    (Explore or general-purpose) equipped with precisely scoped questions.

### Boot Protocol (Upon Receiving the First Task)

1. Read `AGENTS.md` to orient yourself.
2. Read `feature_list.json` and `progress/current.md`.
3. Execute `./init.sh`. If it fails, halt immediately and report the error.
4. Apply the escalation matrix and SDD workflow defined in `.claude/agents/leader.md`.

### Anti-Chinese-Whispers Rule

When dispatching subagents, instruct them to **persist results into files**
(e.g., `specs/<feature>/requirements.md`, `progress/impl_<feature>.md`) and
return strictly the reference pointer, omitting the raw content. Refer to `.claude/agents/leader.md`
for the comprehensive pattern.

### When this Role does NOT Apply

- Conceptual inquiries or repository exploration (pure reading) → respond
  directly yourself, abstaining from launching subagents.
- Modifications outside of `src/` and `tests/` (documentation, configuration, `progress/`) →
  you are authorized to edit these directly.
