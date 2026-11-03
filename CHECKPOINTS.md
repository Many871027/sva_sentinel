# CHECKPOINTS — Final State Evaluation

> In multi-agent systems, the journey is not evaluated; the destination is.
> These are the objective checkpoints that a judge (human or AI) can utilize
> to determine whether the project is healthy.

## C1 — The Harness is Complete

- [ ] The 4 base files exist: `AGENTS.md`, `init.sh`, `feature_list.json`,
      `progress/current.md`.
- [ ] The 3 documentation files exist: `docs/architecture.md`, `docs/conventions.md`,
      `docs/verification.md`.
- [ ] `./init.sh` terminates with exit code 0.

## C2 — The State is Coherent

- [ ] At most one feature is designated as `in_progress` in `feature_list.json`.
- [ ] Every `done` feature possesses associated tests that pass successfully.
- [ ] `progress/current.md` is empty or accurately describes the active session
      (it contains no remnants from previous sessions).

## C3 — Code Adheres to Architecture

- [ ] `src/` contains solely the modules anticipated in `docs/architecture.md`.
- [ ] There are no external dependencies in `requirements.txt` (it must remain empty
      or non-existent).
- [ ] There are no stray `print()` statements for debugging, nor contextless TODOs.

## C4 — Verification is Genuine

- [ ] `tests/` features at least one test suite per module in `src/`.
- [ ] Tests leverage `tempfile.TemporaryDirectory()`, not filesystem mocks.
- [ ] Executing `python3 -m unittest discover -s tests -v` reveals > 0 tests,
      all of which pass successfully.

## C5 — Session Closed Properly

- [ ] There are no suspicious untracked files (`*.tmp`, `__pycache__`
      outside of `.gitignore`).
- [ ] `progress/history.md` contains an entry corresponding to the latest session.
- [ ] The most recently addressed feature reflects its correct state.

## C6 — Spec Driven Development (SDD)

- [ ] Every feature flagged with `"sdd": true` in the `spec_ready`, `in_progress`,
      or `done` state contains a corresponding directory `specs/<name>/` with 3 files:
      `requirements.md`, `design.md`, `tasks.md`.
- [ ] `requirements.md` strictly adheres to EARS notation (refer to `docs/specs.md`).
- [ ] Every `done` feature with `"sdd": true` has all its constituent tasks marked
      as `[x]` in `tasks.md`.
- [ ] Each `R<n>` requirement in `requirements.md` is comprehensively covered by at least
      one specific test in `tests/`.

---

**How to utilize this file:** a reviewer agent (`.claude/agents/reviewer.md`)
iterates through each checkbox, marks `[x]` or `[ ]`, and subsequently rejects
the session closure if any boxes remain unchecked within C1-C6.
