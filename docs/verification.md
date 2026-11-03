# Verification — Demonstrating Functional Correctness

> Golden Rule: **the agent does not assert "it works", it proves it**.
> Every feature concludes with executable evidence, not with unfounded assertions.

## Verification Levels

### Level 1 — Unit Tests (Mandatory)

Every public function within `src/` requires at least one corresponding test in `tests/` that:

1. Validates the happy path.
2. Validates at least one error path if the function is susceptible to failure.

Command:
```bash
python3 -m unittest discover -s tests -v
```

### Level 2 — CLI Integration Test (Mandatory for UI features)

Features introducing CLI commands are verified by executing the actual CLI
against a temporary file:

```python
import subprocess, tempfile, os
with tempfile.TemporaryDirectory() as d:
    env = {**os.environ, "NOTES_FILE": os.path.join(d, "notes.json")}
    out = subprocess.check_output(
        ["python3", "-m", "src.cli", "add", "hello", "--body", "world"],
        env=env, text=True,
    )
    assert "id=" in out
```

### Level 3 — Manual Smoke Test (Optional but Recommended)

Prior to session closure, execute an end-to-end workflow utilizing a temporary
file residing in `/tmp`:

```bash
NOTES_FILE=/tmp/notes_demo.json python3 -m src.cli add "test" --body "x"
NOTES_FILE=/tmp/notes_demo.json python3 -m src.cli list
rm /tmp/notes_demo.json
```

### Level 4 — Requirement Traceability (Mandatory for features with `"sdd": true`)

Every `R<n>` requirement defined in `specs/<name>/requirements.md` must map definitively to at
least one concrete test within `tests/`. The reviewer will reject the work if coverage is deficient.

The implementer documents the mapping within `progress/impl_<name>.md`:

```markdown
## Traceability
- R1 → `test_recent_default_limit`
- R2 → `test_recent_invalid_limit`
- R3 → `test_recent_custom_limit`
```

## Anti-Patterns (Do NOT do)

- ❌ "I have added the command, it should work." → lacks an executable test.
- ❌ A test verifying solely that a function does not raise an exception. → must
  validate the specific outcome.
- ❌ `mock`ing the filesystem. → strictly utilize a real `tempfile.TemporaryDirectory()`.
- ❌ Marking a feature as `done` without executing `./init.sh`.

## Final Verification Prior to Closure

```bash
./init.sh           # must terminate with [OK] Environment ready
```

If `./init.sh` yields red output, **do not** mark anything as `done`. Document the blockage
within `progress/current.md` and transition the status to `blocked` in `feature_list.json`.
