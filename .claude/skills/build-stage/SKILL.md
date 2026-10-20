---
name: build-stage
description: Build one pipeline stage end to end without a human — spec → xfail tests → implementation → gate → independent review. Use when asked to build/implement a stage from specs/.
---

`$ARGUMENTS` starts with the stage name (first word, e.g. `voice`); anything after it is extra context for you, not part of the name. Below, STAGE means that first word only.

Each step is a gate; do not skip ahead.

1. **Spec exists.** `specs/STAGE.md` must exist. If not, stop and ask for it.
2. **Contract first.** Spawn the `spec-writer` subagent with the stage name. Gate: verify yourself with `uv run pytest tests/stages/test_STAGE.py -rxX` — every new test **xfail**, none XPASS. An XPASS means a test that tests nothing: send back to spec-writer.
   - Stage already implemented and the spec changed? spec-writer adds/updates tests for the changed criteria only, marking just those `xfail(strict=True)`.
3. **Implement.** Spawn the `implementer` subagent with the stage name. Gate: `uv run check` green (includes `test_stages_follow_specs_order`) and the xfail markers gone (`git diff tests/` shows only marker lines removed).
4. **Independent review.** Spawn the `reviewer` subagent with ONLY the stage name — do not pass it your reasoning or the implementer's report. Gate: "APPROVE".
   - On "CHANGES REQUESTED": send the findings to `implementer`, then review again. Maximum 2 review rounds; after that stop and report the open findings to the human.
5. **Record.** Append a row to `AI-LOG.md`: stage, review rounds, what the reviewer caught, any human intervention. Commit with exactly the message `stage: STAGE` (never paste the arguments into the message).

Report to the human: stage built (yes/no), test count, review rounds, open findings.
