"""Stop / SubagentStop: the agent may not finish while `uv run check` is red.

A prompt saying "make sure tests pass" is a wish; this hook is the law. It blocks up to MAX_BLOCKS
stops in a row, then lets go and marks the run STUCK so a confused agent can't loop forever.

Tests for a stage that isn't implemented yet are marked xfail(strict=True) by spec-writer, so the
check stays green while the contract exists and turns red the moment it's implemented without
removing the marker (see .claude/agents)."""

from __future__ import annotations

import sys

from _log import ROOT, log, read_event, run, say

MAX_BLOCKS = 5
COUNTER = ROOT / ".claude" / ".stop_attempts"


def block(reason: str) -> int:
    """Block the stop, or let go as STUCK after MAX_BLOCKS consecutive blocks."""
    blocks = int(COUNTER.read_text()) + 1 if COUNTER.exists() else 1
    if blocks > MAX_BLOCKS:
        COUNTER.unlink(missing_ok=True)
        log("stop_gate", ok=False, stuck=True, blocks=MAX_BLOCKS)
        say(f"STUCK: still red after {MAX_BLOCKS} attempts - stopping, needs a human.\n{reason}")
        return 0
    COUNTER.write_text(str(blocks))
    log("stop_gate", ok=False, blocks=blocks)
    say(f"Not done (block {blocks}/{MAX_BLOCKS}).\n{reason}")
    return 2


def main() -> int:
    read_event()  # consume the payload; the decision depends only on `uv run check`
    result = run(["uv", "run", "--quiet", "check"])
    if result.returncode == 0:
        COUNTER.unlink(missing_ok=True)
        log("stop_gate", ok=True)
        return 0
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
    return block(f"`uv run check` is red:\n{tail}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a crash blocks too (never a silent pass), with the same STUCK limit
        sys.exit(block(f"stop_gate hook crashed: {type(exc).__name__}: {exc}"))
