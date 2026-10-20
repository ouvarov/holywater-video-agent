"""PostToolUse: after an edit to a .py file, lint it (and type-check if it's under src/, which is
what `uv run check` type-checks). Exit 2 feeds the errors back to the agent."""

from __future__ import annotations

import sys
from pathlib import Path

from _log import ROOT, log, read_event, run, say


def main() -> int:
    event = read_event()
    path = Path(event.get("tool_input", {}).get("file_path", ""))
    if path.suffix != ".py" or not path.exists():
        return 0
    lint = run(["ruff", "check", "--fix", str(path)])
    in_src = (ROOT / "src") in path.resolve().parents
    types = run(["mypy", str(path)]) if in_src else None
    ok = lint.returncode == 0 and (types is None or types.returncode == 0)
    log("post_edit", file=str(path), ok=ok)
    if ok:
        return 0
    say(f"{path}: fix before continuing\n{lint.stdout}{types.stdout if types else ''}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a broken hook must not silently let bad code through
        say(f"post_edit hook crashed: {type(exc).__name__}: {exc}")
        sys.exit(2)
