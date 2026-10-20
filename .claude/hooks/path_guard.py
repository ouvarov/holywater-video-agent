"""PreToolUse on Write/Edit: agents may only write inside the repo, and never into .claude/ hooks
or settings (the harness must not edit its own guard rails).

Added after a real incident: during /build-stage intake an agent wrote a scratch check script to
C:\\Users\\Public — outside the repo. CLAUDE.md said "work only in src/, tests/…"; a prompt is a
wish, this hook is the law. Exit 2 blocks the tool call and tells the agent why."""

from __future__ import annotations

import sys
from pathlib import Path

from _log import ROOT, log, read_event, say

PROTECTED = (ROOT / ".claude" / "hooks", ROOT / ".claude" / "settings.json")


def main() -> int:
    event = read_event()
    raw = event.get("tool_input", {}).get("file_path", "")
    if not raw:
        return 0
    path = Path(raw)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    inside = path == ROOT or ROOT in path.parents
    protected = any(path == p or p in path.parents for p in PROTECTED)
    if inside and not protected:
        return 0
    log("path_guard", file=str(path), blocked=True)
    where = "outside the repo" if not inside else "into the harness itself"
    say(
        f"Blocked: writing {where} ({path}). Write only inside {ROOT} (src/, tests/, specs/, "
        f"config/). For scratch checks use pytest's tmp_path or `uv run python -c`."
    )
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        say(f"path_guard hook crashed: {type(exc).__name__}: {exc}")
        sys.exit(2)
