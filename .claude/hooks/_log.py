"""Shared hook helpers: repo-root cwd, tolerant event reading, UTF-8 subprocesses,
append-only metrics (logs/harness.jsonl — raw material for Part 5 of the note)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)  # hooks may be launched from any cwd (e.g. after `cd src`)
LOG = ROOT / "logs" / "harness.jsonl"
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def read_event() -> dict[str, Any]:
    """Hook payload from stdin. Tolerates a BOM (Windows shells) and empty input."""
    raw = sys.stdin.buffer.read().decode("utf-8-sig").strip()
    if not raw:
        return {}
    data: dict[str, Any] = json.loads(raw)
    return data


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """UTF-8 in and out, never None stdout (cp1251 consoles used to break this)."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=_ENV
    )


def log(event: str, **fields: object) -> None:
    LOG.parent.mkdir(exist_ok=True)
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": event, **fields}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def say(message: str) -> None:
    """Write to stderr as UTF-8 regardless of the console code page."""
    sys.stderr.buffer.write((message + "\n").encode("utf-8", errors="replace"))
    sys.stderr.flush()
