"""`uv run check` — the single definition of done: lint, format, types, tests."""

from __future__ import annotations

import shutil
import subprocess
import sys

STEPS: list[tuple[str, list[str]]] = [
    ("ruff", ["ruff", "check", "."]),
    ("format", ["ruff", "format", "--check", "."]),
    ("mypy", ["mypy"]),
    ("pytest", ["pytest", "-q"]),
]


def check() -> int:
    # ASCII output only: Windows consoles may not be UTF-8
    failed = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("== ffmpeg: NOT FOUND on PATH - media tests would be skipped silently")
        failed.append("ffmpeg")
    for name, cmd in STEPS:
        print(f"== {name}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("check: " + ("OK" if not failed else "FAILED: " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check())
