"""Run state on disk. A run survives a crash: `--resume <run_id>` continues from the first stage
that is not DONE."""

from __future__ import annotations

import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

from hwva.schemas import Manifest, ModelRef, StageRecord, StageStatus


def runs_root(root: Path | None = None) -> Path:
    """Explicit root, else $HWVA_RUNS, else ./runs (repo root). Resolved per call, not at import."""
    return root if root is not None else Path(os.environ.get("HWVA_RUNS", "runs"))


def run_dir(run_id: str, root: Path | None = None) -> Path:
    return runs_root(root) / run_id


def new_run(
    idea: str, profile: str, models: dict[str, ModelRef], params: dict[str, str],
    root: Path | None = None,
) -> Manifest:  # fmt: skip
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    manifest = Manifest(run_id=run_id, idea=idea, profile=profile, models=models, params=params)
    run_dir(run_id, root).mkdir(parents=True, exist_ok=False)
    save(manifest, root)
    return manifest


def load(run_id: str, root: Path | None = None) -> Manifest:
    path = run_dir(run_id, root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No run {run_id!r} in {runs_root(root)}/")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def save(manifest: Manifest, root: Path | None = None) -> None:
    path = run_dir(manifest.run_id, root) / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written manifest


def stage(manifest: Manifest, name: str) -> StageRecord:
    return manifest.stages.setdefault(name, StageRecord())


def is_done(manifest: Manifest, name: str) -> bool:
    return stage(manifest, name).status is StageStatus.DONE
