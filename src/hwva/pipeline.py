"""The DAG. Order is fixed in code; a stage runs only after the previous one passed its gate.
The LLM works inside stages, never decides what runs next."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hwva import manifest as mf
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import Manifest, StageRecord, StageStatus
from hwva.stages import (
    assemble,
    final_check,
    intake,
    quality_check,
    script,
    shot_1,
    shot_2,
    voice,
)

StageFn = Callable[[Context], StageOutput]


@dataclass(frozen=True)
class Stage:
    name: str
    run: StageFn
    max_attempts: int = 1


# Filled in as stages are built (/build-stage). Order here IS the pipeline.
STAGES: list[Stage] = [
    Stage("intake", intake.run),
    Stage("script", script.run, max_attempts=3),  # bad LLM answer -> ProviderError -> retried
    Stage("voice", voice.run),  # owns its TTS->STT retries (up to 4); GateFailed is final
    Stage("shot_1", shot_1.run, max_attempts=2),  # retryable ProviderError -> one more try
    Stage("shot_2", shot_2.run, max_attempts=2),  # same retry rules as shot_1
    Stage("assemble", assemble.run),  # local ffmpeg, deterministic: no retries
    Stage("final_check", final_check.run),  # owns its single STT retry; GateFailed is final
    Stage("quality_check", quality_check.run),  # soft only: never raises GateFailed
]


def run(ctx: Context, stages: list[Stage] | None = None, root: Path | None = None) -> Manifest:
    for st in stages if stages is not None else STAGES:
        if mf.is_done(ctx.manifest, st.name):
            continue  # resume: finished stages are not recomputed
        record = mf.stage(ctx.manifest, st.name)
        record.status, record.error, record.attempts = StageStatus.PENDING, None, 0
        record.started_at, record.finished_at = datetime.now(UTC), None
        mf.save(ctx.manifest, root)
        if not _run_with_retries(ctx, st, record, root):
            return ctx.manifest
    return ctx.manifest


def _run_with_retries(ctx: Context, st: Stage, record: StageRecord, root: Path | None) -> bool:
    committed = 0.0  # cost persisted via ctx.commit_cost during the current attempt
    commits = False  # the stage books its cost itself (also 0.0 on a resume that reuses all)

    def commit_cost(usd: float) -> None:
        nonlocal committed, commits
        committed += usd
        commits = True
        record.cost_usd += usd
        ctx.manifest.cost_usd += usd
        mf.save(ctx.manifest, root)

    ctx.commit_cost = commit_cost
    while True:
        record.attempts += 1
        committed = 0.0
        try:
            out = st.run(ctx)
        except ProviderError as exc:
            record.error = f"ProviderError: {exc}"
            if exc.retryable and record.attempts < st.max_attempts:
                mf.save(ctx.manifest, root)
                continue
            return _finish(ctx, record, StageStatus.FAILED, root)
        except GateFailed as exc:
            record.error = f"GateFailed: {exc}"
            return _finish(ctx, record, StageStatus.FAILED, root)
        except Exception as exc:  # a bug or a tool crash must still leave a readable manifest
            record.error = f"Unexpected {type(exc).__name__}: {exc}"
            return _finish(ctx, record, StageStatus.FAILED, root)
        record.artifacts, record.data = out.artifacts, out.data
        if commits:
            # committed part (and earlier attempts' commits) is in the record already
            record.cost_usd += out.cost_usd - committed
        else:
            record.cost_usd = out.cost_usd  # stages that never commit: unchanged behaviour
        ctx.manifest.cost_usd += out.cost_usd - committed  # never count committed cost twice
        record.error = None
        return _finish(ctx, record, StageStatus.DONE, root)


def _finish(ctx: Context, record: StageRecord, status: StageStatus, root: Path | None) -> bool:
    record.status, record.finished_at = status, datetime.now(UTC)
    mf.save(ctx.manifest, root)
    return status is StageStatus.DONE
