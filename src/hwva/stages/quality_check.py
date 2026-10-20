"""quality_check — soft quality signals on final.mp4 (judge + SyncNet). Never blocks.

The judge is not trusted on lip-sync, verbatim or technical facts (that is final_check's job);
everything here is recorded and nothing raises `GateFailed`.

Contract: specs/quality_check.md.
Data: `hwva.schemas.QualityReport`; artifact: quality.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel

from hwva import sync
from hwva.context import Context, StageOutput
from hwva.schemas import (
    AssembleData,
    JudgeOutcome,
    JudgeVerdict,
    QualityReport,
    QualitySync,
    Script,
    SyncNetOutcome,
    SyncReport,
)

FINAL = "final.mp4"
AUDIO = "line.wav"
SCRIPT = "script.json"
QUALITY = "quality.json"
NO_JUDGE = "no judge in profile"
NO_SYNC = "no sync measurement"
PROBE_TIMEOUT_S = 30


def run(ctx: Context) -> StageOutput:
    final = _artifact(ctx, "assemble", "video", FINAL)
    report = QualityReport(
        judge=_judge(ctx, final), syncnet=_syncnet(ctx, final), sync=_sync_numbers(ctx)
    )
    (ctx.run_dir / QUALITY).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return StageOutput(artifacts={"quality": QUALITY}, data=report.model_dump(mode="json"))


def _sync_numbers(ctx: Context) -> QualitySync:
    """shot_1's recorded lip-sync numbers, copied (never re-measured); soft, never raises."""
    record = ctx.manifest.stages.get("shot_1")
    raw = record.data.get("sync") if record is not None else None
    if raw is None:
        return QualitySync(skipped=NO_SYNC)
    try:
        sync = SyncReport.model_validate(raw)
    except ValueError as exc:  # pydantic ValidationError
        return QualitySync(skipped=f"sync record unreadable: {_error(exc)}")
    return QualitySync(
        syncnet_confidence=sync.syncnet_confidence,
        syncnet_offset_frames=sync.syncnet_offset_frames,
        verdict=sync.verdict,
        method=sync.method,
        shift_s=sync.shift_s,
        phrases=sync.phrases,
        confidence_global=sync.confidence_global,
        confidence_piecewise=sync.confidence_piecewise,
        take=sync.take,
        lag_s=sync.lag_s,
        peak_r=sync.peak_r,
        r_at_0=sync.r_at_0,
    )


def _artifact(ctx: Context, stage: str, key: str, default: str) -> Path:
    record = ctx.manifest.stages.get(stage)
    name = record.artifacts.get(key) if record is not None else None
    return ctx.run_dir / (name or default)


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# --- judge --------------------------------------------------------------------------------------


def _judge(ctx: Context, final: Path) -> JudgeOutcome:
    judge = ctx.providers.judge
    if judge is None:
        return JudgeOutcome(skipped=NO_JUDGE)
    if not final.is_file():
        return JudgeOutcome(error=f"{FINAL} not found in the run dir")
    script_path = _artifact(ctx, "script", "script", SCRIPT)
    try:
        script = Script.model_validate_json(script_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return JudgeOutcome(error=f"script not readable: {_error(exc)}")
    try:
        raw: object = judge.judge(final, script)  # one call: a soft signal isn't worth a retry
        verdict = _validate(raw)
    except Exception as exc:  # soft stage: any judge failure is recorded, never raised
        return JudgeOutcome(error=_error(exc) or "judge failed")
    duration_s, why = _duration(ctx, final)
    if duration_s is None:
        return JudgeOutcome(verdict=verdict, filter_skipped=why)
    kept = [d for d in verdict.defects if _seconds(d.timestamp) <= duration_s]
    dropped = len(verdict.defects) - len(kept)
    return JudgeOutcome(
        verdict=verdict.model_copy(update={"defects": kept}), hallucinated_defects=dropped
    )


def _validate(raw: object) -> JudgeVerdict:
    """Re-validate whatever the provider returned: the model may send prose or a broken dict."""
    if isinstance(raw, str | bytes):
        return JudgeVerdict.model_validate_json(raw)
    if isinstance(raw, BaseModel):
        return JudgeVerdict.model_validate(raw.model_dump(mode="json"))
    return JudgeVerdict.model_validate(raw)


def _seconds(mmss: str) -> int:
    minutes, seconds = mmss.split(":")
    return int(minutes) * 60 + int(seconds)


def _duration(ctx: Context, final: Path) -> tuple[float | None, str]:
    """Duration of final.mp4 from ffprobe; falls back to what assemble recorded.

    Returns `(seconds, "")` or `(None, why the length could not be read)`."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
             str(final)],
            capture_output=True, text=True, check=True, timeout=PROBE_TIMEOUT_S,
        ).stdout  # fmt: skip
        return float(json.loads(out)["format"]["duration"]), ""
    except (
        OSError,
        subprocess.SubprocessError,  # CalledProcessError, TimeoutExpired
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        probe_error = _error(exc)
    record = ctx.manifest.stages.get("assemble")
    fallback = "assemble recorded no duration"
    if record is not None and record.data:
        try:
            return AssembleData.model_validate(record.data).duration_s, ""
        except ValueError as exc:
            fallback = f"assemble duration invalid: {_error(exc)}"
    return None, f"video length unreadable (ffprobe: {probe_error}; {fallback}); defects kept"


# --- SyncNet on the final (required dependency; the runner is hwva.sync.syncnet_measure) -------


def _syncnet(ctx: Context, final: Path) -> SyncNetOutcome:
    # the audio as it is in final.mp4 (placement included), so where the line sits in the final
    # doesn't show up as a lip-sync error: final_check's extract, else the file assemble placed
    audio = _artifact(ctx, "final_check", "stt_audio", "")
    if not audio.name or not audio.is_file():
        audio = _artifact(ctx, "assemble", "audio", AUDIO)
        record = ctx.manifest.stages.get("assemble")
        offset = (record.data or {}).get("audio_offset_s") if record is not None else None
        if isinstance(offset, int | float) and offset < 0:  # the file is not what the final has
            return SyncNetOutcome(
                skipped=f"no stt_audio and {audio.name} was trimmed by {-offset:g} s in the final"
            )
    if not audio.is_file():
        audio = _artifact(ctx, "voice", "audio", AUDIO)
    if not final.is_file() or not audio.is_file():
        return SyncNetOutcome(skipped=f"{final.name} or {audio.name} not found in the run dir")
    try:
        result = sync.syncnet_measure(final, audio)  # looked up at call time (tests stub it)
    except sync.NoFaceError as exc:
        return SyncNetOutcome(skipped=str(exc) or "no face found by SyncNet")
    except sync.SyncError as exc:  # never fails the stage: not importable, weights, run, result
        return SyncNetOutcome(error=str(exc) or type(exc).__name__)
    return SyncNetOutcome(
        av_offset_frames=result.offset_frames,
        confidence=result.confidence,
        min_dist=result.min_dist,
        seconds=result.seconds,
    )
