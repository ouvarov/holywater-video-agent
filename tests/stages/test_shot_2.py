"""Acceptance tests for the `shot_2` stage (specs/shot_2.md)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import duration, make_audio

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import fakes
from hwva.providers.base import Capabilities, ProviderError, VideoRequest, VideoResult
from hwva.schemas import (
    SHOT_2_SKIPPED,
    LineSource,
    Script,
    Shot,
    ShotData,
    StageStatus,
    VoiceData,
)
from hwva.stages import shot_2

MakeCtx = Callable[..., Context]

LINE = "Я тут головний"
CHARACTER = "a ginger cat with green eyes"
VISUAL_1 = "the cat sits on a throne of cushions and speaks to camera"
VISUAL_2 = "the cat jumps off the throne and walks away proudly"
CAMERA_2 = "wide tracking shot"
LAST_FRAME = "shot_1.last.png"
ARTIFACTS = {"video": "shot_2.mp4"}
COST = 0.37


# --- helpers ------------------------------------------------------------------------------------


@dataclass
class RecordingVideo:
    """Real FakeVideo output; records requests, raises the queued errors one per call."""

    capabilities: Capabilities = field(
        default_factory=lambda: Capabilities(
            accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
        )
    )
    errors: list[ProviderError] = field(default_factory=list)
    requests: list[VideoRequest] = field(default_factory=list)
    out_paths: list[Path] = field(default_factory=list)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.requests.append(request)
        self.out_paths.append(out_path)
        if self.errors:
            raise self.errors.pop(0)
        result = fakes.FakeVideo().generate(request, out_path)
        return VideoResult(
            video_path=result.video_path, last_frame=result.last_frame, cost_usd=COST
        )


def _shots(*durations: float) -> list[Shot]:
    return [
        Shot(
            visual_prompt=VISUAL_1 if i == 0 else VISUAL_2,
            duration_s=d,
            camera="close-up" if i == 0 else CAMERA_2,
            has_line=i == 0,
        )
        for i, d in enumerate(durations)
    ]


def _ctx(
    make_ctx: MakeCtx,
    *,
    shots: list[Shot] | None = None,
    video: Any = None,
    last_frame_on_disk: bool = True,
) -> Context:
    """Context as it is after `shot_1`: script, audio, voice + shot_1 records DONE, the clip and
    its local last frame in the run dir."""
    params = {"line": LINE, "line_source": LineSource.INPUT.value, "lang": "uk"}
    ctx = make_ctx(idea=f'Кіт каже: "{LINE}"', params=params, video=video or RecordingVideo())
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    script = Script(
        title="test",
        line=LINE,
        line_source=LineSource.INPUT,
        lang="uk",
        character=CHARACTER,
        shots=shots or _shots(5, 4),
    )
    (ctx.run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    rec = mf.stage(ctx.manifest, "script")
    rec.status, rec.artifacts = StageStatus.DONE, {"script": "script.json"}

    make_audio(ctx.run_dir / "line.wav", 2.0, LINE)
    voice = VoiceData(audio_duration_s=2.0, planned_total_s=9.0, attempts=1)
    rec = mf.stage(ctx.manifest, "voice")
    rec.status = StageStatus.DONE
    rec.artifacts = {"audio": "line.wav", "transcript": "transcript.txt"}
    rec.data = voice.model_dump(mode="json")

    # shot_1 output made by the real fake provider: shot_1.mp4 + shot_1.last.png
    first = fakes.FakeVideo().generate(
        VideoRequest(prompt="shot 1", duration_s=2.0, audio_path=ctx.run_dir / "line.wav"),
        ctx.run_dir / "shot_1.mp4",
    )
    assert first.last_frame is not None
    assert Path(first.last_frame).resolve() == (ctx.run_dir / LAST_FRAME).resolve()
    if not last_frame_on_disk:
        (ctx.run_dir / LAST_FRAME).unlink()
    rec = mf.stage(ctx.manifest, "shot_1")
    rec.status = StageStatus.DONE
    rec.artifacts = {"video": "shot_1.mp4", "shot_1_last_frame": LAST_FRAME}
    rec.data = ShotData(duration_s=5.0).model_dump(mode="json")
    return ctx


def _video(ctx: Context) -> RecordingVideo:
    video = ctx.providers.video
    assert isinstance(video, RecordingVideo)
    return video


def _shot_2_stage() -> pipeline.Stage:
    stage = next((s for s in pipeline.STAGES if s.name == "shot_2"), None)
    assert stage is not None, "shot_2 is not registered in pipeline.STAGES"
    return stage


# --- AC: one-shot script -> skipped, DONE --------------------------------------------------------


def test_one_shot_script_is_skipped(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5))
    out = shot_2.run(ctx)
    assert isinstance(out, StageOutput)
    assert out.data == SHOT_2_SKIPPED == {"skipped": "one-shot script"}
    assert out.artifacts == {}
    assert out.cost_usd == 0.0
    assert _video(ctx).requests == []  # nothing generated, nothing paid
    assert not (ctx.run_dir / "shot_2.mp4").exists()


def test_one_shot_script_skips_even_without_last_frame(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5), last_frame_on_disk=False)
    out = shot_2.run(ctx)
    assert out.data == SHOT_2_SKIPPED
    assert _video(ctx).requests == []


def test_one_shot_script_is_done_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 1
    assert record.artifacts == {}
    assert record.data == {"skipped": "one-shot script"}
    assert record.cost_usd == 0.0


# --- AC: request carries shot 1's last frame as reference and no audio ---------------------------


def test_request_references_shot_1_last_frame(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_2.run(ctx)
    video = _video(ctx)
    assert len(video.requests) == 1
    ref = video.requests[0].reference_image
    assert ref is not None
    assert "://" not in ref  # the local copy, not an (expiring) provider URL
    assert Path(ref).resolve() == (ctx.run_dir / LAST_FRAME).resolve()


def test_request_has_no_audio(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_2.run(ctx)
    req = _video(ctx).requests[0]
    assert req.audio_path is None
    assert req.aspect_ratio == "9:16"


def test_prompt_has_same_character_and_second_shot(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_2.run(ctx)
    prompt = _video(ctx).requests[0].prompt
    assert CHARACTER in prompt
    assert VISUAL_2 in prompt
    assert CAMERA_2 in prompt
    assert VISUAL_1 not in prompt


def test_prompt_forbids_logos_and_brands(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_2.run(ctx)
    assert "no logos, brand names or readable text" in _video(ctx).requests[0].prompt.lower()


@pytest.mark.parametrize("shot_s", [3.0, 4.5, 6.0])
def test_duration_is_second_shot_duration(make_ctx: MakeCtx, no_stdin: None, shot_s: float) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5, shot_s))
    out = shot_2.run(ctx)
    assert _video(ctx).requests[0].duration_s == pytest.approx(shot_s, abs=0.01)
    assert ShotData.model_validate(out.data).duration_s == pytest.approx(shot_s, abs=0.01)
    assert duration(ctx.run_dir / ARTIFACTS["video"]) == pytest.approx(shot_s, abs=0.1)


def test_clip_in_run_dir_artifacts_and_cost(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    out = shot_2.run(ctx)
    assert _video(ctx).out_paths[0].resolve() == (ctx.run_dir / ARTIFACTS["video"]).resolve()
    assert out.artifacts == ARTIFACTS
    assert (ctx.run_dir / ARTIFACTS["video"]).is_file()
    assert out.cost_usd == pytest.approx(COST)


def test_two_shot_script_is_done_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5, 4))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE
    assert record.artifacts == ARTIFACTS
    assert ShotData.model_validate(record.data).duration_s == pytest.approx(4.0, abs=0.01)
    assert record.cost_usd == pytest.approx(COST)
    req = _video(ctx).requests[0]
    assert req.audio_path is None
    assert req.reference_image is not None
    assert Path(req.reference_image).resolve() == (ctx.run_dir / LAST_FRAME).resolve()


# --- AC: missing last-frame file -> GateFailed (not a crash) -------------------------------------


def test_missing_last_frame_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, last_frame_on_disk=False)
    with pytest.raises(GateFailed) as exc:
        shot_2.run(ctx)
    assert "frame" in str(exc.value).lower()
    assert _video(ctx).requests == []  # nothing generated, nothing paid


def test_missing_last_frame_fails_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, last_frame_on_disk=False)
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")  # not "Unexpected"
    assert _video(ctx).requests == []


# --- Behaviour: same retry rules as shot_1; reference rejected by moderation -> fails at once ----


def test_reference_rejected_by_moderation_fails_after_one_attempt(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    err = ProviderError("content moderation: face filter rejected the input image", retryable=False)
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[err]))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert len(_video(ctx).requests) == 1
    assert record.error is not None
    assert not record.error.startswith("Unexpected")
    assert "face filter" in record.error  # the provider's reason reaches the manifest


def test_retryable_error_is_retried_once_then_succeeds(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[ProviderError("timeout", retryable=True)]))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 2
    assert len(_video(ctx).requests) == 2
    assert (ctx.run_dir / ARTIFACTS["video"]).is_file()


def test_retryable_error_twice_fails_after_two_attempts(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    errors = [ProviderError("timeout", retryable=True), ProviderError("timeout", retryable=True)]
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=errors))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 2
    assert len(_video(ctx).requests) == 2


# === spec change: stale clip, moderation message, clip gates, cost bookkeeping, no subprocess ===

_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


@dataclass
class ClipVideo:
    """Provider returning `out_path` as the clip. clip: "ok" (real fake clip) | "missing" (no
    file) | "empty" (zero bytes). `on_generated` runs after the clip exists (e.g. arm a trap)."""

    clip: str = "ok"
    on_generated: Callable[[], None] | None = None
    capabilities: Capabilities = field(
        default_factory=lambda: Capabilities(
            accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
        )
    )
    requests: list[VideoRequest] = field(default_factory=list)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.requests.append(request)
        out_path.unlink(missing_ok=True)
        if self.clip == "ok":
            fakes.FakeVideo().generate(request, out_path)
        elif self.clip == "empty":
            out_path.write_bytes(b"")
        if self.on_generated is not None:
            self.on_generated()
        return VideoResult(video_path=out_path, last_frame=None, cost_usd=COST)


def _assert_failed_once_gate(result: Any, video: ClipVideo) -> None:
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert len(video.requests) == 1  # the paid clip is never generated a second time


# --- AC: one-shot script with a stale shot_2.mp4 -> skipped, DONE, the file is gone -------------


def test_one_shot_script_removes_stale_clip(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5))
    stale = ctx.run_dir / ARTIFACTS["video"]
    stale.write_bytes(b"stale clip from an earlier two-shot run")
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE
    assert record.data == SHOT_2_SKIPPED
    assert record.artifacts == {}
    assert not stale.exists()
    assert _video(ctx).requests == []


# --- AC: moderation rejection -> GateFailed (type asserted) naming the one-shot fallback ---------


def test_moderation_rejection_raises_gate_failed_with_fallback(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    err = ProviderError("content moderation: face filter rejected the input image", retryable=False)
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[err]))
    with pytest.raises(GateFailed) as exc:
        shot_2.run(ctx)
    msg = str(exc.value).lower()
    assert "one-shot" in msg  # the fallback is named
    assert "face filter" in msg  # and the provider's reason is kept
    assert len(_video(ctx).requests) == 1


def test_moderation_rejection_is_gate_failed_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    err = ProviderError("content moderation: face filter rejected the input image", retryable=False)
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[err]))
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert "one-shot" in record.error.lower()


# --- AC: missing or zero-byte clip -> FAILED after 1 attempt, GateFailed, one generate ----------


@pytest.mark.parametrize("clip", ["missing", "empty"])
def test_missing_or_empty_clip_fails_gate_once(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, clip: str
) -> None:
    video = ClipVideo(clip=clip)
    ctx = _ctx(make_ctx, video=video)
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)


@pytest.mark.parametrize("clip", ["missing", "empty"])
def test_missing_or_empty_clip_raises_gate_failed(
    make_ctx: MakeCtx, no_stdin: None, clip: str
) -> None:
    video = ClipVideo(clip=clip)
    ctx = _ctx(make_ctx, video=video)
    with pytest.raises(GateFailed) as exc:
        shot_2.run(ctx)
    msg = str(exc.value).lower()
    assert "shot_2.mp4" in msg or "clip" in msg  # says what is wrong
    assert len(video.requests) == 1


@dataclass
class ElsewhereVideo:
    """Provider that reports its clip at a path other than `out_path`, never writes `out_path`
    and leaves nothing at the reported path."""

    capabilities: Capabilities = field(
        default_factory=lambda: Capabilities(
            accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
        )
    )
    requests: list[VideoRequest] = field(default_factory=list)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.requests.append(request)
        elsewhere = out_path.with_name("provider_download.mp4")
        assert elsewhere.resolve() != out_path.resolve()
        elsewhere.unlink(missing_ok=True)
        return VideoResult(video_path=elsewhere, last_frame=None, cost_usd=COST)


def test_stale_clip_does_not_mask_missing_provider_clip(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    video = ElsewhereVideo()
    ctx = _ctx(make_ctx, video=video)
    stale = ctx.run_dir / ARTIFACTS["video"]
    stale_bytes = b"stale clip from an earlier run"
    stale.write_bytes(stale_bytes)

    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert len(video.requests) == 1  # the paid clip is never generated a second time

    stale.write_bytes(stale_bytes)  # direct call: same situation again
    with pytest.raises(GateFailed):
        shot_2.run(ctx)
    assert len(video.requests) == 2  # exactly one more generate


# --- AC: clip cost saved before the clip check; never double-counted ----------------------------


@pytest.mark.parametrize("clip", ["missing", "empty"])
def test_clip_cost_saved_when_clip_check_fails(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, clip: str
) -> None:
    video = ClipVideo(clip=clip)
    ctx = _ctx(make_ctx, video=video)
    ctx.manifest.cost_usd = 0.1  # spent by earlier stages
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)
    saved = mf.load(ctx.manifest.run_id, runs_root)  # what is on disk, not just in memory
    assert saved.stages["shot_2"].cost_usd == pytest.approx(COST)
    assert saved.cost_usd == pytest.approx(0.1 + COST)


def test_clip_cost_counted_once_on_success(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    video = ClipVideo()
    ctx = _ctx(make_ctx, video=video)
    ctx.manifest.cost_usd = 0.1
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE, record.error
    assert record.cost_usd == pytest.approx(COST)
    assert result.cost_usd == pytest.approx(0.1 + COST)  # not 0.1 + 2 * COST
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.stages["shot_2"].cost_usd == pytest.approx(COST)
    assert saved.cost_usd == pytest.approx(0.1 + COST)


# --- AC: the stage calls no subprocess (TimeoutExpired from subprocess.run does not affect it) --


@dataclass
class TimeoutTrap:
    """Once armed (after the provider made the clip), every subprocess.run / Popen raises
    TimeoutExpired and is recorded. Before that the real ones run (the fake provider needs them)."""

    armed: bool = False
    calls: list[str] = field(default_factory=list)

    def arm(self) -> None:
        self.armed = True

    def run(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if self.armed:
            self.calls.append(f"run {cmd!r}")
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 120)
        return _REAL_RUN(cmd, *args, **kwargs)

    def popen(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if self.armed:
            self.calls.append(f"Popen {cmd!r}")
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 120)
        return _REAL_POPEN(cmd, *args, **kwargs)


def test_stage_calls_no_subprocess(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trap = TimeoutTrap()
    video = ClipVideo(on_generated=trap.arm)
    ctx = _ctx(make_ctx, video=video)
    monkeypatch.setattr(subprocess, "run", trap.run)
    monkeypatch.setattr(subprocess, "Popen", trap.popen)
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.DONE, record.error
    assert record.attempts == 1
    assert trap.calls == []
    assert len(video.requests) == 1


# --- AC: money cap (HWVA_MAX_COST_USD) before the paid video call -------------------------------


def test_cost_cap_reached_stops_before_the_video_call(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx)
    ctx.max_cost_usd = 0.75
    ctx.manifest.cost_usd = 0.8
    result = pipeline.run(ctx, [_shot_2_stage()], root=runs_root)
    record = result.stages["shot_2"]
    assert record.status is StageStatus.FAILED and record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")
    assert "cost cap reached" in record.error and "0.75" in record.error
    assert "HWVA_MAX_COST_USD" in record.error
    assert _video(ctx).requests == []


def test_one_shot_script_is_skipped_even_with_the_cap_reached(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, shots=_shots(5))
    ctx.max_cost_usd = 0.5
    ctx.manifest.cost_usd = 0.9
    assert shot_2.run(ctx).data == SHOT_2_SKIPPED


def test_below_the_cap_runs(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    ctx.max_cost_usd = 1.0
    ctx.manifest.cost_usd = 0.99
    shot_2.run(ctx)
    assert len(_video(ctx).requests) == 1
