"""Acceptance tests for the `shot_1` stage (specs/shot_1.md)."""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from conftest import (
    StubSyncNetMeasure,
    SyncNetAnswer,
    duration,
    install_syncnet_measure,
    make_audio,
    make_video,
    probe,
    read_wav,
    speech_wav,
)

from hwva import manifest as mf
from hwva import pipeline, sync
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import fakes, seedance
from hwva.providers.base import (
    Capabilities,
    ImageResult,
    ProviderError,
    VideoRequest,
    VideoResult,
)
from hwva.schemas import (
    LineSource,
    ModelRef,
    Script,
    Shot,
    ShotData,
    StageStatus,
    SyncReport,
    VoiceData,
)
from hwva.stages import shot_1

MakeCtx = Callable[..., Context]

LINE = "Я тут головний"
CHARACTER = "a ginger cat with green eyes"
VISUAL = "the cat sits on a throne of cushions and speaks to camera"
CAMERA = "slow push-in close-up"
ARTIFACTS = {"video": "shot_1.mp4", "shot_1_last_frame": "shot_1.last.png"}
# Lip-sync check (specs/shot_1.md): the stage also writes sync.json
SYNC_ARTIFACTS = {**ARTIFACTS, "sync": "sync.json"}
AUDIO = "line.wav"
COST = 0.42


# --- helpers ------------------------------------------------------------------------------------


@dataclass
class RecordingVideo:
    """Real FakeVideo output, configurable capabilities / failures / where the last frame lives.

    last_frame modes:
      "local"     — FakeVideo default: <out>.last.png next to the clip
      "elsewhere" — a local path outside the run dir
      "url"       — an http URL (served by `frame_server`); nothing left on disk in the run dir
    """

    capabilities: Capabilities = field(
        default_factory=lambda: Capabilities(
            accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
        )
    )
    errors: list[ProviderError] = field(default_factory=list)  # raised one per call, in order
    last_frame_mode: str = "local"
    elsewhere_dir: Path | None = None
    url_base: str | None = None
    # if set, the provider's frame ("elsewhere"/"url") holds these bytes instead of the real frame,
    # so a test can tell the provider's frame apart from one extracted from the local clip
    provider_frame_bytes: bytes | None = None
    requests: list[VideoRequest] = field(default_factory=list)
    out_paths: list[Path] = field(default_factory=list)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.requests.append(request)
        self.out_paths.append(out_path)
        if self.errors:
            raise self.errors.pop(0)
        result = fakes.FakeVideo().generate(request, out_path)
        assert result.last_frame is not None
        frame = Path(result.last_frame)
        last: str = str(frame)
        if self.last_frame_mode == "elsewhere":
            assert self.elsewhere_dir is not None
            moved = self.elsewhere_dir / "provider_frame.png"
            shutil.move(frame, moved)
            if self.provider_frame_bytes is not None:
                moved.write_bytes(self.provider_frame_bytes)
            last = str(moved)
        elif self.last_frame_mode == "url":
            assert self.elsewhere_dir is not None and self.url_base is not None
            served = self.elsewhere_dir / "frame.png"
            shutil.move(frame, served)
            if self.provider_frame_bytes is not None:
                served.write_bytes(self.provider_frame_bytes)
            last = f"{self.url_base}/frame.png"
        return VideoResult(video_path=result.video_path, last_frame=last, cost_usd=COST)


@dataclass
class FrameServer:
    base_url: str
    root: Path
    hits: list[str]


@pytest.fixture
def frame_server(tmp_path: Path) -> Iterator[FrameServer]:
    """Local HTTP server (127.0.0.1, no internet) standing in for the provider's CDN."""
    root = tmp_path / "cdn"
    root.mkdir()
    hits: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield FrameServer(f"http://{host!s}:{port}", root, hits)
    finally:
        server.shutdown()
        server.server_close()


def _shots(*durations: float) -> list[Shot]:
    return [
        Shot(
            visual_prompt=VISUAL if i == 0 else f"shot {i}",
            duration_s=d,
            camera=CAMERA if i == 0 else "wide",
            has_line=i == 0,
        )
        for i, d in enumerate(durations)
    ]


def _ctx(
    make_ctx: MakeCtx,
    *,
    audio_s: float = 2.0,
    shots: list[Shot] | None = None,
    video: Any = None,
    character: str = CHARACTER,
    **overrides: Any,
) -> Context:
    """Context as it is after `voice`: script.json and line.wav on disk, voice record DONE.
    `overrides` replace other providers (e.g. image=...)."""
    params = {"line": LINE, "line_source": LineSource.INPUT.value, "lang": "uk"}
    ctx = make_ctx(
        idea=f'Кіт каже: "{LINE}"', params=params, video=video or RecordingVideo(), **overrides
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    script = Script(
        title="test",
        line=LINE,
        line_source=LineSource.INPUT,
        lang="uk",
        character=character,
        shots=shots or _shots(5, 4),
    )
    (ctx.run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    rec = mf.stage(ctx.manifest, "script")
    rec.status, rec.artifacts = StageStatus.DONE, {"script": "script.json"}

    make_audio(ctx.run_dir / AUDIO, audio_s, LINE)
    (ctx.run_dir / "transcript.txt").write_text(LINE, encoding="utf-8")
    voice = VoiceData(audio_duration_s=audio_s, planned_total_s=max(5, audio_s) + 4, attempts=1)
    rec = mf.stage(ctx.manifest, "voice")
    rec.status = StageStatus.DONE
    rec.artifacts = {"audio": AUDIO, "transcript": "transcript.txt"}
    rec.data = voice.model_dump(mode="json")
    return ctx


def _video(ctx: Context) -> RecordingVideo:
    video = ctx.providers.video
    assert isinstance(video, RecordingVideo)
    return video


def _data(out: StageOutput) -> ShotData:
    assert isinstance(out, StageOutput)
    return ShotData.model_validate(out.data)


def _shot_1_stage() -> pipeline.Stage:
    stage = next((s for s in pipeline.STAGES if s.name == "shot_1"), None)
    assert stage is not None, "shot_1 is not registered in pipeline.STAGES"
    return stage


# --- AC: request carries the audio path, 9:16, duration = max(shot, audio) ----------------------


def test_request_carries_our_audio_and_vertical_format(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_1.run(ctx)
    video = _video(ctx)
    assert len(video.requests) == 1
    req = video.requests[0]
    assert req.audio_path is not None
    assert Path(req.audio_path).resolve() == (ctx.run_dir / AUDIO).resolve()
    assert req.aspect_ratio == "9:16"


def test_prompt_has_character_visual_and_camera(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_1.run(ctx)
    prompt = _video(ctx).requests[0].prompt
    assert CHARACTER in prompt
    assert VISUAL in prompt
    assert CAMERA in prompt


def test_prompt_forbids_logos_and_brands(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    shot_1.run(ctx)
    assert "no logos, brand names or readable text" in _video(ctx).requests[0].prompt.lower()


@pytest.mark.parametrize(
    ("shot_s", "audio_s", "expected"),
    [(5.0, 2.0, 5.0), (5.0, 6.5, 6.5), (4.0, 4.0, 4.0)],
    ids=["shot_longer", "audio_longer", "equal"],
)
def test_duration_is_max_of_shot_and_audio(
    make_ctx: MakeCtx, no_stdin: None, shot_s: float, audio_s: float, expected: float
) -> None:
    ctx = _ctx(make_ctx, audio_s=audio_s, shots=_shots(shot_s, 4))
    out = shot_1.run(ctx)
    req = _video(ctx).requests[0]
    assert req.duration_s == pytest.approx(expected, abs=0.01)
    assert _data(out).duration_s == pytest.approx(expected, abs=0.01)
    assert duration(ctx.run_dir / ARTIFACTS["video"]) == pytest.approx(expected, abs=0.1)


def test_clip_downloaded_into_run_dir_and_cost_reported(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    out = shot_1.run(ctx)
    assert _video(ctx).out_paths[0].resolve() == (ctx.run_dir / ARTIFACTS["video"]).resolve()
    assert out.artifacts == SYNC_ARTIFACTS
    assert (ctx.run_dir / ARTIFACTS["video"]).is_file()
    assert out.cost_usd == pytest.approx(COST)


# --- AC: model without audio input -> GateFailed; needs_image without image -> GateFailed --------


def test_model_without_audio_input_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    caps = Capabilities(accepts_audio_input=False, native_speech=True, max_duration_s=30)
    ctx = _ctx(make_ctx, video=RecordingVideo(capabilities=caps))
    with pytest.raises(GateFailed):
        shot_1.run(ctx)
    assert _video(ctx).requests == []  # nothing generated, nothing paid


def test_needs_image_without_image_provider_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    """shot_1 makes the reference image itself; without an `image` role in the
    profile there is still no image -> GateFailed, nothing paid."""
    caps = Capabilities(accepts_audio_input=True, needs_image=True, max_duration_s=30)
    ctx = _ctx(make_ctx, video=RecordingVideo(capabilities=caps), image=None)
    with pytest.raises(GateFailed) as exc:
        shot_1.run(ctx)
    assert "image" in str(exc.value).lower()
    assert _video(ctx).requests == []


def test_capability_gates_fail_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    caps = Capabilities(accepts_audio_input=False, max_duration_s=30)
    ctx = _ctx(make_ctx, video=RecordingVideo(capabilities=caps))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")


# --- AC: last frame artifact is an existing local file -------------------------------------------


def test_last_frame_is_local_file_in_run_dir(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    out = shot_1.run(ctx)
    assert out.artifacts["shot_1_last_frame"] == "shot_1.last.png"
    assert (ctx.run_dir / "shot_1.last.png").is_file()


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PROVIDER_FRAME = PNG_MAGIC + b"provider-frame-not-from-our-clip"


def _assert_frame_from_local_clip(frame: Path, provider_frame: Path) -> None:
    assert frame.is_file() and frame.stat().st_size > 0
    assert frame.read_bytes() != provider_frame.read_bytes()  # not the provider's frame
    assert frame.read_bytes()[:8] == PNG_MAGIC
    stream = probe(frame)["streams"][0]
    assert (stream["width"], stream["height"]) == (720, 1280)  # a real frame of our 9:16 clip


def test_last_frame_outside_run_dir_is_ignored(
    make_ctx: MakeCtx, no_stdin: None, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "provider_tmp"
    elsewhere.mkdir()
    video = RecordingVideo(
        last_frame_mode="elsewhere", elsewhere_dir=elsewhere, provider_frame_bytes=PROVIDER_FRAME
    )
    ctx = _ctx(make_ctx, video=video)
    out = shot_1.run(ctx)
    assert out.artifacts["shot_1_last_frame"] == "shot_1.last.png"
    assert len(video.requests) == 1
    _assert_frame_from_local_clip(
        ctx.run_dir / out.artifacts["shot_1_last_frame"], elsewhere / "provider_frame.png"
    )


def test_last_frame_url_is_never_fetched(
    make_ctx: MakeCtx, no_stdin: None, frame_server: FrameServer
) -> None:
    video = RecordingVideo(
        last_frame_mode="url",
        elsewhere_dir=frame_server.root,
        url_base=frame_server.base_url,
        provider_frame_bytes=PROVIDER_FRAME,
    )
    ctx = _ctx(make_ctx, video=video)
    out = shot_1.run(ctx)
    assert frame_server.hits == []  # never re-downloaded
    name = out.artifacts["shot_1_last_frame"]
    assert name == "shot_1.last.png"
    assert "://" not in name  # never keep only the URL
    assert len(video.requests) == 1
    _assert_frame_from_local_clip(ctx.run_dir / name, frame_server.root / "frame.png")


def test_last_frame_survives_in_manifest_after_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE
    assert record.artifacts == SYNC_ARTIFACTS
    assert (ctx.run_dir / record.artifacts["shot_1_last_frame"]).is_file()
    assert ShotData.model_validate(record.data).duration_s == pytest.approx(5.0, abs=0.01)
    assert record.cost_usd == pytest.approx(COST)


# --- AC: non-retryable provider error -> FAILED after 1 attempt ---------------------------------


def test_provider_error_propagates_from_run(make_ctx: MakeCtx, no_stdin: None) -> None:
    err = ProviderError("content moderation: rejected", retryable=False)
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[err]))
    with pytest.raises(ProviderError) as exc:
        shot_1.run(ctx)
    assert exc.value.retryable is False  # not swallowed, not turned into a retryable error


def test_non_retryable_error_fails_after_one_attempt(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    err = ProviderError("content moderation: rejected", retryable=False)
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[err]))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert len(_video(ctx).requests) == 1
    assert record.error is not None and "moderation" in record.error


def test_retryable_error_is_retried_once_then_succeeds(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=[ProviderError("timeout", retryable=True)]))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 2
    assert len(_video(ctx).requests) == 2
    assert (ctx.run_dir / ARTIFACTS["shot_1_last_frame"]).is_file()


def test_retryable_error_twice_fails_after_two_attempts(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    errors = [ProviderError("timeout", retryable=True), ProviderError("timeout", retryable=True)]
    ctx = _ctx(make_ctx, video=RecordingVideo(errors=errors))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 2
    assert len(_video(ctx).requests) == 2


# === spec change: local last-frame extraction, cost bookkeeping, file/ffmpeg gates ==============
# Every test below targets a criterion added to specs/shot_1.md and is xfail until implemented.

_REAL_RUN = subprocess.run
VIDEO_NAME = ARTIFACTS["video"]
FRAME_NAME = ARTIFACTS["shot_1_last_frame"]
RENDER_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 30

FailFn = Callable[[list[str], dict[str, Any]], Any]


@dataclass
class FfSpy:
    """Wraps `subprocess.run`; records ffmpeg/ffprobe calls made AFTER the clip was generated
    (i.e. by the stage, not by the fake provider). `fail`, if set, replaces an armed call."""

    armed: bool = False
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)
    fail: FailFn | None = None

    def __call__(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        argv = [str(c) for c in cmd] if isinstance(cmd, list | tuple) else [str(cmd)]
        tool = Path(argv[0]).stem.lower() if argv else ""
        if self.armed and tool in ("ffmpeg", "ffprobe"):
            self.calls.append((argv, dict(kwargs)))
            if self.fail is not None:
                return self.fail(argv, kwargs)
        return _REAL_RUN(cmd, *args, **kwargs)

    def by_tool(self, tool: str) -> list[tuple[list[str], dict[str, Any]]]:
        return [(a, k) for a, k in self.calls if Path(a[0]).stem.lower() == tool]

    def sseof_calls(self) -> list[tuple[list[str], dict[str, Any]]]:
        return [(a, k) for a, k in self.by_tool("ffmpeg") if "-sseof" in a]


@pytest.fixture
def ff_spy(monkeypatch: pytest.MonkeyPatch) -> FfSpy:
    spy = FfSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    return spy


@dataclass
class ClipOnlyVideo:
    """Provider that downloads only the clip and returns no `last_frame` (a model without
    `return_last_frame`). clip: "ok" | "missing" | "empty". `url_frame` simulates a provider that
    still hands back a CDN URL for the frame."""

    spy: FfSpy | None = None
    clip: str = "ok"
    url_frame: str | None = None
    capabilities: Capabilities = field(
        default_factory=lambda: Capabilities(
            accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0
        )
    )
    requests: list[VideoRequest] = field(default_factory=list)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.requests.append(request)
        if self.clip == "ok":
            make_video(out_path, request.duration_s)
        elif self.clip == "empty":
            out_path.write_bytes(b"")
        if self.spy is not None:
            self.spy.armed = True  # from here on every ffmpeg/ffprobe call is the stage's
        return VideoResult(video_path=out_path, last_frame=self.url_frame, cost_usd=COST)


def _arg_after(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    i = argv.index(flag) + 1
    return argv[i] if i < len(argv) else None


def _is_single_frame(argv: list[str]) -> bool:
    pairs = (("-frames:v", "1"), ("-vframes", "1"), ("-update", "1"))
    return any(_arg_after(argv, f) == v for f, v in pairs)


def _same(a: str | Path | None, b: Path) -> bool:
    return a is not None and Path(a).resolve() == b.resolve()


def _assert_failed_once_gate(result: Any, video: ClipOnlyVideo) -> None:
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert len(video.requests) == 1  # the paid clip is never generated a second time


def _nonzero_exit(argv: list[str], kwargs: dict[str, Any]) -> Any:
    if kwargs.get("check"):
        raise subprocess.CalledProcessError(1, argv, output="", stderr="decode error")
    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="decode error")


def _timeout(argv: list[str], kwargs: dict[str, Any]) -> Any:
    raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or RENDER_TIMEOUT_S)


def _sseof_ok_but(ctx: Context, png: bytes | None) -> FailFn:
    """-sseof call 'succeeds' (exit 0) but leaves no PNG (png=None) or the given bytes."""

    def fake(argv: list[str], kwargs: dict[str, Any]) -> Any:
        if "-sseof" not in argv:
            return _REAL_RUN(argv, **kwargs)
        frame = ctx.run_dir / FRAME_NAME
        if png is None:
            frame.unlink(missing_ok=True)
        else:
            frame.write_bytes(png)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return fake


# --- AC: last frame extracted by ffmpeg (-sseof, one frame) from the local clip ------------------


def test_last_frame_extracted_locally_without_provider_frame(
    make_ctx: MakeCtx, no_stdin: None, ff_spy: FfSpy
) -> None:
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    out = shot_1.run(ctx)
    assert len(video.requests) == 1
    assert out.artifacts == SYNC_ARTIFACTS
    clip, frame = ctx.run_dir / VIDEO_NAME, ctx.run_dir / FRAME_NAME
    calls = ff_spy.sseof_calls()
    assert len(calls) == 1, f"expected one ffmpeg -sseof call, got {ff_spy.calls}"
    argv, _ = calls[0]
    assert _same(_arg_after(argv, "-i"), clip)  # from the LOCAL clip
    assert _is_single_frame(argv)
    assert any(_same(a, frame) for a in argv[1:])
    assert frame.is_file() and frame.stat().st_size > 0
    assert frame.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    stream = probe(frame)["streams"][0]
    assert (stream["width"], stream["height"]) == (720, 1280)  # a frame of OUR vertical clip


def test_last_frame_extracted_locally_even_if_provider_returns_url(
    make_ctx: MakeCtx, no_stdin: None, ff_spy: FfSpy, frame_server: FrameServer
) -> None:
    (frame_server.root / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-our-frame")
    video = ClipOnlyVideo(spy=ff_spy, url_frame=f"{frame_server.base_url}/frame.png")
    ctx = _ctx(make_ctx, video=video)
    shot_1.run(ctx)
    assert len(video.requests) == 1
    assert len(ff_spy.sseof_calls()) == 1
    frame = ctx.run_dir / FRAME_NAME
    assert frame.read_bytes() != (frame_server.root / "frame.png").read_bytes()


def test_pipeline_done_with_one_generate_when_provider_has_no_frame(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, ff_spy: FfSpy
) -> None:
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert record.attempts == 1
    assert len(video.requests) == 1
    assert (ctx.run_dir / record.artifacts["shot_1_last_frame"]).is_file()


# --- AC: clip cost saved before the frame step, never double-counted -----------------------------


def test_clip_cost_saved_when_frame_step_fails(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, ff_spy: FfSpy
) -> None:
    ff_spy.fail = _nonzero_exit
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    ctx.manifest.cost_usd = 0.1  # spent by earlier stages
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)
    saved = mf.load(ctx.manifest.run_id, runs_root)  # what is on disk, not just in memory
    assert saved.stages["shot_1"].cost_usd == pytest.approx(COST)
    assert saved.cost_usd == pytest.approx(0.1 + COST)


def test_clip_cost_counted_once_on_success(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, ff_spy: FfSpy
) -> None:
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    ctx.manifest.cost_usd = 0.1
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert record.cost_usd == pytest.approx(COST)
    assert result.cost_usd == pytest.approx(0.1 + COST)  # not 0.1 + 2 * COST
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.stages["shot_1"].cost_usd == pytest.approx(COST)
    assert saved.cost_usd == pytest.approx(0.1 + COST)


# --- AC: missing or zero-byte clip -> FAILED after 1 attempt, GateFailed, one generate ----------


@pytest.mark.parametrize("clip", ["missing", "empty"])
def test_missing_or_empty_clip_fails_gate_once(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, clip: str
) -> None:
    video = ClipOnlyVideo(clip=clip)
    ctx = _ctx(make_ctx, video=video)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)


@pytest.mark.parametrize("clip", ["missing", "empty"])
def test_missing_or_empty_clip_raises_gate_failed(
    make_ctx: MakeCtx, no_stdin: None, clip: str
) -> None:
    ctx = _ctx(make_ctx, video=ClipOnlyVideo(clip=clip))
    with pytest.raises(GateFailed):
        shot_1.run(ctx)


# --- AC: frame extraction fails / missing / zero-byte PNG -> GateFailed, one generate ------------


@pytest.mark.parametrize("mode", ["nonzero_exit", "no_png", "empty_png"])
def test_frame_extraction_failure_fails_gate_once(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, ff_spy: FfSpy, mode: str
) -> None:
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    modes: dict[str, FailFn] = {
        "nonzero_exit": _nonzero_exit,
        "no_png": _sseof_ok_but(ctx, None),
        "empty_png": _sseof_ok_but(ctx, b""),
    }
    ff_spy.fail = modes[mode]
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)
    assert ff_spy.sseof_calls(), "the stage never tried to extract the frame with ffmpeg -sseof"


# --- AC: ffmpeg/ffprobe timeouts (120 s render, 30 s probe); TimeoutExpired -> GateFailed --------


def test_ffmpeg_and_ffprobe_calls_pass_timeouts(
    make_ctx: MakeCtx, no_stdin: None, ff_spy: FfSpy
) -> None:
    ctx = _ctx(make_ctx, video=ClipOnlyVideo(spy=ff_spy))
    shot_1.run(ctx)
    assert ff_spy.sseof_calls(), "no ffmpeg frame extraction happened"
    for argv, kwargs in ff_spy.by_tool("ffmpeg"):
        assert kwargs.get("timeout") == RENDER_TIMEOUT_S, argv
    for argv, kwargs in ff_spy.by_tool("ffprobe"):
        assert kwargs.get("timeout") == PROBE_TIMEOUT_S, argv


def test_timeout_expired_fails_gate_once(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, ff_spy: FfSpy
) -> None:
    ff_spy.fail = _timeout  # every ffmpeg/ffprobe call of the stage hangs
    video = ClipOnlyVideo(spy=ff_spy)
    ctx = _ctx(make_ctx, video=video)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    _assert_failed_once_gate(result, video)
    assert ff_spy.calls, "the stage made no ffmpeg/ffprobe call"


def test_timeout_expired_raises_gate_failed_from_run(
    make_ctx: MakeCtx, no_stdin: None, ff_spy: FfSpy
) -> None:
    ff_spy.fail = _timeout
    ctx = _ctx(make_ctx, video=ClipOnlyVideo(spy=ff_spy))
    with pytest.raises(GateFailed):
        shot_1.run(ctx)


# --- AC: Seedance never pays twice for one clip (real provider, HTTP mocked) --------------------

ARK_TASKS = seedance.DEFAULT_BASE_URL + seedance.TASKS_PATH
ARK_VIDEO_URL = "https://tos.example.com/shot_1.mp4?X-Tos-Signature=abc"
ARK_TOKENS = 200_000
ARK_PRICE = 2.5  # USD per million tokens
ARK_CLIP_COST = ARK_TOKENS * ARK_PRICE / 1_000_000


@dataclass
class ArkApi:
    """MockTransport handler: POST -> {"id": "cgt-<n>"}; task GET -> next status of `polls`
    (the last one repeats); any other GET -> the clip bytes."""

    clip: bytes
    polls: list[str]
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"id": f"cgt-{len(self.posts())}"})
        url = str(request.url)
        if not url.startswith(ARK_TASKS):
            return httpx.Response(200, content=self.clip)
        status = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        body: dict[str, Any] = {"id": url.rsplit("/", 1)[-1], "status": status}
        if status == "succeeded":
            body["content"] = {"video_url": ARK_VIDEO_URL}
            body["usage"] = {"completion_tokens": ARK_TOKENS}
        return httpx.Response(200, json=body)

    def posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]


def _seedance(monkeypatch: pytest.MonkeyPatch, api: ArkApi) -> seedance.SeedanceVideo:
    """Priced SeedanceVideo whose every poll deadline is 0 s: one poll per attempt."""
    monkeypatch.setenv(seedance.API_KEY_ENV, "test-ark-key-not-real")
    monkeypatch.delenv(seedance.BASE_URL_ENV, raising=False)
    ref = ModelRef(provider="seedance", model="seedance-test-model", version="2.5")
    return seedance.SeedanceVideo(
        ref, transport=httpx.MockTransport(api), poll_interval_s=0.0, timeout_s=0.0,
        usd_per_million_tokens=ARK_PRICE,
    )  # fmt: skip


def _mp4(tmp_path: Path) -> bytes:
    return make_video(tmp_path / "served.mp4", 5).read_bytes()


def test_seedance_poll_timeout_then_pipeline_retry_makes_exactly_one_post(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    api = ArkApi(clip=_mp4(tmp_path), polls=["running", "succeeded"])
    ctx = _ctx(make_ctx, video=_seedance(monkeypatch, api))

    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)

    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert record.attempts == 2  # attempt 1 timed out while polling
    assert len(api.posts()) == 1
    assert (ctx.run_dir / VIDEO_NAME).read_bytes() == api.clip
    assert result.cost_usd == pytest.approx(ARK_CLIP_COST)


def test_seedance_clip_cost_counted_once_when_resumed_after_the_frame_step_failed(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, ff_spy: FfSpy,
) -> None:  # fmt: skip
    api = ArkApi(clip=_mp4(tmp_path), polls=["succeeded"])
    ctx = _ctx(make_ctx, video=_seedance(monkeypatch, api))
    ctx.manifest.cost_usd = 0.1  # spent by earlier stages
    ff_spy.armed, ff_spy.fail = True, _nonzero_exit

    first = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    assert first.stages["shot_1"].status is StageStatus.FAILED
    assert mf.load(ctx.manifest.run_id, runs_root).cost_usd == pytest.approx(0.1 + ARK_CLIP_COST)

    ff_spy.fail = None
    resumed = Context(mf.load(ctx.manifest.run_id, runs_root), ctx.providers, ctx.run_dir, False)
    result = pipeline.run(resumed, [_shot_1_stage()], root=runs_root)

    assert result.stages["shot_1"].status is StageStatus.DONE, result.stages["shot_1"].error
    assert len(api.posts()) == 1
    assert result.cost_usd == pytest.approx(0.1 + ARK_CLIP_COST)
    assert mf.load(ctx.manifest.run_id, runs_root).cost_usd == pytest.approx(0.1 + ARK_CLIP_COST)


# --- AC: needs_image -> character.png made once by the image provider (live) ---------

NEEDS_IMAGE = Capabilities(accepts_audio_input=True, needs_image=True, max_duration_s=15)
IMAGE = "character.png"
IMAGE_COST = 0.0387
PNG_BYTES = PNG_MAGIC + b"character"


@dataclass
class SpyImage:
    """Writes a PNG-looking file at a price; `write=False` leaves nothing behind."""

    cost: float = IMAGE_COST
    write: bool = True
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, out_path: Path) -> ImageResult:
        self.prompts.append(prompt)
        if self.write:
            out_path.write_bytes(PNG_BYTES)
        return ImageResult(path=out_path, cost_usd=self.cost)


def _image_ctx(make_ctx: MakeCtx, **kwargs: Any) -> tuple[Context, SpyImage]:
    image = kwargs.pop("image", None) or SpyImage()
    video = kwargs.pop("video", None) or RecordingVideo(capabilities=NEEDS_IMAGE)
    return _ctx(make_ctx, video=video, image=image, **kwargs), image


def test_needs_image_makes_a_cartoon_character_image_and_sends_it(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx, image = _image_ctx(make_ctx, character="a tired office worker in a grey hoodie")
    out = shot_1.run(ctx)

    (prompt,) = image.prompts
    low = prompt.lower()
    assert "a tired office worker in a grey hoodie" in prompt
    for wanted in ("stylized 3d", "cartoon", "not a real person", "head and shoulders",
                   "mouth closed", "looking at the camera"):  # fmt: skip
        assert wanted in low, wanted
    assert "anthropomorphic" not in low  # a human character
    (request,) = _video(ctx).requests
    assert request.reference_image is not None
    assert Path(request.reference_image).resolve() == (ctx.run_dir / IMAGE).resolve()
    assert request.audio_path is not None  # still our audio
    assert "reference image" in request.prompt.lower()
    assert out.artifacts == {**SYNC_ARTIFACTS, "character_image": IMAGE}
    assert out.cost_usd == pytest.approx(IMAGE_COST + COST)


def test_image_prompt_forbids_logos_and_brands(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx, image = _image_ctx(make_ctx)
    shot_1.run(ctx)
    assert "no logos, brand names or readable text" in image.prompts[0].lower()


@pytest.mark.parametrize("character", ["a ginger cat with green eyes", "a green parrot pirate"])
def test_animal_character_is_asked_as_an_anthropomorphic_cartoon_with_a_mouth(
    make_ctx: MakeCtx, no_stdin: None, character: str
) -> None:
    """Live: a beak lip-synced worse than a mouth."""
    ctx, image = _image_ctx(make_ctx, character=character)
    shot_1.run(ctx)
    low = image.prompts[0].lower()
    assert "anthropomorphic" in low and "expressive mouth" in low
    assert "not a real person" in low  # the cartoon rules still hold


def test_existing_character_image_is_reused_no_second_image(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx, image = _image_ctx(make_ctx)
    (ctx.run_dir / IMAGE).write_bytes(PNG_BYTES + b"from an earlier attempt")
    out = shot_1.run(ctx)
    assert image.prompts == []
    assert out.cost_usd == pytest.approx(COST)
    assert _video(ctx).requests[0].reference_image is not None


def test_zero_byte_character_image_is_regenerated(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx, image = _image_ctx(make_ctx)
    (ctx.run_dir / IMAGE).write_bytes(b"")
    shot_1.run(ctx)
    assert len(image.prompts) == 1
    assert (ctx.run_dir / IMAGE).read_bytes() == PNG_BYTES


def test_image_provider_leaving_no_file_fails_gate_without_video(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx, _ = _image_ctx(make_ctx, image=SpyImage(write=False))
    with pytest.raises(GateFailed) as info:
        shot_1.run(ctx)
    assert "image" in str(info.value).lower()
    assert _video(ctx).requests == []


def test_image_cost_is_saved_before_the_video_and_the_retry_reuses_the_image(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    video = RecordingVideo(capabilities=NEEDS_IMAGE, errors=[ProviderError("busy")])
    ctx, image = _image_ctx(make_ctx, video=video)
    ctx.manifest.cost_usd = 0.1

    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)

    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert record.attempts == 2
    assert len(image.prompts) == 1  # the retry reused character.png
    assert record.cost_usd == pytest.approx(IMAGE_COST + COST)
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.cost_usd == pytest.approx(0.1 + IMAGE_COST + COST)  # counted once each
    assert record.artifacts["character_image"] == IMAGE


def test_image_cost_is_on_disk_when_the_video_fails_for_good(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    rejected = ProviderError("moderation: may contain real person", retryable=False)
    video = RecordingVideo(capabilities=NEEDS_IMAGE, errors=[rejected])
    ctx, _ = _image_ctx(make_ctx, video=video)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    assert result.stages["shot_1"].status is StageStatus.FAILED
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.stages["shot_1"].cost_usd == pytest.approx(IMAGE_COST)
    assert saved.cost_usd == pytest.approx(IMAGE_COST)


def test_no_image_needed_no_image_call(make_ctx: MakeCtx, no_stdin: None) -> None:
    image = SpyImage()
    ctx = _ctx(make_ctx, image=image)  # default caps: needs_image False
    out = shot_1.run(ctx)
    assert image.prompts == []
    assert _video(ctx).requests[0].reference_image is None
    assert "character_image" not in out.artifacts


# --- AC: money cap (HWVA_MAX_COST_USD) before every paid call ------------------------------------


def test_cost_cap_reached_stops_before_the_video_call(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    ctx.max_cost_usd = 1.25
    ctx.manifest.cost_usd = 1.25
    with pytest.raises(GateFailed) as info:
        shot_1.run(ctx)
    msg = str(info.value)
    assert "cost cap reached" in msg and "1.25" in msg and "HWVA_MAX_COST_USD" in msg
    assert _video(ctx).requests == []


def test_cost_cap_reached_stops_before_the_image_call(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx, image = _image_ctx(make_ctx)
    ctx.max_cost_usd = 0.5
    ctx.manifest.cost_usd = 0.6
    with pytest.raises(GateFailed, match="cost cap reached"):
        shot_1.run(ctx)
    assert image.prompts == [] and _video(ctx).requests == []


def test_image_cost_reaching_the_cap_stops_the_video_and_stays_saved(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx, image = _image_ctx(make_ctx)
    ctx.max_cost_usd = 1.0
    ctx.manifest.cost_usd = 0.98  # the image (0.0387) takes it over the cap
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and "cost cap reached" in record.error
    assert len(image.prompts) == 1 and _video(ctx).requests == []
    assert mf.load(ctx.manifest.run_id, runs_root).cost_usd == pytest.approx(0.98 + IMAGE_COST)


def test_below_the_cap_runs(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    ctx.max_cost_usd = 1.0
    ctx.manifest.cost_usd = 0.99
    shot_1.run(ctx)
    assert len(_video(ctx).requests) == 1


def test_default_cap_is_three_dollars(make_ctx: MakeCtx) -> None:
    assert _ctx(make_ctx).max_cost_usd == 3.0


# === Lip-sync check (specs/shot_1.md "Lip-sync check", specs/sync.md; v2) ==============
# SyncNet decides; phrases are aligned. `hwva.sync.syncnet_measure` is stubbed (conftest
# `install_syncnet_measure`, answers keyed by the audio file name, or "<video>|<audio>" for takes);
# phrases / motion / render run for real on synthetic media (real ffmpeg). The stage must look the
# functions up at call time (`from hwva import sync` ... `sync.syncnet_measure(...)`).
#
# Replaced v1 tests (specs/shot_1.md v2 removed `lag_s`/`peak_r` from the record and "peak r" from
# the messages): the old `sync.measure` stubs (`StubMeasure`, `IN_SYNC`/`LAGGING`/`NO_SYNC`...).

SHOT_AUDIO = "shot_audio.wav"
LEAD_IN = 0.7
TAKE_2 = "shot_1.take2.mp4"
SYNC_JSON = "sync.json"
GLOBAL = "sync_global.wav"
PIECEWISE = "sync_piecewise.wav"
ALIGNED = "aligned_line.wav"
ALIGNED_ARTIFACTS = {**SYNC_ARTIFACTS, "aligned_audio": ALIGNED}
DEFAULT_BOX = [0.20, 0.80, 0.42, 0.60]
# shot_audio.wav in clip time: a 0.7 s lead-in, three speech runs, pauses of 0.3 s and 0.2 s
CLIP_PHRASES = [(0.7, 1.5), (1.8, 2.4), (2.6, 3.3)]
SHOT_AUDIO_S = 3.3
LINE_S = SHOT_AUDIO_S - LEAD_IN
CLIP_S = 5.0  # max(shot 5.0, lead-in + line 3.3)
RATE = 24000  # speech_wav default


def _lead_ctx(
    make_ctx: MakeCtx,
    *,
    lead_in: float = LEAD_IN,
    audio_s: float = 2.0,
    shots: list[Shot] | None = None,
    video: Any = None,
    **overrides: Any,
) -> Context:
    """Context after a `voice` with the lead-in: line.wav + shot_audio.wav (lead-in + line) on
    disk, voice data `lead_in_s` / `shot_audio_duration_s`, artifact `shot_audio`."""
    shots = shots or _shots(5, 4)
    ctx = _ctx(make_ctx, audio_s=audio_s, shots=shots, video=video, **overrides)
    sent = lead_in + audio_s
    make_audio(ctx.run_dir / SHOT_AUDIO, sent)  # a plain tone: one long "phrase"
    planned = max(shots[0].duration_s, sent) + sum(s.duration_s for s in shots[1:])
    voice = VoiceData(
        audio_duration_s=audio_s, planned_total_s=planned, attempts=1,
        lead_in_s=lead_in, shot_audio_duration_s=sent,
    )  # fmt: skip
    rec = ctx.manifest.stages["voice"]
    rec.artifacts = {"audio": AUDIO, "transcript": "transcript.txt", "shot_audio": SHOT_AUDIO}
    rec.data = voice.model_dump(mode="json")
    return ctx


def _v2_ctx(make_ctx: MakeCtx, *, video: Any = None, **overrides: Any) -> Context:
    """`_lead_ctx` whose shot_audio.wav has real phrases (CLIP_PHRASES), so the piecewise
    alignment has something to work on."""
    ctx = _lead_ctx(make_ctx, audio_s=LINE_S, video=video, **overrides)
    speech_wav(ctx.run_dir / SHOT_AUDIO, CLIP_PHRASES, SHOT_AUDIO_S)
    return ctx


def _answers(
    raw: SyncNetAnswer,
    global_: SyncNetAnswer = (0, 9.0),
    piecewise: SyncNetAnswer = (0, 8.0),
) -> dict[str, SyncNetAnswer]:
    return {SHOT_AUDIO: raw, GLOBAL: global_, PIECEWISE: piecewise}


@dataclass
class TakesVideo(RecordingVideo):
    """RecordingVideo that raises the given error on the given call numbers (1-based) only."""

    fail_on: dict[int, ProviderError] = field(default_factory=dict)
    calls: int = 0

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        self.calls += 1
        if self.calls in self.fail_on:
            self.requests.append(request)
            self.out_paths.append(out_path)
            raise self.fail_on[self.calls]
        return super().generate(request, out_path)


def _sync_data(data: dict[str, Any]) -> SyncReport:
    assert data.get("sync") is not None, "shot_1 data has no `sync` record"
    return SyncReport.model_validate(data["sync"])


def _sync_json(ctx: Context) -> SyncReport:
    path = ctx.run_dir / SYNC_JSON
    assert path.is_file(), f"{SYNC_JSON} not written"
    return SyncReport.model_validate_json(path.read_text(encoding="utf-8"))


def _names(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def _lag_s(track: Path, reference: Path) -> float:
    """Where `reference`'s content sits in `track` (same sample rate): track[t] ~ ref[t - lag]."""
    a, rate_a, _ = read_wav(track)
    b, rate_b, _ = read_wav(reference)
    assert rate_a == rate_b
    x, y = a.astype(np.float64), b.astype(np.float64)
    size = 1 << int(len(x) + len(y)).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size)), size)
    i = int(np.argmax(corr))
    return (i - size if i > size // 2 else i) / rate_a


def _raw_calls(stub: StubSyncNetMeasure) -> list[str]:
    """Video names of the raw measurements (SyncNet on the audio sent to the model)."""
    return [v.name for v, a in stub.calls if a.name in (SHOT_AUDIO, AUDIO)]


# --- AC: request carries shot_audio.wav, 9:16, duration = max(shot, lead-in + audio) -------------
# Unchanged criteria (not xfail): no stub needed, the measurement outcome does not matter here.


@pytest.mark.parametrize(
    ("shot_s", "audio_s", "expected"),
    [(5.0, 2.0, 5.0), (5.0, 4.6, 5.3), (5.0, 6.5, 7.2)],
    ids=["shot_longer", "lead_in_tips_it", "audio_longer"],
)
def test_request_carries_shot_audio_and_duration_includes_lead_in(
    make_ctx: MakeCtx, no_stdin: None, shot_s: float, audio_s: float, expected: float
) -> None:
    ctx = _lead_ctx(make_ctx, audio_s=audio_s, shots=_shots(shot_s, 4))
    out = shot_1.run(ctx)
    (req,) = _video(ctx).requests
    assert req.audio_path is not None
    assert Path(req.audio_path).resolve() == (ctx.run_dir / SHOT_AUDIO).resolve()
    assert req.aspect_ratio == "9:16"
    assert req.duration_s == pytest.approx(expected, abs=0.01)
    assert _data(out).duration_s == pytest.approx(expected, abs=0.01)
    assert duration(ctx.run_dir / VIDEO_NAME) == pytest.approx(expected, abs=0.1)


def test_old_run_without_shot_audio_sends_line_wav(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, audio_s=6.5)  # no shot_audio artifact, no lead_in_s
    shot_1.run(ctx)
    (req,) = _video(ctx).requests
    assert req.audio_path is not None
    assert Path(req.audio_path).resolve() == (ctx.run_dir / AUDIO).resolve()
    assert req.duration_s == pytest.approx(6.5, abs=0.01)  # max(5, 0 + 6.5)


def test_old_run_without_shot_audio_measures_line_wav(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run from before the lead-in: line.wav is the audio sent, so SyncNet measures it."""
    ctx = _ctx(make_ctx, audio_s=6.5)
    stub = install_syncnet_measure(monkeypatch, {AUDIO: (0, 8.0), PIECEWISE: (0, 7.0)})
    out = shot_1.run(ctx)
    video, audio = stub.calls[0]
    assert (video.name, audio.name) == (VIDEO_NAME, AUDIO)
    assert _sync_data(out.data).verdict == "pass"


# --- AC: sync.json + data `sync` with the v2 fields; sync.json equals data `sync` ----------------


def test_sync_is_measured_by_syncnet_and_recorded(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    frame_ready: list[bool] = []
    stub = install_syncnet_measure(monkeypatch, _answers((-6, 8.63, 7.1), (1, 9.0), (0, 9.2)))
    real_call = stub.__call__

    def watched(video: Path, audio: Path) -> Any:
        frame_ready.append((ctx.run_dir / FRAME_NAME).is_file())
        return real_call(video, audio)

    monkeypatch.setattr(sync, "syncnet_measure", watched, raising=False)
    out = shot_1.run(ctx)

    video, audio = stub.calls[0]  # the raw measurement comes first
    assert (video.name, audio.name) == (VIDEO_NAME, SHOT_AUDIO)
    assert frame_ready and frame_ready[0], "SyncNet ran before the last frame was extracted"
    assert sorted(stub.audio_names()) == sorted([SHOT_AUDIO, GLOBAL, PIECEWISE])
    assert set(stub.video_names()) == {VIDEO_NAME}
    assert out.artifacts == ALIGNED_ARTIFACTS

    record = _sync_data(out.data)
    assert record == _sync_json(ctx)  # same content
    assert record.verdict == "correct"
    assert record.syncnet_confidence == 8.63
    assert record.syncnet_offset_frames == -6 and record.syncnet_min_dist == 7.1
    assert record.shift_s == pytest.approx(-0.24, abs=1e-9)
    assert record.thresholds is not None
    assert (record.thresholds.sync_conf, record.thresholds.weak_conf) == (5.0, 3.0)
    assert record.method == "piecewise"
    assert (record.confidence_global, record.confidence_piecewise) == (9.0, 9.2)
    assert record.piecewise_error is None
    assert len(record.phrases) == 3
    assert record.aligned_audio == ALIGNED
    assert record.box == pytest.approx(DEFAULT_BOX)
    assert record.take == 1 and record.rejected is None
    assert record.error is None and record.warning is None
    assert (record.lag_s, record.peak_r, record.r_at_0) == (None, None, None)  # v1 fields unused
    assert len(_video(ctx).requests) == 1


# --- AC: confidence 8.6, offset -6 -> correct, shift -0.24; sync_global.wav 0.24 s earlier --------


def test_correct_offset_moves_the_global_track_earlier(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6)))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert record.verdict == "correct" and record.shift_s == pytest.approx(-0.24, abs=1e-9)
    track = ctx.run_dir / GLOBAL
    assert track.is_file()
    assert _lag_s(track, ctx.run_dir / SHOT_AUDIO) == pytest.approx(-0.24, abs=0.01)
    assert duration(track) == pytest.approx(duration(ctx.run_dir / VIDEO_NAME), abs=0.05)


# --- AC: piecewise kept only when its confidence >= the global one (equal -> piecewise) ----------


@pytest.mark.parametrize(
    ("piecewise_conf", "method", "chosen"),
    [(9.2, "piecewise", PIECEWISE), (9.0, "piecewise", PIECEWISE), (8.0, "global", GLOBAL)],
    ids=["piecewise_better", "equal", "global_better"],
)
def test_the_better_candidate_becomes_aligned_line_wav(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch,
    piecewise_conf: float, method: str, chosen: str,
) -> None:  # fmt: skip
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), (1, 9.0), (0, piecewise_conf)))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert record.method == method
    assert (record.confidence_global, record.confidence_piecewise) == (9.0, piecewise_conf)
    assert (ctx.run_dir / GLOBAL).is_file() and (ctx.run_dir / PIECEWISE).is_file()
    aligned = ctx.run_dir / ALIGNED
    assert aligned.read_bytes() == (ctx.run_dir / chosen).read_bytes()
    assert out.artifacts["aligned_audio"] == ALIGNED and record.aligned_audio == ALIGNED


# --- AC: pass / no_sync -> shift 0, global confidence = raw, no SyncNet run on sync_global.wav ----


@pytest.mark.parametrize(
    ("raw", "verdict"), [((1, 8.0), "pass"), ((-1, 6.0), "pass"), ((-7, 1.7), "no_sync")]
)
def test_zero_shift_reuses_the_raw_confidence(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch,
    raw: tuple[int, float], verdict: str,
) -> None:  # fmt: skip
    ctx = _v2_ctx(make_ctx)
    stub = install_syncnet_measure(monkeypatch, _answers(raw, (5, 99.0), (0, 0.5)))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert record.verdict == verdict and record.shift_s == 0.0
    assert record.confidence_global == raw[1]  # the raw one, not the 99.0 a re-run would give
    assert GLOBAL not in stub.audio_names(), "SyncNet must not re-measure an unshifted track"
    track = ctx.run_dir / GLOBAL
    assert track.is_file()  # still rendered (shift 0)
    assert _lag_s(track, ctx.run_dir / SHOT_AUDIO) == pytest.approx(0.0, abs=0.01)
    assert record.method == "global"  # piecewise 0.5 < raw
    assert (ctx.run_dir / ALIGNED).read_bytes() == track.read_bytes()


# --- AC: piecewise SyncNet run raising SyncError -> global, piecewise_error set, DONE -------------


def test_piecewise_syncnet_failure_falls_back_to_global(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    error = sync.SyncError("SyncNet: cv2 exploded on sync_piecewise")
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), (1, 9.0), error))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    report = _sync_data(record.data)
    assert report.method == "global"
    assert report.piecewise_error and "cv2 exploded" in report.piecewise_error
    assert report.confidence_global == 9.0 and report.confidence_piecewise is None
    assert (ctx.run_dir / ALIGNED).read_bytes() == (ctx.run_dir / GLOBAL).read_bytes()
    assert record.artifacts["aligned_audio"] == ALIGNED


def test_global_syncnet_failure_keeps_global_with_a_null_confidence(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sync.md step 4: a SyncError in step 2's SyncNet run -> global with confidence_global null,
    piecewise not chosen (even with a high confidence), the reason recorded."""
    ctx = _v2_ctx(make_ctx)
    error = sync.SyncError("SyncNet: decoder died on the global track")
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), error, (0, 9.9)))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert record.method == "global"
    assert record.confidence_global is None
    assert (ctx.run_dir / ALIGNED).read_bytes() == (ctx.run_dir / GLOBAL).read_bytes()
    assert "decoder died on the global track" in record.model_dump_json()  # the reason


def test_global_render_failure_leaves_no_aligned_audio(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """shot_1.md: a failure of the global render itself -> no aligned_audio, method null, a
    warning, the stage still succeeds (assemble then uses lead_in_s + shift_s)."""
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6)))
    real_render = sync.render

    def render(audio: Path, phrases: Any, shifts: Any, duration_s: float, out: Path) -> None:
        if Path(out).name == GLOBAL:
            raise sync.SyncError("render: disk full")
        real_render(audio, phrases, shifts, duration_s, out)

    monkeypatch.setattr(sync, "render", render)
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    report = _sync_data(record.data)
    assert report.verdict == "correct" and report.shift_s == pytest.approx(-0.24, abs=1e-9)
    assert report.method is None and report.aligned_audio is None
    assert "aligned_audio" not in record.artifacts
    assert not (ctx.run_dir / ALIGNED).exists()
    assert report.warning and "\n" not in report.warning


# --- AC: phrases in clip time, a shift each; aligned_line.wav lasts the clip ----------------------


def test_phrases_are_recorded_and_the_aligned_track_lasts_the_clip(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), (1, 9.0), (0, 9.5)))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert len(record.phrases) == 3, record.phrases
    for phrase, (start, end) in zip(record.phrases, CLIP_PHRASES, strict=True):
        assert phrase.start_s == pytest.approx(start, abs=0.03)  # clip time, lead-in included
        assert phrase.end_s == pytest.approx(end, abs=0.03)
        assert abs(phrase.shift_s) <= 0.5 + 1e-9
        assert phrase.shift_s == round(phrase.shift_s, 3)  # 3 decimals in the record
    clip_s = duration(ctx.run_dir / VIDEO_NAME)
    assert clip_s == pytest.approx(CLIP_S, abs=0.1)
    assert duration(ctx.run_dir / ALIGNED) == pytest.approx(clip_s, abs=0.05)
    _, rate, channels = read_wav(ctx.run_dir / ALIGNED)
    assert (rate, channels) == (RATE, 1)  # shot_audio's own rate, mono: never resampled


# --- AC: weak -> DONE, warning, shift applied; no_sync -> DONE, warning, alignment recorded -------


def test_weak_is_a_warning_and_still_corrected(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 4.0), (0, 4.4), (0, 4.2)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    report = _sync_data(record.data)
    assert report.verdict == "weak"
    assert report.shift_s == pytest.approx(-0.24, abs=1e-9)  # offset / 25
    assert report.warning and "\n" not in report.warning and "4" in report.warning
    assert report.method == "global" and record.artifacts["aligned_audio"] == ALIGNED
    assert len(_video(ctx).requests) == 1


def test_no_sync_is_a_warning_and_the_alignment_still_runs(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-7, 1.71), piecewise=(0, 2.95)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert len(_video(ctx).requests) == 1
    report = _sync_data(record.data)
    assert report.verdict == "no_sync" and report.shift_s == 0.0
    assert report.warning and "\n" not in report.warning
    assert "1.71" in report.warning  # names the SyncNet confidence
    assert report.method == "piecewise"  # 2.95 >= the raw 1.71
    assert report.aligned_audio == ALIGNED and record.artifacts["aligned_audio"] == ALIGNED
    assert record.artifacts["video"] == VIDEO_NAME
    assert _sync_json(ctx) == report


# --- AC: raw SyncError / NoFaceError -> DONE, error, warning, no alignment; also with reject ------


@pytest.mark.parametrize("reject", [None, "1"], ids=["switch_off", "switch_on"])
@pytest.mark.parametrize("kind", ["SyncError", "NoFaceError"])
def test_measurement_error_is_recorded_and_never_rejected(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch,
    reject: str | None, kind: str,
) -> None:  # fmt: skip
    if reject is not None:
        monkeypatch.setenv("HWVA_SYNC_REJECT", reject)
    ctx = _v2_ctx(make_ctx)
    error = getattr(sync, kind)("no face found by SyncNet")
    stub = install_syncnet_measure(monkeypatch, _answers(error))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert len(_video(ctx).requests) == 1  # a local tool failure never pays for a new clip
    assert stub.audio_names() == [SHOT_AUDIO]  # error -> stop: no alignment runs
    report = _sync_data(record.data)
    assert report.verdict == "error"
    assert report.syncnet_confidence is None and report.syncnet_offset_frames is None
    assert report.error and "no face found" in report.error
    assert report.warning and "\n" not in report.warning
    assert report.shift_s == 0.0 and report.take == 1
    assert report.method is None and report.aligned_audio is None and report.phrases == []
    assert record.artifacts == SYNC_ARTIFACTS  # no aligned_audio
    assert not (ctx.run_dir / ALIGNED).exists()
    assert _sync_json(ctx) == report


# --- AC: invalid thresholds -> GateFailed naming the variable, before any paid call ---------------


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({"HWVA_SYNC_CONF": "abc"}, ("HWVA_SYNC_CONF",)),
        ({"HWVA_SYNC_WEAK_CONF": "-1"}, ("HWVA_SYNC_WEAK_CONF",)),
        ({"HWVA_SYNC_CONF": "2"}, ("HWVA_SYNC_CONF", "HWVA_SYNC_WEAK_CONF")),  # weak 3 > sync 2
    ],
    ids=["conf-abc", "weak-negative", "weak-gt-sync"],
)
def test_invalid_thresholds_fail_before_any_paid_call(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str], named: tuple[str, ...],
) -> None:  # fmt: skip
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    ctx = _v2_ctx(make_ctx)
    with pytest.raises(GateFailed) as err:
        shot_1.run(ctx)
    assert any(name in str(err.value) for name in named), str(err.value)
    assert _video(ctx).requests == []


def test_thresholds_from_env_are_used_and_recorded(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_CONF", "6")
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 5.5)))  # correct with the default 5
    record = _sync_data(shot_1.run(ctx).data)
    assert record.verdict == "weak"
    assert record.thresholds is not None
    assert (record.thresholds.sync_conf, record.thresholds.weak_conf) == (6.0, 3.0)


def test_verdict_uses_the_raw_confidence_and_the_record_rounds_it(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((0, 4.99964, 7.12345)))
    record = _sync_data(shot_1.run(ctx).data)
    assert record.verdict == "weak"  # 4.99964 < 5 ...
    assert record.syncnet_confidence == 5.0  # ... though the record shows 5.0 (3 decimals)
    assert record.syncnet_min_dist == 7.123


# --- motion metric: diagnostic only (recorded as `motion`; its failure changes nothing) -----------


@dataclass
class MotionStub:
    outcome: Any  # a MotionResult-like answer or an exception to raise
    boxes: list[Any] = field(default_factory=list)

    def __call__(self, video: Path, audio: Path, *, box: Any = None) -> Any:
        self.boxes.append(box)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return sync.MotionResult(**self.outcome)


def test_motion_is_recorded_with_the_box_from_env(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_MOUTH_BOX", "0.1,0.9,0.3,0.7")
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6)))
    motion = MotionStub({"lag_s": 0.25, "peak_r": 0.36, "r_at_0": 0.1})
    monkeypatch.setattr(sync, "measure", motion)
    record = _sync_data(shot_1.run(ctx).data)
    assert record.motion is not None
    assert (record.motion.lag_s, record.motion.peak_r, record.motion.r_at_0) == (0.25, 0.36, 0.1)
    assert motion.boxes and tuple(motion.boxes[0]) == pytest.approx((0.1, 0.9, 0.3, 0.7))
    assert record.box == pytest.approx([0.1, 0.9, 0.3, 0.7])
    assert record.shift_s == pytest.approx(-0.24, abs=1e-9)  # SyncNet decides, not the motion lag


def test_motion_failure_leaves_motion_null_and_changes_nothing(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), (1, 9.0), (0, 9.5)))
    monkeypatch.setattr(sync, "measure", MotionStub(sync.SyncError("mouth crop timed out")))
    out = shot_1.run(ctx)
    record = _sync_data(out.data)
    assert record.motion is None
    assert record.verdict == "correct" and record.error is None and record.warning is None
    assert record.method == "piecewise" and out.artifacts == ALIGNED_ARTIFACTS


# --- AC: HWVA_SYNC_REJECT=1, take 1 no_sync, take 2 correct -> two calls, take 2 kept & aligned ---


def _takes(take_1: SyncNetAnswer, take_2: SyncNetAnswer) -> dict[str, SyncNetAnswer]:
    return {
        f"{VIDEO_NAME}|{SHOT_AUDIO}": take_1,
        f"{TAKE_2}|{SHOT_AUDIO}": take_2,
        GLOBAL: (0, 9.0),
        PIECEWISE: (0, 8.0),
    }


def test_reject_regenerates_once_into_take_2_and_aligns_take_2(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    ctx = _v2_ctx(make_ctx)
    stub = install_syncnet_measure(monkeypatch, _takes((-7, 1.7), (-6, 8.6)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error

    video = _video(ctx)
    assert _names(video.out_paths) == [VIDEO_NAME, TAKE_2]  # a new path = a new provider task
    assert video.requests[1] == video.requests[0]  # the same request
    assert record.artifacts["video"] == TAKE_2
    assert record.artifacts["aligned_audio"] == ALIGNED
    assert (ctx.run_dir / record.artifacts["shot_1_last_frame"]).is_file()
    assert _raw_calls(stub) == [VIDEO_NAME, TAKE_2]  # both takes measured
    aligned_videos = {v.name for v, a in stub.calls if a.name in (GLOBAL, PIECEWISE)}
    assert aligned_videos == {TAKE_2}, "only the kept take is aligned"

    report = _sync_data(record.data)
    assert report.take == 2 and report.verdict == "correct"
    assert report.shift_s == pytest.approx(-0.24, abs=1e-9) and report.method == "global"
    assert report.rejected is not None
    assert report.rejected.verdict == "no_sync" and report.rejected.syncnet_confidence == 1.7
    assert report.rejected.take == 1 and report.rejected.rejected is None
    assert report.rejected.method is None and report.rejected.aligned_audio is None
    assert _sync_json(ctx) == report

    assert record.cost_usd == pytest.approx(2 * COST)  # both clips
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.cost_usd == pytest.approx(2 * COST)


def test_reject_switch_on_and_in_sync_makes_one_call(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((0, 8.0)))
    out = shot_1.run(ctx)
    assert len(_video(ctx).requests) == 1
    assert out.artifacts["video"] == VIDEO_NAME
    report = _sync_data(out.data)
    assert report.take == 1 and report.verdict == "pass" and report.rejected is None


# --- AC: HWVA_SYNC_REJECT=1, both takes no_sync -> FAILED naming both SyncNet confidences ---------


def test_reject_two_takes_without_sync_fails_naming_both_confidences(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _takes((-7, 1.7), (3, 1.2)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert "no lip-sync after 2 takes" in record.error
    assert "1.7" in record.error and "1.2" in record.error
    assert "peak r" not in record.error  # v1 wording is gone
    assert _names(_video(ctx).out_paths) == [VIDEO_NAME, TAKE_2]  # never a third clip

    report = _sync_json(ctx)  # written before GateFailed
    assert report.take == 2 and report.verdict == "no_sync" and report.syncnet_confidence == 1.2
    assert report.rejected is not None
    assert report.rejected.take == 1 and report.rejected.syncnet_confidence == 1.7
    saved = mf.load(ctx.manifest.run_id, runs_root)
    assert saved.cost_usd == pytest.approx(2 * COST)  # each clip committed right after its call


# --- AC: HWVA_SYNC_REJECT=1, take 1 weak -> one call, DONE (weak is never rejected) ---------------


def test_reject_never_rejects_weak(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 3.4)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert len(_video(ctx).requests) == 1
    report = _sync_data(record.data)
    assert report.verdict == "weak" and report.take == 1 and report.rejected is None


# --- AC: switch off (unset / "0" / anything but "1") and no_sync -> one call, DONE ----------------


@pytest.mark.parametrize("value", [None, "0", "", "true"], ids=["unset", "0", "empty", "true"])
def test_reject_switch_off_unless_exactly_one(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is not None:
        monkeypatch.setenv("HWVA_SYNC_REJECT", value)
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-7, 1.7)))
    out = shot_1.run(ctx)
    assert len(_video(ctx).requests) == 1
    report = _sync_data(out.data)
    assert report.verdict == "no_sync" and report.syncnet_confidence == 1.7
    assert out.artifacts["video"] == VIDEO_NAME


# --- AC: HWVA_SYNC_REJECT=1, take 1 no_sync and the cost cap reached -> no second call -----------


def test_reject_stops_at_the_cost_cap(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    ctx = _v2_ctx(make_ctx)
    ctx.max_cost_usd = COST  # take 1 reaches it
    stub = install_syncnet_measure(monkeypatch, _answers((-7, 1.7)))
    result = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and "cost cap reached" in record.error, record.error
    assert len(_video(ctx).requests) == 1
    assert _raw_calls(stub) == [VIDEO_NAME]  # take 1 was measured (that is what asked for take 2)
    assert mf.load(ctx.manifest.run_id, runs_root).cost_usd == pytest.approx(COST)


# --- Behaviour: resume of a stage that failed mid-take-2 -----------------------------------------


def test_resume_after_take_2_failed_asks_for_the_same_two_clips(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Take 1 is asked for at its own path again (the provider reuses its task: no new payment),
    measured again (same verdict), then take 2 at its own path."""
    monkeypatch.setenv("HWVA_SYNC_REJECT", "1")
    video = TakesVideo(fail_on={2: ProviderError("moderation: rejected", retryable=False)})
    ctx = _v2_ctx(make_ctx, video=video)
    stub = install_syncnet_measure(monkeypatch, _takes((-7, 1.7), (-6, 8.6)))
    first = pipeline.run(ctx, [_shot_1_stage()], root=runs_root)
    assert first.stages["shot_1"].status is StageStatus.FAILED
    assert _names(video.out_paths) == [VIDEO_NAME, TAKE_2]

    resumed = Context(mf.load(ctx.manifest.run_id, runs_root), ctx.providers, ctx.run_dir, False)
    result = pipeline.run(resumed, [_shot_1_stage()], root=runs_root)
    record = result.stages["shot_1"]
    assert record.status is StageStatus.DONE, record.error
    assert _names(video.out_paths) == [VIDEO_NAME, TAKE_2, VIDEO_NAME, TAKE_2]
    assert _raw_calls(stub) == [VIDEO_NAME, VIDEO_NAME, TAKE_2]
    assert record.artifacts["video"] == TAKE_2
    report = _sync_data(record.data)
    assert report.take == 2 and report.rejected is not None
    assert report.rejected.verdict == "no_sync"


def test_same_syncnet_answers_give_the_same_aligned_track(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic given the same SyncNet answers, so a resume gives the same result."""
    ctx = _v2_ctx(make_ctx)
    install_syncnet_measure(monkeypatch, _answers((-6, 8.6), (1, 9.0), (0, 9.5)))
    first = _sync_data(shot_1.run(ctx).data)
    first_bytes = (ctx.run_dir / ALIGNED).read_bytes()
    second = _sync_data(shot_1.run(ctx).data)
    assert second == first
    assert (ctx.run_dir / ALIGNED).read_bytes() == first_bytes


# --- the record's warning comes from the one builder, `hwva.sync.warning_text` --------------------

WARNING_SENTINEL = "SENTINEL: built by hwva.sync.warning_text"


@pytest.mark.parametrize("case", ["weak", "no_sync", "error"])
def test_warning_comes_from_sync_warning_text(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    ctx = _v2_ctx(make_ctx)
    raw: SyncNetAnswer = {
        "weak": (-6, 4.0),
        "no_sync": (-7, 1.7),
        "error": sync.SyncError("no face found"),
    }[case]
    stub = install_syncnet_measure(monkeypatch, _answers(raw))
    built: list[tuple[Any, ...]] = []

    def warning_text(*args: Any, **kwargs: Any) -> str:
        built.append((*args, *kwargs.values()))
        return WARNING_SENTINEL

    monkeypatch.setattr(sync, "warning_text", warning_text)
    report = _sync_data(shot_1.run(ctx).data)
    assert stub.audio_names()[:1] == [SHOT_AUDIO], "the verdict comes from SyncNet"
    assert report.verdict == case
    assert built and built[0][0] == case, "shot_1 did not call hwva.sync.warning_text"
    assert report.warning == WARNING_SENTINEL
    assert _sync_json(ctx).warning == WARNING_SENTINEL
