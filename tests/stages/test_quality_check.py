"""Acceptance tests for the `quality_check` stage (specs/quality_check.md). Real ffmpeg, no network.

The judge is faked (conftest: FakeJudge, CrashingJudge, InvalidJudge, HallucinatingJudge).
SyncNet is never really run here: conftest's autouse `no_real_syncnet` makes `syncnet_python`
unimportable, and the tests that need it install a fake module in `sys.modules` (imported lazily
by the shared runner `hwva.sync.syncnet_measure` since v2) or monkeypatch that runner."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import logging
import re
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    CrashingJudge,
    FakeJudge,
    HallucinatingJudge,
    InvalidJudge,
    install_syncnet_measure,
    judge_verdict,
    make_audio,
    make_video,
    needs_ffmpeg,
)
from pydantic import ValidationError

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import (
    AssembleData,
    JudgeVerdict,
    LineSource,
    QualityReport,
    Script,
    Shot,
    ShotData,
    StageStatus,
    SyncReport,
)
from hwva.stages import quality_check

pytestmark = [
    needs_ffmpeg,
]

MakeCtx = Callable[..., Context]

LINE = "Я тут головний"
FINAL = "final.mp4"
AUDIO = "line.wav"
SCRIPT = "script.json"
QUALITY = "quality.json"
VIDEO_S = 9.0  # final.mp4 length: shots 5 + 4
NO_JUDGE = "no judge in profile"
WEIGHT_FILES = ("syncnet_v2.model", "sfd_face.pth")


# --- context ------------------------------------------------------------------------------------


def _script() -> Script:
    return Script(
        title="test",
        line=LINE,
        line_source=LineSource.INPUT,
        lang="uk",
        character="a ginger cat with green eyes",
        shots=[
            Shot(visual_prompt="cat speaking", duration_s=5.0, has_line=True),
            Shot(visual_prompt="cat reacting", duration_s=4.0),
        ],
    )


def _ctx(
    make_ctx: MakeCtx,
    *,
    with_final: bool = True,
    with_script: bool = True,
    **providers: Any,
) -> Context:
    """Context as it is after `final_check` passed: script.json, line.wav, final.mp4 (9 s)."""
    params = {"line": LINE, "line_source": LineSource.INPUT.value, "lang": "uk"}
    ctx = make_ctx(idea=f'Кіт каже: "{LINE}"', params=params, **providers)
    run_dir = ctx.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    if with_script:
        (run_dir / SCRIPT).write_text(_script().model_dump_json(indent=2), encoding="utf-8")
        rec = mf.stage(ctx.manifest, "script")
        rec.status, rec.artifacts = StageStatus.DONE, {"script": SCRIPT}

    make_audio(run_dir / AUDIO, 3.0, text=LINE)
    rec = mf.stage(ctx.manifest, "voice")
    rec.status, rec.artifacts = StageStatus.DONE, {"audio": AUDIO}

    if with_final:
        make_video(run_dir / FINAL, VIDEO_S, audio=True)
        rec = mf.stage(ctx.manifest, "assemble")
        rec.status, rec.artifacts = StageStatus.DONE, {"video": FINAL}
        rec.data = AssembleData(duration_s=VIDEO_S).model_dump(mode="json")
    rec = mf.stage(ctx.manifest, "final_check")
    rec.status = StageStatus.DONE
    return ctx


def _run(ctx: Context) -> QualityReport:
    """Run the stage; it must not raise, must write quality.json, and data must equal the file."""
    out = quality_check.run(ctx)
    assert isinstance(out, StageOutput)
    path = ctx.run_dir / QUALITY
    assert path.is_file(), f"{QUALITY} not written"
    assert QUALITY in out.artifacts.values(), out.artifacts
    report = QualityReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert QualityReport.model_validate(out.data) == report, "data must be the same as quality.json"
    return report


def _stage() -> pipeline.Stage:
    stage = next((s for s in pipeline.STAGES if s.name == "quality_check"), None)
    assert stage is not None, "quality_check is not registered in pipeline.STAGES"
    return stage


# --- SyncNet fakes ------------------------------------------------------------------------------


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """$HWVA_SYNCNET_WEIGHTS pointing to a dir with both weight files present."""
    d = tmp_path / "weights"
    d.mkdir()
    for name in WEIGHT_FILES:
        (d / name).write_bytes(b"fake weights")
    monkeypatch.setenv("HWVA_SYNCNET_WEIGHTS", str(d))
    return d


NO_FACE: tuple[Any, ...] = ([], [], [], 0.0, 0.0, "", False)  # what inference returns, no face
FACE: tuple[Any, ...] = ([-6], [4.74], [7.12], 4.74, 7.12, "{}", True)


class SyncNetCalls:
    def __init__(self) -> None:
        self.init: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.inference: list[tuple[tuple[Any, ...], dict[str, Any]]] = []


def _install_syncnet(
    monkeypatch: pytest.MonkeyPatch,
    result: tuple[Any, ...] = NO_FACE,
    exc: Exception | None = None,
) -> SyncNetCalls:
    """Put a fake `syncnet_python` (with SyncNetPipeline) into sys.modules."""
    calls = SyncNetCalls()

    class SyncNetPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.init.append((args, kwargs))

        def inference(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            calls.inference.append((args, kwargs))
            if exc is not None:
                raise exc
            return result

    module = types.ModuleType("syncnet_python")
    module.SyncNetPipeline = SyncNetPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "syncnet_python", module)
    return calls


def _strings(value: Any) -> list[str]:
    """All str/Path leaves of nested args (weights may be passed as kwargs or as a cfg dict)."""
    if isinstance(value, str | Path):
        return [str(value)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


def _inference_paths(call: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[Path, Path]:
    args, kwargs = call
    video = kwargs.get("video_path", args[0] if args else None)
    audio = kwargs.get("audio_path", args[1] if len(args) > 1 else None)
    assert video is not None and audio is not None, call
    return Path(video).resolve(), Path(audio).resolve()


# --- AC: never raises GateFailed; always writes quality.json ------------------------------------


def test_good_judge_verdict_is_recorded(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = FakeJudge(judge_verdict("00:03"))
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict == judge.verdict
    assert report.judge.hallucinated_defects == 0
    assert report.judge.error is None and report.judge.skipped is None


def test_judge_gets_final_video_and_script(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = FakeJudge()
    ctx = _ctx(make_ctx, judge=judge)
    _run(ctx)
    assert len(judge.calls) == 1, "judge is called exactly once, on the final video"
    video, script = judge.calls[0]
    assert Path(video).resolve() == (ctx.run_dir / FINAL).resolve()
    assert script == _script()  # character + shot prompts + number of shots from script.json


def test_all_fail_verdict_does_not_block(make_ctx: MakeCtx, no_stdin: None) -> None:
    """The judge says everything is wrong: recorded as-is, still no GateFailed."""
    judge = FakeJudge(judge_verdict("00:01", "00:04", verdict="fail", score=1))
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict == judge.verdict


def test_all_fail_verdict_stage_is_done_in_pipeline(
    make_ctx: MakeCtx, runs_root: Path, no_stdin: None
) -> None:
    """Registered last in STAGES; through the runner a damning verdict still ends DONE."""
    assert pipeline.STAGES[-1].name == "quality_check", "quality_check must run last"
    ctx = _ctx(make_ctx, judge=FakeJudge(judge_verdict(verdict="fail", score=1)))
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = mf.load(result.run_id, runs_root).stages["quality_check"]
    assert record.status is StageStatus.DONE, record.error
    assert QUALITY in record.artifacts.values()
    saved = QualityReport.model_validate_json((ctx.run_dir / QUALITY).read_text(encoding="utf-8"))
    assert QualityReport.model_validate(record.data) == saved


def test_missing_final_video_still_writes_quality(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, with_final=False, judge=FakeJudge())
    _run(ctx)


def test_missing_script_records_judge_error(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = FakeJudge()
    ctx = _ctx(make_ctx, with_script=False, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict is None
    assert report.judge.error or report.judge.skipped


# --- AC: missing judge -> skipped record --------------------------------------------------------


def test_no_judge_in_profile_is_skipped(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)  # dev profile: judge is None
    assert ctx.providers.judge is None
    report = _run(ctx)
    assert report.judge.skipped == NO_JUDGE
    assert report.judge.verdict is None and report.judge.error is None


# --- AC: crashing / invalid judge -> error recorded, DONE ---------------------------------------


def _validation_error() -> ValidationError:
    try:
        JudgeVerdict.model_validate_json('{"face_present": ')
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


CRASHES: list[tuple[Exception, str]] = [
    (ProviderError("judge quota exceeded", retryable=False), "judge quota exceeded"),
    (ProviderError("judge timeout"), "judge timeout"),  # retryable: still not a stage failure
    (RuntimeError("judge exploded"), "judge exploded"),
    (_validation_error(), ""),  # provider couldn't parse the model's JSON
]


@pytest.mark.parametrize(("exc", "message"), CRASHES, ids=["moderation", "timeout", "bug", "json"])
def test_crashing_judge_error_is_recorded(
    make_ctx: MakeCtx, no_stdin: None, exc: Exception, message: str
) -> None:
    judge = CrashingJudge(exc)
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert judge.calls == 1, "a soft signal is not worth a second paid judge call"
    assert report.judge.error, "error must be recorded"
    assert message in report.judge.error
    assert report.judge.verdict is None and report.judge.skipped is None


@pytest.mark.parametrize(("exc", "message"), CRASHES, ids=["moderation", "timeout", "bug", "json"])
def test_crashing_judge_stage_is_done_in_pipeline(
    make_ctx: MakeCtx, runs_root: Path, no_stdin: None, exc: Exception, message: str
) -> None:
    ctx = _ctx(make_ctx, judge=CrashingJudge(exc))
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = mf.load(result.run_id, runs_root).stages["quality_check"]
    assert record.status is StageStatus.DONE, record.error
    report = QualityReport.model_validate(record.data)
    assert report.judge.error and message in report.judge.error


INVALID: list[object] = [
    "Sure! The video looks great.",  # prose instead of JSON
    '{"face_present": {"verdict": "pass"',  # truncated JSON text
    {"face_present": {"verdict": "maybe", "score": 9, "evidence": "", "timestamp": None}},
    None,
]


@pytest.mark.parametrize("raw", INVALID, ids=["prose", "truncated", "bad-dict", "none"])
def test_invalid_judge_answer_error_is_recorded(
    make_ctx: MakeCtx, runs_root: Path, no_stdin: None, raw: object
) -> None:
    judge = InvalidJudge(raw)
    ctx = _ctx(make_ctx, judge=judge)
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = mf.load(result.run_id, runs_root).stages["quality_check"]
    assert record.status is StageStatus.DONE, record.error
    report = QualityReport.model_validate_json((ctx.run_dir / QUALITY).read_text(encoding="utf-8"))
    assert report.judge.error, "invalid judge answer must be recorded as an error"
    assert report.judge.verdict is None and report.judge.skipped is None
    assert QualityReport.model_validate(record.data) == report


# --- AC: out-of-range defect timestamps are dropped and counted ---------------------------------


def test_hallucinated_defects_are_dropped_and_counted(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = HallucinatingJudge("00:02", "00:12", "00:08", "01:05")  # video is 9 s
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict is not None
    assert [d.timestamp for d in report.judge.verdict.defects] == ["00:02", "00:08"]
    assert report.judge.hallucinated_defects == 2
    # the rubric part is untouched by post-processing
    assert report.judge.verdict.face_present == judge.verdict.face_present
    assert report.judge.verdict.matches_prompt == judge.verdict.matches_prompt


def test_all_defects_beyond_video_are_dropped(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = HallucinatingJudge("00:30", "02:00", "10:00")
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict is not None
    assert report.judge.verdict.defects == []
    assert report.judge.hallucinated_defects == 3


def test_defects_inside_video_are_kept(make_ctx: MakeCtx, no_stdin: None) -> None:
    judge = HallucinatingJudge("00:00", "00:05", "00:08")
    ctx = _ctx(make_ctx, judge=judge)
    report = _run(ctx)
    assert report.judge.verdict is not None
    assert report.judge.verdict.defects == judge.verdict.defects
    assert report.judge.hallucinated_defects == 0


# --- AC: SyncNet not importable / weights missing -> syncnet.error ----------
# Replaces v1's "SyncNet missing -> skipped" and "without weights -> skipped": SyncNet is a required
# dependency now (specs/quality_check.md "Acceptance criteria": "SyncNet not importable ->
# syncnet.error ...; weights missing -> syncnet.error naming the files").


def test_syncnet_not_importable_is_an_error(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, judge=FakeJudge())  # conftest: `import syncnet_python` raises ImportError
    report = _run(ctx)
    s = report.syncnet
    assert s.error, "a required dependency that does not import is an error, not 'skipped'"
    assert "syncnet" in s.error.lower(), s.error  # names what is missing
    assert s.skipped is None and s.av_offset_frames is None
    assert report.judge.verdict is not None


def test_syncnet_without_weights_is_an_error_naming_the_files(
    make_ctx: MakeCtx, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    calls = _install_syncnet(monkeypatch, result=FACE)
    empty = tmp_path / "no_weights"
    empty.mkdir()
    monkeypatch.setenv("HWVA_SYNCNET_WEIGHTS", str(empty))
    ctx = _ctx(make_ctx)
    report = _run(ctx)
    s = report.syncnet
    assert s.error, "missing weights are an error now"
    for name in WEIGHT_FILES:
        assert name in s.error, s.error
    assert s.skipped is None and s.av_offset_frames is None
    assert calls.inference == [], "no inference without weights"


def test_syncnet_runs_through_the_shared_runner(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    """quality_check measures the final with `hwva.sync.syncnet_measure` (the ONE place that runs
    SyncNet, specs/sync.md), looked up at call time."""
    stub = install_syncnet_measure(monkeypatch, {AUDIO: (-2, 6.25, 7.5)})
    ctx = _ctx(make_ctx)
    report = _run(ctx)
    assert [(v.name, a.name) for v, a in stub.calls] == [(FINAL, AUDIO)]
    s = report.syncnet
    assert s.skipped is None and s.error is None
    assert (s.av_offset_frames, s.confidence, s.min_dist) == (-2, 6.25, 7.5)
    assert s.seconds is not None and s.seconds >= 0


@pytest.mark.parametrize("kind", ["NoFaceError", "SyncError"])
def test_shared_runner_failures_map_to_skipped_or_error(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch, no_stdin: None, kind: str
) -> None:
    from hwva import sync

    install_syncnet_measure(monkeypatch, {AUDIO: getattr(sync, kind)("no face found by SyncNet")})
    ctx = _ctx(make_ctx, judge=FakeJudge())
    s = _run(ctx).syncnet
    if kind == "NoFaceError":
        assert s.skipped and s.error is None  # no face (the dev fakes): skipped
    else:
        assert s.error and "no face found" in s.error and s.skipped is None
    assert s.av_offset_frames is None


def test_syncnet_no_face_is_skipped(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    calls = _install_syncnet(monkeypatch, result=NO_FACE)
    ctx = _ctx(make_ctx)
    report = _run(ctx)
    assert len(calls.inference) == 1
    assert report.syncnet.skipped, "no face must be a skipped record"
    assert report.syncnet.av_offset_frames is None
    assert report.syncnet.confidence is None and report.syncnet.min_dist is None


def test_syncnet_crash_does_not_block(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    _install_syncnet(monkeypatch, exc=RuntimeError("torch out of memory"))
    ctx = _ctx(make_ctx, judge=FakeJudge())
    report = _run(ctx)
    assert report.syncnet.error or report.syncnet.skipped
    assert report.syncnet.av_offset_frames is None
    assert report.judge.verdict is not None, "a SyncNet failure must not lose the judge result"


def test_syncnet_measurement_is_recorded(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    calls = _install_syncnet(monkeypatch, result=FACE)
    ctx = _ctx(make_ctx)
    report = _run(ctx)

    assert len(calls.init) == 1
    args, kwargs = calls.init[0]
    assert kwargs.get("device") == "cpu"
    passed = {Path(s).resolve() for s in _strings((args, kwargs))}
    for name in WEIGHT_FILES:
        assert (weights / name).resolve() in passed, f"{name} from $HWVA_SYNCNET_WEIGHTS not used"

    assert len(calls.inference) == 1
    video, audio = _inference_paths(calls.inference[0])
    assert video == (ctx.run_dir / FINAL).resolve()
    assert audio == (ctx.run_dir / AUDIO).resolve()

    s = report.syncnet
    assert s.skipped is None and s.error is None
    assert s.av_offset_frames == -6
    assert s.confidence == pytest.approx(4.74)
    assert s.min_dist == pytest.approx(7.12)
    assert s.seconds is not None and s.seconds >= 0

    raw = (ctx.run_dir / QUALITY).read_text(encoding="utf-8")
    assert raw.index('"av_offset_frames"') < raw.index('"confidence"'), "offset is reported first"


# === Changed acceptance criteria (spec diff): robustness, timeouts, console =====================

TIMED_OUT = re.compile(r"timed?\s*out", re.IGNORECASE)  # "TimeoutExpired", "timed out", ...
PROBE_TIMEOUT_S = 30
RENDER_TIMEOUT_S = 120


def _exe(cmd: Any) -> str:
    """'ffmpeg' / 'ffprobe' / ... from an argv list (full paths and .exe allowed)."""
    first = cmd.split()[0] if isinstance(cmd, str) else str(next(iter(cmd)))
    return Path(first).stem.lower()


def _cmd_of(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return args[0] if args else kwargs["args"]


def _judge_exactly_one(j: Any) -> bool:
    return sum(x is not None for x in (j.skipped, j.error, j.verdict)) == 1


def _syncnet_exactly_one(s: Any) -> bool:
    metric = (s.av_offset_frames, s.confidence, s.min_dist)
    has_metric = any(x is not None for x in metric)
    if has_metric and not all(x is not None for x in metric):
        return False  # a half-filled metric is not a metric
    return sum((s.skipped is not None, s.error is not None, has_metric)) == 1


def _unreadable_length(ctx: Context) -> None:
    """final.mp4 that ffprobe cannot read, and assemble recorded no duration."""
    (ctx.run_dir / FINAL).write_bytes(b"not a video at all")
    ctx.manifest.stages["assemble"].data = {}


class _BrokenImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Makes `import syncnet_python` (any import style) raise `exc` while executing the module."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "syncnet_python" or fullname.startswith("syncnet_python."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        raise self.exc


def _break_syncnet_import(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    monkeypatch.delitem(sys.modules, "syncnet_python", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenImport(exc), *sys.meta_path])


class _ProcSpy:
    """Intercepts at the Popen level: records every process spawned while `active` (its argv and
    the timeouts it was waited with). With `expire=True` any timed wait raises TimeoutExpired, i.e.
    the process is 'slower than any timeout' - it only fires if someone actually set a timeout."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, expire: bool = False) -> None:
        self.active = False
        self.expire = expire
        self.procs: list[tuple[list[str], list[float]]] = []
        popen = subprocess.Popen
        orig_init, orig_comm, orig_wait = popen.__init__, popen.communicate, popen.wait
        spy = self

        def note(proc: Any, timeout: float | None) -> None:
            slot = getattr(proc, "_hwva_timeouts", None)
            if slot is None or timeout is None:
                return
            slot.append(float(timeout))
            if spy.expire:
                raise subprocess.TimeoutExpired(proc.args, timeout)

        def init(proc: Any, *args: Any, **kwargs: Any) -> None:
            orig_init(proc, *args, **kwargs)
            if spy.active:
                cmd = proc.args
                argv = cmd.split() if isinstance(cmd, str) else [str(a) for a in cmd]
                proc._hwva_timeouts = []
                spy.procs.append((argv, proc._hwva_timeouts))

        def communicate(proc: Any, input: Any = None, timeout: float | None = None) -> Any:
            note(proc, timeout)
            return orig_comm(proc, input, timeout)

        def wait(proc: Any, timeout: float | None = None) -> Any:
            note(proc, timeout)
            return orig_wait(proc, timeout)

        monkeypatch.setattr(popen, "__init__", init)
        monkeypatch.setattr(popen, "communicate", communicate)
        monkeypatch.setattr(popen, "wait", wait)


class _FfmpegSyncNet:
    """What the stub SyncNet saw during its inference."""

    def __init__(self) -> None:
        self.calls = 0
        self.scenedetect_levels: list[int] = []


def _install_ffmpeg_syncnet(
    monkeypatch: pytest.MonkeyPatch, clip: Path, spy: _ProcSpy | None = None
) -> _FfmpegSyncNet:
    """Stub `syncnet_python` whose inference behaves like the real one on the console: warns on the
    `pyscenedetect` logger and runs ffmpeg both ways syncnet_pipeline.py does - `subprocess.run`
    with captured output and ffmpeg-python `.run()` (inherits the console). Works on `clip`."""
    ffmpeg = pytest.importorskip("ffmpeg")
    seen = _FfmpegSyncNet()

    class SyncNetPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inference(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            seen.calls += 1
            log = logging.getLogger("pyscenedetect")
            seen.scenedetect_levels.append(log.getEffectiveLevel())
            log.warning("VideoManager is deprecated and will be removed.")
            work = clip.parent
            if spy is not None:
                spy.active = True
            try:
                subprocess.run(  # like _extract_audio_from_video / _crop
                    ["ffmpeg", "-y", "-i", str(clip), "-ac", "1", "-ar", "16000",
                     "-acodec", "pcm_s16le", "-f", "wav", str(work / "speech.wav")],
                    capture_output=True, text=True, check=True,
                )  # fmt: skip
                (  # like inference() step 1
                    ffmpeg.input(str(clip))
                    .output(str(work / "video.avi"), **{"q:v": 2}, r=25)
                    .overwrite_output()
                    .run()
                )
            finally:
                if spy is not None:
                    spy.active = False
            return FACE

    module = types.ModuleType("syncnet_python")
    module.SyncNetPipeline = SyncNetPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "syncnet_python", module)
    return seen


def _tiny_clip(tmp_path: Path) -> Path:
    d = tmp_path / "syncnet_work"
    d.mkdir()
    return make_video(d / "clip.mp4", 0.5, size="64x64", audio=True)


# --- AC: SyncNet never fails the stage (import / run / result reading) --------------------------


IMPORT_ERRORS: list[BaseException] = [
    ImportError("DLL load failed while importing _C: The specified module could not be found."),
    OSError("[WinError 126] Error loading torch\\lib\\c10.dll or one of its dependencies."),
    RuntimeError("operator torchvision::nms does not exist"),
]


@pytest.mark.parametrize("exc", IMPORT_ERRORS, ids=["dll-importerror", "oserror", "runtimeerror"])
def test_syncnet_broken_import_is_error(
    make_ctx: MakeCtx,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_stdin: None,
    exc: BaseException,
) -> None:
    """An installed-but-broken extra (e.g. torch DLL on Windows) is an error, not 'missing'."""
    _break_syncnet_import(monkeypatch, exc)
    ctx = _ctx(make_ctx, judge=FakeJudge())
    report = _run(ctx)
    assert report.syncnet.error, "a broken SyncNet import must be recorded as syncnet.error"
    assert report.syncnet.skipped is None
    assert report.syncnet.av_offset_frames is None
    assert report.judge.verdict is not None, "a SyncNet failure must not lose the judge result"


def _numpy_two_tracks() -> tuple[Any, ...]:
    np = pytest.importorskip("numpy")
    return (
        np.array([-6, 2]),
        np.array([4.74, 1.2], dtype=np.float32),
        np.array([7.12, 9.0], dtype=np.float32),
        np.float32(4.74),
        np.float32(7.12),
        "{}",
        np.bool_(True),
    )


MALFORMED: dict[str, Callable[[], tuple[Any, ...]]] = {
    "numpy-arrays": _numpy_two_tracks,
    "offset-not-a-number": lambda: (["x"], [4.74], [7.12], 4.74, 7.12, "{}", True),
    "confs-none": lambda: ([-6], None, [7.12], 4.74, 7.12, "{}", True),
    "max-conf-none": lambda: ([-6], [4.74], [7.12], None, 7.12, "{}", True),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_syncnet_odd_result_never_raises(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None, case: str
) -> None:
    """numpy arrays / malformed values -> syncnet.error or a valid metric; never an exception."""
    _install_syncnet(monkeypatch, result=MALFORMED[case]())
    ctx = _ctx(make_ctx)
    report = _run(ctx)
    s = report.syncnet
    assert _syncnet_exactly_one(s), s
    if s.error is None:  # accepted only if it is the right measurement
        assert s.av_offset_frames == -6, s
        assert s.confidence == pytest.approx(4.74, abs=1e-3)
        assert s.min_dist == pytest.approx(7.12, abs=1e-3)


RUN_FAILURES: dict[str, Callable[[pytest.MonkeyPatch], Any]] = {
    "inference-raises": lambda mp: _install_syncnet(mp, exc=RuntimeError("cv2 error")),
    "short-tuple": lambda mp: _install_syncnet(mp, result=([-6], [4.74], 4.74)),
}


@pytest.mark.parametrize("case", list(RUN_FAILURES))
def test_syncnet_run_failure_is_error(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None, case: str
) -> None:
    """Already satisfied by the current stage (not xfail): a crash while running -> error only."""
    RUN_FAILURES[case](monkeypatch)
    ctx = _ctx(make_ctx)
    report = _run(ctx)
    assert report.syncnet.error
    assert _syncnet_exactly_one(report.syncnet), report.syncnet


# --- AC: video length unreadable -> defects kept, judge.filter_skipped -------------------------


def _ffprobe_absent(monkeypatch: pytest.MonkeyPatch, ctx: Context) -> None:
    real_popen = subprocess.Popen

    def popen(args: Any, *a: Any, **kw: Any) -> Any:
        if _exe(args) == "ffprobe":
            raise FileNotFoundError(2, "The system cannot find the file specified", "ffprobe")
        return real_popen(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    ctx.manifest.stages["assemble"].data = {}


def _ffprobe_fails(monkeypatch: pytest.MonkeyPatch, ctx: Context) -> None:
    _unreadable_length(ctx)  # real ffprobe on a garbage file


UNREADABLE: dict[str, Callable[[pytest.MonkeyPatch, Context], None]] = {
    "ffprobe-fails": _ffprobe_fails,
    "ffprobe-absent": _ffprobe_absent,
}


@pytest.mark.parametrize("case", list(UNREADABLE))
def test_unreadable_length_keeps_defects_and_says_why(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch, no_stdin: None, case: str
) -> None:
    judge = HallucinatingJudge("00:02", "00:12", "01:05")  # 00:12 / 01:05 would be dropped on 9 s
    ctx = _ctx(make_ctx, judge=judge)
    UNREADABLE[case](monkeypatch, ctx)
    report = _run(ctx)
    j = report.judge
    assert j.verdict is not None and j.error is None and j.skipped is None
    assert j.verdict.defects == judge.verdict.defects, "defects must be kept as returned"
    assert j.hallucinated_defects == 0
    assert j.filter_skipped, "judge.filter_skipped must say why the filter did not run"


# --- AC: exactly one of skipped / error / verdict|metric ----------------------------------------


JUDGES: dict[str, Callable[[], Any]] = {
    "no-judge": lambda: None,
    "crashing": lambda: CrashingJudge(RuntimeError("judge exploded")),
    "ok": lambda: FakeJudge(judge_verdict("00:03")),
    "ok-unreadable-length": lambda: HallucinatingJudge("00:02", "00:12"),
}

SYNCNETS: dict[str, Callable[[pytest.MonkeyPatch], Any]] = {
    "missing": lambda mp: None,  # autouse fixture: not installed
    "no-face": lambda mp: _install_syncnet(mp, result=NO_FACE),
    "import-oserror": lambda mp: _break_syncnet_import(mp, OSError("[WinError 126] c10.dll")),
    "inference-error": lambda mp: _install_syncnet(mp, exc=RuntimeError("torch exploded")),
    "numpy-metric": lambda mp: _install_syncnet(mp, result=_numpy_two_tracks()),
    "metric": lambda mp: _install_syncnet(mp, result=FACE),
}

SCENARIOS: list[tuple[str, str]] = [
    ("no-judge", "missing"),
    ("crashing", "no-face"),
    ("ok", "import-oserror"),
    ("ok-unreadable-length", "inference-error"),
    ("ok", "numpy-metric"),
    ("crashing", "metric"),
]


def test_exactly_one_outcome_in_every_scenario(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    """Every judge kind and every SyncNet kind appears at least once; each run must satisfy the
    invariant in both records (filter_skipped does not count: it qualifies a verdict)."""
    for judge_kind, syncnet_kind in SCENARIOS:
        with monkeypatch.context() as mp:
            SYNCNETS[syncnet_kind](mp)
            judge = JUDGES[judge_kind]()
            ctx = _ctx(make_ctx, judge=judge) if judge is not None else _ctx(make_ctx)
            if judge_kind == "ok-unreadable-length":
                _unreadable_length(ctx)
            report = _run(ctx)
        where = f"{judge_kind} / {syncnet_kind}"
        assert _judge_exactly_one(report.judge), (where, report.judge)
        assert _syncnet_exactly_one(report.syncnet), (where, report.syncnet)


# --- AC: every ffmpeg/ffprobe subprocess has a timeout ------------------------------------------


def test_stage_subprocesses_have_timeouts(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    """Spy on subprocess.run: ffprobe probes -> timeout=30, ffmpeg renders/analysis -> 120."""
    ctx = _ctx(make_ctx, judge=HallucinatingJudge("00:02", "00:12"))
    calls: list[tuple[Any, dict[str, Any]]] = []
    real_run = subprocess.run

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((_cmd_of(args, kwargs), kwargs))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    _run(ctx)
    media = [(cmd, kw) for cmd, kw in calls if _exe(cmd) in {"ffmpeg", "ffprobe"}]
    assert any(_exe(cmd) == "ffprobe" for cmd, _ in media), "the video length is probed"
    for cmd, kw in media:
        expected = PROBE_TIMEOUT_S if _exe(cmd) == "ffprobe" else RENDER_TIMEOUT_S
        assert kw.get("timeout") == expected, (cmd, kw.get("timeout"))


def test_ffprobe_timeout_is_recorded_in_filter_skipped(
    make_ctx: MakeCtx, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    judge = HallucinatingJudge("00:02", "00:12")
    ctx = _ctx(make_ctx, judge=judge)
    ctx.manifest.stages["assemble"].data = {}  # no fallback length
    real_run = subprocess.run

    def hanging_ffprobe(*args: Any, **kwargs: Any) -> Any:
        cmd = _cmd_of(args, kwargs)
        if _exe(cmd) == "ffprobe":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or PROBE_TIMEOUT_S)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", hanging_ffprobe)
    report = _run(ctx)  # must not raise
    j = report.judge
    assert j.verdict is not None and j.error is None
    assert j.verdict.defects == judge.verdict.defects, "defects kept as returned"
    assert j.filter_skipped and TIMED_OUT.search(j.filter_skipped), j.filter_skipped


def test_syncnet_ffmpeg_runs_quiet_with_timeout(
    make_ctx: MakeCtx,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_stdin: None,
) -> None:
    """SyncNet spawns ffmpeg itself (subprocess.run and ffmpeg-python `.run()`); the stage must make
    every such process run with `-loglevel error` and be waited with a 120 s timeout."""
    ctx = _ctx(make_ctx)
    spy = _ProcSpy(monkeypatch)
    seen = _install_ffmpeg_syncnet(monkeypatch, _tiny_clip(tmp_path), spy)
    report = _run(ctx)
    assert seen.calls == 1
    ffmpegs = [(argv, t) for argv, t in spy.procs if _exe(argv) == "ffmpeg"]
    assert len(ffmpegs) >= 2, f"stub SyncNet should have spawned ffmpeg twice: {spy.procs}"
    for argv, timeouts in ffmpegs:
        assert "-loglevel" in argv and argv[argv.index("-loglevel") + 1] == "error", argv
        assert timeouts and max(timeouts) == pytest.approx(RENDER_TIMEOUT_S), (argv, timeouts)
    assert report.syncnet.av_offset_frames == -6, "the measurement itself is unaffected"


def test_syncnet_ffmpeg_timeout_is_recorded_as_error(
    make_ctx: MakeCtx,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_stdin: None,
) -> None:
    """SyncNet's ffmpeg is slower than any timeout: the process only 'expires' if a timeout was
    actually set, so this also proves the stage sets one."""
    ctx = _ctx(make_ctx, judge=FakeJudge())
    spy = _ProcSpy(monkeypatch, expire=True)
    _install_ffmpeg_syncnet(monkeypatch, _tiny_clip(tmp_path), spy)
    report = _run(ctx)  # must not raise
    s = report.syncnet
    assert s.error and TIMED_OUT.search(s.error), s
    assert s.skipped is None and s.av_offset_frames is None
    assert report.judge.verdict is not None


# --- AC: console stays clean --------------------------------------------------------------------


def test_stage_with_stub_syncnet_writes_nothing_to_console(
    make_ctx: MakeCtx,
    weights: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    no_stdin: None,
) -> None:
    ctx = _ctx(make_ctx, judge=FakeJudge())
    seen = _install_ffmpeg_syncnet(monkeypatch, _tiny_clip(tmp_path))
    scenedetect = logging.getLogger("pyscenedetect")
    old_level = scenedetect.level
    handler = logging.StreamHandler(sys.stderr)  # what a CLI user would see
    scenedetect.addHandler(handler)
    scenedetect.setLevel(logging.INFO)
    try:
        capfd.readouterr()
        report = _run(ctx)
        out, err = capfd.readouterr()
        level_after = scenedetect.level
    finally:
        scenedetect.removeHandler(handler)
        scenedetect.setLevel(old_level)

    assert seen.calls == 1
    assert out == "", f"stdout not clean: {out!r}"
    assert err == "", f"stderr not clean: {err!r}"
    assert seen.scenedetect_levels and min(seen.scenedetect_levels) >= logging.ERROR, (
        "pyscenedetect must be ERROR+ during the SyncNet call"
    )
    assert level_after == logging.INFO, "pyscenedetect level must be restored after the call"
    assert report.syncnet.av_offset_frames == -6


def test_pyscenedetect_error_level_deprecations_are_silenced(
    make_ctx: MakeCtx,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    no_stdin: None,
) -> None:
    """Real pyscenedetect logs its deprecation messages at ERROR: none may reach the console or
    any log handler during a stage run, and the logger level is restored afterwards."""
    ctx = _ctx(make_ctx, judge=FakeJudge())

    class SyncNetPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inference(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            log = logging.getLogger("pyscenedetect")
            log.error("VideoManager is deprecated and will be removed.")
            log.error("base_timecode argument is deprecated and has no effect.")
            return FACE

    module = types.ModuleType("syncnet_python")
    module.SyncNetPipeline = SyncNetPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "syncnet_python", module)
    scenedetect = logging.getLogger("pyscenedetect")
    old_level = scenedetect.level
    handler = logging.StreamHandler(sys.stderr)  # what a CLI user would see
    scenedetect.addHandler(handler)
    try:
        capsys.readouterr()
        report = _run(ctx)
        out, err = capsys.readouterr()
        level_after = scenedetect.level
    finally:
        scenedetect.removeHandler(handler)

    assert report.syncnet.av_offset_frames == -6
    assert [r.getMessage() for r in caplog.records if r.name == "pyscenedetect"] == []
    assert "deprecated" not in out and "deprecated" not in err, (out, err)
    assert level_after == old_level, "pyscenedetect level must be restored after the call"


# === Lip-sync numbers (specs/quality_check.md; see specs/sync.md) ===============

STT_AUDIO = "final_audio.wav"


def _with_sync(ctx: Context, report: SyncReport | None) -> None:
    """shot_1 DONE; with `report` as its data `sync` (None = a run from before the measurement)."""
    rec = mf.stage(ctx.manifest, "shot_1")
    rec.status = StageStatus.DONE
    rec.artifacts = {"video": "shot_1.mp4", "shot_1_last_frame": "shot_1.last.png"}
    shot = ShotData(duration_s=5.0, sync=report)
    rec.data = shot.model_dump(mode="json")
    if report is not None:
        rec.artifacts["sync"] = "sync.json"


TAKE_1 = SyncReport(
    lag_s=0.25, peak_r=0.43, r_at_0=0.1, verdict="no_sync", shift_s=0.0,
    box=[0.2, 0.8, 0.42, 0.6], warning="lip-sync not detected (peak r 0.43 < 0.6)",
)  # fmt: skip
TAKE_2 = SyncReport(
    lag_s=0.5, peak_r=0.71, r_at_0=0.16, verdict="correct", shift_s=0.5,
    box=[0.2, 0.8, 0.42, 0.6], take=2, rejected=TAKE_1,
)  # fmt: skip


# v1 records (TAKE_1 / TAKE_2 above). Regression guard (not xfail): "a v1 record -> copied without
# error, the new fields null". v1's "same lag_s/peak_r" is no longer asked (specs/quality_check.md
# v2 lists the copied fields; lag_s/peak_r are not among them).
@pytest.mark.parametrize("report", [TAKE_2, TAKE_1], ids=["correct_take_2", "no_sync"])
def test_v1_sync_record_is_copied_without_error(
    make_ctx: MakeCtx, no_stdin: None, report: SyncReport
) -> None:
    ctx = _ctx(make_ctx)
    _with_sync(ctx, report)
    quality = _run(ctx)
    raw = json.loads((ctx.run_dir / QUALITY).read_text(encoding="utf-8"))["sync"]
    assert raw is not None, "quality.json has no `sync`"
    for key in ("verdict", "shift_s", "take"):
        assert raw[key] == getattr(report, key), key
    for key in ("syncnet_confidence", "syncnet_offset_frames", "method", "confidence_global",
                "confidence_piecewise"):  # fmt: skip
        assert raw[key] is None, key
    assert quality.sync is not None and quality.sync.skipped is None
    assert quality.sync.verdict == report.verdict


V2_RECORD = SyncReport(
    verdict="correct", syncnet_confidence=8.63, syncnet_offset_frames=-6, syncnet_min_dist=7.1,
    shift_s=-0.24, thresholds={"sync_conf": 5.0, "weak_conf": 3.0}, method="piecewise",
    confidence_global=8.71, confidence_piecewise=8.86,
    phrases=[{"start_s": 0.7, "end_s": 1.5, "shift_s": -0.083},
             {"start_s": 1.8, "end_s": 2.4, "shift_s": 0.042},
             {"start_s": 2.6, "end_s": 3.3, "shift_s": 0.0}],
    aligned_audio="aligned_line.wav", box=[0.2, 0.8, 0.42, 0.6], take=2,
    rejected=SyncReport(verdict="no_sync", syncnet_confidence=1.7, syncnet_offset_frames=-7,
                        box=[0.2, 0.8, 0.42, 0.6]),
)  # fmt: skip


def test_v2_sync_record_is_copied(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    _with_sync(ctx, V2_RECORD)
    quality = _run(ctx)
    raw = json.loads((ctx.run_dir / QUALITY).read_text(encoding="utf-8"))["sync"]
    assert raw["skipped"] is None
    assert raw["syncnet_confidence"] == 8.63 and raw["syncnet_offset_frames"] == -6
    assert raw["verdict"] == "correct" and raw["method"] == "piecewise"
    assert raw["shift_s"] == -0.24 and raw["take"] == 2
    assert (raw["confidence_global"], raw["confidence_piecewise"]) == (8.71, 8.86)
    assert [p["shift_s"] for p in raw["phrases"]] == [-0.083, 0.042, 0.0]
    assert [(p["start_s"], p["end_s"]) for p in raw["phrases"]] == [
        (0.7, 1.5), (1.8, 2.4), (2.6, 3.3)
    ]  # fmt: skip
    assert quality.sync is not None and quality.sync.method == "piecewise"


def test_no_sync_record_is_skipped(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)
    _with_sync(ctx, None)
    _run(ctx)
    raw = json.loads((ctx.run_dir / QUALITY).read_text(encoding="utf-8"))["sync"]
    assert raw is not None and raw.get("skipped") == "no sync measurement", raw
    assert all(raw.get(k) is None for k in ("lag_s", "peak_r", "r_at_0", "verdict")), raw


def test_no_shot_1_record_at_all_is_skipped(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx)  # _ctx records no shot_1 stage
    report = _run(ctx)
    assert report.sync is not None and report.sync.skipped == "no sync measurement"


def test_sync_numbers_never_block(make_ctx: MakeCtx, no_stdin: None, runs_root: Path) -> None:
    ctx = _ctx(make_ctx)
    _with_sync(ctx, TAKE_1)  # no lip-sync detected: still a soft signal
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = mf.load(result.run_id, runs_root).stages["quality_check"]
    assert record.status is StageStatus.DONE, record.error
    report = QualityReport.model_validate(record.data)
    assert report.sync is not None and report.sync.verdict == "no_sync"


# --- SyncNet gets the audio as it is in final.mp4: final_check's stt_audio when recorded ---------


def test_syncnet_gets_final_checks_stt_audio(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    calls = _install_syncnet(monkeypatch, result=FACE)
    ctx = _ctx(make_ctx)
    make_audio(ctx.run_dir / STT_AUDIO, VIDEO_S)  # the audio track of final.mp4, as extracted
    rec = ctx.manifest.stages["final_check"]
    rec.artifacts = {"report": "final_report.json", "stt_audio": STT_AUDIO}
    _run(ctx)
    assert len(calls.inference) == 1
    video, audio = _inference_paths(calls.inference[0])
    assert video == (ctx.run_dir / FINAL).resolve()
    assert audio == (ctx.run_dir / STT_AUDIO).resolve()


def test_syncnet_gets_assembles_audio_without_stt_audio(
    make_ctx: MakeCtx, weights: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    """No stt_audio from final_check -> assemble's artifact `audio` (the file placed), not
    line.wav (specs/quality_check.md v2)."""
    calls = _install_syncnet(monkeypatch, result=FACE)
    ctx = _ctx(make_ctx)
    make_audio(ctx.run_dir / "aligned_line.wav", 5.0)
    ctx.manifest.stages["assemble"].artifacts["audio"] = "aligned_line.wav"
    report = _run(ctx)
    assert len(calls.inference) == 1
    video, audio = _inference_paths(calls.inference[0])
    assert video == (ctx.run_dir / FINAL).resolve()
    assert audio == (ctx.run_dir / "aligned_line.wav").resolve()
    assert report.syncnet.av_offset_frames == -6
