"""Acceptance tests for the `final_check` stage (specs/final_check.md). Real ffmpeg, no network.

Synthetic media: line.wav is a 3 s rising chirp (200 -> 1100 Hz) — not periodic, so the
cross-correlation has one sharp peak at the true offset. A faithful final.mp4 carries that chirp
from t=0, padded with silence, over a testsrc picture (no black, no freeze, no cuts)."""

from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import ScriptedStt, duration, speech_wav

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import fakes
from hwva.providers.base import ProviderError
from hwva.schemas import (
    FINAL_HARD_CHECKS,
    FINAL_SOFT_CHECKS,
    SHOT_2_SKIPPED,
    AssembleData,
    FinalCheck,
    FinalReport,
    Lang,
    LineSource,
    Script,
    Shot,
    ShotData,
    StageStatus,
    VoiceData,
)
from hwva.stages import final_check

MakeCtx = Callable[..., Context]
BuildFinal = Callable[[Path, Path, float], None]  # (final.mp4 path, line.wav path, seconds)

LINE = "Я тут головний"
MISHEARD = "Я тут найголовніший"
FINAL = "final.mp4"
REPORT = "final_report.json"
ARTIFACTS = {"report": REPORT, "stt_audio": "final_audio.wav"}
SIZE = "720x1280"
LINE_S = 3.0
CHIRP = "aevalsrc=exprs=0.5*sin(2*PI*(200*t+150*t*t)):s=44100:d={d}"


# --- media helpers ------------------------------------------------------------------------------


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _lavfi(source: str) -> list[str]:
    return ["-f", "lavfi", "-i", source]


def _testsrc(seconds: float, size: str = SIZE, src: str = "testsrc") -> str:
    return f"{src}=size={size}:rate=24:duration={seconds}"


def _color(color: str, seconds: float, size: str = SIZE) -> str:
    return f"color=c={color}:size={size}:rate=24:duration={seconds}"


def _line_wav(path: Path, *, lead_s: float = 0.0, gain_db: float = 0.0) -> Path:
    """'TTS output': the chirp, optionally starting with `lead_s` of silence and/or attenuated."""
    af = f"volume={gain_db}dB"
    if lead_s:
        af += f",adelay={int(lead_s * 1000)}:all=1"
    _ffmpeg(*_lavfi(CHIRP.format(d=LINE_S)), "-af", af, str(path))
    return path


def _final(
    path: Path,
    seconds: float,
    *,
    audio: list[str] | None,
    video: list[str] | None = None,
    delay_ms: int = 0,
    audio_streams: int = 1,
) -> Path:
    """final.mp4 the way assemble writes it (H.264 + AAC, audio padded to the video length).
    `video`: lavfi sources concatenated in order; `audio`: ffmpeg input args, None = no audio."""
    sources = video or [_testsrc(seconds)]
    args: list[str] = []
    for src in sources:
        args += _lavfi(src)
    k = len(sources)
    graph = "".join(f"[{i}:v]" for i in range(k)) + f"concat=n={k}:v=1:a=0,format=yuv420p[v]"
    maps = ["-map", "[v]"]
    if audio is not None:
        args += audio
        chain = f"[{k}:a]" + (f"adelay={delay_ms}:all=1," if delay_ms else "")
        chain += f"apad=whole_dur={seconds}"
        labels = [f"[a{i}]" for i in range(audio_streams)]
        chain += (f",asplit={audio_streams}" if audio_streams > 1 else "") + "".join(labels)
        graph += ";" + chain
        for label in labels:
            maps += ["-map", label]
    _ffmpeg(*args, "-filter_complex", graph, *maps, "-c:v", "libx264", "-c:a", "aac", str(path))
    return path


def _faithful(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=["-i", str(line_wav)])


# --- context ------------------------------------------------------------------------------------


def _ctx(
    make_ctx: MakeCtx,
    *,
    shots: tuple[float, ...] = (5.0, 4.0),
    build_final: BuildFinal = _faithful,
    line_kw: dict[str, Any] | None = None,
    line: str = LINE,
    lang: Lang = "uk",
    script_line: str | None = None,
    **providers: Any,
) -> Context:
    """Context as it is after `assemble`: script.json, line.wav, final.mp4, all upstream DONE.
    The dev FakeStt 'hears' the line via fakes.LAST_SPOKEN, as if FakeTts had spoken it here."""
    params = {"line": line, "line_source": LineSource.INPUT.value, "lang": lang}
    ctx = make_ctx(idea=f'Кіт каже: "{line}"', params=params, **providers)
    run_dir = ctx.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    total = sum(shots)

    script = Script(
        title="test",
        line=script_line or line,
        line_source=LineSource.INPUT,
        lang=lang,
        character="a ginger cat with green eyes",
        shots=[
            Shot(visual_prompt=f"shot {i + 1}", duration_s=d, has_line=i == 0)
            for i, d in enumerate(shots)
        ],
    )
    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    rec = mf.stage(ctx.manifest, "script")
    rec.status, rec.artifacts = StageStatus.DONE, {"script": "script.json"}

    line_wav = _line_wav(run_dir / "line.wav", **(line_kw or {}))
    (run_dir / "line.wav.txt").write_text(line, encoding="utf-8")
    (run_dir / fakes.LAST_SPOKEN).write_text(line, encoding="utf-8")
    rec = mf.stage(ctx.manifest, "voice")
    rec.status = StageStatus.DONE
    rec.artifacts = {"audio": "line.wav", "transcript": "transcript.txt"}
    rec.data = VoiceData(
        audio_duration_s=duration(line_wav), planned_total_s=min(total, 30.0), attempts=1
    ).model_dump(mode="json")

    rec = mf.stage(ctx.manifest, "shot_1")
    rec.status = StageStatus.DONE
    rec.artifacts = {"video": "shot_1.mp4", "shot_1_last_frame": "shot_1.last.png"}
    rec.data = ShotData(duration_s=shots[0]).model_dump(mode="json")
    rec = mf.stage(ctx.manifest, "shot_2")
    rec.status = StageStatus.DONE
    if len(shots) > 1:
        rec.artifacts = {"video": "shot_2.mp4"}
        rec.data = ShotData(duration_s=shots[1]).model_dump(mode="json")
    else:
        rec.artifacts, rec.data = {}, dict(SHOT_2_SKIPPED)

    build_final(run_dir / FINAL, line_wav, total)
    rec = mf.stage(ctx.manifest, "assemble")
    rec.status, rec.artifacts = StageStatus.DONE, {"video": FINAL}
    rec.data = AssembleData(duration_s=total).model_dump(mode="json")
    return ctx


class ProbingStt:
    """Always 'hears' `text`; records the lang and the duration of the file it was given."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Path] = []
        self.langs: list[Lang] = []
        self.durations: list[float] = []

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        self.calls.append(audio_path)
        self.langs.append(lang)
        self.durations.append(duration(audio_path))
        return self.text


# --- report helpers -----------------------------------------------------------------------------


def _report(ctx: Context) -> FinalReport:
    path = ctx.run_dir / REPORT
    assert path.is_file(), f"{REPORT} not written"
    return FinalReport.model_validate_json(path.read_text(encoding="utf-8"))


def _check(report: FinalReport, name: str) -> FinalCheck:
    found = [c for c in report.checks if c.name == name]
    assert len(found) == 1, f"check {name!r} appears {len(found)} times in the report"
    return found[0]


def _number(value: Any) -> float:
    assert isinstance(value, int | float) and not isinstance(value, bool), value
    return float(value)


def _passes(ctx: Context) -> FinalReport:
    out = final_check.run(ctx)
    assert isinstance(out, StageOutput)
    assert out.artifacts == ARTIFACTS
    report = _report(ctx)
    assert FinalReport.model_validate(out.data) == report  # data = same summary
    assert report.verdict == "pass", report.failed
    assert report.failed == []
    return report


def _fails(ctx: Context, check: str) -> FinalReport:
    with pytest.raises(GateFailed) as err:
        final_check.run(ctx)
    assert REPORT in str(err.value), "GateFailed message must point to the report"
    report = _report(ctx)
    assert report.verdict == "fail"
    assert check in report.failed, report.failed
    failed = _check(report, check)
    assert failed.hard and not failed.passed
    return report


def _stage() -> pipeline.Stage:
    stage = next((s for s in pipeline.STAGES if s.name == "final_check"), None)
    assert stage is not None, "final_check is not registered in pipeline.STAGES"
    return stage


# --- AC: valid file (fakes) passes and writes the report with raw values ------------------------


@pytest.mark.parametrize("shots", [(5.0, 4.0), (9.0,)])
def test_valid_file_passes_and_writes_report(
    make_ctx: MakeCtx, no_stdin: None, shots: tuple[float, ...]
) -> None:
    ctx = _ctx(make_ctx, shots=shots, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    assert report.transcript == LINE
    names = [c.name for c in report.checks]
    assert sorted(names) == sorted(FINAL_HARD_CHECKS + FINAL_SOFT_CHECKS)
    for name in FINAL_HARD_CHECKS:
        c = _check(report, name)
        assert c.hard and c.passed, c
        assert c.threshold.strip(), f"{name}: threshold missing"
    for name in FINAL_SOFT_CHECKS:
        assert not _check(report, name).hard
    assert _check(report, "scene_cuts").value["shots"] == len(shots)


def test_report_records_raw_values(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    report = _passes(ctx)

    fmt = _check(report, "format").value
    assert _number(fmt["duration_s"]) == pytest.approx(9.0, abs=0.2)
    assert (fmt["width"], fmt["height"]) == (720, 1280)
    assert (fmt["video_streams"], fmt["audio_streams"]) == (1, 1)

    assert _number(_check(report, "black_frames").value["black_s"]) == pytest.approx(0, abs=0.05)
    assert _number(_check(report, "leading_silence").value["leading_silence_s"]) <= 0.1
    # a raw measurement, not a boolean: the chirp at 0.5 amplitude is ~ -10 LUFS
    lufs = _number(_check(report, "loudness").value["integrated_lufs"])
    assert lufs == pytest.approx(-9.7, abs=3)

    match = _check(report, "audio_match").value
    assert _number(match["peak"]) >= 0.9
    assert abs(_number(match["offset_ms"])) <= 50

    spoken = _check(report, "line").value
    assert spoken["attempts"] == 1
    assert spoken["diffs"] == [[]]

    for name, key in (("freeze", "freeze_s"), ("true_peak", "true_peak_dbtp")):
        _number(_check(report, name).value[key])
    _number(_check(report, "scene_cuts").value["cuts"])


def test_valid_file_passes_with_dev_fake_stt(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    """Default dev profile (FakeStt hears the last line FakeTts spoke in the run dir), through
    the pipeline: DONE, report artifact, data = FinalReport, no spend."""
    ctx = _ctx(make_ctx)
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = result.stages["final_check"]
    assert record.status is StageStatus.DONE, record.error
    assert record.artifacts == ARTIFACTS
    assert record.attempts == 1
    assert record.cost_usd == 0.0
    data = FinalReport.model_validate(record.data)
    assert data.verdict == "pass"
    assert data == _report(ctx)


# --- AC: format gate — too short / too long / horizontal / no audio -----------------------------


def _short(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, 5.0, audio=["-i", str(line_wav)])


def _long(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, 31.0, audio=["-i", str(line_wav)])


def _horizontal(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=["-i", str(line_wav)], video=[_testsrc(seconds, "1280x720")])


def _square(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=["-i", str(line_wav)], video=[_testsrc(seconds, "720x720")])


def _no_audio(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=None)


def _two_audio(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=["-i", str(line_wav)], audio_streams=2)


@pytest.mark.parametrize(
    ("shots", "build"),
    [
        pytest.param((5.0,), _short, id="too_short_5s"),
        pytest.param((16.0, 15.0), _long, id="too_long_31s"),
        pytest.param((5.0, 4.0), _horizontal, id="horizontal"),
        pytest.param((5.0, 4.0), _square, id="square"),
        pytest.param((5.0, 4.0), _no_audio, id="no_audio"),
        pytest.param((5.0, 4.0), _two_audio, id="two_audio_streams"),
    ],
)
def test_bad_format_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, shots: tuple[float, ...], build: BuildFinal
) -> None:
    ctx = _ctx(make_ctx, shots=shots, build_final=build, stt=ScriptedStt(LINE))
    _fails(ctx, "format")


def test_format_records_measured_duration(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, shots=(5.0,), build_final=_short, stt=ScriptedStt(LINE))
    report = _fails(ctx, "format")
    assert _number(_check(report, "format").value["duration_s"]) == pytest.approx(5.0, abs=0.2)


# --- AC: black frames gate — 2 s of black fails, 0.5 s passes -----------------------------------


def _black(seconds_black: float) -> BuildFinal:
    def build(path: Path, line_wav: Path, seconds: float) -> None:
        video = [_color("black", seconds_black), _testsrc(seconds - seconds_black)]
        _final(path, seconds, audio=["-i", str(line_wav)], video=video)

    return build


def test_two_seconds_of_black_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_black(2.0), stt=ScriptedStt(LINE))
    report = _fails(ctx, "black_frames")
    assert _number(_check(report, "black_frames").value["black_s"]) == pytest.approx(2.0, abs=0.1)


def test_half_second_of_black_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_black(0.5), stt=ScriptedStt(LINE))
    report = _passes(ctx)
    assert _number(_check(report, "black_frames").value["black_s"]) == pytest.approx(0.5, abs=0.1)


# --- AC: leading silence gate — 1 s fails, 0.3 s passes -----------------------------------------


def test_one_second_of_leading_silence_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    """The TTS itself starts with 1 s of silence and final.mp4 carries it faithfully: only the
    leading-silence gate fails (audio_match still matches — it IS our audio)."""
    ctx = _ctx(make_ctx, line_kw={"lead_s": 1.0}, stt=ScriptedStt(LINE))
    report = _fails(ctx, "leading_silence")
    assert report.failed == ["leading_silence"]
    value = _check(report, "leading_silence").value["leading_silence_s"]
    assert _number(value) == pytest.approx(1.0, abs=0.1)


def test_short_leading_silence_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, line_kw={"lead_s": 0.3}, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    assert _number(_check(report, "leading_silence").value["leading_silence_s"]) <= 0.5


# --- AC: loudness gate — silent audio -----------------------------------------------------------


def _digital_silence(path: Path, line_wav: Path, seconds: float) -> None:
    """Muxing 'lost' the audio: the stream exists but carries zeros."""
    _final(path, seconds, audio=_lavfi(f"aevalsrc=exprs=0:s=44100:d={LINE_S}"))


def test_silent_audio_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_digital_silence, stt=ScriptedStt(LINE))
    _fails(ctx, "loudness")


def test_quiet_audio_fails_gate_with_raw_loudness(make_ctx: MakeCtx, no_stdin: None) -> None:
    """Our audio, faithfully muxed, but 50 dB down (~ -60 LUFS)."""
    ctx = _ctx(make_ctx, line_kw={"gain_db": -50.0}, stt=ScriptedStt(LINE))
    report = _fails(ctx, "loudness")
    lufs = _number(_check(report, "loudness").value["integrated_lufs"])
    assert lufs < -35
    assert lufs == pytest.approx(-59.7, abs=3)


# --- AC: audio is OUR TTS — other source fails, offset > 50 ms fails, small offset passes -------


def _other_tone(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=_lavfi(f"sine=frequency=440:duration={LINE_S}"))


def _other_chirp(path: Path, line_wav: Path, seconds: float) -> None:
    falling = f"aevalsrc=exprs=0.5*sin(2*PI*(1100*t-150*t*t)):s=44100:d={LINE_S}"
    _final(path, seconds, audio=_lavfi(falling))


@pytest.mark.parametrize("build", [_other_tone, _other_chirp], ids=["other_tone", "other_chirp"])
def test_audio_from_other_source_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, build: BuildFinal
) -> None:
    ctx = _ctx(make_ctx, build_final=build, stt=ScriptedStt(LINE))
    report = _fails(ctx, "audio_match")
    assert _number(_check(report, "audio_match").value["peak"]) < 0.9


def _delayed(ms: int) -> BuildFinal:
    def build(path: Path, line_wav: Path, seconds: float) -> None:
        _final(path, seconds, audio=["-i", str(line_wav)], delay_ms=ms)

    return build


def test_audio_offset_over_50ms_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    """Our chirp, but 300 ms late (still < 0.5 s leading silence): wrong offset -> not ours."""
    ctx = _ctx(make_ctx, build_final=_delayed(300), stt=ScriptedStt(LINE))
    report = _fails(ctx, "audio_match")
    assert report.failed == ["audio_match"]


def test_audio_offset_within_50ms_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_delayed(30), stt=ScriptedStt(LINE))
    report = _passes(ctx)
    match = _check(report, "audio_match").value
    assert _number(match["peak"]) >= 0.9
    assert _number(match["offset_ms"]) == pytest.approx(30, abs=10)


# --- AC: line survived — STT on the final audio, one retry --------------------------------------


def test_mishearing_twice_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ScriptedStt(MISHEARD)
    ctx = _ctx(make_ctx, stt=stt)
    report = _fails(ctx, "line")
    assert report.failed == ["line"]
    assert len(stt.calls) == 2  # one retry, not more
    spoken = _check(report, "line").value
    assert spoken["attempts"] == 2
    assert len(spoken["diffs"]) == 2
    assert all(d for d in spoken["diffs"])
    assert report.transcript == MISHEARD


def test_different_number_as_digits_twice_fails_gate(make_ctx: MakeCtx, no_stdin: None) -> None:
    line = "У мене дві гривні"
    stt = ScriptedStt("У мене 5 гривні")
    ctx = _ctx(make_ctx, line=line, stt=stt)
    report = _fails(ctx, "line")
    assert report.failed == ["line"]
    assert _check(report, "line").value["diffs"][-1] == ["expected 'дві', heard '5'"]


def test_mismatch_once_then_match_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ScriptedStt(MISHEARD, LINE)
    ctx = _ctx(make_ctx, stt=stt)
    report = _passes(ctx)
    assert len(stt.calls) == 2
    spoken = _check(report, "line").value
    assert spoken["attempts"] == 2
    assert spoken["diffs"][0] and spoken["diffs"][1] == []
    assert report.transcript == LINE


def test_match_first_time_calls_stt_once(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ScriptedStt(LINE)
    ctx = _ctx(make_ctx, stt=stt)
    _passes(ctx)
    assert len(stt.calls) == 1


def test_stt_hears_audio_extracted_from_final(make_ctx: MakeCtx, no_stdin: None) -> None:
    """STT gets the audio track of final.mp4 (9 s, padded), not line.wav (3 s) or the mp4."""
    stt = ProbingStt(LINE)
    ctx = _ctx(make_ctx, stt=stt)
    _passes(ctx)
    assert stt.calls
    assert stt.calls[0].name not in {"line.wav", FINAL}
    assert stt.durations[0] == pytest.approx(9.0, abs=0.3)


def test_stt_gets_the_run_language(make_ctx: MakeCtx, no_stdin: None) -> None:
    line = "I am the boss here"
    stt = ProbingStt(line)
    ctx = _ctx(make_ctx, line=line, lang="en", stt=stt)
    report = _passes(ctx)
    assert stt.langs == ["en"]
    assert report.transcript == line


def test_line_is_compared_with_params_not_script(make_ctx: MakeCtx, no_stdin: None) -> None:
    """params["line"] is the single source of truth."""
    ctx = _ctx(make_ctx, script_line=MISHEARD, stt=ScriptedStt(LINE))
    _passes(ctx)


def test_mishearing_fails_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    stt = ScriptedStt(MISHEARD)
    ctx = _ctx(make_ctx, stt=stt)
    result = pipeline.run(ctx, [_stage()], root=runs_root)
    record = result.stages["final_check"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1  # the stage owns its STT retry; the pipeline doesn't add one
    assert record.error is not None and record.error.startswith("GateFailed"), record.error
    assert REPORT in record.error
    assert len(stt.calls) == 2
    assert "line" in _report(ctx).failed


# --- AC: soft signals present in the report but never fail the stage -----------------------------


def _frozen(path: Path, line_wav: Path, seconds: float) -> None:
    _final(path, seconds, audio=["-i", str(line_wav)], video=[_color("gray", seconds)])


def _hard_cut(path: Path, line_wav: Path, seconds: float) -> None:
    video = [_testsrc(5.0), _testsrc(seconds - 5.0, src="testsrc2")]
    _final(path, seconds, audio=["-i", str(line_wav)], video=video)


def test_frozen_video_is_soft(make_ctx: MakeCtx, no_stdin: None) -> None:
    """A static picture for the whole video: freezedetect fires, the stage still passes."""
    ctx = _ctx(make_ctx, build_final=_frozen, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    freeze = _check(report, "freeze")
    assert not freeze.hard
    assert _number(freeze.value["freeze_s"]) > 0


def test_hot_true_peak_is_soft(make_ctx: MakeCtx, no_stdin: None) -> None:
    """Audio clipped +12 dB: true peak above 0 dBTP is recorded, the stage still passes."""
    ctx = _ctx(make_ctx, line_kw={"gain_db": 12.0}, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    peak = _check(report, "true_peak")
    assert not peak.hard
    assert _number(peak.value["true_peak_dbtp"]) > 0


def test_cut_count_mismatch_is_soft(make_ctx: MakeCtx, no_stdin: None) -> None:
    """One-shot script, but the picture has a hard cut: recorded, the stage still passes."""
    ctx = _ctx(make_ctx, shots=(9.0,), build_final=_hard_cut, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    cuts = _check(report, "scene_cuts")
    assert not cuts.hard
    assert cuts.value["shots"] == 1
    assert _number(cuts.value["cuts"]) >= 1


@pytest.mark.parametrize(
    ("build", "line_kw", "shots"),
    [
        pytest.param(_frozen, {}, (5.0, 4.0), id="frozen"),
        pytest.param(_faithful, {"gain_db": 12.0}, (5.0, 4.0), id="hot"),
        pytest.param(_hard_cut, {}, (9.0,), id="cut_mismatch"),
    ],
)
def test_soft_signals_never_listed_as_failed(
    make_ctx: MakeCtx,
    no_stdin: None,
    build: BuildFinal,
    line_kw: dict[str, Any],
    shots: tuple[float, ...],
) -> None:
    ctx = _ctx(make_ctx, shots=shots, build_final=build, line_kw=line_kw, stt=ScriptedStt(LINE))
    report = _passes(ctx)
    assert not set(report.failed) & set(FINAL_SOFT_CHECKS)
    for name in FINAL_SOFT_CHECKS:
        assert not _check(report, name).hard
    assert all(_check(report, n).passed for n in FINAL_HARD_CHECKS)


# === spec change: the five new acceptance criteria ==============================================

EXTRACTED = "final_audio.wav"
STT_ERROR = "stt quota exceeded"


class FlakyStt:
    """Plays `outcomes` in order (last one repeats): a str is a transcript, an Exception is
    raised. Records every call."""

    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        self.calls.append(audio_path)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _assert_other_measurements(report: FinalReport) -> None:
    """Every check is in the report, and the ones that don't depend on STT were measured."""
    assert sorted(c.name for c in report.checks) == sorted(FINAL_HARD_CHECKS + FINAL_SOFT_CHECKS)
    for name in FINAL_HARD_CHECKS:
        if name != "line":
            assert _check(report, name).passed, _check(report, name)
    _number(_check(report, "format").value["duration_s"])
    _number(_check(report, "loudness").value["integrated_lufs"])
    _number(_check(report, "audio_match").value["peak"])


# --- AC: STT raising counts as a failed attempt (max 2 STT calls in total) ----------------------


def test_stt_raise_then_match_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = FlakyStt(ProviderError(STT_ERROR, retryable=True), LINE)
    ctx = _ctx(make_ctx, stt=stt)
    out = final_check.run(ctx)
    report = _report(ctx)
    assert report.verdict == "pass", report.failed
    assert FinalReport.model_validate(out.data) == report
    assert len(stt.calls) == 2
    assert report.transcript == LINE
    assert _check(report, "line").passed


def test_stt_raising_twice_fails_line_and_still_writes_report(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    stt = FlakyStt(ProviderError(STT_ERROR, retryable=True))
    ctx = _ctx(make_ctx, stt=stt)
    report = _fails(ctx, "line")
    assert len(stt.calls) == 2  # raise counts as an attempt: one retry, not more
    assert report.failed == ["line"]
    assert STT_ERROR in json.dumps(_check(report, "line").value, ensure_ascii=False)
    _assert_other_measurements(report)


def test_stt_raise_then_mishear_fails_after_two_calls(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = FlakyStt(ProviderError(STT_ERROR, retryable=True), MISHEARD, LINE)
    ctx = _ctx(make_ctx, stt=stt)
    report = _fails(ctx, "line")
    assert len(stt.calls) == 2  # raise + mishear = both attempts used; a 3rd is never made
    assert report.failed == ["line"]


# --- AC: missing / malformed script.json only affects scene_cuts --------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="missing"),
        pytest.param("{not json", id="not_json"),
        pytest.param('{"title": "no shots"}', id="schema_invalid"),
    ],
)
def test_bad_script_only_affects_scene_cuts(
    make_ctx: MakeCtx, no_stdin: None, content: str | None
) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    script = ctx.run_dir / "script.json"
    if content is None:
        script.unlink()
    else:
        script.write_text(content, encoding="utf-8")
    out = final_check.run(ctx)  # scene_cuts is soft: the verdict doesn't change
    report = _report(ctx)
    assert FinalReport.model_validate(out.data) == report
    assert report.verdict == "pass", report.failed
    cuts = _check(report, "scene_cuts")
    assert not cuts.hard
    assert cuts.value["shots"] is None
    assert str(cuts.value.get("error", "")).strip(), cuts.value
    assert all(_check(report, n).passed for n in FINAL_HARD_CHECKS)


# --- AC: extracted final_audio.wav is in the artifacts ------------------------------------------


def test_extracted_audio_is_listed_in_artifacts(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ProbingStt(LINE)
    ctx = _ctx(make_ctx, stt=stt)
    out = final_check.run(ctx)
    assert out.artifacts == ARTIFACTS
    assert out.artifacts["stt_audio"] == EXTRACTED
    assert (ctx.run_dir / EXTRACTED).is_file()
    assert stt.calls and stt.calls[0].name == EXTRACTED  # the listed file is the one STT heard


# --- AC: missing / unknown lang is an error, never a default to Ukrainian -----------------------


@pytest.mark.parametrize("lang", [None, "de", ""], ids=["missing", "unknown_de", "empty"])
def test_missing_or_unknown_lang_is_error(
    make_ctx: MakeCtx, no_stdin: None, lang: str | None
) -> None:
    stt = ScriptedStt(LINE)
    ctx = _ctx(make_ctx, stt=stt)
    if lang is None:
        del ctx.manifest.params["lang"]
    else:
        ctx.manifest.params["lang"] = lang
    report = _fails(ctx, "line")
    assert stt.calls == []  # STT is not called with a guessed language
    value = _check(report, "line").value
    assert "lang" in str(value.get("error", "")).lower(), value
    assert report.failed == ["line"]
    _assert_other_measurements(report)


# --- AC: every ffmpeg/ffprobe call has a timeout; TimeoutExpired -> recorded error --------------

PROBE_TIMEOUT_S = 30
RENDER_TIMEOUT_S = 120
_REAL_RUN = subprocess.run


def _cmd(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    cmd = args[0] if args else kwargs["args"]
    return [str(c) for c in cmd]


def _is_probe(cmd: list[str]) -> bool:
    return Path(cmd[0]).stem == "ffprobe"


def _is_video_analysis(cmd: list[str]) -> bool:
    return "-filter:v" in cmd


def _is_audio_analysis(cmd: list[str]) -> bool:
    return "-filter:a" in cmd


def _is_decode(cmd: list[str]) -> bool:
    return "f32le" in cmd


def _is_extraction(cmd: list[str]) -> bool:
    return cmd[-1].endswith(EXTRACTED)


def test_every_tool_call_has_the_right_timeout(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probes 30 s; analysis / extraction / decode 120 s. Every call kind is exercised."""
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    calls: list[tuple[list[str], Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((_cmd(args, kwargs), kwargs.get("timeout")))
        return _REAL_RUN(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    final_check.run(ctx)
    cmds = [cmd for cmd, _ in calls]
    kinds = {
        "probe": any(_is_probe(c) for c in cmds),
        "video analysis": any(_is_video_analysis(c) for c in cmds),
        "audio analysis": any(_is_audio_analysis(c) for c in cmds),
        "decode": any(_is_decode(c) for c in cmds),
        "extraction": any(_is_extraction(c) for c in cmds),
    }
    assert all(kinds.values()), kinds
    for cmd, timeout in calls:
        expected = PROBE_TIMEOUT_S if _is_probe(cmd) else RENDER_TIMEOUT_S
        assert timeout == expected, f"{' '.join(cmd)}: timeout={timeout!r}, want {expected}"


@pytest.mark.parametrize(
    ("matches", "check"),
    [
        pytest.param(_is_probe, "format", id="ffprobe"),
        pytest.param(_is_video_analysis, "black_frames", id="video_analysis"),
        pytest.param(_is_audio_analysis, "leading_silence", id="audio_analysis"),
        pytest.param(_is_decode, "audio_match", id="decode"),
        pytest.param(_is_extraction, "line", id="extraction"),
    ],
)
def test_tool_timeout_is_recorded_and_report_written(
    make_ctx: MakeCtx,
    no_stdin: None,
    monkeypatch: pytest.MonkeyPatch,
    matches: Callable[[list[str]], bool],
    check: str,
) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    hit: list[list[str]] = []

    def hang(*args: Any, **kwargs: Any) -> Any:
        cmd = _cmd(args, kwargs)
        if matches(cmd):
            hit.append(cmd)
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or RENDER_TIMEOUT_S)
        return _REAL_RUN(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", hang)
    report = _fails(ctx, check)  # GateFailed -> report; report written; verdict fail
    assert hit, "the targeted tool call was never made"
    value = _check(report, check).value
    assert "time" in str(value.get("error", "")).lower(), value  # "timed out" / "timeout"


# === audio_offset_s (specs/final_check.md; see specs/sync.md) ===================
# assemble records where it placed the line (`audio_offset_s` in its data, 0 when absent). The
# leading-silence allowance is 0.5 s + offset (value `allowed_s`), the audio match expects the
# line at the offset (value `expected_offset_ms`), +- 50 ms.

OFFSET_S = 1.2


def _record_offset(ctx: Context, offset_s: float) -> None:
    rec = ctx.manifest.stages["assemble"]
    rec.data = AssembleData(duration_s=rec.data["duration_s"], audio_offset_s=offset_s).model_dump(
        mode="json"
    )


def _no_offset(ctx: Context) -> None:
    """assemble data as written before the correction: no audio_offset_s key at all."""
    ctx.manifest.stages["assemble"].data.pop("audio_offset_s", None)


def _allowances(report: FinalReport) -> tuple[float, float]:
    """(leading_silence allowed_s, audio_match expected_offset_ms) as recorded in the report."""
    lead = _check(report, "leading_silence").value
    match = _check(report, "audio_match").value
    assert "allowed_s" in lead, lead
    assert "expected_offset_ms" in match, match
    return _number(lead["allowed_s"]), _number(match["expected_offset_ms"])


# --- AC: no audio_offset_s recorded -> the failing files fail exactly as before (offset 0) -------


def test_no_offset_recorded_means_zero_and_is_recorded(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    _no_offset(ctx)
    report = _passes(ctx)
    assert _allowances(report) == (pytest.approx(0.5), pytest.approx(0.0))


def test_no_offset_one_second_of_leading_silence_still_fails(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, line_kw={"lead_s": 1.0}, stt=ScriptedStt(LINE))
    _no_offset(ctx)
    report = _fails(ctx, "leading_silence")
    assert report.failed == ["leading_silence"]
    assert _allowances(report)[0] == pytest.approx(0.5)


def test_no_offset_late_audio_still_fails_the_match(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_delayed(300), stt=ScriptedStt(LINE))
    _no_offset(ctx)
    report = _fails(ctx, "audio_match")
    assert report.failed == ["audio_match"]
    assert _allowances(report)[1] == pytest.approx(0.0)


# --- AC: offset 1.2 recorded and the line at 1.2 s -> leading_silence and audio_match pass -------


def test_line_at_the_recorded_offset_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, build_final=_delayed(1200), stt=ScriptedStt(LINE))
    _record_offset(ctx, OFFSET_S)
    report = _passes(ctx)
    assert _check(report, "leading_silence").passed and _check(report, "audio_match").passed
    lead = _number(_check(report, "leading_silence").value["leading_silence_s"])
    assert lead == pytest.approx(1.2, abs=0.05)
    match = _check(report, "audio_match").value
    assert _number(match["peak"]) >= 0.9
    assert _number(match["offset_ms"]) == pytest.approx(1200, abs=50)
    assert _allowances(report) == (pytest.approx(1.7), pytest.approx(1200.0))


# --- AC: the same final with no offset recorded -> both fail ------------------------------------


def test_line_at_1_2_s_without_a_recorded_offset_fails_both(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, build_final=_delayed(1200), stt=ScriptedStt(LINE))
    _no_offset(ctx)
    report = _fails(ctx, "leading_silence")
    assert set(report.failed) == {"leading_silence", "audio_match"}
    assert _allowances(report) == (pytest.approx(0.5), pytest.approx(0.0))


# --- AC: offset 1.2 recorded, 2.0 s of leading silence -> leading_silence fails -------------------


def test_speech_later_than_offset_plus_half_a_second_fails(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, build_final=_delayed(2000), stt=ScriptedStt(LINE))
    _record_offset(ctx, OFFSET_S)
    report = _fails(ctx, "leading_silence")
    lead = _number(_check(report, "leading_silence").value["leading_silence_s"])
    assert lead == pytest.approx(2.0, abs=0.05)
    assert _allowances(report)[0] == pytest.approx(1.7)  # the offset, no more


# --- AC: offset 1.2 recorded and the line placed at 0 -> audio_match fails ----------------------


def test_line_at_zero_when_the_record_says_1_2_fails_the_match(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))  # faithful at 0
    _record_offset(ctx, OFFSET_S)
    report = _fails(ctx, "audio_match")
    assert report.failed == ["audio_match"]  # leading silence 0 <= 1.7 still passes
    assert _number(_check(report, "audio_match").value["offset_ms"]) == pytest.approx(0, abs=50)
    assert _allowances(report)[1] == pytest.approx(1200.0)


# --- AC: the match is searched up to the recorded offset even when the final is shorter than ------
# offset + line.wav (assemble cut line.wav's trailing silence): the final is zero-padded.
# line.wav = 2.5 s speech + 0.6 s trailing silence, offset 1.2, a 4.0 s final, the line at 1.2 s.

SHORT_SPEECH_S = 2.5
TRAILING_S = 0.6
SHORT_FINAL_S = 4.0


def _cut_trailing_silence(path: Path, line_wav: Path, seconds: float) -> None:
    """Rewrites line.wav as 2.5 s chirp + 0.6 s silence, then a final whose audio is that line at
    1.2 s, cut at the video end (as assemble does): 1.2 + 3.1 = 4.3 s > the 4.0 s final."""
    _ffmpeg(*_lavfi(CHIRP.format(d=SHORT_SPEECH_S)), "-af", f"apad=pad_dur={TRAILING_S}",
            str(line_wav))  # fmt: skip
    end = f"{seconds:.3f}"
    _ffmpeg(
        *_lavfi(_testsrc(seconds)), "-i", str(line_wav),
        "-filter_complex",
        f"[0:v]format=yuv420p[v];[1:a]adelay=1200:all=1,apad=whole_dur={end},atrim=end={end}[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", str(path),
    )  # fmt: skip


def test_match_is_found_at_the_offset_when_the_final_is_shorter_than_offset_plus_line(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(
        make_ctx, shots=(SHORT_FINAL_S,), build_final=_cut_trailing_silence, stt=ScriptedStt(LINE)
    )
    _record_offset(ctx, OFFSET_S)
    line_wav, final = ctx.run_dir / "line.wav", ctx.run_dir / FINAL
    # preconditions: the case really is "final shorter than offset + line.wav"
    assert duration(line_wav) == pytest.approx(SHORT_SPEECH_S + TRAILING_S, abs=0.02)
    assert duration(final) == pytest.approx(SHORT_FINAL_S, abs=0.1)
    assert duration(final) < OFFSET_S + duration(line_wav)

    with contextlib.suppress(GateFailed):  # a 4 s final fails `format`; only audio_match matters
        final_check.run(ctx)
    match = _check(_report(ctx), "audio_match")
    assert match.passed, match.value
    assert _number(match.value["peak"]) >= 0.9
    assert _number(match.value["offset_ms"]) == pytest.approx(1200, abs=50)
    assert _allowances(_report(ctx))[1] == pytest.approx(1200.0)


# === Aligned audio (specs/final_check.md "Aligned audio", v2) ========================
# The match compares against the audio assemble actually placed (artifact `audio`, absent ->
# line.wav) at assemble's `audio_offset_s` (negative: the reference is compared from its trimmed
# start); the leading-silence allowance is 0.5 s + `speech_start_s` (absent -> audio_offset_s).
# line.wav = two phrases back to back; aligned_line.wav = the same phrases moved one by one.

ALIGNED = "aligned_line.wav"
LINE_PHRASES = [(0.0, 0.8), (0.85, 1.55)]
ALIGNED_PHRASES = [(1.0, 1.8), (2.3, 3.0)]  # phrase 2 moved 0.45 s relative to line.wav at 1.0


def _aligned_case(
    make_ctx: MakeCtx,
    final_from: str,
    *,
    offset_s: float = 0.0,
    speech_start_s: float = 1.0,
) -> Context:
    """Context after an aligned assemble: line.wav, aligned_line.wav, the record (artifact
    `audio`, `audio_offset_s`, `speech_start_s`) and a final.mp4 built from `final_from`:
    "aligned" (the track at 0), "line_at_1" (line.wav at 1.0 s: phrases NOT moved),
    "aligned_late" (the track 0.7 s later), "aligned_trimmed" (the track from 0.3 s on)."""
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))
    run_dir = ctx.run_dir
    line = speech_wav(run_dir / "line.wav", LINE_PHRASES, 1.55)
    aligned = speech_wav(run_dir / ALIGNED, ALIGNED_PHRASES, 5.0)
    final, total = run_dir / FINAL, 9.0
    if final_from == "aligned":
        _final(final, total, audio=["-i", str(aligned)])
    elif final_from == "line_at_1":
        _final(final, total, audio=["-i", str(line)], delay_ms=1000)
    elif final_from == "aligned_late":
        _final(final, total, audio=["-i", str(aligned)], delay_ms=700)
    elif final_from == "aligned_trimmed":
        _final(final, total, audio=["-ss", "0.3", "-i", str(aligned)])
    else:
        raise AssertionError(final_from)
    rec = ctx.manifest.stages["assemble"]
    rec.artifacts = {"video": FINAL, "audio": ALIGNED}
    rec.data = AssembleData(
        duration_s=total, audio_offset_s=offset_s, audio_offset_wanted_s=0.0,
        audio_source="aligned", speech_start_s=speech_start_s,
    ).model_dump(mode="json")  # fmt: skip
    return ctx


def test_harness_aligned_track_really_moves_the_second_phrase() -> None:
    (a1, _), (b1, _) = LINE_PHRASES
    (a2, _), (b2, _) = ALIGNED_PHRASES
    assert a2 - a1 == pytest.approx(1.0)  # phrase 1 at the same place as line.wav at 1.0 s
    assert (b2 - b1) - (a2 - a1) >= 0.4  # phrase 2 moved by >= 0.4 s: the shapes differ


# --- AC: aligned final + record -> leading_silence (1.0 <= 1.5) and audio_match pass --------------


def test_aligned_final_passes_against_the_placed_track(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _aligned_case(make_ctx, "aligned")
    report = _passes(ctx)
    lead = _check(report, "leading_silence").value
    assert _number(lead["leading_silence_s"]) == pytest.approx(1.0, abs=0.05)
    assert _number(lead["allowed_s"]) == pytest.approx(1.5)  # 0.5 + speech_start_s
    match = _check(report, "audio_match").value
    assert match["reference"] == ALIGNED
    assert _number(match["peak"]) >= 0.9
    assert _number(match["offset_ms"]) == pytest.approx(0, abs=50)
    assert _number(match["expected_offset_ms"]) == pytest.approx(0.0)


# --- AC: same record, final from the contiguous line.wav at 1.0 s -> audio_match fails ------------


def test_unmoved_line_against_an_aligned_record_fails_the_match(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _aligned_case(make_ctx, "line_at_1")
    report = _fails(ctx, "audio_match")
    assert report.failed == ["audio_match"]  # leading silence 1.0 <= 1.5 still passes
    assert _check(report, "leading_silence").passed
    match = _check(report, "audio_match").value
    assert match["reference"] == ALIGNED
    assert _number(match["peak"]) < 0.9


# --- AC: same record, 1.7 s of leading silence -> leading_silence fails ---------------------------


def test_speech_later_than_the_first_phrase_plus_half_a_second_fails(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _aligned_case(make_ctx, "aligned_late")
    report = _fails(ctx, "leading_silence")
    lead = _check(report, "leading_silence").value
    assert _number(lead["leading_silence_s"]) == pytest.approx(1.7, abs=0.05)
    assert _number(lead["allowed_s"]) == pytest.approx(1.5)  # speech_start_s + 0.5, no more


# --- AC: a record without artifact `audio` -> compared with line.wav exactly as before -----------


def test_without_audio_artifact_the_reference_is_line_wav(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(LINE))  # assemble artifacts: {"video": FINAL} only
    report = _passes(ctx)
    assert _check(report, "audio_match").value["reference"] == "line.wav"
    assert _allowances(report) == (pytest.approx(0.5), pytest.approx(0.0))


def test_without_audio_artifact_another_source_still_fails(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _ctx(make_ctx, build_final=_other_chirp, stt=ScriptedStt(LINE))
    report = _fails(ctx, "audio_match")
    assert _check(report, "audio_match").value["reference"] == "line.wav"


# --- AC: audio_offset_s -0.3 and a final from the track with its first 0.3 s trimmed -> passes ----


def test_negative_offset_compares_the_trimmed_reference(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _aligned_case(make_ctx, "aligned_trimmed", offset_s=-0.3, speech_start_s=0.7)
    report = _passes(ctx)
    match = _check(report, "audio_match")
    assert match.passed and match.value["reference"] == ALIGNED
    assert _number(match.value["peak"]) >= 0.9
    lead = _check(report, "leading_silence").value
    assert _number(lead["leading_silence_s"]) == pytest.approx(0.7, abs=0.05)
    assert _number(lead["allowed_s"]) == pytest.approx(1.2)
