"""Acceptance tests for the `assemble` stage (specs/assemble.md). Real ffmpeg, no network."""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import sys
from array import array
from collections.abc import Callable
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from conftest import duration, make_audio, make_video, probe, speech_wav

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.schemas import (
    SHOT_2_SKIPPED,
    AssembleData,
    LineSource,
    Script,
    Shot,
    ShotData,
    StageStatus,
    SyncReport,
    VoiceData,
)
from hwva.stages import assemble

MakeCtx = Callable[..., Context]

LINE = "Я тут головний"
FINAL = "final.mp4"
# v2: artifact `audio` = the file placed (line.wav here: no aligned track recorded)
ARTIFACTS = {"video": FINAL, "audio": "line.wav"}
TTS_HZ = 440  # make_audio / FakeTts tone
SHOT_HZ = 880  # audio the "video model" put into its own clips
SAMPLE_RATE = 16000
SILENCE = 300  # |s16 sample| below this counts as silence (440 Hz sine peaks ~4000)


# --- helpers ------------------------------------------------------------------------------------


def _color_video(path: Path, seconds: float, color: str, size: str = "720x1280") -> Path:
    """Solid-colour clip with its own 880 Hz audio, so shot order can be read off the frames."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"color=c={color}:size={size}:rate=24:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency={SHOT_HZ}:duration={seconds}",
         "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


def _ctx(
    make_ctx: MakeCtx,
    *,
    shot_s: tuple[float, ...] = (3.0, 2.0),
    audio_s: float = 2.0,
    make_shot: Callable[[Path, float, int], Path] | None = None,
) -> Context:
    """Context as it is after `shot_2`: script, line.wav (440 Hz), shot clips WITH their own
    880 Hz audio, all upstream records DONE. One entry in `shot_s` = one-shot script."""
    params = {"line": LINE, "line_source": LineSource.INPUT.value, "lang": "uk"}
    ctx = make_ctx(idea=f'Кіт каже: "{LINE}"', params=params)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    shots = [
        Shot(visual_prompt=f"shot {i + 1}", duration_s=d, has_line=i == 0)
        for i, d in enumerate(shot_s)
    ]
    script = Script(
        title="test",
        line=LINE,
        line_source=LineSource.INPUT,
        lang="uk",
        character="a ginger cat with green eyes",
        shots=shots,
    )
    (ctx.run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    rec = mf.stage(ctx.manifest, "script")
    rec.status, rec.artifacts = StageStatus.DONE, {"script": "script.json"}

    make_audio(ctx.run_dir / "line.wav", audio_s, LINE)
    rec = mf.stage(ctx.manifest, "voice")
    rec.status = StageStatus.DONE
    rec.artifacts = {"audio": "line.wav", "transcript": "transcript.txt"}
    rec.data = VoiceData(
        audio_duration_s=audio_s, planned_total_s=sum(shot_s), attempts=1
    ).model_dump(mode="json")

    build = make_shot or (lambda p, s, i: make_video(p, s, audio=True, tone=SHOT_HZ))
    build(ctx.run_dir / "shot_1.mp4", shot_s[0], 0)
    rec = mf.stage(ctx.manifest, "shot_1")
    rec.status = StageStatus.DONE
    rec.artifacts = {"video": "shot_1.mp4", "shot_1_last_frame": "shot_1.last.png"}
    rec.data = ShotData(duration_s=shot_s[0]).model_dump(mode="json")

    rec = mf.stage(ctx.manifest, "shot_2")
    rec.status = StageStatus.DONE
    if len(shot_s) > 1:
        build(ctx.run_dir / "shot_2.mp4", shot_s[1], 1)
        rec.artifacts = {"video": "shot_2.mp4"}
        rec.data = ShotData(duration_s=shot_s[1]).model_dump(mode="json")
    else:
        rec.artifacts, rec.data = {}, dict(SHOT_2_SKIPPED)
    return ctx


def _streams(path: Path, kind: str) -> list[dict[str, Any]]:
    return [s for s in probe(path)["streams"] if s["codec_type"] == kind]


def _pcm(path: Path) -> array[int]:
    """First audio stream decoded to mono 16 kHz signed 16-bit samples."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    samples = array("h")
    samples.frombytes(out.stdout[: len(out.stdout) // 2 * 2])
    return samples


def _dominant_hz(samples: array[int], start_s: float, end_s: float) -> float:
    """Zero-crossing frequency estimate; exact enough for a pure sine, and a 440+880 mix reads
    as ~880, so mixing the shots' audio in is caught too."""
    window = samples[int(start_s * SAMPLE_RATE) : int(end_s * SAMPLE_RATE)]
    crossings = sum(1 for a, b in pairwise(window) if (a < 0) != (b < 0))
    return crossings / 2 / (len(window) / SAMPLE_RATE)


def _peak(samples: array[int], start_s: float, end_s: float) -> int:
    window = samples[int(start_s * SAMPLE_RATE) : int(end_s * SAMPLE_RATE)]
    return max((abs(x) for x in window), default=0)


def _rgb_at(path: Path, t: float) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    r, g, b = out.stdout[:3]
    return r, g, b


def _assemble_stage() -> pipeline.Stage:
    stage = next((s for s in pipeline.STAGES if s.name == "assemble"), None)
    assert stage is not None, "assemble is not registered in pipeline.STAGES"
    return stage


# --- AC: exactly one video and one audio stream --------------------------------------------------


@pytest.mark.parametrize("shot_s", [(3.0, 2.0), (3.0,)])
def test_exactly_one_video_and_one_audio_stream(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...]
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s)
    out = assemble.run(ctx)
    assert isinstance(out, StageOutput)
    assert out.artifacts == ARTIFACTS
    final = ctx.run_dir / FINAL
    assert final.is_file()
    streams = probe(final)["streams"]
    assert len(streams) == 2
    assert len(_streams(final, "video")) == 1
    assert len(_streams(final, "audio")) == 1


# --- Behaviour: common format (H.264, yuv420p, 24 fps, 720x1280, AAC) ---------------------------


def test_reencodes_to_common_format(make_ctx: MakeCtx, no_stdin: None) -> None:
    def mixed(path: Path, seconds: float, i: int) -> Path:
        # the second clip comes in a different size: assemble must normalise it
        size = "720x1280" if i == 0 else "1080x1920"
        return make_video(path, seconds, size=size, audio=True, tone=SHOT_HZ)

    ctx = _ctx(make_ctx, make_shot=mixed)
    assemble.run(ctx)
    final = ctx.run_dir / FINAL
    (v,) = _streams(final, "video")
    (a,) = _streams(final, "audio")
    assert v["codec_name"] == "h264"
    assert v["pix_fmt"] == "yuv420p"
    assert (v["width"], v["height"]) == (720, 1280)
    assert Fraction(v["avg_frame_rate"]) == 24
    assert a["codec_name"] == "aac"


# --- AC: duration = sum of shot durations (±0.2 s); never time-stretched -------------------------


@pytest.mark.parametrize("shot_s", [(3.0, 2.0), (4.0, 5.0), (3.0,)])
def test_duration_is_sum_of_shots(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...]
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s)
    out = assemble.run(ctx)
    final = ctx.run_dir / FINAL
    total = sum(shot_s)
    assert duration(final) == pytest.approx(total, abs=0.2)
    (v,) = _streams(final, "video")
    (a,) = _streams(final, "audio")
    assert float(v["duration"]) == pytest.approx(total, abs=0.2)
    assert float(a["duration"]) == pytest.approx(total, abs=0.2)  # padded to the video length
    data = AssembleData.model_validate(out.data)
    assert data.duration_s == pytest.approx(duration(final), abs=0.05)


def test_shots_concatenated_in_order(make_ctx: MakeCtx, no_stdin: None) -> None:
    colors = ("red", "blue")
    ctx = _ctx(make_ctx, make_shot=lambda p, s, i: _color_video(p, s, colors[i]))
    assemble.run(ctx)
    final = ctx.run_dir / FINAL
    r, _, b = _rgb_at(final, 1.5)  # inside shot 1 (0..3 s)
    assert r > 150 and b < 100
    r, _, b = _rgb_at(final, 4.0)  # inside shot 2 (3..5 s)
    assert b > 150 and r < 100


# --- AC: the audio is line.wav, not the shots' own audio ----------------------------------------


@pytest.mark.parametrize("shot_s", [(3.0, 2.0), (3.0,)])
def test_audio_is_line_wav_not_shot_audio(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...]
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=2.0)
    # precondition: the shots really carry their own 880 Hz audio
    shot_pcm = _pcm(ctx.run_dir / "shot_1.mp4")
    assert _dominant_hz(shot_pcm, 0.2, 1.8) == pytest.approx(SHOT_HZ, rel=0.05)

    assemble.run(ctx)
    samples = _pcm(ctx.run_dir / FINAL)
    assert _dominant_hz(samples, 0.2, 1.8) == pytest.approx(TTS_HZ, rel=0.05)


def test_audio_starts_at_zero_and_is_padded_with_silence(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    assemble.run(ctx)
    final = ctx.run_dir / FINAL
    (a,) = _streams(final, "audio")
    assert float(a.get("start_time", 0.0)) == pytest.approx(0.0, abs=0.05)
    samples = _pcm(final)
    assert _peak(samples, 0.0, 0.1) > SILENCE  # the line starts at once, no delay
    assert _peak(samples, 2.3, 4.8) < SILENCE  # after line.wav: silence, not the shots' audio
    assert len(samples) / SAMPLE_RATE == pytest.approx(5.0, abs=0.2)


def test_line_audio_not_stretched(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, shot_s=(4.0, 5.0), audio_s=2.5)
    assemble.run(ctx)
    samples = _pcm(ctx.run_dir / FINAL)
    loud = [i for i, x in enumerate(samples) if abs(x) > SILENCE]
    assert loud
    spoken_s = (loud[-1] - loud[0]) / SAMPLE_RATE
    assert spoken_s == pytest.approx(2.5, abs=0.1)  # same length as line.wav
    assert _dominant_hz(samples, 0.2, 2.3) == pytest.approx(TTS_HZ, rel=0.05)  # same pitch


def test_assemble_done_in_pipeline(make_ctx: MakeCtx, no_stdin: None, runs_root: Path) -> None:
    ctx = _ctx(make_ctx)
    result = pipeline.run(ctx, [_assemble_stage()], root=runs_root)
    record = result.stages["assemble"]
    assert record.status is StageStatus.DONE
    assert record.artifacts == ARTIFACTS
    assert AssembleData.model_validate(record.data).duration_s == pytest.approx(5.0, abs=0.2)
    assert record.cost_usd == 0.0
    assert _dominant_hz(_pcm(ctx.run_dir / FINAL), 0.2, 1.8) == pytest.approx(TTS_HZ, rel=0.05)


# --- AC: END OF SPEECH (silencedetect n=-40dB d=0.3), not file length, vs. video length ----------


def _speech_then_silence(path: Path, speech_s: float, silence_s: float) -> Path:
    """line.wav = `speech_s` of 440 Hz tone followed by `silence_s` of digital silence."""
    af = ["-af", f"apad=pad_dur={silence_s}"] if silence_s > 0 else []
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency={TTS_HZ}:duration={speech_s}", *af, str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.mark.parametrize(
    ("shot_s", "speech_s", "file_s"),
    [
        ((3.0,), 2.0, 3.5),  # trailing silence 0.5 s past the video end
        ((3.0,), 2.6, 3.4),  # speech ends 0.4 s before the video end, silence runs past it
        ((2.0, 1.0), 2.0, 3.6),  # two shots; silence runs 0.6 s past the total
    ],
)
def test_trailing_silence_past_video_end_is_ok(
    make_ctx: MakeCtx,
    no_stdin: None,
    shot_s: tuple[float, ...],
    speech_s: float,
    file_s: float,
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=2.0)
    line = _speech_then_silence(ctx.run_dir / "line.wav", speech_s, file_s - speech_s)
    total = sum(shot_s)
    assert duration(line) == pytest.approx(file_s, abs=0.05)  # precondition: file > video
    assert duration(line) > total

    out = assemble.run(ctx)
    final = ctx.run_dir / FINAL
    assert duration(final) == pytest.approx(total, abs=0.2)
    assert AssembleData.model_validate(out.data).duration_s == pytest.approx(total, abs=0.2)
    assert len(_streams(final, "video")) == 1
    assert len(_streams(final, "audio")) == 1
    samples = _pcm(final)
    # the whole spoken part survives, nothing stretched
    assert _peak(samples, speech_s - 0.15, speech_s - 0.05) > SILENCE
    assert _dominant_hz(samples, 0.2, speech_s - 0.2) == pytest.approx(TTS_HZ, rel=0.05)


@pytest.mark.parametrize(
    ("shot_s", "overrun_s", "silence_s"),
    [
        ((2.0,), 0.01, 0.0),
        ((2.0,), 0.03, 0.0),
        ((2.0, 1.0), 0.01, 0.0),
        ((2.0, 1.0), 0.03, 0.0),
        ((2.0,), 0.01, 0.5),
        ((2.0,), 0.03, 0.5),
        ((2.0,), 0.1, 0.0),
        ((2.0,), 0.2, 0.0),
        ((2.0,), 0.3, 0.0),
        ((2.0, 1.0), 0.1, 0.0),
        ((2.0,), 0.1, 0.5),  # speech overruns, then trailing silence
        ((2.0,), 0.3, 0.5),
        ((2.0, 1.0), 0.1, 0.5),
        ((2.0, 1.0), 0.2, 0.5),
        ((2.0,), 1.0, 0.5),
    ],
)
def test_speech_past_video_end_fails_gate(
    make_ctx: MakeCtx,
    no_stdin: None,
    shot_s: tuple[float, ...],
    overrun_s: float,
    silence_s: float,
) -> None:
    """Spec: speech running past the video end by ANY amount -> GateFailed, no tolerance;
    the audio is never trimmed."""
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=2.0)
    _speech_then_silence(ctx.run_dir / "line.wav", sum(shot_s) + overrun_s, silence_s)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: audio longer than the video -> GateFailed (a pure tone is speech up to EOF) -------------


@pytest.mark.parametrize(("shot_s", "audio_s"), [((2.0,), 3.0), ((2.0, 1.0), 4.0)])
def test_audio_longer_than_video_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...], audio_s: float
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=audio_s)
    with pytest.raises(GateFailed):
        assemble.run(ctx)


@pytest.mark.parametrize(
    ("shot_s", "overrun_s"),
    [
        ((2.0,), 0.06),
        ((2.0,), 0.1),
        ((2.0, 1.0), 0.1),
        ((2.0,), 0.01),
        ((2.0,), 0.03),
        ((2.0, 1.0), 0.01),
        ((2.0, 1.0), 0.03),
    ],
)
def test_audio_slightly_longer_than_video_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...], overrun_s: float
) -> None:
    """Spec: ANY audio longer than the video -> GateFailed. A tiny overrun must not be absorbed
    by a tolerance and then trimmed away with -t: that would clip the verified line."""
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=sum(shot_s) + overrun_s)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: a short non-silent burst at EOF is speech -----------------------------------------------


def _burst_at_eof(path: Path, lead_s: float, gap_s: float, burst_s: float) -> Path:
    """line.wav = `lead_s` of tone, `gap_s` of silence, then a `burst_s` tone ending at EOF."""
    srcs = [f"anullsrc=r=44100:cl=mono:d={gap_s}", f"sine=f={TTS_HZ}:d={burst_s}:r=44100"]
    if lead_s > 0:
        srcs.insert(0, f"sine=f={TTS_HZ}:d={lead_s}:r=44100")
    inputs = [arg for src in srcs for arg in ("-f", "lavfi", "-i", src)]
    graph = "".join(f"[{i}:a]" for i in range(len(srcs))) + f"concat=n={len(srcs)}:v=0:a=1[a]"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", graph, "-map", "[a]", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.mark.parametrize(
    ("shot_s", "lead_s", "gap_s"),
    [
        ((2.0,), 1.0, 1.2),  # speech, silence, then a 40 ms burst at 2.20-2.24 s
        ((2.0,), 0.0, 2.2),  # only silence, then the burst past the video end
        ((2.0, 1.0), 1.5, 1.6),  # two shots; burst at 3.10-3.14 s
    ],
)
def test_short_burst_at_eof_past_video_end_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...], lead_s: float, gap_s: float
) -> None:
    """A short final non-silent burst at EOF is still speech: the speech-end detector must not
    drop it (silencedetect reports no silence after it, and it is shorter than d=0.3)."""
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=2.0)
    line = _burst_at_eof(ctx.run_dir / "line.wav", lead_s, gap_s, 0.04)
    assert duration(line) == pytest.approx(lead_s + gap_s + 0.04, abs=0.02)  # precondition
    assert lead_s + gap_s > sum(shot_s)  # the burst lies entirely past the video end
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: malformed script.json -> GateFailed -----------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param("{not json", id="not_json"),
        pytest.param('{"title": "x"}', id="missing_fields"),
        pytest.param("[1, 2, 3]", id="wrong_type"),
        pytest.param(
            '{"title": "t", "line": "x", "line_source": "input", "lang": "uk", '
            '"character": "c", "shots": []}',
            id="no_shots",
        ),
    ],
)
def test_malformed_script_fails_gate(make_ctx: MakeCtx, no_stdin: None, content: str) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    (ctx.run_dir / "script.json").write_text(content, encoding="utf-8")
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: missing ffmpeg binary -> GateFailed -----------------------------------------------------


def _is_tool(cmd: Any, *names: str) -> bool:
    if isinstance(cmd, (list, tuple)) and cmd:
        return Path(str(cmd[0])).stem in names
    if isinstance(cmd, str):
        return Path(cmd.split()[0]).stem in names
    return False


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param(("ffmpeg", "ffprobe"), id="both"),
        pytest.param(("ffmpeg",), id="ffmpeg_only"),
        pytest.param(("ffprobe",), id="ffprobe_only"),
    ],
)
def test_missing_ffmpeg_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, missing: tuple[str, ...]
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    real_which = shutil.which

    def no_binary(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_tool(cmd, *missing):
            name = cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd).split()[0]
            raise FileNotFoundError(2, "The system cannot find the file specified", name)
        return _real_run(cmd, *args, **kwargs)

    def which(name: str, *args: Any, **kwargs: Any) -> str | None:
        return None if Path(name).stem in missing else real_which(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_binary)
    monkeypatch.setattr(shutil, "which", which)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    monkeypatch.undo()
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: every ffmpeg/ffprobe call has a timeout; TimeoutExpired -> GateFailed --------------------

# Spec values (specs/assemble.md), deliberately not imported from the stage.
RENDER_TIMEOUT_S = 120  # renders and analysis (ffmpeg)
PROBE_TIMEOUT_S = 30  # ffprobe


@pytest.mark.parametrize("shot_s", [(3.0, 2.0), (3.0,)])
def test_every_tool_call_has_timeout(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, shot_s: tuple[float, ...]
) -> None:
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=2.0)
    calls: list[tuple[Any, dict[str, Any]]] = []

    def spy(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((cmd, dict(kwargs)))
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    assemble.run(ctx)
    monkeypatch.undo()

    probes = [kw for cmd, kw in calls if _is_tool(cmd, "ffprobe")]
    ffmpegs = [kw for cmd, kw in calls if _is_tool(cmd, "ffmpeg")]
    assert probes, "no ffprobe call was seen"
    assert ffmpegs, "no ffmpeg call was seen"
    assert len(probes) + len(ffmpegs) == len(calls)  # nothing else is shelled out
    assert all(kw.get("timeout") == PROBE_TIMEOUT_S for kw in probes), probes
    assert all(kw.get("timeout") == RENDER_TIMEOUT_S for kw in ffmpegs), ffmpegs


def test_tool_output_is_decoded_as_utf8_with_replacement(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a non-UTF-8 byte in ffmpeg's stderr (cp1251 file name, broken metadata) must not crash
    ctx = _ctx(make_ctx, shot_s=(3.0,), audio_s=2.0)
    calls: list[dict[str, Any]] = []

    def spy(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    assemble.run(ctx)
    monkeypatch.undo()
    assert calls
    assert all(kw.get("encoding") == "utf-8" and kw.get("errors") == "replace" for kw in calls)


def test_non_utf8_tool_stderr_is_a_gate_failure_not_a_crash() -> None:
    # 0x81 is invalid in UTF-8 and undefined in cp1252: strict decoding would raise
    script = "import sys; sys.stderr.buffer.write(b'bad \\x81\\xff byte'); sys.exit(1)"
    with pytest.raises(GateFailed, match=r"bad \S+ byte"):
        assemble._run_tool([sys.executable, "-c", script], "ffmpeg render of final video")


def _is_render(ctx: Context, cmd: Any) -> bool:
    return (
        _is_tool(cmd, "ffmpeg")
        and isinstance(cmd, (list, tuple))
        and Path(str(cmd[-1])).suffix == ".mp4"
        and Path(str(cmd[-1])).parent == ctx.run_dir
        and not Path(str(cmd[-1])).name.startswith("shot_")
    )


def _is_analysis(cmd: Any) -> bool:
    return _is_tool(cmd, "ffmpeg") and any("silencedetect" in str(a) for a in cmd)


_TIMEOUT_AT: dict[str, Callable[[Context, Any], bool]] = {
    "ffprobe": lambda ctx, cmd: _is_tool(cmd, "ffprobe"),
    "silencedetect": lambda ctx, cmd: _is_analysis(cmd),
    "render": _is_render,
}


@pytest.mark.parametrize("where", list(_TIMEOUT_AT))
def test_timeout_expired_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    hit = _TIMEOUT_AT[where]
    fired: list[Any] = []

    def hang(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if hit(ctx, cmd):
            fired.append(cmd)
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    monkeypatch.undo()
    assert fired, f"no {where} call was made"
    assert not (ctx.run_dir / FINAL).exists()


# --- shots without their own audio stream --------------------------------------------------------


@pytest.mark.parametrize("shot_s", [(3.0, 2.0), (3.0,)])
def test_shots_without_audio_stream_assemble(
    make_ctx: MakeCtx, no_stdin: None, shot_s: tuple[float, ...]
) -> None:
    ctx = _ctx(
        make_ctx,
        shot_s=shot_s,
        audio_s=2.0,
        make_shot=lambda p, s, i: make_video(p, s, audio=False),
    )
    assert _streams(ctx.run_dir / "shot_1.mp4", "audio") == []  # precondition: silent clip
    assemble.run(ctx)
    final = ctx.run_dir / FINAL
    assert len(_streams(final, "video")) == 1
    assert len(_streams(final, "audio")) == 1
    assert duration(final) == pytest.approx(sum(shot_s), abs=0.2)
    samples = _pcm(final)
    assert _dominant_hz(samples, 0.2, 1.8) == pytest.approx(TTS_HZ, rel=0.05)


# --- two-shot script with shot 2 missing -> GateFailed --------------------------------------------
# The expected shot count comes from the script artifact (manifest.stages["script"].artifacts
# ["script"] -> script.json, `len(Script.shots)`). A two-shot script must never be silently
# assembled from shot 1 only.


def _drop_shot_2_record(ctx: Context) -> None:
    del ctx.manifest.stages["shot_2"]


def _drop_shot_2_artifact(ctx: Context) -> None:
    ctx.manifest.stages["shot_2"].artifacts = {}


def _delete_shot_2_file(ctx: Context) -> None:
    (ctx.run_dir / "shot_2.mp4").unlink()


@pytest.mark.parametrize(
    "break_shot_2",
    [
        pytest.param(_drop_shot_2_record, id="no_record"),
        pytest.param(_drop_shot_2_artifact, id="no_video_artifact"),
        pytest.param(_delete_shot_2_file, id="file_missing"),
    ],
)
def test_two_shot_script_with_shot_2_missing_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, break_shot_2: Callable[[Context], None]
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    script = Script.model_validate_json((ctx.run_dir / "script.json").read_text(encoding="utf-8"))
    assert len(script.shots) == 2  # precondition: the script promises two shots
    break_shot_2(ctx)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


def test_audio_longer_than_video_fails_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, shot_s=(2.0,), audio_s=3.0)
    result = pipeline.run(ctx, [_assemble_stage()], root=runs_root)
    record = result.stages["assemble"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")


# --- corrupt shot input -> GateFailed (not CalledProcessError / ValueError) ----------------------


def _zero_byte(path: Path) -> None:
    path.write_bytes(b"")


def _garbage(path: Path) -> None:
    path.write_bytes(b"this is not an mp4 file\x00\x01\x02" * 64)


def _truncated(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:200])


_CORRUPT = [
    pytest.param(_zero_byte, id="zero_byte"),
    pytest.param(_garbage, id="garbage"),
    pytest.param(_truncated, id="truncated"),
]


@pytest.mark.parametrize("corrupt", _CORRUPT)
@pytest.mark.parametrize("shot_name", ["shot_1.mp4", "shot_2.mp4"])
def test_corrupt_shot_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, shot_name: str, corrupt: Callable[[Path], None]
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    corrupt(ctx.run_dir / shot_name)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


@pytest.mark.parametrize("corrupt", _CORRUPT)
def test_corrupt_shot_fails_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, corrupt: Callable[[Path], None]
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    corrupt(ctx.run_dir / "shot_1.mp4")
    result = pipeline.run(ctx, [_assemble_stage()], root=runs_root)
    record = result.stages["assemble"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and record.error.startswith("GateFailed"), record.error


# --- rendered final.mp4 is verified: 1 video + 1 audio stream, duration = shot total ±0.2 s ------
# ffmpeg is real; the only stand-in is a wrapper around subprocess.run that lets the real render
# happen and then damages the file it wrote to the run dir, as a broken encoder/filter graph would.

_real_run = subprocess.run


def _remux(src: Path, dst: Path, *args: str) -> None:
    _real_run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *args, str(dst)], check=True)


def _extra_audio(src: Path, dst: Path) -> None:
    _real_run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-i", str(src),
         "-map", "0:v", "-map", "0:a", "-map", "1:a", "-c", "copy", str(dst)],
        check=True,
    )  # fmt: skip


def _no_audio(src: Path, dst: Path) -> None:
    _remux(src, dst, "-map", "0:v", "-c", "copy")


def _extra_video(src: Path, dst: Path) -> None:
    _real_run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-i", str(src),
         "-map", "0:v", "-map", "1:v", "-map", "0:a", "-c", "copy", str(dst)],
        check=True,
    )  # fmt: skip


def _too_short(src: Path, dst: Path) -> None:
    _remux(src, dst, "-t", "4.5", "-c:v", "libx264", "-c:a", "aac")  # 5 s expected


def _too_long(src: Path, dst: Path) -> None:
    _remux(src, dst, "-vf", "tpad=stop_mode=clone:stop_duration=1", "-af", "apad=pad_dur=1",
           "-c:v", "libx264", "-c:a", "aac")  # fmt: skip


_DAMAGE = [
    pytest.param(_extra_audio, id="two_audio_streams"),
    pytest.param(_no_audio, id="no_audio_stream"),
    pytest.param(_extra_video, id="two_video_streams"),
    pytest.param(_too_short, id="duration_minus_0.5s"),
    pytest.param(_too_long, id="duration_plus_1s"),
]


def _bad_output(path: Path) -> bool:
    n_v, n_a = len(_streams(path, "video")), len(_streams(path, "audio"))
    return (n_v, n_a) != (1, 1) or abs(duration(path) - 5.0) > 0.2


def _intercept_render(
    ctx: Context,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    damage: Callable[[Path, Path], None],
) -> list[Path]:
    """Patch subprocess.run: ffmpeg writing an .mp4 into the run dir (other than a shot) renders
    to a scratch file for real, then `damage` writes the broken result to the requested path.
    Returns the list of paths that were damaged (filled in as the stage runs)."""
    damaged: list[Path] = []

    def render_then_damage(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        out = Path(cmd[-1]) if isinstance(cmd, list) and cmd else None
        is_render = (
            out is not None
            and Path(str(cmd[0])).stem == "ffmpeg"
            and out.suffix == ".mp4"
            and out.parent == ctx.run_dir
            and not out.name.startswith("shot_")
        )
        if not is_render:
            return _real_run(cmd, *args, **kwargs)
        assert out is not None
        pristine = tmp_path / f"pristine{out.suffix}"
        result = _real_run([*cmd[:-1], str(pristine)], *args, **kwargs)
        damage(pristine, out)
        keep = tmp_path / f"damaged_{len(damaged)}{out.suffix}"
        shutil.copyfile(out, keep)  # survives even if the stage deletes the bad output
        damaged.append(keep)
        return result

    monkeypatch.setattr(subprocess, "run", render_then_damage)
    return damaged


@pytest.mark.parametrize("damage", _DAMAGE)
def test_render_interceptor_really_damages_output(
    make_ctx: MakeCtx,
    no_stdin: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    damage: Callable[[Path, Path], None],
) -> None:
    """Harness check (not xfail): the interceptor fires on the real render and the file it leaves
    really violates the stream/duration contract, whether or not the stage catches it."""
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    damaged = _intercept_render(ctx, monkeypatch, tmp_path, damage)
    with contextlib.suppress(GateFailed):
        assemble.run(ctx)
    monkeypatch.undo()
    assert damaged, "the render was not intercepted"
    assert all(_bad_output(p) for p in damaged)


@pytest.mark.parametrize("damage", _DAMAGE)
def test_rendered_output_is_verified(
    make_ctx: MakeCtx,
    no_stdin: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    damage: Callable[[Path, Path], None],
) -> None:
    ctx = _ctx(make_ctx, shot_s=(3.0, 2.0), audio_s=2.0)
    _intercept_render(ctx, monkeypatch, tmp_path, damage)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    monkeypatch.undo()
    assert not (ctx.run_dir / FINAL).exists()  # a failed re-probe leaves no final.mp4


# === Lip-sync correction (specs/assemble.md; see specs/sync.md) ================
# The line starts at audio_offset_s = max(0, voice lead_in_s + shot_1 sync.shift_s).

CHIRP = "aevalsrc=exprs=0.5*sin(2*PI*(200*t+150*t*t)):s=44100:d={d}"  # not periodic: one xcorr peak
LINE_S = 3.0
BOX = [0.20, 0.80, 0.42, 0.60]


def _chirp_line(path: Path, seconds: float = LINE_S) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", CHIRP.format(d=seconds),
         str(path)],
        check=True,
    )  # fmt: skip
    return path


def _sync_record(shift_s: float) -> SyncReport:
    return SyncReport(
        lag_s=shift_s, peak_r=0.71, r_at_0=0.16, verdict="correct", shift_s=shift_s, box=BOX
    )


def _shift_ctx(
    make_ctx: MakeCtx,
    *,
    lead_in: float | None,
    shift: float | None,
    shot_s: tuple[float, ...] = (5.0, 4.0),
    chirp: bool = True,
    audio_s: float = 2.0,
) -> Context:
    """`_ctx` plus the recorded lead-in (voice data) and lip-sync shift (shot_1 data `sync`);
    None = not recorded (a run from before the correction). line.wav = a 3 s chirp."""
    ctx = _ctx(make_ctx, shot_s=shot_s, audio_s=audio_s)
    line = ctx.run_dir / "line.wav"
    if chirp:
        _chirp_line(line)
    voice = VoiceData.model_validate(ctx.manifest.stages["voice"].data)
    if lead_in is not None:
        voice = voice.model_copy(update={"lead_in_s": lead_in})
    voice = voice.model_copy(update={"audio_duration_s": duration(line)})
    ctx.manifest.stages["voice"].data = voice.model_dump(mode="json")
    if shift is not None:
        shot = ShotData(duration_s=shot_s[0], sync=_sync_record(shift))
        ctx.manifest.stages["shot_1"].data = shot.model_dump(mode="json")
        ctx.manifest.stages["shot_1"].artifacts["sync"] = "sync.json"
    return ctx


def _floats(path: Path) -> npt.NDArray[np.float64]:
    return np.asarray(_pcm(path), dtype=np.float64)


def _xcorr_lag_s(final: Path, ref: Path) -> float:
    """Lag (s) of `ref` (line.wav) inside the final audio: argmax of the cross-correlation."""
    a, b = _floats(final), _floats(ref)
    size = 1 << int(len(a) + len(b) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)[: len(a)]
    return float(np.argmax(corr)) / SAMPLE_RATE


def _leading_silence_s(path: Path) -> float:
    """silencedetect=n=-40dB:d=0.4: end of a silence that starts at 0, else 0."""
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
         "-af", "silencedetect=n=-40dB:d=0.4", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stderr  # fmt: skip
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log)]
    if not starts or starts[0] > 0.05:
        return 0.0
    return ends[0] if ends else duration(path)


# --- AC: lead-in 0.7 + shift 0.5 -> offset 1.2; 1.2 s of silence, then the TTS at 1.2 s ---------


def test_lead_in_plus_shift_moves_the_line_to_the_offset(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5)
    out = assemble.run(ctx)
    assert out.data["audio_offset_s"] == 1.2
    assert AssembleData.model_validate(out.data).audio_offset_s == 1.2
    final = ctx.run_dir / FINAL
    assert _leading_silence_s(final) == pytest.approx(1.2, abs=0.05)  # silent first 1.2 s
    assert _xcorr_lag_s(final, ctx.run_dir / "line.wav") == pytest.approx(1.2, abs=0.05)
    assert duration(final) == pytest.approx(9.0, abs=0.2)  # still the shot total
    samples = _pcm(final)
    spoken = [i for i, x in enumerate(samples) if abs(x) > SILENCE]
    assert (spoken[-1] - spoken[0]) / SAMPLE_RATE == pytest.approx(LINE_S, abs=0.1)  # whole line
    assert len(_streams(final, "audio")) == 1 and len(_streams(final, "video")) == 1


@pytest.mark.parametrize(
    ("lead_in", "shift", "expected"),
    [(0.7, None, 0.7), (0.0, 0.3, 0.3), (0.7, -0.5, 0.2), (0.7, 0.123456, 0.823)],
    ids=["lead_in_only", "shift_only", "negative_shift", "rounded_to_ms"],
)
def test_audio_offset_is_lead_in_plus_shift(
    make_ctx: MakeCtx, no_stdin: None, lead_in: float, shift: float | None, expected: float
) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=lead_in, shift=shift)
    out = assemble.run(ctx)
    assert out.data["audio_offset_s"] == pytest.approx(expected, abs=1e-9)
    lag = _xcorr_lag_s(ctx.run_dir / FINAL, ctx.run_dir / "line.wav")
    assert lag == pytest.approx(expected, abs=0.05)


def test_offset_is_recorded_in_the_manifest(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5)
    result = pipeline.run(ctx, [_assemble_stage()], root=runs_root)
    record = result.stages["assemble"]
    assert record.status is StageStatus.DONE, record.error
    assert AssembleData.model_validate(record.data).audio_offset_s == 1.2


# --- AC: an older run (no lead_in_s, no sync record) -> offset 0, the same output as before -----
# Regression guards (not xfail): today's behaviour IS the required behaviour for these.


def test_older_run_has_offset_zero(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=None, shift=None)
    out = assemble.run(ctx)
    assert out.data["audio_offset_s"] == 0.0
    final = ctx.run_dir / FINAL
    assert _peak(_pcm(final), 0.0, 0.1) > SILENCE  # the line starts at once
    assert _xcorr_lag_s(final, ctx.run_dir / "line.wav") == pytest.approx(0.0, abs=0.05)


# --- AC: negative shift larger than the lead-in -> offset 0, never a trimmed line ----------------


_NUMBER_RE = re.compile(r"[-−]?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    """Every number written in `text` (a leading '-' or U+2212 minus is the sign)."""
    return [float(m.replace("−", "-")) for m in _NUMBER_RE.findall(text)]


def _names_value(numbers: list[float], value: float) -> bool:
    return any(abs(n - value) <= 1e-3 for n in numbers)


def _new_assumptions(ctx: Context, before: list[str]) -> list[str]:
    return [a for a in ctx.manifest.assumptions if a not in before]


def test_negative_shift_beyond_the_lead_in_clamps_to_zero(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=-0.9)
    before = list(ctx.manifest.assumptions)
    out = assemble.run(ctx)
    assert out.data["audio_offset_s"] == 0.0
    final = ctx.run_dir / FINAL
    samples = _pcm(final)
    assert _peak(samples, 0.0, 0.1) > SILENCE  # starts at 0 ...
    spoken = [i for i, x in enumerate(samples) if abs(x) > SILENCE]
    assert (spoken[-1] - spoken[0]) / SAMPLE_RATE == pytest.approx(LINE_S, abs=0.1)  # ... whole
    assert _xcorr_lag_s(final, ctx.run_dir / "line.wav") == pytest.approx(0.0, abs=0.05)
    # changed: the wanted (unclamped) offset is recorded, and the clamp is an assumption
    assert out.data["audio_offset_wanted_s"] == pytest.approx(-0.2, abs=1e-9)
    assert AssembleData.model_validate(out.data).audio_offset_wanted_s == pytest.approx(-0.2)
    (note,) = _new_assumptions(ctx, before)
    assert _names_value(_numbers(note), -0.2) and _names_value(_numbers(note), 0.0), note
    assert "negative shift" in note.lower(), note


# --- AC: speech that fits at 0 but runs past the video end after the offset -> offset REDUCED ----
# changed: was GateFailed. Shots 5 + 4 s, line.wav = an 8.25 s chirp: fits at 0, at the
# wanted 0.7 + 0.5 = 1.2 s it would end at 9.45 s; reduced to 9.0 - 8.25 = 0.75 s it ends at 9.0 s.

LONG_LINE_S = 8.25
REDUCED_S = 9.0 - LONG_LINE_S  # 0.75


def _long_line_ctx(make_ctx: MakeCtx) -> Context:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5, chirp=False)
    line = _chirp_line(ctx.run_dir / "line.wav", LONG_LINE_S)
    voice = VoiceData.model_validate(ctx.manifest.stages["voice"].data)
    voice = voice.model_copy(update={"audio_duration_s": duration(line)})
    ctx.manifest.stages["voice"].data = voice.model_dump(mode="json")
    assert duration(line) == pytest.approx(LONG_LINE_S, abs=0.01)  # precondition: fits at 0 ...
    assert 1.2 + duration(line) > 9.0  # ... but not at the wanted offset
    return ctx


def test_speech_past_the_video_end_after_the_offset_reduces_the_offset(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _long_line_ctx(make_ctx)
    before = list(ctx.manifest.assumptions)
    out = assemble.run(ctx)  # DONE, no GateFailed
    data = AssembleData.model_validate(out.data)
    assert data.audio_offset_s == pytest.approx(REDUCED_S, abs=1e-3)  # video_s - speech end
    assert out.data["audio_offset_wanted_s"] == pytest.approx(1.2, abs=1e-9)  # unchanged
    (note,) = _new_assumptions(ctx, before)
    numbers = _numbers(note)
    assert _names_value(numbers, 1.2) and _names_value(numbers, REDUCED_S), note  # both values
    assert "shot 1 too short" in note.lower(), note  # the reason

    final = ctx.run_dir / FINAL
    assert duration(final) == pytest.approx(9.0, abs=0.2)
    samples = _pcm(final)
    spoken = [i for i, x in enumerate(samples) if abs(x) > SILENCE]
    assert (spoken[-1] - spoken[0]) / SAMPLE_RATE == pytest.approx(LONG_LINE_S, abs=0.1)  # whole
    assert _xcorr_lag_s(final, ctx.run_dir / "line.wav") == pytest.approx(REDUCED_S, abs=0.05)


def test_reduced_offset_final_passes_final_check(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    from hwva.providers import fakes
    from hwva.schemas import FinalReport

    ctx = _long_line_ctx(make_ctx)
    (ctx.run_dir / fakes.LAST_SPOKEN).write_text(LINE, encoding="utf-8")  # what FakeStt hears
    final_check_stage = next(s for s in pipeline.STAGES if s.name == "final_check")
    result = pipeline.run(ctx, [_assemble_stage(), final_check_stage], root=runs_root)
    assert result.stages["assemble"].status is StageStatus.DONE, result.stages["assemble"].error
    assert result.stages["assemble"].data["audio_offset_s"] == pytest.approx(REDUCED_S, abs=1e-3)
    assert any(_names_value(_numbers(a), 1.2) for a in result.assumptions), result.assumptions
    record = result.stages["final_check"]
    assert record.status is StageStatus.DONE, record.error
    report = FinalReport.model_validate(record.data)
    assert report.verdict == "pass", report.failed
    checks = {c.name: c for c in report.checks}
    assert checks["audio_match"].value["offset_ms"] == pytest.approx(REDUCED_S * 1000, abs=50)


# --- AC: speech that does not fit even at offset 0 -> GateFailed (unchanged) ---------------------
# Regression guard (not xfail): today's behaviour IS the required behaviour.


def test_speech_that_does_not_fit_at_zero_fails_even_with_an_offset(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    # line 3.0 s in a 2.0 s video: no offset >= 0 can make it fit
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5, shot_s=(2.0,), chirp=False, audio_s=3.0)
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


# --- AC: offset applied = wanted -> no assumption added ------------------------------------------


def test_offset_applied_as_wanted_adds_no_assumption(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5)  # 3 s line in 9 s: 1.2 fits
    before = list(ctx.manifest.assumptions)
    out = assemble.run(ctx)
    assert out.data["audio_offset_s"] == 1.2
    assert out.data["audio_offset_wanted_s"] == pytest.approx(1.2, abs=1e-9)
    assert ctx.manifest.assumptions == before


# --- AC: the shifted final passes final_check (end to end, real ffmpeg, fakes) -------------------


def test_shifted_final_passes_final_check(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    from hwva.providers import fakes
    from hwva.schemas import FinalReport

    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5)
    (ctx.run_dir / fakes.LAST_SPOKEN).write_text(LINE, encoding="utf-8")  # what FakeStt hears
    final_check_stage = next(s for s in pipeline.STAGES if s.name == "final_check")
    result = pipeline.run(ctx, [_assemble_stage(), final_check_stage], root=runs_root)
    assert result.stages["assemble"].status is StageStatus.DONE, result.stages["assemble"].error
    record = result.stages["final_check"]
    assert record.status is StageStatus.DONE, record.error
    report = FinalReport.model_validate(record.data)
    assert report.verdict == "pass", report.failed
    checks = {c.name: c for c in report.checks}
    assert checks["leading_silence"].value["leading_silence_s"] == pytest.approx(1.2, abs=0.05)
    assert checks["audio_match"].value["offset_ms"] == pytest.approx(1200, abs=50)


# === Aligned audio (specs/assemble.md "Aligned audio", v2; see specs/sync.md) =========
# shot_1 records `aligned_audio` = aligned_line.wav (clip time, lead-in and phrase shifts baked in):
# it is placed at 0, never delayed by lead_in_s + shift_s; moved earlier (leading silence trimmed)
# only when its speech would run past the video end.

ALIGNED = "aligned_line.wav"
TWO_PHRASES = [(1.0, 1.8), (2.3, 3.0)]


def _aligned_ctx(
    make_ctx: MakeCtx,
    *,
    runs: list[tuple[float, float]] | None = None,
    track_s: float = 5.0,
    shot_s: tuple[float, ...] = (5.0, 4.0),
) -> Context:
    """`_shift_ctx` (lead-in 0.7, shift -0.24 recorded, line.wav a 3 s chirp) plus shot_1's
    aligned track (`speech_wav` phrases) recorded as artifact `aligned_audio`."""
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=-0.24, shot_s=shot_s)
    speech_wav(ctx.run_dir / ALIGNED, runs or TWO_PHRASES, track_s)
    record = SyncReport(
        verdict="correct", syncnet_confidence=8.6, syncnet_offset_frames=-6, shift_s=-0.24,
        method="piecewise", aligned_audio=ALIGNED, box=BOX,
    )  # fmt: skip
    rec = ctx.manifest.stages["shot_1"]
    rec.data = ShotData(duration_s=shot_s[0], sync=record).model_dump(mode="json")
    rec.artifacts["sync"] = "sync.json"
    rec.artifacts["aligned_audio"] = ALIGNED
    return ctx


def _ncc(final: Path, ref: Path, *, trim_s: float = 0.0) -> tuple[float, float]:
    """(peak, lag_s) of the normalized cross-correlation of `ref` (its first `trim_s` dropped)
    slid over the final's audio, both decoded to 16 kHz mono; 1.0 = an exact (scaled) copy."""
    a = _floats(final)
    b = _floats(ref)[round(trim_s * SAMPLE_RATE) :]
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    n, lags = len(b), len(a) - len(b) + 1
    size = 1 << int(len(a) + n).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)[:lags]
    csum = np.concatenate(([0.0], np.cumsum(a * a)))
    window = csum[n : n + lags] - csum[:lags]
    energy = float(np.dot(b, b))
    # windows of (near) silence correlate with nothing: AAC's noise floor would blow up the ratio
    ncc = np.where(
        window > 0.01 * energy,
        corr / np.maximum(np.sqrt(np.maximum(window, 0.0) * energy), 1e-9),
        0.0,
    )
    best = int(np.argmax(ncc))
    return float(ncc[best]), best / SAMPLE_RATE


# --- AC: aligned track recorded -> placed at 0, its own silence first, no extra delay -------------


def test_aligned_track_is_the_audio_placed_at_zero(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _aligned_ctx(make_ctx)
    out = assemble.run(ctx)
    data = AssembleData.model_validate(out.data)
    assert data.audio_source == "aligned"
    assert data.audio_offset_s == 0.0
    assert data.audio_offset_wanted_s == 0.0  # not 0.7 - 0.24: the shift is in the file
    assert data.speech_start_s == pytest.approx(1.0, abs=0.05)
    assert out.artifacts == {"video": FINAL, "audio": ALIGNED}

    final = ctx.run_dir / FINAL
    peak, lag = _ncc(final, ctx.run_dir / ALIGNED)
    assert peak >= 0.9, peak
    assert lag == pytest.approx(0.0, abs=0.05)  # no adelay of lead_in_s + shift_s on top
    assert _leading_silence_s(final) == pytest.approx(1.0, abs=0.05)  # the first 1.0 s silent
    assert duration(final) == pytest.approx(9.0, abs=0.2)  # still the shot total
    samples = _pcm(final)
    assert _peak(samples, 3.1, 8.9) < SILENCE  # after the track: silence (shot 2 gets silence)
    assert len(_streams(final, "audio")) == 1 and len(_streams(final, "video")) == 1


# --- AC: speech 0.3 s past the video end, >= 0.3 s leading silence -> moved 0.3 s earlier --------


def test_aligned_track_too_long_moves_earlier_and_passes_final_check(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    from hwva.providers import fakes
    from hwva.schemas import FinalReport

    # one 9.0 s shot; the track is 9.3 s with its speech running to the end, 0.8 s of silence first
    ctx = _aligned_ctx(make_ctx, runs=[(0.8, 4.0), (4.5, 9.3)], track_s=9.3, shot_s=(9.0,))
    (ctx.run_dir / fakes.LAST_SPOKEN).write_text(LINE, encoding="utf-8")  # what FakeStt hears
    before = list(ctx.manifest.assumptions)
    final_check_stage = next(s for s in pipeline.STAGES if s.name == "final_check")
    result = pipeline.run(ctx, [_assemble_stage(), final_check_stage], root=runs_root)

    record = result.stages["assemble"]
    assert record.status is StageStatus.DONE, record.error
    data = AssembleData.model_validate(record.data)
    assert data.audio_source == "aligned"
    assert data.audio_offset_s == pytest.approx(-0.3, abs=1e-3)
    assert data.speech_start_s == pytest.approx(0.5, abs=0.05)
    assert record.artifacts["audio"] == ALIGNED
    notes = [a for a in result.assumptions if a not in before]
    assert len(notes) == 1, notes
    numbers = _numbers(notes[0])
    assert _names_value(numbers, -0.3) and _names_value(numbers, 0.0), notes[0]
    assert "shot 1 too short" in notes[0].lower(), notes[0]

    final = ctx.run_dir / FINAL
    assert duration(final) == pytest.approx(9.0, abs=0.2)
    peak, lag = _ncc(final, ctx.run_dir / ALIGNED, trim_s=0.3)  # the track from 0.3 s on, at 0
    assert peak >= 0.9 and lag == pytest.approx(0.0, abs=0.05), (peak, lag)
    samples = _pcm(final)
    assert _peak(samples, 8.8, 8.95) > SILENCE  # speech complete: it runs to the end
    assert _peak(samples, 0.0, 0.45) < SILENCE  # 0.8 - 0.3 s of silence remain

    fc = result.stages["final_check"]
    assert fc.status is StageStatus.DONE, fc.error
    assert FinalReport.model_validate(fc.data).verdict == "pass"


def test_aligned_track_overflow_beyond_its_leading_silence_fails_gate(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    # 0.3 s overflow but only 0.2 s of leading silence: trimming would cut speech
    ctx = _aligned_ctx(make_ctx, runs=[(0.2, 4.0), (4.5, 9.3)], track_s=9.3, shot_s=(9.0,))
    with pytest.raises(GateFailed):
        assemble.run(ctx)
    assert not (ctx.run_dir / FINAL).exists()


def test_aligned_track_leading_silence_shorter_than_0_3_s_can_still_be_trimmed(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    """Review fix: 0.25 s of leading silence covers a 0.1 s overflow (silencedetect
    with d=0.3 used to read it as 0 and refuse)."""
    ctx = _aligned_ctx(make_ctx, runs=[(0.25, 4.0), (4.5, 9.1)], track_s=9.1, shot_s=(9.0,))
    data = AssembleData.model_validate(assemble.run(ctx).data)
    assert data.audio_offset_s == pytest.approx(-0.1, abs=1e-3)
    assert data.speech_start_s == pytest.approx(0.15, abs=0.03)


# --- AC: no aligned_audio artifact -> the line path, as before; audio_source "line" ---------------


def test_without_aligned_audio_line_wav_is_placed_as_before(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _shift_ctx(make_ctx, lead_in=0.7, shift=0.5)
    out = assemble.run(ctx)
    data = AssembleData.model_validate(out.data)
    assert data.audio_source == "line"
    assert data.audio_offset_s == 1.2 and data.audio_offset_wanted_s == pytest.approx(1.2)
    assert data.speech_start_s == pytest.approx(1.2, abs=0.05)  # the TTS starts at once
    assert out.artifacts == {"video": FINAL, "audio": "line.wav"}
    assert _xcorr_lag_s(ctx.run_dir / FINAL, ctx.run_dir / "line.wav") == pytest.approx(
        1.2, abs=0.05
    )


# --- AC: the aligned final passes final_check (end to end, real ffmpeg, fakes) --------------------


def test_aligned_final_passes_final_check(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    from hwva.providers import fakes
    from hwva.schemas import FinalReport

    ctx = _aligned_ctx(make_ctx)
    (ctx.run_dir / fakes.LAST_SPOKEN).write_text(LINE, encoding="utf-8")  # what FakeStt hears
    final_check_stage = next(s for s in pipeline.STAGES if s.name == "final_check")
    result = pipeline.run(ctx, [_assemble_stage(), final_check_stage], root=runs_root)
    assert result.stages["assemble"].status is StageStatus.DONE, result.stages["assemble"].error
    record = result.stages["final_check"]
    assert record.status is StageStatus.DONE, record.error
    report = FinalReport.model_validate(record.data)
    assert report.verdict == "pass", report.failed
    checks = {c.name: c for c in report.checks}
    assert checks["audio_match"].value["reference"] == ALIGNED
    assert checks["audio_match"].value["offset_ms"] == pytest.approx(0, abs=50)
    assert checks["leading_silence"].value["leading_silence_s"] == pytest.approx(1.0, abs=0.05)
    assert checks["leading_silence"].value["allowed_s"] == pytest.approx(1.5, abs=0.05)
