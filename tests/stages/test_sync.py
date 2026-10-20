"""Acceptance tests for the lip-sync gate and phrase alignment (specs/sync.md, v2).
Not a pipeline stage: the code lives in `src/hwva/sync.py`; the stages that use it (shot_1,
assemble, final_check, quality_check, eval) are tested in their own files. Real ffmpeg, synthetic
media, SyncNet stubbed (a fake `syncnet_python` in sys.modules), no network.

v2: SyncNet decides (verdict + global shift); our motion metric is diagnostic only. v1's motion
verdict (`verdict(peak_r, lag_s)`, `MIN_PEAK_R` / `PASS_LAG_S` / `MAX_LAG_S`) is removed by the
spec, and so are its tests; the motion lag-detection tests stay (without verdict asserts).

Synthetic media for the motion metric (numpy + ffmpeg, 180x320 gray, 24 fps, lossless H.264 so a
static region stays exactly static):
- audio: a 440 Hz tone switched on/off in bursts of irregular length (seeded);
- video: while the "mouth" is on, the mouth box flickers between two brightness levels every frame,
  while it is off the box is static. Its on/off pattern is the audio's, delayed / advanced /
  unchanged, or a region ABOVE the box carries it instead.

Contract pinned here where specs/sync.md leaves details open:
- `global_shift(verdict, offset_frames) -> float`: the global `shift_s` of the Verdict table.
- `Thresholds(sync_conf=..., weak_conf=...)` is constructible by keywords; `verdict(...)` takes it
  as `thresholds=`.
- `Phrase(start_s=..., end_s=...)`, `SyncNetResult(offset_frames=..., confidence=..., min_dist=...,
  seconds=...)` are constructible by keywords.
- `choose_shifts` returns `k/24` exactly (the record rounds, the render does not)."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import logging
import math
import os
import subprocess
import sys
import types
import wave
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from conftest import (
    NO_FACE_RESULT,
    SYNCNET_WEIGHT_FILES,
    SYNCNET_WEIGHTS_SCRIPT,
    install_syncnet,
    needs_ffmpeg,
    read_wav,
    speech_wav,
    syncnet_result,
)

from hwva import sync
from hwva.schemas import SyncReport

pytestmark = [needs_ffmpeg]


REPO = Path(__file__).resolve().parents[2]
FPS = 24
SAMPLE_RATE = 16000
WIDTH, HEIGHT = 180, 320
SECONDS = 6.0
DELAY = 12  # frames = 0.5 s
BOX = (0.20, 0.80, 0.42, 0.60)  # spec value, deliberately not imported from hwva.sync
ABOVE = (0.05, 0.35)  # a region above y 0.42: outside the default mouth box
ON, OFF_A, OFF_B, STATIC = 208, 48, 128, 128  # mouth-box brightness levels
TIMEOUT_S = 120
_REAL_RUN = subprocess.run
VERDICTS = {"pass", "correct", "weak", "no_sync", "error"}

Pattern = npt.NDArray[np.bool_]
Series = npt.NDArray[np.float64]


# --- synthetic media (motion metric) --------------------------------------------------------------


def _pattern(seconds: float = SECONDS, seed: int = 7) -> Pattern:
    """Per-frame on/off: off for the first 6 frames, then bursts of 3..13 frames."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    on = np.zeros(n, dtype=bool)
    t, state = 6, True
    while t < n:
        k = int(rng.integers(3, 14))
        on[t : t + k] = state
        t += k
        state = not state
    return on


def _delayed(on: Pattern, frames: int) -> Pattern:
    """Positive `frames`: the same pattern later (mouth AFTER the audio); negative: earlier."""
    if frames >= 0:
        return np.concatenate([np.zeros(frames, dtype=bool), on[: len(on) - frames]])
    return np.concatenate([on[-frames:], np.zeros(-frames, dtype=bool)])


def _wav(path: Path, on: Pattern, *, silent: bool = False, frames: int | None = None) -> Path:
    """Mono 16 kHz 16-bit tone whose envelope is `on` (one value per 1/24 s)."""
    n = int(len(on) / FPS * SAMPLE_RATE) if frames is None else frames
    t = np.arange(n) / SAMPLE_RATE
    env = on[np.minimum((t * FPS).astype(int), len(on) - 1)].astype(np.float64)
    signal = 0.0 if silent else 0.5 * np.sin(2 * np.pi * 440 * t) * env
    samples = (np.asarray(signal, dtype=np.float64) * np.ones(n) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return path


def _video(
    path: Path,
    on: Pattern,
    *,
    mode: str = "follow",
    region: tuple[float, float] = (BOX[2], BOX[3]),
    seed: int = 3,
) -> Path:
    """mode "follow": the region flickers while `on`; "noise": random brightness every frame;
    "static": never changes. The region spans x 0.20-0.80 and `region` in y."""
    rng = np.random.default_rng(seed)
    y0, y1 = int(region[0] * HEIGHT), int(region[1] * HEIGHT)
    x0, x1 = int(BOX[0] * WIDTH), int(BOX[1] * WIDTH)
    frames = np.full((len(on), HEIGHT, WIDTH), STATIC, dtype=np.uint8)
    for i, value in enumerate(on):
        if mode == "follow" and value:
            frames[i, y0:y1, x0:x1] = OFF_A if i % 2 else ON
        elif mode == "noise":
            frames[i, y0:y1, x0:x1] = int(rng.integers(30, 220))
    _REAL_RUN(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)],
        input=frames.tobytes(), check=True,
    )  # fmt: skip
    return path


def _clip(tmp_path: Path, shift_frames: int, **video_kw: Any) -> tuple[Path, Path]:
    """(video, audio): the audio follows `_pattern()`, the video's pattern is shifted."""
    on = _pattern()
    audio = _wav(tmp_path / "audio.wav", on)
    video = _video(tmp_path / "clip.mp4", _delayed(on, shift_frames), **video_kw)
    return video, audio


def _box_brightness(video: Path, rows: tuple[float, float]) -> npt.NDArray[np.float64]:
    """Mean brightness of x 0.25-0.75 x `rows` per decoded frame (harness check only)."""
    raw = _REAL_RUN(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH).astype(np.float64)
    y0, y1 = int(rows[0] * HEIGHT) + 2, int(rows[1] * HEIGHT) - 2
    x0, x1 = int(0.25 * WIDTH), int(0.75 * WIDTH)
    result: npt.NDArray[np.float64] = frames[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return result


def _rounded(value: float) -> bool:
    return value == round(value, 3)


# --- harness check (not xfail): the synthetic media really are what the tests claim ---------------


def test_harness_synthetic_media_are_as_described(tmp_path: Path) -> None:
    on = _pattern()
    assert 0.3 < on.mean() < 0.7  # bursts and gaps
    video, audio = _clip(tmp_path, DELAY)
    level = _box_brightness(video, (BOX[2], BOX[3]))
    shifted = _delayed(on, DELAY)
    assert len(level) == len(on)
    flicker = np.abs(np.diff(level)) > 100
    assert flicker[1:][shifted[1:-1] & shifted[2:]].all()  # on -> moves every frame
    assert not (np.abs(np.diff(level))[~shifted[:-1] & ~shifted[1:]] > 1).any()  # off -> static
    with wave.open(str(audio), "rb") as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    edges = [int(np.ceil(i * SAMPLE_RATE / FPS)) for i in range(141)]  # hop i = [i/24, (i+1)/24)
    rms = np.array([np.sqrt(np.mean(samples[a:b] ** 2)) for a, b in pairwise(edges)])
    assert (rms[on[:140]] > 5000).all() and (rms[~on[:140]] < 1).all()

    above = _video(tmp_path / "above.mp4", shifted, region=ABOVE)
    assert np.ptp(_box_brightness(above, (BOX[2], BOX[3]))) == 0  # the mouth box never changes
    assert np.ptp(_box_brightness(above, ABOVE)) > 100


def test_harness_speech_wav_has_the_runs_it_claims(tmp_path: Path) -> None:
    path = speech_wav(tmp_path / "s.wav", [(0.5, 1.0), (1.5, 2.0)], 2.5, floor=100)
    samples, rate, channels = read_wav(path)
    assert (rate, channels, len(samples)) == (24000, 1, 60000)
    assert (samples[:12000] == 100).all() and (samples[24000:36000] == 100).all()
    assert np.abs(samples[12100:23900]).max() > 10000


# --- AC: motion metric - lag detection (kept from v1, no verdict) ---------------------------------
# Regression guards (not xfail): v1's `measure` already finds these lags.


def test_mouth_lagging_by_half_a_second_is_measured(tmp_path: Path) -> None:
    video, audio = _clip(tmp_path, DELAY)
    result = sync.measure(video, audio)
    assert result.lag_s == pytest.approx(0.5, abs=0.05)
    assert result.r_at_0 < result.peak_r
    assert all(_rounded(v) for v in (result.lag_s, result.peak_r, result.r_at_0))


def test_mouth_in_sync_has_no_lag(tmp_path: Path) -> None:
    video, audio = _clip(tmp_path, 0)
    result = sync.measure(video, audio)
    assert abs(result.lag_s) < 0.08
    assert result.r_at_0 == pytest.approx(result.peak_r, abs=0.05)


def test_mouth_leading_by_half_a_second_is_a_negative_lag(tmp_path: Path) -> None:
    video, audio = _clip(tmp_path, -DELAY)
    result = sync.measure(video, audio)
    assert result.lag_s == pytest.approx(-0.5, abs=0.05)


def test_motion_outside_the_mouth_box_is_not_seen(tmp_path: Path) -> None:
    """The crop is really used: the box itself is static, so nothing correlates."""
    video, audio = _clip(tmp_path, DELAY, region=ABOVE)
    result = sync.measure(video, audio)
    assert result.peak_r == 0 and result.r_at_0 == 0
    assert result.lag_s == 0  # every r(L) is 0: the smallest |L| wins the tie


def test_the_box_argument_is_the_crop(tmp_path: Path) -> None:
    """The same clip, measured with a box over the moving region: now the lag is found."""
    video, audio = _clip(tmp_path, DELAY, region=ABOVE)
    result = sync.measure(video, audio, box=(0.20, 0.80, ABOVE[0], ABOVE[1]))
    assert result.lag_s == pytest.approx(0.5, abs=0.05)


def test_static_video_raises_nothing(tmp_path: Path) -> None:
    video, audio = _clip(tmp_path, 0, mode="static")
    result = sync.measure(video, audio)
    assert result.peak_r == 0 and result.r_at_0 == 0 and result.lag_s == 0


def test_silent_audio_raises_nothing(tmp_path: Path) -> None:
    on = _pattern()
    audio = _wav(tmp_path / "silent.wav", on, silent=True)
    video = _video(tmp_path / "clip.mp4", on)
    result = sync.measure(video, audio)
    assert result.peak_r == 0 and result.lag_s == 0


# --- AC: `MotionResult` has no verdict (v2) -------------------------------------------------------


def test_motion_result_has_no_verdict_and_no_shift(tmp_path: Path) -> None:
    video, audio = _clip(tmp_path, DELAY)
    result = sync.measure(video, audio)
    assert isinstance(result, sync.MotionResult)
    assert result.lag_s == pytest.approx(0.5, abs=0.05)
    assert not hasattr(result, "verdict"), "the motion metric has no verdict any more"
    assert not hasattr(result, "shift_s"), "the motion metric proposes no global shift"
    built = sync.MotionResult(lag_s=0.5, peak_r=0.4, r_at_0=0.1)
    assert (built.lag_s, built.peak_r, built.r_at_0) == (0.5, 0.4, 0.1)


def test_constants_are_the_v2_spec_values() -> None:
    assert pytest.approx(BOX) == sync.MOUTH_BOX
    assert sync.FPS == 24
    assert sync.SYNC_CONF == 5.0
    assert sync.WEAK_CONF == 3.0
    assert sync.SYNCNET_FPS == 25
    assert sync.PASS_OFFSET_FRAMES == 1
    assert sync.MAX_PHRASE_SHIFT_S == 0.5
    assert sync.MIN_GAP_S == 0.12
    assert sync.PAD_S == 0.08
    assert sync.FADE_S == 0.015
    assert sync.PHRASE_NOISE_DB == -40
    assert sync.PHRASE_MIN_PAUSE_S == 0.15
    assert issubclass(sync.SyncError, RuntimeError)
    assert issubclass(sync.NoFaceError, sync.SyncError)
    for removed in ("MIN_PEAK_R", "PASS_LAG_S", "MAX_LAG_S"):  # v1's motion thresholds
        assert not hasattr(sync, removed), f"{removed} is removed in v2 (specs/sync.md)"


# --- AC: ffmpeg failure / timeout / unreadable file -> SyncError; ffmpeg calls pass timeout -------


@pytest.mark.parametrize("broken", ["garbage", "missing"])
def test_unreadable_video_is_a_sync_error(tmp_path: Path, broken: str) -> None:
    audio = _wav(tmp_path / "audio.wav", _pattern())
    video = tmp_path / "clip.mp4"
    if broken == "garbage":
        video.write_bytes(b"this is not an mp4 file\x00\x01" * 64)
    with pytest.raises(sync.SyncError) as err:
        sync.measure(video, audio)
    assert str(err.value).strip()


def test_unreadable_audio_is_a_sync_error(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.mp4", _pattern())
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-but-not-really" * 32)
    with pytest.raises(sync.SyncError):
        sync.measure(video, audio)


def test_empty_audio_is_a_sync_error(tmp_path: Path) -> None:
    video = _video(tmp_path / "clip.mp4", _pattern())
    audio = _wav(tmp_path / "empty.wav", _pattern(), frames=0)
    with pytest.raises(sync.SyncError):
        sync.measure(video, audio)


def test_a_single_frame_is_a_sync_error(tmp_path: Path) -> None:
    on = _pattern()
    video = _video(tmp_path / "one.mp4", on[:1])
    audio = _wav(tmp_path / "audio.wav", on)
    with pytest.raises(sync.SyncError):
        sync.measure(video, audio)


def _is_tool(cmd: Any) -> bool:
    argv = cmd if isinstance(cmd, list | tuple) else [cmd]
    return Path(str(argv[0])).stem.lower() in ("ffmpeg", "ffprobe")


@pytest.mark.parametrize("failure", ["nonzero_exit", "timeout"])
def test_ffmpeg_failure_is_a_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    video, audio = _clip(tmp_path, DELAY)
    hit: list[Any] = []

    def broken(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_tool(cmd):
            return _REAL_RUN(cmd, *args, **kwargs)
        hit.append(cmd)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or TIMEOUT_S)
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"decode error")
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"decode error")

    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(sync.SyncError) as err:
        sync.measure(video, audio)
    assert hit, "measure made no ffmpeg call through subprocess.run"
    if failure == "timeout":
        assert "time" in str(err.value).lower()  # "timed out" / "timeout"


def test_every_ffmpeg_call_passes_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, audio = _clip(tmp_path, DELAY)
    calls: list[tuple[Any, Any]] = []

    def spy(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_tool(cmd):
            calls.append((cmd, kwargs.get("timeout")))
        return _REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    result = sync.measure(video, audio)
    assert result.lag_s == pytest.approx(0.5, abs=0.05)
    assert calls, "measure made no ffmpeg call through subprocess.run"
    for cmd, timeout in calls:
        expected = 30 if Path(str(cmd[0])).stem.lower() == "ffprobe" else TIMEOUT_S
        assert timeout == expected, cmd


# --- AC: HWVA_SYNC_MOUTH_BOX parsed; invalid -> ValueError ----------------------------------------


def test_mouth_box_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sync.box_from_env() == pytest.approx(BOX)  # unset (conftest) -> MOUTH_BOX
    monkeypatch.setenv("HWVA_SYNC_MOUTH_BOX", "0.1,0.9,0.3,0.7")
    assert sync.box_from_env() == pytest.approx((0.1, 0.9, 0.3, 0.7))
    monkeypatch.setenv("HWVA_SYNC_MOUTH_BOX", " 0, 1 , 0 ,1 ")
    assert sync.box_from_env() == pytest.approx((0.0, 1.0, 0.0, 1.0))


@pytest.mark.parametrize("value", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_empty_mouth_box_is_the_default(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HWVA_SYNC_MOUTH_BOX", value)
    assert sync.box_from_env() == pytest.approx(BOX)


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "0.1,0.9,0.3",  # three numbers
        "0.1,0.9,0.3,0.7,0.8",  # five
        "0.9,0.1,0.3,0.7",  # x0 > x1
        "0.5,0.5,0.3,0.7",  # x0 == x1
        "0.1,0.9,0.7,0.3",  # y0 > y1
        "-0.1,0.9,0.3,0.7",  # below 0
        "0.1,1.1,0.3,0.7",  # above 1
        "0.1,0.9,0.3,nan",
    ],
)
def test_invalid_mouth_box_is_a_value_error(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HWVA_SYNC_MOUTH_BOX", value)
    with pytest.raises(ValueError):
        sync.box_from_env()


# === Verdict (pure) ===============================================================================


@pytest.mark.parametrize(
    ("confidence", "offset", "expected"),
    [
        (5.0, 0, "pass"),
        (5.0, 1, "pass"),
        (5.0, -1, "pass"),
        (5.0, 2, "correct"),
        (5.0, -2, "correct"),
        (8.63, -6, "correct"),
        (4.999, 0, "weak"),
        (4.999, -6, "weak"),
        (4.9996, 0, "weak"),  # the raw confidence decides, never the 3-decimal record (5.0)
        (3.0, 0, "weak"),
        (3.0, 7, "weak"),
        (2.999, 0, "no_sync"),
        (2.999, -7, "no_sync"),
        (1.71, -7, "no_sync"),
    ],
)
def test_verdict_boundaries_with_the_defaults(
    confidence: float, offset: int, expected: str
) -> None:
    assert sync.verdict(confidence, offset) == expected
    explicit = sync.Thresholds(sync_conf=sync.SYNC_CONF, weak_conf=sync.WEAK_CONF)
    assert sync.verdict(confidence, offset, thresholds=explicit) == expected


@pytest.mark.parametrize(
    ("confidence", "offset", "expected"),
    [(8.0, 0, "pass"), (8.0, 3, "correct"), (7.99, 0, "weak"), (2.0, 5, "weak"),
     (1.99, 0, "no_sync"), (5.0, 0, "weak")],
)  # fmt: skip
def test_verdict_honours_custom_thresholds(confidence: float, offset: int, expected: str) -> None:
    thresholds = sync.Thresholds(sync_conf=8.0, weak_conf=2.0)
    assert sync.verdict(confidence, offset, thresholds=thresholds) == expected


@pytest.mark.parametrize(
    ("verdict", "offset", "expected"),
    [
        ("correct", -6, -0.24),
        ("correct", 2, 0.08),
        ("weak", -6, -0.24),
        ("weak", 3, 0.12),
        ("pass", 1, 0.0),
        ("pass", -1, 0.0),
        ("no_sync", -7, 0.0),  # the offset of a clip without sync means nothing
        ("error", -6, 0.0),
    ],
)
def test_global_shift_is_offset_over_25_for_correct_and_weak(
    verdict: str, offset: int, expected: float
) -> None:
    assert sync.global_shift(verdict, offset) == pytest.approx(expected, abs=1e-9)


# --- AC: thresholds_from_env ----------------------------------------------------------------------


def test_thresholds_from_env_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    t = sync.thresholds_from_env()  # unset (conftest)
    assert (t.sync_conf, t.weak_conf) == (5.0, 3.0)
    monkeypatch.setenv("HWVA_SYNC_CONF", "6.5")
    monkeypatch.setenv("HWVA_SYNC_WEAK_CONF", "2")
    t = sync.thresholds_from_env()
    assert (t.sync_conf, t.weak_conf) == (6.5, 2.0)
    monkeypatch.delenv("HWVA_SYNC_WEAK_CONF")
    monkeypatch.setenv("HWVA_SYNC_CONF", " 4 ")
    t = sync.thresholds_from_env()
    assert (t.sync_conf, t.weak_conf) == (4.0, 3.0)
    monkeypatch.setenv("HWVA_SYNC_CONF", "3")  # weak == sync is allowed
    assert sync.thresholds_from_env().sync_conf == 3.0


@pytest.mark.parametrize("value", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
def test_empty_thresholds_are_the_defaults(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HWVA_SYNC_CONF", value)
    monkeypatch.setenv("HWVA_SYNC_WEAK_CONF", value)
    t = sync.thresholds_from_env()
    assert (t.sync_conf, t.weak_conf) == (5.0, 3.0)


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({"HWVA_SYNC_CONF": "abc"}, ("HWVA_SYNC_CONF",)),
        ({"HWVA_SYNC_WEAK_CONF": "abc"}, ("HWVA_SYNC_WEAK_CONF",)),
        ({"HWVA_SYNC_CONF": "-1"}, ("HWVA_SYNC_CONF",)),
        ({"HWVA_SYNC_WEAK_CONF": "-1"}, ("HWVA_SYNC_WEAK_CONF",)),
        # weak > sync: either variable may be the one named
        ({"HWVA_SYNC_CONF": "2"}, ("HWVA_SYNC_CONF", "HWVA_SYNC_WEAK_CONF")),
        ({"HWVA_SYNC_WEAK_CONF": "6"}, ("HWVA_SYNC_CONF", "HWVA_SYNC_WEAK_CONF")),
    ],
    ids=["conf-abc", "weak-abc", "conf-negative", "weak-negative", "weak-gt-sync-1",
         "weak-gt-sync-2"],
)  # fmt: skip
def test_invalid_thresholds_are_a_value_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], named: tuple[str, ...]
) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError) as err:
        sync.thresholds_from_env()
    assert any(name in str(err.value) for name in named), str(err.value)


# --- AC: warning text (one builder) ---------------------------------------------------------------


def test_warning_text_per_verdict() -> None:
    t = sync.Thresholds(sync_conf=5.0, weak_conf=3.0)
    weak = sync.warning_text("weak", 3.4, t)
    assert weak is not None and "weak" in weak.lower() and "3.4" in weak and "5" in weak
    no_sync = sync.warning_text("no_sync", 1.71, t)
    assert no_sync is not None and "not detected" in no_sync and "1.71" in no_sync
    assert "3" in no_sync.replace("1.71", "")  # names the weak threshold
    error = sync.warning_text("error", None, t, error="no face found")
    assert error is not None and "error" in error.lower() and "no face found" in error
    for text in (weak, no_sync, error):
        assert "\n" not in text
    assert sync.warning_text("pass", 8.6, t) is None
    assert sync.warning_text("correct", 8.6, t) is None


# === syncnet_measure (stub `syncnet_python`) ======================================================


def _media(tmp_path: Path) -> tuple[Path, Path]:
    """Files that exist (the stub never reads them)."""
    video, audio = tmp_path / "clip.mp4", tmp_path / "shot_audio.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    return video, audio


def _inference_paths(call: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[Path, Path]:
    args, kwargs = call
    video = kwargs.get("video_path", args[0] if args else None)
    audio = kwargs.get("audio_path", args[1] if len(args) > 1 else None)
    assert video is not None and audio is not None, call
    return Path(video).resolve(), Path(audio).resolve()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str | Path):
        return [str(value)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


def test_syncnet_measure_reads_the_offset_at_the_max_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path
) -> None:
    result_tuple = ([2, -6, 1], [1.2, 8.63, 3.0], [9.0, 7.1, 8.0], 8.63, 7.1, "{}", True)
    calls = install_syncnet(monkeypatch, result=result_tuple)
    video, audio = _media(tmp_path)
    result = sync.syncnet_measure(video, audio)
    assert isinstance(result, sync.SyncNetResult)
    assert result.offset_frames == -6 and isinstance(result.offset_frames, int)
    assert result.confidence == pytest.approx(8.63)
    assert result.min_dist == pytest.approx(7.1)
    assert result.seconds >= 0

    ((args, kwargs),) = calls.init
    assert kwargs.get("device") == "cpu"
    passed = {Path(s).resolve() for s in _strings((args, kwargs))}
    for name in SYNCNET_WEIGHT_FILES:
        assert (syncnet_weights / name).resolve() in passed, f"{name} not passed to SyncNet"
    (call,) = calls.inference
    assert _inference_paths(call) == (video.resolve(), audio.resolve())


def test_syncnet_measure_reads_numpy_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path
) -> None:
    result_tuple = (
        np.array([2, -6]),
        np.array([1.2, 8.63], dtype=np.float32),
        np.array([9.0, 7.1], dtype=np.float32),
        np.float32(8.63),
        np.float32(7.1),
        "{}",
        np.bool_(True),
    )
    install_syncnet(monkeypatch, result=result_tuple)
    result = sync.syncnet_measure(*_media(tmp_path))
    assert result.offset_frames == -6 and type(result.offset_frames) is int
    assert result.confidence == pytest.approx(8.63, abs=1e-3)
    assert type(result.confidence) is float
    assert result.min_dist == pytest.approx(7.1, abs=1e-3)


@pytest.mark.parametrize(
    "result_tuple",
    [NO_FACE_RESULT, ([], [], [], 0.0, 0.0, "", True), ([-6], [8.6], [7.0], 8.6, 7.0, "", False)],
    ids=["no-face", "face-but-no-offsets", "has-face-false"],
)
def test_syncnet_measure_without_a_face_is_no_face_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path, result_tuple: Any
) -> None:
    install_syncnet(monkeypatch, result=result_tuple)
    with pytest.raises(sync.NoFaceError):
        sync.syncnet_measure(*_media(tmp_path))


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


@pytest.mark.parametrize(
    "exc",
    [
        None,  # conftest: `syncnet_python` is None in sys.modules -> ImportError
        ImportError("DLL load failed while importing _C"),
        OSError("[WinError 126] Error loading torch\\lib\\c10.dll"),
        RuntimeError("operator torchvision::nms does not exist"),
    ],
    ids=["not-importable", "dll-importerror", "oserror", "runtimeerror"],
)
def test_syncnet_import_failure_is_a_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path, exc: Exception | None
) -> None:
    if exc is not None:
        monkeypatch.delitem(sys.modules, "syncnet_python", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_BrokenImport(exc), *sys.meta_path])
    with pytest.raises(sync.SyncError) as err:
        sync.syncnet_measure(*_media(tmp_path))
    assert not isinstance(err.value, sync.NoFaceError)
    assert str(err.value).strip()


@pytest.mark.parametrize(
    "missing",
    [SYNCNET_WEIGHT_FILES, SYNCNET_WEIGHT_FILES[:1], SYNCNET_WEIGHT_FILES[1:]],
    ids=["both", "syncnet", "s3fd"],
)
def test_missing_weights_are_a_sync_error_naming_the_files_and_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path,
    missing: tuple[str, ...],
) -> None:  # fmt: skip
    calls = install_syncnet(monkeypatch, result=syncnet_result(-6, 8.63))
    for name in missing:
        (syncnet_weights / name).unlink()
    with pytest.raises(sync.SyncError) as err:
        sync.syncnet_measure(*_media(tmp_path))
    message = str(err.value)
    for name in missing:
        assert name in message, message
    for name in set(SYNCNET_WEIGHT_FILES) - set(missing):
        assert name not in message, f"{name} is present: {message}"
    assert SYNCNET_WEIGHTS_SCRIPT in message, message
    assert calls.inference == [], "no inference without weights"


def test_weights_default_to_the_weights_dir_of_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_syncnet(monkeypatch, result=syncnet_result(-1, 6.0))
    monkeypatch.delenv("HWVA_SYNCNET_WEIGHTS", raising=False)
    (tmp_path / "weights").mkdir()
    for name in SYNCNET_WEIGHT_FILES:
        (tmp_path / "weights" / name).write_bytes(b"w")
    monkeypatch.chdir(tmp_path)
    assert sync.syncnet_measure(*_media(tmp_path)).offset_frames == -1


@pytest.mark.parametrize("missing", ["video", "audio"])
def test_missing_media_is_a_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path, missing: str
) -> None:
    calls = install_syncnet(monkeypatch, result=syncnet_result(-6, 8.63))
    video, audio = _media(tmp_path)
    (video if missing == "video" else audio).unlink()
    with pytest.raises(sync.SyncError):
        sync.syncnet_measure(video, audio)
    assert calls.inference == []


def test_raising_inference_is_a_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path
) -> None:
    install_syncnet(monkeypatch, exc=RuntimeError("cv2 exploded"))
    with pytest.raises(sync.SyncError) as err:
        sync.syncnet_measure(*_media(tmp_path))
    assert "cv2 exploded" in str(err.value)
    assert not isinstance(err.value, sync.NoFaceError)


ODD_RESULTS: dict[str, object] = {
    "not-a-tuple": "8.63",
    "short-tuple": ([-6], [8.63], 8.63),
    "offset-not-a-number": (["x"], [8.63], [7.1], 8.63, 7.1, "{}", True),
    "confs-none": ([-6], None, [7.1], 8.63, 7.1, "{}", True),
    "max-conf-none": ([-6], [8.63], [7.1], None, 7.1, "{}", True),
    "lengths-differ": ([-6, 2], [8.63], [7.1], 8.63, 7.1, "{}", True),
}


@pytest.mark.parametrize("case", list(ODD_RESULTS))
def test_odd_result_is_a_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path, case: str
) -> None:
    install_syncnet(monkeypatch, result=ODD_RESULTS[case])
    with pytest.raises(sync.SyncError):
        sync.syncnet_measure(*_media(tmp_path))


def test_syncnet_runs_quiet_and_bounded_only_for_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syncnet_weights: Path
) -> None:
    """The `_BoundedFfmpeg` / `_quiet_syncnet` mechanism moved here from quality_check: during
    `inference` Popen is the bounded one and pyscenedetect is silenced; both restored after."""
    scenedetect = logging.getLogger("pyscenedetect")
    monkeypatch.setattr(scenedetect, "level", logging.INFO)
    popen_before = subprocess.Popen
    seen: list[tuple[Any, int]] = []

    def during() -> None:
        seen.append((subprocess.Popen, scenedetect.level))

    install_syncnet(monkeypatch, result=syncnet_result(-6, 8.63), during=during)
    sync.syncnet_measure(*_media(tmp_path))
    ((popen, level),) = seen
    assert popen is sync._BoundedFfmpeg
    assert issubclass(sync._BoundedFfmpeg, popen_before)
    assert level >= logging.CRITICAL
    assert subprocess.Popen is popen_before
    assert scenedetect.level == logging.INFO
    assert callable(sync._quiet_syncnet)


def test_syncnet_import_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`hwva.sync` imports without SyncNet (conftest makes it unimportable): the import happens
    inside `syncnet_measure` only."""
    import importlib

    reloaded = importlib.reload(sync)
    assert callable(reloaded.syncnet_measure)


# === Phrases ======================================================================================

THREE_PHRASES = [(0.7, 1.6), (1.9, 2.5), (2.7, 3.5)]  # 0.7 s lead-in, pauses 0.3 s and 0.2 s


def test_phrases_are_the_speech_runs_split_at_real_pauses(tmp_path: Path) -> None:
    audio = speech_wav(tmp_path / "shot_audio.wav", THREE_PHRASES, 4.0)
    found = sync.phrases(audio)
    assert len(found) == 3, found
    for phrase, (start, end) in zip(found, THREE_PHRASES, strict=True):
        assert phrase.start_s == pytest.approx(start, abs=0.03)
        assert phrase.end_s == pytest.approx(end, abs=0.03)
    assert all(a.end_s < b.start_s for a, b in pairwise(found))  # time order


def test_a_short_gap_is_not_a_split_and_speech_to_eof_ends_at_the_file_end(tmp_path: Path) -> None:
    audio = speech_wav(tmp_path / "a.wav", [(0.5, 1.2), (1.3, 2.0)], 2.0)  # 0.1 s gap, no tail
    found = sync.phrases(audio)
    assert len(found) == 1, found
    assert found[0].start_s == pytest.approx(0.5, abs=0.03)
    assert found[0].end_s == pytest.approx(2.0, abs=0.03)


def test_silence_has_no_phrases(tmp_path: Path) -> None:
    assert sync.phrases(speech_wav(tmp_path / "s.wav", [], 2.0)) == []


def test_phrases_of_an_unreadable_file_is_a_sync_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF-but-not-really" * 32)
    with pytest.raises(sync.SyncError):
        sync.phrases(bad)


def test_phrases_ffmpeg_calls_pass_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = speech_wav(tmp_path / "shot_audio.wav", THREE_PHRASES, 4.0)
    calls: list[Any] = []

    def spy(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_tool(cmd):
            calls.append(kwargs.get("timeout"))
        return _REAL_RUN(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    assert len(sync.phrases(audio)) == 3
    assert calls and all(t is not None and t <= TIMEOUT_S for t in calls), calls


# === choose_shifts (pure numpy) ===================================================================


def _phrase(start: float, end: float) -> Any:
    return sync.Phrase(start_s=start, end_s=end)


def _frames(start: float, end: float) -> tuple[int, int]:
    return math.floor(start * FPS), math.ceil(end * FPS)


def _shifted_motion(
    n: int, envelope: Series, phrases: list[tuple[float, float]], shifts: list[int]
) -> Series:
    """Motion that repeats each phrase's envelope `shift` frames later (truncated at the clip)."""
    motion = np.zeros(n)
    for (start, end), k in zip(phrases, shifts, strict=True):
        a, b = _frames(start, end)
        for t in range(a, b):
            if 0 <= t + k < n:
                motion[t + k] += envelope[t]
    return motion


def _random_envelope(n: int, phrases: list[tuple[float, float]], seed: int = 5) -> Series:
    rng = np.random.default_rng(seed)
    envelope = np.zeros(n)
    for start, end in phrases:
        a, b = _frames(start, end)
        envelope[a:b] = rng.uniform(0.2, 1.0, b - a)
    return envelope


def _is_step(value: float) -> bool:
    return abs(value * FPS - round(value * FPS)) < 1e-9


def test_each_phrase_gets_its_own_delay() -> None:
    clip_s, n = 8.0, 192
    spans = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    envelope = _random_envelope(n, spans)
    motion = _shifted_motion(n, envelope, spans, [6, -4, 0])  # +0.25 s, -0.167 s, 0
    shifts = sync.choose_shifts(motion, envelope, [_phrase(*s) for s in spans], clip_s)
    assert len(shifts) == 3
    for got, want in zip(shifts, [6 / 24, -4 / 24, 0.0], strict=True):
        assert got == pytest.approx(want, abs=1 / 24 + 1e-9)
    assert all(_is_step(s) for s in shifts), "shifts are k/24 exactly"


def test_order_and_the_min_gap_hold_when_the_best_shifts_would_collide() -> None:
    """P1 wants +8 frames (ends at 2.333 s), P2 wants -4 (starts at 2.333 s): no gap."""
    clip_s, n = 6.0, 144
    spans = [(1.0, 2.0), (2.5, 3.25)]
    envelope = _random_envelope(n, spans, seed=9)
    motion = _shifted_motion(n, envelope, spans, [8, -4])
    shifts = sync.choose_shifts(motion, envelope, [_phrase(*s) for s in spans], clip_s)
    assert shifts[0] == pytest.approx(8 / 24, abs=1e-9)  # left to right: P1 placed first
    p1_end = spans[0][1] + shifts[0]
    p2_start = spans[1][0] + shifts[1]
    assert p2_start >= p1_end + 0.12 - 1e-9, (shifts, p1_end, p2_start)
    assert all(_is_step(s) for s in shifts)


def test_shifts_stay_within_half_a_second() -> None:
    clip_s, n = 6.0, 144
    spans = [(2.0, 3.0)]
    envelope = _random_envelope(n, spans, seed=3)
    motion = _shifted_motion(n, envelope, spans, [18])  # 0.75 s: out of range
    (shift,) = sync.choose_shifts(motion, envelope, [_phrase(*spans[0])], clip_s)
    assert abs(shift) <= 0.5 + 1e-9
    assert _is_step(shift)


@pytest.mark.parametrize(
    ("span", "delay"), [((3.0, 3.8), 12), ((0.1, 0.8), -12)], ids=["clip-end", "clip-start"]
)
def test_a_phrase_is_never_pushed_out_of_the_clip(span: tuple[float, float], delay: int) -> None:
    clip_s, n = 4.0, 96
    envelope = _random_envelope(n, [span], seed=4)
    motion = _shifted_motion(n, envelope, [span], [delay])
    (shift,) = sync.choose_shifts(motion, envelope, [_phrase(*span)], clip_s)
    assert span[0] + shift >= -1e-9
    assert span[1] + shift <= clip_s + 1e-9
    assert _is_step(shift)


def test_motion_series_covers_the_whole_clip_when_the_audio_is_shorter(tmp_path: Path) -> None:
    """Live regression: the series were cut to the (shorter) audio, so the last
    phrase of a Ukrainian clip had no room to move later -> "no feasible shift". The motion
    spans the clip; the envelope is padded with silence to it."""
    on = _pattern()  # 6 s of video
    video = _video(tmp_path / "clip.mp4", on)
    audio = _wav(tmp_path / "short.wav", on[: 4 * FPS])  # 4 s of audio
    motion, envelope = sync.motion_series(video, audio)
    assert len(motion) == len(envelope) == len(on)
    assert (envelope[4 * FPS + 1 :] == 0).all()
    phrase = _phrase(3.3, 3.95)  # ends at the audio end; the motion has it 0.25 s later
    shifted = _shifted_motion(len(on), _random_envelope(len(on), [(3.3, 3.95)]), [(3.3, 3.95)], [6])
    env = _random_envelope(len(on), [(3.3, 3.95)])
    (shift,) = sync.choose_shifts(shifted, env, [phrase], len(on) / FPS)
    assert shift == pytest.approx(6 / FPS, abs=1 / FPS + 1e-9)


def test_flat_motion_gives_no_shift() -> None:
    clip_s, n = 6.0, 144
    spans = [(1.0, 2.0), (2.5, 3.25), (4.0, 5.0)]
    envelope = _random_envelope(n, spans)
    shifts = sync.choose_shifts(np.ones(n), envelope, [_phrase(*s) for s in spans], clip_s)
    assert list(shifts) == [0.0, 0.0, 0.0]


def test_a_tie_goes_to_the_smallest_shift_then_the_negative_one() -> None:
    """Period-4 +-1 patterns: scores are exact and equal at k = +-2, +-6, +-10 (the best);
    smallest |k| -> 2, negative first -> -2/24."""
    clip_s, n = 2.0, 48
    span = (0.5, 1.5)  # frames 12..36
    pattern = np.array([1.0, 1.0, -1.0, -1.0])
    envelope = np.full(n, 2.0)
    envelope[12:36] = 2.0 + np.tile(pattern, 6)  # z-score of the phrase = the +-1 pattern
    motion = 2.0 + np.array([pattern[(j - 14) % 4] for j in range(n)])  # matches at k = +-2
    (shift,) = sync.choose_shifts(motion, envelope, [_phrase(*span)], clip_s)
    assert shift == pytest.approx(-2 / 24, abs=1e-9)


def test_no_feasible_shift_is_a_sync_error() -> None:
    """P1 can only move later (it starts at 0), so it ends >= 1.8 s; P2 must start >= 1.92 s but
    cannot move later (it ends at the clip end): nothing fits."""
    clip_s, n = 2.0, 48
    spans = [(0.0, 1.8), (1.85, 2.0)]
    envelope = _random_envelope(n, spans)
    with pytest.raises(sync.SyncError) as err:
        sync.choose_shifts(np.ones(n), envelope, [_phrase(*s) for s in spans], clip_s)
    assert "no feasible shift" in str(err.value)


# === render =======================================================================================

RATE = 24000  # the source's own rate: neither 16 k nor 44.1/48 k, so a resample shows
FLOOR = 100  # constant level between the speech runs (-50 dBFS): makes the cut padding visible
PAD = round(0.08 * RATE)  # 1920 samples
FADE = round(0.015 * RATE)  # 360 samples


def _s(seconds: float) -> int:
    return round(seconds * RATE)


def _assert_identical_near(out: Any, src: Any, src_a: int, src_b: int, at: int) -> None:
    """src[src_a:src_b] appears unchanged in `out` starting at `at` (+-1 sample of rounding)."""
    want = src[src_a:src_b]
    for d in (0, -1, 1):
        got = out[at + d : at + d + len(want)]
        if len(got) == len(want) and np.array_equal(got, want):
            return
    raise AssertionError(f"speech samples [{src_a}:{src_b}] not found unchanged at {at}")


def test_render_moves_phrases_without_touching_their_samples(tmp_path: Path) -> None:
    spans = [(0.7, 1.6), (2.0, 2.6), (2.9, 3.5)]
    shifts = [0.25, 2 / 24, 0.0]
    src_path = speech_wav(tmp_path / "shot_audio.wav", spans, 4.0, rate=RATE, floor=FLOOR)
    out_path = tmp_path / "sync_piecewise.wav"
    sync.render(src_path, [_phrase(*s) for s in spans], shifts, 4.0, out_path)
    src, _, _ = read_wav(src_path)
    out, rate, channels = read_wav(out_path)
    assert (rate, channels) == (RATE, 1), "same rate, mono: never resampled"
    assert abs(len(out) - _s(4.0)) <= 1, len(out)

    for (start, end), shift in zip(spans, shifts, strict=True):
        a, b, sh = _s(start), _s(end), _s(shift)
        _assert_identical_near(out, src, a, b, a + sh)  # speech: identical, no stretch
        # padding beyond the fades: the source's own samples
        assert (out[a - PAD + FADE + 2 + sh : a + sh] == FLOOR).all()
        assert (out[b + sh : b + PAD - FADE - 2 + sh] == FLOOR).all()
        # linear fade-in at the cut start, fade-out at the cut end (inside the padding)
        p, q = a - PAD + sh, b + PAD + sh  # cut start, cut end (exclusive) in the output
        assert abs(int(out[p])) <= 1 and abs(int(out[q - 1])) <= 1
        assert abs(int(out[p + FADE // 2]) - FLOOR // 2) <= 2
        assert abs(int(out[q - 1 - FADE // 2]) - FLOOR // 2) <= 2
        assert (np.diff(out[p : p + FADE].astype(int)) >= 0).all()
        assert (np.diff(out[q - FADE : q].astype(int)) <= 0).all()
        # nothing just outside the cut
        assert out[p - 2] == 0 and out[q + 1] == 0

    assert (out[: _s(0.62 + 0.25) - 2] == 0).all()  # before the first cut
    assert (out[_s(1.68 + 0.25) + 2 : _s(1.92 + 2 / 24) - 2] == 0).all()  # between cuts
    assert (out[_s(3.58) + 2 :] == 0).all()  # after the last cut


def test_render_adds_overlapping_cuts(tmp_path: Path) -> None:
    """Paddings that overlap in the output are mixed (added), not overwritten."""
    spans = [(1.0, 1.5), (1.62, 2.0)]  # cuts 0.92-1.58 and 1.54-2.08 overlap at 1.54-1.58
    src_path = speech_wav(tmp_path / "a.wav", spans, 2.5, rate=RATE, floor=FLOOR)
    out_path = tmp_path / "out.wav"
    sync.render(src_path, [_phrase(*s) for s in spans], [0.0, 0.0], 2.5, out_path)
    out, _, _ = read_wav(out_path)
    both = out[_s(1.557) : _s(1.563)]  # past P2's fade-in, before P1's fade-out
    assert (np.abs(both.astype(int) - 2 * FLOOR) <= 1).all(), both


def test_render_drops_what_falls_outside_the_track(tmp_path: Path) -> None:
    """P1's cut (clamped to the file start) moved 0.04 s earlier loses its first 0.04 s; P2's
    padding past `duration_s` is cut. Speech stays whole."""
    spans = [(0.05, 0.6), (3.0, 3.5)]
    src_path = speech_wav(tmp_path / "a.wav", spans, 4.0, rate=RATE, floor=FLOOR)
    out_path = tmp_path / "out.wav"
    sync.render(src_path, [_phrase(*s) for s in spans], [-0.04, 0.0], 3.55, out_path)
    src, _, _ = read_wav(src_path)
    out, rate, _ = read_wav(out_path)
    assert rate == RATE and abs(len(out) - _s(3.55)) <= 1
    _assert_identical_near(out, src, _s(0.05), _s(0.6), _s(0.05) - _s(0.04))
    _assert_identical_near(out, src, _s(3.0), _s(3.5), _s(3.0))
    assert (out[: _s(0.005)] == FLOOR).all()  # source samples 0.04.. s: no fade, no shift clamp
    assert (out[-10:] == FLOOR).all()  # the fade-out lay beyond the end: dropped


def test_render_of_a_missing_source_is_a_sync_error(tmp_path: Path) -> None:
    with pytest.raises(sync.SyncError):
        sync.render(tmp_path / "missing.wav", [_phrase(0.5, 1.0)], [0.0], 2.0, tmp_path / "o.wav")


# --- review fixes: edge fades, short runs, no silent cut ----------------------------


def test_render_never_fades_speech_at_the_file_edges(tmp_path: Path) -> None:
    """A phrase starting at 0 and one running to EOF have no padding there: no fade on speech."""
    spans = [(0.0, 0.6), (1.0, 2.0)]  # the second runs to the file end
    src_path = speech_wav(tmp_path / "a.wav", spans, 2.0, rate=RATE, floor=FLOOR)
    out_path = tmp_path / "out.wav"
    sync.render(src_path, [_phrase(*s) for s in spans], [0.0, 0.0], 2.0, out_path)
    src, _, _ = read_wav(src_path)
    out, _, _ = read_wav(out_path)
    assert np.array_equal(out[: _s(0.6)], src[: _s(0.6)])  # the first 15 ms too
    assert np.array_equal(out[_s(1.0) :], src[_s(1.0) :])  # the last 15 ms too


def test_a_short_speech_burst_is_its_own_phrase(tmp_path: Path) -> None:
    runs = [(0.7, 1.5), (1.8, 1.83), (2.1, 2.9)]  # a 30 ms burst between two pauses
    found = sync.phrases(speech_wav(tmp_path / "a.wav", runs, 3.5))
    assert len(found) == 3, found
    assert found[1].start_s == pytest.approx(1.8, abs=0.03)


def test_global_render_refuses_to_cut_speech(tmp_path: Path) -> None:
    """Speech longer than the track even at the earliest start -> SyncError, never a silent cut."""
    spans = [(0.0, 3.0)]
    src_path = speech_wav(tmp_path / "a.wav", spans, 3.0)
    with pytest.raises(sync.SyncError) as err:
        sync.render_global(src_path, [_phrase(*spans[0])], 0.0, 2.5, tmp_path / "o.wav")
    assert "cut" in str(err.value)
    # it fits by moving earlier over the lead-in: allowed, and the applied shift says so
    lead = speech_wav(tmp_path / "b.wav", [(0.7, 3.0)], 3.0)
    applied = sync.render_global(lead, [_phrase(0.7, 3.0)], 0.0, 2.5, tmp_path / "p.wav")
    assert applied == pytest.approx(-0.5)
    assert (
        str(sync.render_global(lead, [_phrase(0.7, 3.0)], -0.0, 3.0, tmp_path / "q.wav")) == "0.0"
    )


# === Record schema (contract; not xfail: the schema is part of this spec-writing step) ============


def test_a_v1_record_still_validates() -> None:
    v1 = {"lag_s": 0.5, "peak_r": 0.71, "r_at_0": 0.16, "verdict": "correct", "shift_s": 0.5,
          "box": [0.2, 0.8, 0.42, 0.6], "take": 1, "error": None, "warning": None,
          "rejected": None}  # fmt: skip
    report = SyncReport.model_validate(v1)
    assert report.verdict == "correct" and report.shift_s == 0.5
    assert report.syncnet_confidence is None and report.method is None
    assert report.phrases == [] and report.aligned_audio is None and report.thresholds is None


def test_a_v2_record_round_trips() -> None:
    record = {
        "verdict": "weak", "syncnet_confidence": 4.0, "syncnet_offset_frames": -6,
        "syncnet_min_dist": 7.1, "shift_s": -0.24,
        "thresholds": {"sync_conf": 5.0, "weak_conf": 3.0}, "method": "piecewise",
        "confidence_global": 4.2, "confidence_piecewise": 4.5, "piecewise_error": None,
        "phrases": [{"start_s": 0.7, "end_s": 1.6, "shift_s": -0.083}],
        "aligned_audio": "aligned_line.wav",
        "motion": {"lag_s": 0.25, "peak_r": 0.36, "r_at_0": 0.1},
        "box": [0.2, 0.8, 0.42, 0.6], "take": 1, "error": None,
        "warning": "weak lip-sync (SyncNet confidence 4 < 5)", "rejected": None,
    }  # fmt: skip
    report = SyncReport.model_validate(record)
    assert SyncReport.model_validate_json(report.model_dump_json()) == report
    assert json.loads(report.model_dump_json())["phrases"][0]["shift_s"] == -0.083


# === Real SyncNet on a clip from disk (local only; skips when absent) ============================


@pytest.mark.syncnet
def test_real_syncnet_on_a_clip_from_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = Path(os.environ.get("HWVA_SYNCNET_TEST_CLIP") or REPO / "missing-clip.mp4")
    audio = Path(os.environ.get("HWVA_SYNCNET_TEST_AUDIO") or REPO / "missing-audio.wav")
    if not clip.is_file() or not audio.is_file():
        pytest.skip(f"no local clip + audio ({clip}, {audio}); set HWVA_SYNCNET_TEST_CLIP/_AUDIO")
    weights = Path(os.environ.get("HWVA_SYNCNET_WEIGHTS") or REPO / "weights")
    if not all((weights / name).is_file() for name in SYNCNET_WEIGHT_FILES):
        pytest.skip(f"SyncNet weights not in {weights} (scripts/{SYNCNET_WEIGHTS_SCRIPT})")
    monkeypatch.setenv("HWVA_SYNCNET_WEIGHTS", str(weights))
    result = sync.syncnet_measure(clip, audio)
    assert type(result.offset_frames) is int
    assert math.isfinite(result.confidence)
    assert sync.verdict(result.confidence, result.offset_frames) in VERDICTS - {"error"}
