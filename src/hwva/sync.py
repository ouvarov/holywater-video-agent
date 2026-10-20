"""Lip-sync gate and phrase alignment (not a pipeline stage), v2: SyncNet decides
whether the talking shot lip-syncs and gives the global shift; our motion metric (mouth-box
motion energy vs the audio envelope) only proposes per-phrase shifts. Runs inside `shot_1`;
`assemble` places the chosen track; `final_check`, `quality_check` and `eval` read the record.

Contract: specs/sync.md. Tests: tests/stages/test_sync.py.

Public API (pinned by the tests):
- `syncnet_measure(video, audio) -> SyncNetResult` — the ONE place that runs SyncNet (lazy
  import); every failure -> `SyncError`, no face -> `NoFaceError`.
- `verdict(confidence, offset_frames, thresholds=...)`, `global_shift(verdict, offset_frames)`,
  `thresholds_from_env()`, `warning_text(verdict, confidence, thresholds, error=None)`.
- `measure(video, audio, *, box) -> MotionResult` — diagnostic only; `motion_series` gives the
  per-frame series `choose_shifts` works on.
- `phrases(audio)`, `choose_shifts(motion, envelope, phrases, clip_s)`, `render(...)`,
  `render_global(...)`, `media_duration(path)`.
- `box_from_env()`, `REJECT_ENV` (`HWVA_SYNC_REJECT`): exactly "1" turns the reject switch on."""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import subprocess
import time
import wave
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hwva.schemas import SyncVerdict

Box = tuple[float, float, float, float]  # x0, x1, y0, y1 as fractions of the frame
Series = npt.NDArray[np.float64]

# --- SyncNet verdict (PROVISIONAL thresholds: n=8 clips, 5 human-rated, specs/sync.md) -----------
SYNC_CONF = 5.0
WEAK_CONF = 3.0
SYNCNET_FPS = 25
PASS_OFFSET_FRAMES = 1
SYNC_CONF_ENV = "HWVA_SYNC_CONF"
WEAK_CONF_ENV = "HWVA_SYNC_WEAK_CONF"
WEIGHTS_ENV = "HWVA_SYNCNET_WEIGHTS"
DEFAULT_WEIGHTS = "weights"
SYNCNET_WEIGHTS = "syncnet_v2.model"
S3FD_WEIGHTS = "sfd_face.pth"
WEIGHTS_SCRIPT = "scripts/get-syncnet-weights.ps1"
SYNCNET_PACKAGE = "syncnet_python"
RENDER_TIMEOUT_S = 120  # every untimed wait of SyncNet's own ffmpeg

# --- motion metric -------------------------------------------------------------------------------
MOUTH_BOX: Box = (0.20, 0.80, 0.42, 0.60)  # lower-centre face of a centred head-and-shoulders shot
FPS = 24
MAX_LAG_FRAMES = 24  # r(L) for L in -24..+24
CROP_SIZE = (96, 48)  # width, height of the scaled mouth crop
FFMPEG_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 30
SAMPLE_RATE = 16000  # audio is decoded to mono at this rate for the envelope
BOX_ENV = "HWVA_SYNC_MOUTH_BOX"
REJECT_ENV = "HWVA_SYNC_REJECT"

# --- phrases and alignment -----------------------------------------------------------------------
PHRASE_NOISE_DB = -40
PHRASE_MIN_PAUSE_S = 0.15
MAX_PHRASE_SHIFT_S = 0.5
MIN_GAP_S = 0.12
PAD_S = 0.08
FADE_S = 0.015
_EPS = 1e-9


class SyncError(RuntimeError):
    """A measurement / alignment step could not run (SyncNet, ffmpeg, IO, odd result)."""


class NoFaceError(SyncError):
    """SyncNet found no face in the clip."""


@dataclass(frozen=True)
class Thresholds:
    sync_conf: float = SYNC_CONF
    weak_conf: float = WEAK_CONF


@dataclass(frozen=True)
class SyncNetResult:
    offset_frames: int  # at the highest confidence, 25 fps frames
    confidence: float  # max_conf (LSE-C)
    min_dist: float  # LSE-D
    seconds: float  # wall time


@dataclass(frozen=True)
class MotionResult:
    """Our motion metric, 3 decimals. Diagnostic only: no verdict, no shift."""

    lag_s: float  # positive = the mouth moves AFTER the audio
    peak_r: float
    r_at_0: float


@dataclass(frozen=True)
class Phrase:
    start_s: float
    end_s: float


# === SyncNet ======================================================================================


def syncnet_measure(video: Path, audio: Path) -> SyncNetResult:
    video, audio = Path(video), Path(audio)
    for what, path in (("video", video), ("audio", audio)):
        if not path.is_file():
            raise SyncError(f"SyncNet: {what} {path} not found")
    weights = Path(os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS)
    syncnet_w, s3fd_w = weights / SYNCNET_WEIGHTS, weights / S3FD_WEIGHTS
    missing = [p.name for p in (syncnet_w, s3fd_w) if not p.is_file()]
    if missing:
        raise SyncError(
            f"SyncNet weights missing in {weights}: {', '.join(missing)} "
            f"(run {WEIGHTS_SCRIPT}, or set {WEIGHTS_ENV})"
        )
    try:
        module: Any = importlib.import_module(SYNCNET_PACKAGE)
    except Exception as exc:  # not installed, or installed but broken (torch DLL on Windows)
        raise SyncError(f"SyncNet: import {SYNCNET_PACKAGE} failed: {_error(exc)}") from exc
    start = time.perf_counter()
    try:
        with _quiet_syncnet():
            pipe = module.SyncNetPipeline(
                {"s3fd_weights": str(s3fd_w), "syncnet_weights": str(syncnet_w)}, device="cpu"
            )
            result: object = pipe.inference(video_path=str(video), audio_path=str(audio))
    except Exception as exc:  # incl. TimeoutExpired from the bounded ffmpeg
        raise SyncError(f"SyncNet failed: {_error(exc)}") from exc
    return _read_result(result, round(time.perf_counter() - start, 3))


def _read_result(result: object, seconds: float) -> SyncNetResult:
    """Parse `inference()`'s 7-tuple (lists or numpy); anything odd is a SyncError."""
    if not isinstance(result, tuple | list) or len(result) != 7:
        raise SyncError(f"SyncNet: unexpected result {type(result).__name__}")
    offsets_raw, confs_raw, _dists, max_conf, min_dist, _faces, has_face = result
    try:
        offsets = [int(o) for o in offsets_raw]
        face = bool(has_face)
    except (TypeError, ValueError) as exc:
        raise SyncError(f"SyncNet: unreadable result: {_error(exc)}") from exc
    if not face or not offsets:
        raise NoFaceError("no face found by SyncNet")
    try:
        confs = [float(c) for c in confs_raw]
        confidence, distance = float(max_conf), float(min_dist)
    except (TypeError, ValueError) as exc:
        raise SyncError(f"SyncNet: unreadable result: {_error(exc)}") from exc
    if len(confs) != len(offsets):
        raise SyncError(f"SyncNet: {len(offsets)} offsets but {len(confs)} confidences")
    if not (math.isfinite(confidence) and math.isfinite(distance)):
        raise SyncError(f"SyncNet: non-finite result (confidence {confidence}, dist {distance})")
    best = max(range(len(offsets)), key=lambda i: confs[i])
    return SyncNetResult(
        offset_frames=offsets[best], confidence=confidence, min_dist=distance, seconds=seconds
    )


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class _BoundedFfmpeg(subprocess.Popen[Any]):
    """Popen for SyncNet's ffmpeg: quiet, output not inherited, and every wait bounded."""

    def __init__(self, args: Any, *rest: Any, **kwargs: Any) -> None:
        self._hwva_killed = False
        if isinstance(args, list | tuple) and args and Path(str(args[0])).stem.lower() == "ffmpeg":
            argv = [str(a) for a in args]
            for flag in ("-loglevel", "-v"):  # drop any level SyncNet set; ours goes first
                while flag in argv[1:]:
                    i = argv.index(flag, 1)
                    del argv[i : i + 2]
            args = [argv[0], "-loglevel", "error", *argv[1:]]
            for stream in ("stdout", "stderr"):
                if kwargs.get(stream) is None:
                    kwargs[stream] = subprocess.DEVNULL
        super().__init__(args, *rest, **kwargs)

    def kill(self) -> None:
        self._hwva_killed = True
        super().kill()

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[Any, Any]:
        if timeout is not None or self._hwva_killed:
            return super().communicate(input, timeout)
        try:
            return super().communicate(input, RENDER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._reap()
            raise

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None or self._hwva_killed:
            return super().wait(timeout)
        try:
            return super().wait(RENDER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._reap()
            raise

    def _reap(self) -> None:
        self.kill()
        super().communicate()  # unbounded after kill: collects the dead process, closes pipes


@contextmanager
def _quiet_syncnet() -> Iterator[None]:
    # syncnet_python spawns ffmpeg via subprocess.run (captured, no timeout) and ffmpeg-python
    # `.run()` (inherits the console, no timeout) and we can't edit it. For the duration of the
    # call only, `subprocess.Popen` is swapped for `_BoundedFfmpeg` (ffmpeg argv gets
    # `-loglevel error`, uninherited output goes to DEVNULL, untimed waits get 120 s and the process
    # is killed on expiry) and the `pyscenedetect` logger is raised to CRITICAL (it logs its
    # deprecation messages at ERROR); both restored in finally. Process-global, which is fine:
    # stages run sequentially.
    scenedetect = logging.getLogger("pyscenedetect")
    old_level, old_popen = scenedetect.level, subprocess.Popen
    scenedetect.setLevel(logging.CRITICAL)
    setattr(subprocess, "Popen", _BoundedFfmpeg)  # noqa: B010 - mypy rejects assigning a type
    try:
        yield
    finally:
        setattr(subprocess, "Popen", old_popen)  # noqa: B010
        scenedetect.setLevel(old_level)


# === Verdict ======================================================================================


def verdict(
    confidence: float, offset_frames: int, thresholds: Thresholds | None = None
) -> SyncVerdict:
    """The rule of the "Verdict" table, on the RAW confidence."""
    t = thresholds or Thresholds()
    if confidence >= t.sync_conf:
        return "pass" if abs(offset_frames) <= PASS_OFFSET_FRAMES else "correct"
    if confidence >= t.weak_conf:
        return "weak"
    return "no_sync"


def global_shift(result_verdict: str, offset_frames: int) -> float:
    """The global shift in s (negative = audio earlier): offset/25 for correct / weak, else 0."""
    if result_verdict in ("correct", "weak"):
        return offset_frames / SYNCNET_FPS
    return 0.0


def thresholds_from_env() -> Thresholds:
    sync_conf = _env_threshold(SYNC_CONF_ENV, SYNC_CONF)
    weak_conf = _env_threshold(WEAK_CONF_ENV, WEAK_CONF)
    if weak_conf > sync_conf:
        raise ValueError(
            f"{WEAK_CONF_ENV}={weak_conf:g} is above {SYNC_CONF_ENV}={sync_conf:g}: "
            "the weak threshold must not exceed the sync one"
        )
    return Thresholds(sync_conf=sync_conf, weak_conf=weak_conf)


def _env_threshold(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: not a number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}={raw!r}: need a finite number >= 0")
    return value


def warning_text(
    result_verdict: SyncVerdict,
    confidence: float | None,
    thresholds: Thresholds | None = None,
    error: str | None = None,
) -> str | None:
    """The one-line warning of the record (specs/sync.md "Record"); None for pass / correct."""
    t = thresholds or Thresholds()
    shown = "n/a" if confidence is None else f"{round(confidence, 3):g}"
    if result_verdict == "weak":
        return f"weak lip-sync (SyncNet confidence {shown} < {t.sync_conf:g})"
    if result_verdict == "no_sync":
        return f"lip-sync not detected (SyncNet confidence {shown} < {t.weak_conf:g})"
    if result_verdict == "error":
        detail = " ".join((error or "unknown").split())
        return f"lip-sync measurement error: {detail}"
    return None


def box_from_env() -> Box:
    raw = os.environ.get(BOX_ENV)
    if raw is None or not raw.strip():
        return MOUTH_BOX
    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError(f"{BOX_ENV}={raw!r}: expected 4 numbers x0,x1,y0,y1")
    x0, x1, y0, y1 = (float(p) for p in parts)  # "abc" -> ValueError
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):  # nan fails every comparison
        raise ValueError(f"{BOX_ENV}={raw!r}: need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return (x0, x1, y0, y1)


# === Motion metric (diagnostic; per-frame series for the phrase shifts) ==========================


def motion_series(video: Path, audio: Path, *, box: Box = MOUTH_BOX) -> tuple[Series, Series]:
    """(motion, envelope) at 24 fps, raw (not z-scored), over the WHOLE clip: the envelope is
    cut or padded with silence to the clip's frame count, so a phrase near the end of a short
    audio can still move later (live: cut to the audio, the last phrase had no room)."""
    motion = _motion(_mouth_frames(video, box))
    envelope = _envelope(_samples(audio))[: len(motion)]
    if len(envelope) < 2:
        raise SyncError(f"lip-sync: only {len(envelope)} audio frame(s) in {audio.name}")
    return motion, np.pad(envelope, (0, len(motion) - len(envelope)))


def measure(video: Path, audio: Path, *, box: Box = MOUTH_BOX) -> MotionResult:
    motion = _motion(_mouth_frames(video, box))
    envelope = _envelope(_samples(audio))
    n = min(len(motion), len(envelope))
    if n < 2:
        raise SyncError(f"lip-sync: only {n} common frame(s) of {video.name} and {audio.name}")
    m, e = _zscore(motion[:n]), _zscore(envelope[:n])
    best_lag, best_r = 0, _r(m, e, 0)
    for k in range(1, MAX_LAG_FRAMES + 1):  # by growing |L|: a tie keeps the smallest |L|
        for lag in (k, -k):
            r = _r(m, e, lag)
            if r > best_r:
                best_lag, best_r = lag, r
    return MotionResult(
        lag_s=round(best_lag / FPS, 3), peak_r=round(best_r, 3), r_at_0=round(_r(m, e, 0), 3)
    )


def media_duration(path: Path, *, video: bool = False) -> float:
    """ffprobe duration of a media file (`video=True`: of its first video stream, the length
    assemble checks speech against; the container's when the stream has none). Any failure ->
    SyncError."""
    select = ["-select_streams", "v:0", "-show_entries", "stream=duration:format=duration"]
    entries = select if video else ["-show_entries", "format=duration"]
    out = _run(
        ["ffprobe", "-v", "error", *entries, "-of", "json", str(path)],
        f"duration of {Path(path).name}",
        timeout=PROBE_TIMEOUT_S,
    )
    try:
        data = json.loads(out.stdout)
        streams = data.get("streams") or [{}]
        value = streams[0].get("duration") if video else None
        if value in (None, "N/A"):
            value = data["format"]["duration"]
        return float(value)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SyncError(f"lip-sync: no duration for {Path(path).name}: {_error(exc)}") from exc


def _run(
    cmd: list[str], what: str, timeout: float = FFMPEG_TIMEOUT_S
) -> subprocess.CompletedProcess[bytes]:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as err:
        raise SyncError(f"lip-sync: {cmd[0]} {what} timed out after {timeout:g} s") from err
    except subprocess.CalledProcessError as err:
        raise SyncError(f"lip-sync: {cmd[0]} {what} failed: {_text(err.stderr)}") from err
    except OSError as err:
        raise SyncError(f"lip-sync: {cmd[0]} {what} could not start: {err}") from err
    if out.returncode != 0:
        raise SyncError(
            f"lip-sync: {cmd[0]} {what} failed (exit {out.returncode}): {_text(out.stderr)}"
        )
    return out


def _ffmpeg(args: list[str], what: str) -> bytes:
    out = _run(["ffmpeg", "-v", "error", "-nostdin", *args], what)
    return bytes(out.stdout or b"")


def _text(stderr: object) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    lines = str(stderr or "").strip().splitlines()
    return lines[-1] if lines else "no message"


def _mouth_frames(video: Path, box: Box) -> npt.NDArray[np.uint8]:
    x0, x1, y0, y1 = box
    w, h = CROP_SIZE
    crop = (
        f"crop=trunc(iw*{x1 - x0:.6f}):trunc(ih*{y1 - y0:.6f})"
        f":trunc(iw*{x0:.6f}):trunc(ih*{y0:.6f})"
    )
    vf = f"fps={FPS},format=gray,{crop},scale={w}:{h}"
    raw = _ffmpeg(
        ["-i", str(video), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        f"frames of {video.name}",
    )
    count = len(raw) // (w * h)
    if count < 2:
        raise SyncError(f"lip-sync: {video.name} has {count} frame(s), need at least 2")
    return np.frombuffer(raw[: count * w * h], dtype=np.uint8).reshape(count, h, w)


def _samples(audio: Path) -> Series:
    raw = _ffmpeg(
        ["-i", str(audio), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        f"audio of {Path(audio).name}",
    )
    samples = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4").astype(np.float64)
    if samples.size == 0:
        raise SyncError(f"lip-sync: {Path(audio).name} has no audio samples")
    return samples


def _motion(frames: npt.NDArray[np.uint8]) -> Series:
    """motion[t] = mean |frame t - frame t-1|, motion[0] = 0."""
    as_float = frames.astype(np.float64)
    diffs = np.abs(np.diff(as_float, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], diffs])


def _envelope(samples: Series) -> Series:
    """RMS over hop t = [t/FPS, (t+1)/FPS) s."""
    hops = math.ceil(len(samples) * FPS / SAMPLE_RATE)
    edges = [math.ceil(t * SAMPLE_RATE / FPS) for t in range(hops + 1)]
    values = [float(np.sqrt(np.mean(samples[a:b] ** 2))) for a, b in pairwise(edges) if b > a]
    return np.asarray(values, dtype=np.float64)


def _zscore(x: Series) -> Series:
    std = float(np.std(x)) if len(x) else 0.0
    if std == 0 or not math.isfinite(std):
        return np.zeros_like(x)
    result: Series = (x - float(np.mean(x))) / std
    return result


def _r(m: Series, e: Series, lag: int) -> float:
    n = len(m)
    if lag >= 0:
        a, b = m[lag:], e[: n - lag]
    else:
        a, b = m[: n + lag], e[-lag:]
    if len(a) == 0:
        return 0.0
    return float(np.mean(a * b))


# === Phrases ======================================================================================

_NUMBER = rb"(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)"  # ffmpeg may print 1e-05
_SILENCE_START = re.compile(rb"silence_start:\s*" + _NUMBER)
_SILENCE_END = re.compile(rb"silence_end:\s*" + _NUMBER)
MIN_RUN_S = 0.005  # a speech run shorter than this is a rounding artefact, not speech


def phrases(audio: Path) -> list[Phrase]:
    """Speech runs of `audio`, split at pauses >= 0.15 s below -40 dB (ffmpeg silencedetect)."""
    total_s = len(_samples(audio)) / SAMPLE_RATE  # also: unreadable / empty -> SyncError
    detect = f"silencedetect=n={PHRASE_NOISE_DB}dB:d={PHRASE_MIN_PAUSE_S}"
    out = _run(
        ["ffmpeg", "-v", "info", "-nostdin", "-hide_banner", "-i", str(audio), "-vn",
         "-af", detect, "-f", "null", "-"],
        f"silencedetect of {Path(audio).name}",
    )  # fmt: skip
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in (out.stderr or b"").splitlines():
        if m := _SILENCE_START.search(line):
            start = float(m.group(1))
        elif (m := _SILENCE_END.search(line)) and start is not None:
            silences.append((max(0.0, start), min(total_s, float(m.group(1)))))
            start = None
    if start is not None:  # silence to the end of the file
        silences.append((max(0.0, start), total_s))
    result: list[Phrase] = []
    cursor = 0.0
    for a, b in silences:
        if a - cursor > MIN_RUN_S:  # every real run is a phrase, however short
            result.append(Phrase(start_s=round(cursor, 3), end_s=round(a, 3)))
        cursor = max(cursor, b)
    if total_s - cursor > MIN_RUN_S:  # speech to the end of the file
        result.append(Phrase(start_s=round(cursor, 3), end_s=round(total_s, 3)))
    return result


# === Per-phrase shifts ============================================================================


def choose_shifts(
    motion: Series, envelope: Series, phrase_list: Sequence[Phrase], clip_s: float
) -> list[float]:
    """Per-phrase shift k/24 s (|k| <= 12), left to right: order kept, gap >= 0.12 s, inside the
    clip; the best mean(z(phrase envelope) * z(motion)); a tie -> smallest |k|, negative first."""
    m = _zscore(np.asarray(motion, dtype=np.float64))
    env = np.asarray(envelope, dtype=np.float64)
    max_k = round(MAX_PHRASE_SHIFT_S * FPS)
    ks = [0] + [s * k for k in range(1, max_k + 1) for s in (-1, 1)]
    shifts: list[float] = []
    prev_end: float | None = None
    for i, phrase in enumerate(phrase_list):
        a, b = math.floor(phrase.start_s * FPS), math.ceil(phrase.end_s * FPS)
        e = _zscore(env[a:b])
        best: tuple[float, int] | None = None
        for k in ks:
            shift = k / FPS
            if phrase.start_s + shift < -_EPS or phrase.end_s + shift > clip_s + _EPS:
                continue
            if a + k < 0 or b + k > len(m) or len(e) != b - a:
                continue
            if prev_end is not None and phrase.start_s + shift < prev_end + MIN_GAP_S - _EPS:
                continue
            score = float(np.mean(e * m[a + k : b + k])) if b > a else 0.0
            if best is None or score > best[0]:
                best = (score, k)
        if best is None:
            raise SyncError(f"piecewise: no feasible shift for phrase {i}")
        shift = best[1] / FPS
        shifts.append(shift)
        prev_end = phrase.end_s + shift
    return shifts


# === Render =======================================================================================


def render(
    audio: Path,
    phrase_list: Sequence[Phrase],
    shifts: Sequence[float],
    duration_s: float,
    out: Path,
) -> None:
    """Each phrase cut with 80 ms padding and 15 ms linear fades, mixed into a silent track of
    `duration_s` at its new place. The fades lie in the padding actually present (none where the
    cut meets the file edge), so speech samples are copied exactly: never resampled, stretched
    or faded."""
    samples, rate = _read_pcm(Path(audio))
    n_out = round(duration_s * rate)
    track = np.zeros(n_out, dtype=np.float64)
    pad, fade = round(PAD_S * rate), round(FADE_S * rate)
    for phrase, shift in zip(phrase_list, shifts, strict=True):
        speech_a, speech_b = round(phrase.start_s * rate), round(phrase.end_s * rate)
        a, b = max(0, speech_a - pad), min(len(samples), speech_b + pad)
        if b <= a:
            continue
        cut = samples[a:b].astype(np.float64)
        head = max(0, min(fade, speech_a - a))  # only the padding present in front
        tail = max(0, min(fade, b - speech_b))  # ... and behind
        if head:
            cut[:head] *= np.arange(head, dtype=np.float64) / head
        if tail:
            cut[-tail:] *= (np.arange(tail, dtype=np.float64) / tail)[::-1]
        dest = a + round(shift * rate)
        lo, hi = max(0, dest), min(n_out, dest + len(cut))
        if hi > lo:
            track[lo:hi] += cut[lo - dest : hi - dest]
    data = np.clip(np.round(track), -32768, 32767).astype("<i2")
    try:
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())
    except OSError as exc:
        raise SyncError(f"render: cannot write {Path(out).name}: {exc}") from exc


def render_global(
    audio: Path, phrase_list: Sequence[Phrase], shift_s: float, duration_s: float, out: Path
) -> float:
    """The global alignment: every phrase moved by `shift_s`, clamped so the first speech starts
    >= 0 and the last ends <= `duration_s` (the start wins). Returns the applied shift. Speech
    that cannot fit the track even then -> SyncError: the line is never cut silently (the caller
    then records no aligned track and assemble's own gate names the problem)."""
    applied = shift_s
    if phrase_list:
        first, last = phrase_list[0].start_s, phrase_list[-1].end_s
        applied = max(min(applied, duration_s - last), -first)
        if last + applied > duration_s + 1.0 / (2 * FPS):
            raise SyncError(
                f"aligned track: speech ends at {last + applied:.3f} s, after the clip end "
                f"{duration_s:.3f} s; the line would be cut"
            )
    applied = round(applied, 6) + 0.0  # never -0.0 in the record
    render(audio, phrase_list, [applied] * len(phrase_list), duration_s, out)
    return applied


def _read_pcm(audio: Path) -> tuple[npt.NDArray[np.int16], int]:
    """Mono PCM 16-bit samples at the file's own rate (a stereo file is averaged)."""
    try:
        with wave.open(str(audio), "rb") as w:
            width, rate, channels = w.getsampwidth(), w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise SyncError(f"render: cannot read {audio.name}: {exc}") from exc
    if width != 2:
        raise SyncError(f"render: {audio.name} is not 16-bit PCM")
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).round().astype("<i2")
    return samples.copy(), rate
