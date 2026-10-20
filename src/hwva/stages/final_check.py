"""final_check — prove final.mp4 is right without anyone watching it.

Only deterministic, provable checks. Every check is always measured and written to
final_report.json (raw value, threshold, pass/fail) — also when a hard gate fails, so the report
explains the failure before `GateFailed` is raised. Soft signals are recorded, never block.
Tool failures and timeouts (30 s probes, 120 s analysis/decode/extraction), STT errors, a
missing/unknown lang and a bad script.json become failed check values, never a crash before the
report.

Contract: specs/final_check.md.
Data: `hwva.schemas.FinalReport`;
artifacts: {"report": "final_report.json", "stt_audio": "final_audio.wav"} (the audio STT heard)."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import FinalCheck, FinalReport, Lang, Script
from hwva.text import diff_words

FINAL = "final.mp4"
AUDIO = "line.wav"
REPORT = "final_report.json"
EXTRACTED = "final_audio.wav"  # audio track of final.mp4, inside the run dir, for STT

MIN_S, MAX_S = 8.0, 30.0
BLACK_FILTER = "blackdetect=d=0.1:pix_th=0.10"
MAX_BLACK_S = 1.0
SILENCE_FILTER = "silencedetect=n=-40dB:d=0.4"
MAX_LEAD_SILENCE_S = 0.5
LEAD_START_EPS_S = 0.05  # a silence starting this close to t=0 counts as leading
MIN_LUFS = -35.0
MATCH_RATE = 16000
MIN_PEAK = 0.9
MAX_OFFSET_MS = 50.0
STT_ATTEMPTS = 2  # video is already paid; STT is not deterministic -> one retry
SCDET_FILTER = "scdet=threshold=10"
STDERR_TAIL_LINES = 10
PROBE_TIMEOUT_S = 30
RENDER_TIMEOUT_S = 120  # analysis passes, decode, extraction
LANGS: tuple[Lang, ...] = ("uk", "en")

Signal = npt.NDArray[np.float64]


class _ToolError(Exception):
    """ffmpeg/ffprobe failed; becomes a failed check, never a crash before the report."""


def run(ctx: Context) -> StageOutput:
    line: str = ctx.manifest.params["line"]  # the single source of truth, never script.line
    lang = ctx.manifest.params.get("lang")  # validated in _line_check: never a silent default
    final = ctx.run_dir / FINAL
    line_wav = ctx.run_dir / AUDIO
    if not final.is_file():
        raise GateFailed(f"{FINAL} not found in the run dir; re-run assemble.")
    if not line_wav.is_file():
        raise GateFailed(f"{AUDIO} not found in the run dir; re-run voice.")
    shots, script_error = _shot_count(ctx)
    reference = _placed_audio(ctx, line_wav)
    offset_s = _audio_offset(ctx)
    speech_start_s = _speech_start(ctx, offset_s)

    fmt, has_audio, duration_s = _format_check(final)
    video = _video_signals(final, duration_s)
    audio = _audio_signals(final, duration_s, has_audio, speech_start_s)
    line_check, transcript = _line_check(ctx, final, line, lang, has_audio)

    checks = [
        fmt,
        video["black_frames"],
        audio["leading_silence"],
        audio["loudness"],
        _audio_match_check(final, reference, has_audio, offset_s),
        line_check,
        video["freeze"],
        audio["true_peak"],
        _scene_cuts(video["scene_cuts"], shots, script_error),
    ]
    failed = [c.name for c in checks if c.hard and not c.passed]
    report = FinalReport(
        verdict="fail" if failed else "pass",
        failed=failed,
        transcript=transcript,
        checks=checks,
    )
    (ctx.run_dir / REPORT).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if failed:
        raise GateFailed(f"final check failed: {', '.join(failed)}; see {REPORT} in the run dir.")
    return StageOutput(
        artifacts={"report": REPORT, "stt_audio": EXTRACTED}, data=report.model_dump(mode="json")
    )


# --- inputs -------------------------------------------------------------------------------------


def _shot_count(ctx: Context) -> tuple[int | None, str | None]:
    """Shot count from script.json, or (None, error). Only the soft scene_cuts signal needs it,
    so a missing/malformed script never stops the stage before the report."""
    record = ctx.manifest.stages.get("script")
    name = record.artifacts.get("script") if record is not None else None
    path = ctx.run_dir / (name or "script.json")
    try:
        script = Script.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"script not found at {path.name}"
    except (OSError, ValueError) as err:  # pydantic ValidationError is a ValueError
        return None, f"script {path.name} unreadable: {type(err).__name__}: {err}"
    return len(script.shots), None


def _placed_audio(ctx: Context, line_wav: Path) -> Path:
    """The audio assemble actually placed (artifact `audio`: aligned_line.wav or line.wav);
    absent -> line.wav (older runs). Recorded but missing -> GateFailed, never a silent swap."""
    record = ctx.manifest.stages.get("assemble")
    name = record.artifacts.get("audio") if record is not None else None
    if not name:
        return line_wav
    path = ctx.run_dir / name
    if not path.is_file():
        raise GateFailed(f"{name} (the audio assemble placed) not found; re-run assemble.")
    return path


def _assemble_number(ctx: Context, key: str) -> float | None:
    record = ctx.manifest.stages.get("assemble")
    raw = record.data.get(key) if record is not None else None
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return value if value is not None and math.isfinite(value) else None


def _audio_offset(ctx: Context) -> float:
    """Where assemble placed the audio in final.mp4 (specs/sync.md): assemble's recorded
    `audio_offset_s`, 0 when absent or unreadable (the strict value). Negative only for an
    aligned track trimmed to fit (its first -offset seconds are not in the final)."""
    return _assemble_number(ctx, "audio_offset_s") or 0.0


def _speech_start(ctx: Context, offset_s: float) -> float:
    """First speech in the final (the leading-silence allowance): assemble's `speech_start_s`,
    else the offset (older runs). Never negative."""
    value = _assemble_number(ctx, "speech_start_s")
    return max(0.0, value if value is not None else offset_s)


# --- tools --------------------------------------------------------------------------------------


def _tool(
    cmd: list[str], what: str, timeout: int = RENDER_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        raise _ToolError(f"{what} timed out after {timeout} s") from err
    except subprocess.CalledProcessError as err:
        lines = [ln for ln in (err.stderr or "").strip().splitlines() if ln.strip()]
        tail = "\n".join(lines[-STDERR_TAIL_LINES:]) or f"exit code {err.returncode}"
        raise _ToolError(f"{what} failed: {tail}") from err


def _analyze(path: Path, flag: str, graph: str) -> str:
    """Run ffmpeg filters over one stream type; return stderr (where the filters log)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), flag, graph, "-f", "null", "-"]
    return _tool(cmd, f"ffmpeg analysis of {path.name}").stderr


def _decode(path: Path) -> Signal:
    """First audio stream -> mono 16 kHz float samples."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
             "-ac", "1", "-ar", str(MATCH_RATE), "-f", "f32le", "-"],
            capture_output=True, check=True, timeout=RENDER_TIMEOUT_S,
        )  # fmt: skip
    except subprocess.TimeoutExpired as err:
        raise _ToolError(
            f"decoding audio of {path.name} timed out after {RENDER_TIMEOUT_S} s"
        ) from err
    except subprocess.CalledProcessError as err:
        raise _ToolError(f"decoding audio of {path.name} failed") from err
    return np.frombuffer(out.stdout, dtype="<f4").astype(np.float64)


def _floats(pattern: str, text: str) -> list[float]:
    return [float(m) for m in re.findall(pattern, text)]


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


# --- hard: format -------------------------------------------------------------------------------


def _format_check(path: Path) -> tuple[FinalCheck, bool, float]:
    threshold = "8 <= duration <= 30 s; height > width; exactly 1 video + 1 audio stream"
    try:
        out = _tool(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,width,height:format=duration", "-of", "json", str(path)],
            f"ffprobe of {path.name}", PROBE_TIMEOUT_S,
        ).stdout  # fmt: skip
        data: dict[str, Any] = json.loads(out)
    except (_ToolError, json.JSONDecodeError) as err:
        check = FinalCheck(
            name="format", hard=True, passed=False, value={"error": str(err)}, threshold=threshold
        )
        return check, False, 0.0
    streams: list[dict[str, Any]] = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    n_audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    try:
        duration_s = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration_s = 0.0
    width = int(videos[0].get("width", 0)) if videos else 0
    height = int(videos[0].get("height", 0)) if videos else 0
    passed = MIN_S <= duration_s <= MAX_S and height > width and len(videos) == 1 and n_audio == 1
    value = {
        "duration_s": round(duration_s, 3),
        "width": width,
        "height": height,
        "video_streams": len(videos),
        "audio_streams": n_audio,
    }
    check = FinalCheck(name="format", hard=True, passed=passed, value=value, threshold=threshold)
    return check, n_audio >= 1, duration_s


# --- video pass: black frames (hard), freeze + scene cuts (soft) --------------------------------


def _video_signals(path: Path, duration_s: float) -> dict[str, FinalCheck]:
    black_thr = f"total black <= {MAX_BLACK_S} s ({BLACK_FILTER})"
    try:
        log = _analyze(path, "-filter:v", f"{BLACK_FILTER},freezedetect,{SCDET_FILTER}")
    except _ToolError as err:
        error = {"error": str(err)}
        return {
            "black_frames": FinalCheck(
                name="black_frames", hard=True, passed=False,
                value={"black_s": None, **error}, threshold=black_thr,
            ),
            "freeze": FinalCheck(
                name="freeze", hard=False, passed=False,
                value={"freeze_s": None, **error}, threshold="recorded only",
            ),
            "scene_cuts": FinalCheck(
                name="scene_cuts", hard=False, passed=False,
                value={"cuts": None, **error}, threshold="recorded only",
            ),
        }  # fmt: skip
    black_s = sum(_floats(r"black_duration:\s*([\d.]+)", log))
    return {
        "black_frames": FinalCheck(
            name="black_frames",
            hard=True,
            passed=black_s <= MAX_BLACK_S,
            value={"black_s": round(black_s, 3)},
            threshold=black_thr,
        ),
        "freeze": FinalCheck(
            name="freeze",
            hard=False,
            passed=True,
            value={"freeze_s": round(_freeze_total(log, duration_s), 3)},
            threshold="recorded only (talking heads are nearly static)",
        ),
        "scene_cuts": FinalCheck(
            name="scene_cuts",
            hard=False,
            passed=True,
            value={"cuts": len(re.findall(r"lavfi\.scd\.time:", log))},
            threshold="recorded only; expected cuts = shots - 1",
        ),
    }


def _freeze_total(log: str, duration_s: float) -> float:
    """Sum of freeze durations; a freeze that never ends (frozen until EOF) counts up to the
    file duration."""
    total, open_start = 0.0, None
    for kind, raw in re.findall(r"lavfi\.freezedetect\.freeze_(start|end):\s*([\d.]+)", log):
        t = float(raw)
        if kind == "start":
            open_start = t
        elif open_start is not None:
            total += t - open_start
            open_start = None
    if open_start is not None:
        total += max(0.0, duration_s - open_start)
    return total


def _scene_cuts(check: FinalCheck, shots: int | None, error: str | None) -> FinalCheck:
    value = {**check.value, "shots": shots}
    if error is not None:
        prior = value.get("error")
        value["error"] = f"{prior}; {error}" if prior else error
    return check.model_copy(update={"value": value})


# --- audio pass: leading silence, loudness (hard), true peak (soft) -----------------------------


# name -> (hard, threshold, value keys when nothing could be measured)
_AUDIO_CHECKS: dict[str, tuple[bool, str, dict[str, Any]]] = {
    "leading_silence": (
        True,
        f"leading silence <= {MAX_LEAD_SILENCE_S} s + speech_start_s ({SILENCE_FILTER})",
        {"leading_silence_s": None},
    ),
    "loudness": (True, f">= {MIN_LUFS:g} LUFS (ebur128 integrated)", {"integrated_lufs": None}),
    "true_peak": (False, "recorded only", {"true_peak_dbtp": None}),
}


def _audio_signals(
    path: Path, duration_s: float, has_audio: bool, speech_start_s: float
) -> dict[str, FinalCheck]:
    allowed_s = round(MAX_LEAD_SILENCE_S + speech_start_s, 3)  # the first phrase, no more
    measured: dict[str, tuple[bool, dict[str, Any]]]
    try:
        if not has_audio:
            raise _ToolError("no audio stream")
        log = _analyze(path, "-filter:a", f"{SILENCE_FILTER},ebur128=peak=true")
    except _ToolError as err:
        measured = {
            name: (False, {**empty, "error": str(err)})
            for name, (_, _, empty) in _AUDIO_CHECKS.items()
        }
    else:
        lead = _leading_silence(log, duration_s)
        summary = log[log.rfind("Summary:") :] if "Summary:" in log else log
        lufs = _last(_floats(r"I:\s*(-?[\d.]+|-inf)\s*LUFS", summary))
        peak = _last(_floats(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", summary))
        loud_ok = lufs is not None and math.isfinite(lufs) and lufs >= MIN_LUFS
        measured = {
            "leading_silence": (lead <= allowed_s, {"leading_silence_s": round(lead, 3)}),
            "loudness": (loud_ok, {"integrated_lufs": _finite(lufs)}),
            "true_peak": (True, {"true_peak_dbtp": _finite(peak)}),
        }
    measured["leading_silence"][1]["allowed_s"] = allowed_s
    return {
        name: FinalCheck(
            name=name,
            hard=hard,
            passed=measured[name][0],
            value=measured[name][1],
            threshold=threshold,
        )
        for name, (hard, threshold, _) in _AUDIO_CHECKS.items()
    }


def _last(values: list[float]) -> float | None:
    return values[-1] if values else None


def _leading_silence(log: str, duration_s: float) -> float:
    starts = _floats(r"silence_start:\s*(-?[\d.]+)", log)
    if not starts or starts[0] > LEAD_START_EPS_S:
        return 0.0
    ends = _floats(r"silence_end:\s*(-?[\d.]+)", log)
    return ends[0] if ends else duration_s  # silent to the end


# --- hard: audio is OUR TTS ---------------------------------------------------------------------


def _audio_match_check(
    final: Path, reference: Path, has_audio: bool, offset_s: float
) -> FinalCheck:
    threshold = (
        f"normalized cross-correlation peak >= {MIN_PEAK} with the placed audio at "
        f"|offset - audio_offset_s| <= {MAX_OFFSET_MS:g} ms"
    )
    # a negative offset: the reference's first -offset_s seconds were trimmed, the rest is at 0
    trim_s, place_s = max(0.0, -offset_s), max(0.0, offset_s)
    expected_ms = round(place_s * 1000.0, 1)
    base: dict[str, Any] = {"reference": reference.name, "expected_offset_ms": expected_ms}
    if not has_audio:
        value: dict[str, Any] = {
            "peak": None, "offset_ms": None, **base, "error": "no audio stream"
        }  # fmt: skip
        return FinalCheck(
            name="audio_match", hard=True, passed=False, value=value, threshold=threshold
        )
    try:
        ref = _decode(reference)[round(trim_s * MATCH_RATE) :]
        peak, offset_ms = _xcorr(_decode(final), ref, place_s)
    except _ToolError as err:
        value = {"peak": None, "offset_ms": None, **base, "error": str(err)}
        return FinalCheck(
            name="audio_match", hard=True, passed=False, value=value, threshold=threshold
        )
    passed = peak >= MIN_PEAK and abs(offset_ms - expected_ms) <= MAX_OFFSET_MS
    return FinalCheck(
        name="audio_match",
        hard=True,
        passed=passed,
        value={"peak": round(peak, 4), "offset_ms": round(offset_ms, 1), **base},
        threshold=threshold,
    )


def _xcorr(final: Signal, ref: Signal, offset_s: float = 0.0) -> tuple[float, float]:
    """Peak of the normalized cross-correlation of `ref` slid over `final`, and its lag in ms.
    Each lag is normalized by the energy of `ref` and of the `final` window it covers, so the
    value is 1.0 for an exact (scaled) copy, regardless of volume. `final` is zero-padded so the
    search reaches `offset_s` even when assemble cut ref's trailing silence at the video end."""
    n = len(ref)
    ref_energy = float(np.dot(ref, ref))
    if n == 0 or ref_energy <= 0.0 or len(final) == 0:
        return 0.0, 0.0
    need = n + round(max(offset_s, 0.0) * MATCH_RATE + MATCH_RATE * MAX_OFFSET_MS / 1000.0)
    if len(final) < need:
        final = np.pad(final, (0, need - len(final)))
    lags = len(final) - n + 1
    size = 1 << int(len(final) + n - 1).bit_length()
    spec = np.fft.rfft(final, size) * np.conj(np.fft.rfft(ref, size))
    corr = np.fft.irfft(spec, size)[:lags]
    csum = np.concatenate(([0.0], np.cumsum(final * final)))
    window_energy = csum[n : n + lags] - csum[:lags]
    denom = np.sqrt(np.maximum(window_energy, 0.0) * ref_energy)
    floor = ref_energy * 1e-12  # windows of digital silence correlate with nothing
    ncc = np.where(denom > math.sqrt(floor), corr / np.maximum(denom, 1e-30), 0.0)
    best = int(np.argmax(ncc))
    return float(ncc[best]), best * 1000.0 / MATCH_RATE


# --- hard: line survived ------------------------------------------------------------------------


_LINE_THRESHOLD = f"STT of final audio == line, word for word ({STT_ATTEMPTS} STT attempts max)"


def _line_failed(error: str) -> tuple[FinalCheck, str]:
    value: dict[str, Any] = {"attempts": 0, "diffs": [], "error": error}
    check = FinalCheck(name="line", hard=True, passed=False, value=value, threshold=_LINE_THRESHOLD)
    return check, ""


def _line_check(
    ctx: Context, final: Path, line: str, lang: object, has_audio: bool
) -> tuple[FinalCheck, str]:
    """STT of the extracted final audio vs the line. A raise or a mishear each use one of the
    STT_ATTEMPTS; the check passes only if the last attempt heard the line exactly."""
    if lang not in LANGS:
        return _line_failed(f"lang {lang!r} is missing or unknown (expected one of {LANGS})")
    run_lang: Lang = "en" if lang == "en" else "uk"  # narrowed: lang is one of LANGS
    if not has_audio:
        return _line_failed("no audio stream")
    extracted = ctx.run_dir / EXTRACTED
    try:
        _tool(
            ["ffmpeg", "-y", "-v", "error", "-i", str(final), "-map", "0:a:0",
             "-ac", "1", "-ar", str(MATCH_RATE), "-c:a", "pcm_s16le", str(extracted)],
            f"audio extraction from {final.name}",
        )  # fmt: skip
    except _ToolError as err:
        return _line_failed(str(err))
    diffs: list[list[str]] = []
    errors: list[str] = []
    transcript = ""
    passed = False
    for _ in range(STT_ATTEMPTS):
        try:
            heard = ctx.providers.stt.transcribe(extracted, run_lang)
        except ProviderError as err:  # counts as a failed attempt
            errors.append(f"{type(err).__name__}: {err}")
            passed = False
            continue
        transcript = heard
        diff = diff_words(line, heard, run_lang)
        diffs.append(diff)
        passed = not diff
        if passed:
            break
    value: dict[str, Any] = {"attempts": len(diffs) + len(errors), "diffs": diffs}
    if errors:
        value["errors"] = errors
        if not passed:
            value["error"] = errors[-1]
    check = FinalCheck(
        name="line", hard=True, passed=passed, value=value, threshold=_LINE_THRESHOLD
    )
    return check, transcript
