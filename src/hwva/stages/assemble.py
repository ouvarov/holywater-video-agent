"""assemble — concatenate the shots and put OUR verified TTS on top as the only audio.

The shots' own audio (if the video model produced any) is dropped. The audio is shot_1's
`aligned_line.wav` when recorded (lip-sync v2: lead-in and phrase shifts already in the file,
placed at 0), else line.wav at `audio_offset_s` (voice lead-in + shot_1 lip-sync shift, 0 in
older runs); padded with silence to the video length, never time-stretched. Speech (end of the
last non-silent segment) must end within the video: a wanted offset that does not fit is reduced
to fit, an aligned track is moved earlier over its own leading silence (an assumption is
logged); only silence is ever cut.
The render goes to a temp file in the run dir, is re-probed, and only then renamed to final.mp4.

Contract: specs/assemble.md.
Data: `hwva.schemas.AssembleData`; artifacts: {"video": "final.mp4", "audio": <file placed>}."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hwva.context import Context, GateFailed, StageOutput
from hwva.schemas import AssembleData, Script, SyncReport

RENDER_TIMEOUT_S = 120  # ffmpeg renders and analysis
PROBE_TIMEOUT_S = 30  # ffprobe
SILENCE_FILTER = "silencedetect=n=-40dB:d=0.3"
# the first speech of an aligned track: its leading silence may be shorter than 0.3 s
LEADING_FILTER = "silencedetect=n=-40dB:d=0.05"
# A silence_end this close to EOF is the flush newer ffmpeg emits for silence running to EOF.
EOF_EPSILON_S = 1e-3
TIME_DECIMALS = 3  # video_s is rounded to ms once and used unchanged by gate and render
_SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")

FINAL = "final.mp4"
TMP = "final.tmp.mp4"  # suffix must stay .mp4: ffmpeg picks the muxer from it
AUDIO = "line.wav"
ALIGNED_KEY = "aligned_audio"  # shot_1's artifact: the chosen lip-sync track
SHOT_STAGES = ("shot_1", "shot_2")
WIDTH, HEIGHT, FPS = 720, 1280, 24
DURATION_TOLERANCE_S = 0.2
STDERR_TAIL_LINES = 10


def run(ctx: Context) -> StageOutput:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise GateFailed(f"{tool} not found on PATH; install ffmpeg.")
    shots = _shot_paths(ctx)
    line = ctx.run_dir / AUDIO
    if not line.is_file():
        raise GateFailed(f"{AUDIO} not found in the run dir; re-run voice.")
    aligned = _aligned_track(ctx)

    # Rounded once: the gate and the render's apad/atrim use the very same value.
    video_s = round(sum(_video_duration(p) for p in shots), TIME_DECIMALS)
    if aligned is not None:
        audio, wanted_s = aligned, 0.0  # the lead-in and every shift are in the file already
        _, speech_s = _speech_bounds(audio)
        speech_start = _speech_start(audio)
        offset_s, reason = 0.0, ""
        overflow = round(speech_s - video_s, TIME_DECIMALS)
        if overflow > 0:
            if overflow > speech_start + EOF_EPSILON_S:
                raise GateFailed(
                    f"Speech in {audio.name} ends at {speech_s:.3f} s, {overflow:.3f} s after the "
                    f"video end {video_s:.3f} s, but only {speech_start:.3f} s of leading silence "
                    "can be trimmed; the line would be cut. Re-run shot_1."
                )
            offset_s, reason = -overflow, "shot 1 too short"  # trim leading silence only
    else:
        audio = line
        wanted_s = _wanted_offset(ctx)
        speech_start, speech_s = _speech_bounds(audio)
        if speech_s > video_s:
            raise GateFailed(
                f"Speech in {AUDIO} ends at {speech_s:.3f} s, after the video end {video_s:.3f} s "
                "even at audio offset 0; the line would be cut. Re-run shot_1 (shot 1 must "
                "cover the audio)."
            )
        # Never cut the line, never pay again: a wanted offset that does not fit is reduced.
        offset_s = max(0.0, wanted_s)
        reason = "negative shift larger than the lead-in" if wanted_s < 0 else ""
        if offset_s + speech_s > video_s:
            offset_s = max(0.0, round(video_s - speech_s, TIME_DECIMALS))
            reason = "shot 1 too short"
    if reason:
        note = (
            f"Lip-sync audio offset {offset_s:g} s applied instead of the wanted {wanted_s:g} s "
            f"({reason})."
        )
        if note not in ctx.manifest.assumptions:
            ctx.manifest.assumptions.append(note)

    tmp_path = ctx.run_dir / TMP
    out_path = ctx.run_dir / FINAL
    try:
        _render(shots, audio, video_s, offset_s, tmp_path)
        final_s = _verify_output(tmp_path, video_s)
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return StageOutput(
        artifacts={"video": FINAL, "audio": audio.name},
        data=AssembleData(
            duration_s=final_s,
            audio_offset_s=offset_s,
            audio_offset_wanted_s=wanted_s,
            audio_source="aligned" if aligned is not None else "line",
            speech_start_s=round(max(0.0, speech_start + offset_s), TIME_DECIMALS),
        ).model_dump(mode="json"),
        cost_usd=0.0,
    )


def _aligned_track(ctx: Context) -> Path | None:
    """shot_1's recorded `aligned_audio` (lip-sync v2), or None (older runs, verdict `error`, a
    failed render). Recorded but missing on disk -> GateFailed: never silently fall back."""
    record = ctx.manifest.stages.get("shot_1")
    name = record.artifacts.get(ALIGNED_KEY) if record is not None else None
    if not name:
        return None
    path = ctx.run_dir / name
    if not path.is_file():
        raise GateFailed(f"shot_1 aligned audio not found at {path}; re-run shot_1.")
    return path


def _wanted_offset(ctx: Context) -> float:
    """Wanted start of line.wav in the final: voice `lead_in_s` + shot_1 `sync.shift_s`, rounded
    to ms, unclamped. Both are recorded values (never the env); absent in older runs -> 0."""
    voice = ctx.manifest.stages.get("voice")
    shot = ctx.manifest.stages.get("shot_1")
    lead_in = float((voice.data if voice is not None else {}).get("lead_in_s") or 0.0)
    sync = (shot.data if shot is not None else {}).get("sync")
    shift = 0.0
    if sync is not None:
        try:
            shift = SyncReport.model_validate(sync).shift_s
        except ValidationError as err:
            raise GateFailed(f"shot_1 lip-sync record is malformed ({err}); re-run shot_1") from err
    return round(lead_in + shift, TIME_DECIMALS)


def _expected_shots(ctx: Context) -> int:
    """Number of shots the script promises (`len(Script.shots)`), read from the script artifact."""
    record = ctx.manifest.stages.get("script")
    name = record.artifacts.get("script") if record is not None else None
    if name is None:
        raise GateFailed("Script artifact not recorded in the manifest; re-run script.")
    path = ctx.run_dir / name
    if not path.is_file():
        raise GateFailed(f"Script not found at {path}; re-run script.")
    try:
        script = Script.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as err:
        raise GateFailed(f"Script at {path} is malformed ({err}); re-run script.") from err
    n = len(script.shots)
    if not 1 <= n <= len(SHOT_STAGES):
        raise GateFailed(f"Script has {n} shots; only 1..{len(SHOT_STAGES)} are supported.")
    return n


def _shot_paths(ctx: Context) -> list[Path]:
    """Every shot the script promises must exist; a missing shot is never silently skipped."""
    paths: list[Path] = []
    for stage_name in SHOT_STAGES[: _expected_shots(ctx)]:
        record = ctx.manifest.stages.get(stage_name)
        name = record.artifacts.get("video") if record is not None else None
        if name is None:
            raise GateFailed(f"{stage_name} has no recorded video; re-run {stage_name}.")
        path = ctx.run_dir / name
        if not path.is_file():
            raise GateFailed(f"{stage_name} video not found at {path}; re-run {stage_name}.")
        paths.append(path)
    return paths


def _stderr_tail(err: subprocess.CalledProcessError) -> str:
    raw = err.stderr
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(lines[-STDERR_TAIL_LINES:]) or f"exit code {err.returncode}"


def _run_tool(cmd: list[str], what: str, *, stderr: bool = False) -> str:
    """Run ffmpeg/ffprobe with a timeout; a non-zero exit, a timeout or a missing binary becomes
    GateFailed. Returns stdout (or stderr when `stderr=True`)."""
    timeout = PROBE_TIMEOUT_S if cmd[0] == "ffprobe" else RENDER_TIMEOUT_S
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # a non-UTF-8 byte in ffmpeg's log must not crash the stage
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as err:
        raise GateFailed(f"{what} failed: {_stderr_tail(err)}") from err
    except subprocess.TimeoutExpired as err:
        raise GateFailed(f"{what} timed out after {timeout} s.") from err
    except OSError as err:
        raise GateFailed(f"{what} could not start {cmd[0]} ({err}); install ffmpeg.") from err
    return (out.stderr if stderr else out.stdout) or ""


def _speech_bounds(audio: Path) -> tuple[float, float]:
    """(start of the first, end of the last non-silent segment) of `audio` (silencedetect
    n=-40dB d=0.3). No leading silence -> 0; not ending in silence -> the file length."""
    file_s = _format_duration(audio)
    log = _run_tool(
        ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(audio),
         "-af", SILENCE_FILTER, "-f", "null", "-"],
        f"silence analysis of {audio.name}",
        stderr=True,
    )  # fmt: skip
    events = [(kind, float(value)) for kind, value in _SILENCE_RE.findall(log)]
    start_s = 0.0
    leading = len(events) >= 2 and events[0][0] == "start" and events[0][1] <= EOF_EPSILON_S
    if leading and events[1][0] == "end" and abs(events[1][1] - file_s) > EOF_EPSILON_S:
        start_s = events[1][1]  # leading silence ends where the speech starts
    last_start: float | None = None
    last_end: float | None = None
    for kind, value in events:
        if kind == "start":
            last_start, last_end = value, None
        else:
            last_end = value
    if last_start is None:
        return start_s, file_s
    # The file ends in silence only if no silence_end follows the last silence_start, or that
    # silence_end is AT EOF. Anything after it, however short, is speech.
    if last_end is None or abs(last_end - file_s) <= EOF_EPSILON_S:
        return start_s, max(start_s, last_start)
    return start_s, file_s


def _speech_start(audio: Path) -> float:
    """Start of the first speech in `audio` (silencedetect n=-40dB d=0.05): the leading silence
    an aligned track may lose when moved earlier. No leading silence -> 0."""
    file_s = _format_duration(audio)
    log = _run_tool(
        ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(audio),
         "-af", LEADING_FILTER, "-f", "null", "-"],
        f"leading silence of {audio.name}",
        stderr=True,
    )  # fmt: skip
    events = [(kind, float(value)) for kind, value in _SILENCE_RE.findall(log)]
    if len(events) >= 2 and events[0][0] == "start" and events[0][1] <= EOF_EPSILON_S:
        kind, end = events[1]
        if kind == "end" and abs(end - file_s) > EOF_EPSILON_S:
            return end
    return 0.0


def _render(
    shots: list[Path], audio: Path, video_s: float, offset_s: float, out_path: Path
) -> None:
    n = len(shots)
    norm = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p"
    )
    parts = [f"[{i}:v:0]{norm}[v{i}]" for i in range(n)]
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]")
    # Pad to the video length; atrim only ever cuts trailing silence (speech end <= video_s was
    # checked before the render).
    end = f"{video_s:.{TIME_DECIMALS}f}"  # video_s is already rounded: this is the gate's value
    # The lip-sync correction: silence in front of the line (offset_s is rounded to ms); a
    # negative offset (an aligned track moved earlier) trims its leading silence.
    if offset_s > 0:
        delay = f"adelay=delays={round(offset_s * 1000)}:all=1,"
    elif offset_s < 0:
        delay = f"atrim=start={-offset_s:.{TIME_DECIMALS}f},asetpts=PTS-STARTPTS,"
    else:
        delay = ""
    parts.append(f"[{n}:a:0]{delay}apad=whole_dur={end},atrim=end={end}[a]")
    inputs: list[str] = []
    for p in [*shots, audio]:
        inputs += ["-i", str(p)]
    # The output path must stay the last argv item (tests intercept the render by it).
    _run_tool(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-filter_complex", ";".join(parts),
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac",
         "-movflags", "+faststart", str(out_path)],
        "ffmpeg render of final video",
    )  # fmt: skip


def _verify_output(path: Path, video_s: float) -> float:
    """Re-probe the render: exactly 1 video + 1 audio stream, duration = shot total ±0.2 s."""
    data = _probe(path, "stream=codec_type:format=duration")
    kinds = [s.get("codec_type") for s in data.get("streams", [])]
    n_v, n_a = kinds.count("video"), kinds.count("audio")
    if (n_v, n_a) != (1, 1):
        raise GateFailed(
            f"Rendered video has {n_v} video and {n_a} audio streams; expected exactly 1 + 1."
        )
    final_s = _to_seconds(data.get("format", {}).get("duration"), path, "container")
    if abs(final_s - video_s) > DURATION_TOLERANCE_S:
        raise GateFailed(
            f"Rendered video is {final_s:.3f} s, shots total {video_s:.3f} s "
            f"(tolerance {DURATION_TOLERANCE_S} s)."
        )
    return final_s


def _probe(path: Path, entries: str, *select: str) -> dict[str, Any]:
    stdout = _run_tool(
        ["ffprobe", "-v", "error", *select, "-show_entries", entries, "-of", "json", str(path)],
        f"ffprobe of {path.name}",
    )
    try:
        result: dict[str, Any] = json.loads(stdout)
    except json.JSONDecodeError as err:
        raise GateFailed(f"ffprobe of {path.name} returned unparsable output.") from err
    return result


def _to_seconds(value: Any, path: Path, what: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as err:
        raise GateFailed(f"{path.name}: unreadable {what} duration {value!r}.") from err
    if not seconds > 0:
        raise GateFailed(f"{path.name}: {what} duration is {seconds!r}.")
    return seconds


def _format_duration(path: Path) -> float:
    value = _probe(path, "format=duration").get("format", {}).get("duration")
    return _to_seconds(value, path, "container")


def _video_duration(path: Path) -> float:
    """Duration of the first video stream. If the stream reports none ("N/A"), the container
    duration is used instead; a file with no video stream, or no readable duration at all,
    raises GateFailed."""
    streams: list[dict[str, Any]] = _probe(path, "stream=duration", "-select_streams", "v:0").get(
        "streams", []
    )
    if not streams:
        raise GateFailed(f"{path.name} has no video stream; re-run the shot.")
    value = streams[0].get("duration")
    if value in (None, "N/A"):
        return _format_duration(path)
    return _to_seconds(value, path, "video stream")
