"""voice — TTS of the exact line, duration limits, STT verbatim gate with retries.

The stage owns its retries: a mismatching transcript re-synthesizes the same line (up to
`MAX_ATTEMPTS`); a retryable provider error (429 / 5xx / timeout) from TTS or STT is one of
those attempts too, after a backoff. A duration limit failure is final — the line itself doesn't
fit, a retry won't help — and so is a non-retryable provider error.

Contract: specs/voice.md."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

from hwva import manifest as mf
from hwva import voices
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import Lang, Script, VoiceChoice, VoiceData, VoicePersona
from hwva.text import diff_words

AUDIO = "line.wav"
TRANSCRIPT = "transcript.txt"
SHOT_AUDIO = "shot_audio.wav"  # lead-in silence + line.wav: the audio sent to the video model
LEAD_IN_ENV = "HWVA_LEAD_IN_S"
LEAD_IN_DEFAULT_S = 0.7
LEAD_IN_MAX_S = 2.0
FFMPEG_TIMEOUT_S = 120
SCRIPT = "script.json"
MAX_ATTEMPTS = 4  # TTS->STT attempts in total: mismatches and retryable provider errors alike
MAX_TOTAL_S = 30.0
BACKOFF_S = 2.0  # first wait after a retryable provider error; doubles per error in a row
BACKOFF_MAX_S = 30.0
PROBE_TIMEOUT_S = 30


def run(ctx: Context) -> StageOutput:
    line: str = ctx.manifest.params["line"]
    lang = _as_lang(ctx.manifest.params.get("lang"))
    script = Script.model_validate_json((ctx.run_dir / SCRIPT).read_text(encoding="utf-8"))
    caps = ctx.providers.video.capabilities
    lead_in_s = _lead_in()  # before any TTS call: a bad value must not cost a synthesis
    # Resolved once, before any TTS call (no voices -> GateFailed), or taken from the pin of an
    # earlier run of this stage. Written to the stage record now, so every exit — success,
    # GateFailed, provider error — leaves the pin in the manifest.
    pinned = _choose_voice(ctx, script.voice)
    record = mf.stage(ctx.manifest, "voice")
    record.data["voice"] = pinned.model_dump(mode="json")
    choice = voices.Voice(key=pinned.key, voice=pinned.voice, style=pinned.style)
    persona = choice.persona  # resolved: the persona of the voices.yaml key actually used

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    audio_path = ctx.run_dir / AUDIO
    diffs: list[list[str]] = []
    last_error: ProviderError | None = None  # the last attempt's failure, if it was a provider's
    errors_in_row = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if errors_in_row:
            time.sleep(_backoff(errors_in_row))
        try:
            # exactly the line — no prompt wrapping that could end up spoken
            # the pinned mapping goes along, so the adapter never re-reads config/voices.yaml
            produced = ctx.providers.tts.synthesize(line, lang, audio_path, persona, voice=choice)
            audio_s = _duration(produced)
            planned_total = _check_durations(
                script,
                lead_in_s + audio_s,
                min_audio_s=caps.min_audio_s,
                max_shot_s=caps.max_duration_s,
            )
            transcript = ctx.providers.stt.transcribe(produced, lang)
        except ProviderError as exc:
            if not exc.retryable:
                raise
            last_error = exc  # 429 / 5xx / timeout: counts as an attempt, retried with backoff
            errors_in_row += 1
            continue
        last_error, errors_in_row = None, 0
        diff = diff_words(line, transcript, lang)
        if not diff:
            (ctx.run_dir / TRANSCRIPT).write_text(transcript, encoding="utf-8")
            shot_audio = _make_shot_audio(produced, ctx.run_dir / SHOT_AUDIO, lead_in_s)
            data = VoiceData(
                audio_duration_s=audio_s,
                planned_total_s=planned_total,
                attempts=attempt,
                diffs=diffs,
                voice=pinned,
                lead_in_s=lead_in_s,
                shot_audio_duration_s=_duration(shot_audio),
            )
            return StageOutput(
                artifacts={"audio": AUDIO, "transcript": TRANSCRIPT, "shot_audio": SHOT_AUDIO},
                data=data.model_dump(mode="json"),
            )
        diffs.append(diff)

    if last_error is not None:
        if not diffs:
            raise last_error
        raise ProviderError(  # keep the earlier mismatches visible in the manifest error
            f"{last_error} (after {MAX_ATTEMPTS} attempts; last STT diff: {'; '.join(diffs[-1])})",
            retryable=last_error.retryable,
        ) from last_error
    raise GateFailed(
        f"STT does not match the line after {MAX_ATTEMPTS} attempts; last diff: "
        + "; ".join(diffs[-1])
    )


def _choose_voice(ctx: Context, persona: VoicePersona) -> VoiceChoice:
    """The pin from an earlier run of this stage (success or failed gate), else Script.voice ->
    the config/voices.yaml entry of this run's TTS provider; a fallback is an assumption the
    agent made by itself."""
    record = ctx.manifest.stages.get("voice")
    if record is not None and record.data.get("voice"):
        return VoiceChoice.model_validate(record.data["voice"])  # pinned: config not read again
    provider = ctx.providers.refs["tts"].provider
    try:
        choice = voices.resolve(provider, persona)
    except ValueError as exc:  # missing file / bad YAML / bad entry — the message names the file
        raise GateFailed(f"Invalid voice config: {exc}") from exc
    if choice is None:
        raise GateFailed(f"No voices for TTS provider {provider!r} in {voices.voices_path()}")
    wanted = voices.persona_key(persona)
    if choice.key != wanted:
        note = f"no {wanted} voice for {provider}, used {choice.key}"
        if note not in ctx.manifest.assumptions:
            ctx.manifest.assumptions.append(note)
    return VoiceChoice(persona=persona, key=choice.key, voice=choice.voice, style=choice.style)


def _as_lang(value: str | None) -> Lang:
    return "en" if value == "en" else "uk"


def _lead_in() -> float:
    """`HWVA_LEAD_IN_S` (0 <= x <= 2), default 0.7; anything else -> GateFailed naming it."""
    raw = os.environ.get(LEAD_IN_ENV)
    if raw is None or not raw.strip():
        return LEAD_IN_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0 <= value <= LEAD_IN_MAX_S:  # nan fails too
        raise GateFailed(f"{LEAD_IN_ENV}={raw!r}: expected seconds between 0 and {LEAD_IN_MAX_S:g}")
    return value


def _make_shot_audio(line_wav: Path, out: Path, lead_in_s: float) -> Path:
    """`lead_in_s` of digital silence + line.wav, mono, the sample rate of line.wav."""
    delay_ms = round(lead_in_s * 1000)
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(line_wav),
           "-af", f"adelay=delays={delay_ms}:all=1",
           "-ac", "1", "-c:a", "pcm_s16le", str(out)]  # fmt: skip
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as err:
        raise GateFailed(f"ffmpeg {out.name} timed out after {FFMPEG_TIMEOUT_S} s") from err
    except subprocess.CalledProcessError as err:
        raise GateFailed(f"ffmpeg {out.name} failed: {(err.stderr or '').strip()}") from err
    except OSError as err:  # ffmpeg missing from PATH
        raise GateFailed(f"ffmpeg could not be started for {out.name}: {err}") from err
    return out


def _check_durations(
    script: Script, sent_s: float, *, min_audio_s: float, max_shot_s: float
) -> float:
    """`sent_s` = lead-in + line: the audio sent to the video model. Returns the planned total
    video length; raises GateFailed naming the limit and the value."""
    if sent_s < min_audio_s:
        raise GateFailed(
            f"Audio {sent_s:.2f} s (lead-in + line) is shorter than the video model minimum "
            f"{min_audio_s:g} s"
        )
    speaking_s = max(script.shots[0].duration_s, sent_s)
    if speaking_s > max_shot_s:
        raise GateFailed(
            f"Speaking shot {speaking_s:.2f} s exceeds the video model maximum {max_shot_s:g} s"
        )
    total = speaking_s + sum(shot.duration_s for shot in script.shots[1:])
    if total > MAX_TOTAL_S:
        raise GateFailed(f"Planned video {total:.2f} s exceeds the {MAX_TOTAL_S:g} s limit")
    return total


def _backoff(errors_in_row: int) -> float:
    """Wait after `errors_in_row` provider errors in a row: 2, 4, 8 s, capped."""
    return float(min(BACKOFF_MAX_S, BACKOFF_S * 2 ** (errors_in_row - 1)))


def _duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
             str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=PROBE_TIMEOUT_S,
        )  # fmt: skip
    except subprocess.TimeoutExpired as err:
        raise GateFailed(f"ffprobe of {path.name} timed out after {PROBE_TIMEOUT_S} s") from err
    except subprocess.CalledProcessError as err:
        raise GateFailed(f"ffprobe of {path.name} failed: {(err.stderr or '').strip()}") from err
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as err:
        raise GateFailed(f"ffprobe of {path.name}: no readable duration") from err
