"""Gemini TTS via the Interactions API (`client.interactions.create`, google-genai >= 2.25).

`synthesize` sends the exact line as the only text, asks for audio and writes the returned WAV
(base64 in `output_audio.data`, 24 kHz mono 16-bit) to `out_path`. Nothing is written unless the
bytes are a readable WAV with at least one frame.

Ukrainian is NOT in Gemini TTS's official language list. `lang` is therefore not sent
(`speech_config.language` is left unset, the model reads the language from the text); whether a
Ukrainian line is spoken correctly is checked downstream by the `voice` stage's STT diff, not
assumed here."""

from __future__ import annotations

import base64
import binascii
import io
import wave
from pathlib import Path
from typing import Any

import httpx
from google.genai._gaos.lib.compat_errors import APIError

from hwva import voices
from hwva.providers.base import ProviderError
from hwva.providers.gemini import (
    TIMEOUT_S,
    _Interactions,
    check_completed,
    interactions_from_env,
    is_safety,
    map_error,
)
from hwva.schemas import Lang, ModelRef, VoicePersona

VOICE = "Kore"  # only when no persona is given (callers from before personas)
# Delivery only. The text annotated with it is the line itself, nothing else is sent to speak.
# Default style: no persona, or a config/voices.yaml entry with `style: null`.
STYLE = "Say this naturally, in a calm conversational tone, exactly as written."


class GeminiTts:
    """`TtsProvider`. The model code sent to the API is `ref.version` (the pinned exact code).
    Persona -> voice + style from config/voices.yaml under this provider's name (`gemini-tts`);
    the style goes only into `speech_metadata.style`."""

    def __init__(self, ref: ModelRef, interactions: _Interactions | None = None) -> None:
        self.model = ref.version
        self.provider = ref.provider
        self._interactions = interactions or interactions_from_env()

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: voices.Voice | None = None,
    ) -> Path:
        out_path.unlink(missing_ok=True)  # a stale file must never pass for this call's audio
        # pinned mapping from the stage -> used as is, config/voices.yaml is not read again
        name, style = (voice.voice, voice.style or STYLE) if voice else self._voice(persona)
        try:
            interaction = self._interactions.create(
                model=self.model,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                                "annotations": [{"type": "speech_metadata", "style": style}],
                            }
                        ],
                    }
                ],
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": name}]},
                timeout=TIMEOUT_S,
            )
        except (APIError, httpx.HTTPError) as exc:  # anything else is a bug, not a network issue
            raise map_error(exc) from exc
        out_path.write_bytes(_wav_bytes(interaction))
        return out_path

    def _voice(self, persona: VoicePersona | None) -> tuple[str, str]:
        if persona is None:
            return VOICE, STYLE
        try:
            chosen = voices.resolve(self.provider, persona)
        except ValueError as exc:
            raise ProviderError(str(exc), retryable=False) from exc
        if chosen is None:
            raise ProviderError(
                f"No voices for {self.provider!r} in {voices.voices_path()}", retryable=False
            )
        return chosen.voice, chosen.style or STYLE


def _wav_bytes(interaction: Any) -> bytes:
    detail = check_completed(interaction)
    audio = getattr(interaction, "output_audio", None)
    data = getattr(audio, "data", None)
    if not data:
        message = f"Gemini TTS returned no audio{': ' + detail if detail else ''}"
        raise ProviderError(message, retryable=not is_safety(detail))
    mime = getattr(audio, "mime_type", None)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProviderError(f"Gemini TTS audio is not valid base64: {exc}") from exc
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ProviderError(f"Gemini TTS audio is not WAV (mime_type={mime}, {len(raw)} bytes)")
    try:
        with wave.open(io.BytesIO(raw)) as wav:
            frames = wav.getnframes()
            expected = frames * wav.getnchannels() * wav.getsampwidth()
            got = len(wav.readframes(frames))
    except (wave.Error, EOFError) as exc:
        raise ProviderError(f"Gemini TTS returned an unreadable WAV: {exc}") from exc
    if frames == 0:
        raise ProviderError("Gemini TTS returned an empty WAV (0 frames)")
    if got < expected:  # header promises more than arrived: the end of the line would be lost
        raise ProviderError(f"Gemini TTS WAV is truncated ({got} of {expected} data bytes)")
    return raw
