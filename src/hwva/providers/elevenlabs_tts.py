"""ElevenLabs TTS over plain HTTP (no SDK), same key / HTTP conventions as `scribe.py`.
Facts verified by a manual test (Scribe heard every file word for word, uk and en):
POST /v1/text-to-speech/{voice_id}?output_format=..., `xi-api-key` header, JSON body
`{text, model_id, language_code}`. Free tier: only premade voices work via the API — a library
voice answers 402 `paid_plan_required`. There is no style field: voices.yaml style is null here.

`synthesize` sends the exact line as `text`, asks for raw 24 kHz 16-bit mono PCM and wraps it into
a WAV. Nothing is written unless the result is a readable WAV with at least one frame."""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from urllib.parse import quote

import httpx

from hwva import voices
from hwva.providers.base import ProviderError
from hwva.schemas import Lang, ModelRef, VoicePersona

API_KEY_ENV = "ELEVENLABS_API_KEY"
URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
OUTPUT_FORMAT = "pcm_24000"  # raw signed 16-bit little-endian mono PCM, no header
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
TIMEOUT_S = 60.0
VOICE = "EXAVITQu4vr4xnSDxMaL"  # premade "Sarah"; only when no persona is given
_RETRYABLE_STATUS = {408, 429}  # plus every 5xx


class ElevenLabsTts:
    """`TtsProvider`. The model id sent to the API is `ref.version` (the pinned exact code).
    Persona -> voice id from config/voices.yaml under this provider's name (`elevenlabs-tts`);
    a style, if configured, is ignored (the API has no style field, it is never spoken).

    `transport` exists for tests (httpx.MockTransport); production uses the default one."""

    def __init__(self, ref: ModelRef, *, transport: httpx.BaseTransport | None = None) -> None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
        self.model = ref.version
        self.provider = ref.provider
        self._key = key
        self._client = httpx.Client(transport=transport, timeout=TIMEOUT_S)

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
        voice_id = voice.voice if voice else self._voice(persona)
        try:
            response = self._client.post(
                URL.format(voice_id=quote(voice_id, safe="")),
                params={"output_format": OUTPUT_FORMAT},
                headers={"xi-api-key": self._key},
                json={"text": text, "model_id": self.model, "language_code": lang},
            )
        except httpx.RequestError as exc:  # timeouts, connection errors
            raise ProviderError(
                f"ElevenLabs TTS request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise _http_error(response, voice_id)
        out_path.write_bytes(_wav_bytes(response))
        return out_path

    def _voice(self, persona: VoicePersona | None) -> str:
        if persona is None:
            return VOICE
        try:
            chosen = voices.resolve(self.provider, persona)
        except ValueError as exc:
            raise ProviderError(str(exc), retryable=False) from exc
        if chosen is None:
            raise ProviderError(
                f"No voices for {self.provider!r} in {voices.voices_path()}", retryable=False
            )
        return chosen.voice


def _http_error(response: httpx.Response, voice_id: str) -> ProviderError:
    status = response.status_code
    retryable = status in _RETRYABLE_STATUS or status >= 500
    message = f"ElevenLabs TTS HTTP {status}: {response.text[:300]}"
    if status == 402 and "paid_plan_required" in response.text:
        message = (
            f"ElevenLabs TTS HTTP 402 paid_plan_required: voice {voice_id!r} needs a paid plan "
            f"(the free tier allows premade voices only): {response.text[:300]}"
        )
    return ProviderError(message, retryable=retryable)


def _wav_bytes(response: httpx.Response) -> bytes:
    pcm = response.content
    content_type = response.headers.get("content-type", "")
    if content_type.startswith(("application/json", "text/")):
        raise ProviderError(f"ElevenLabs TTS returned {content_type}, not audio: {pcm[:300]!r}")
    if not pcm:
        raise ProviderError("ElevenLabs TTS returned no audio (0 bytes)")
    if len(pcm) % SAMPLE_WIDTH:  # half a sample: the stream was cut, the line's end may be lost
        raise ProviderError(f"ElevenLabs TTS PCM is truncated ({len(pcm)} bytes, odd length)")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    raw = buf.getvalue()
    with wave.open(io.BytesIO(raw)) as check:  # same rule as gemini_tts: readable, >= 1 frame
        if check.getnframes() == 0 or len(check.readframes(check.getnframes())) != len(pcm):
            raise ProviderError("ElevenLabs TTS produced an unreadable WAV")
    return raw
