"""ElevenLabs Scribe STT over plain HTTP (no SDK). Facts verified by a live call:
POST /v1/speech-to-text, `xi-api-key` header, multipart `model_id` / `language_code` / `file`,
transcript in the JSON field `text`. WAV and MP3 both accepted.

The transcript is returned untouched: punctuation differences ("." vs "!", "?" vs "—") are the
comparison's job (`hwva.text.diff_words`), not the adapter's."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from hwva.providers.base import ProviderError
from hwva.schemas import Lang, ModelRef

API_KEY_ENV = "ELEVENLABS_API_KEY"
URL = "https://api.elevenlabs.io/v1/speech-to-text"
TIMEOUT_S = 60.0
_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
_RETRYABLE_STATUS = {408, 429}  # plus every 5xx


class ScribeStt:
    """`SttProvider`. The model id sent to the API is `ref.version` (the pinned exact code).

    `transport` exists for tests (httpx.MockTransport); production uses the default one."""

    def __init__(self, ref: ModelRef, *, transport: httpx.BaseTransport | None = None) -> None:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"{API_KEY_ENV} is not set (put it in .env, see .env.example)")
        self.model = ref.version
        self._key = key
        self._client = httpx.Client(transport=transport, timeout=TIMEOUT_S)

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        audio = audio_path.read_bytes()
        mime = _MIME.get(audio_path.suffix.lower(), "application/octet-stream")
        try:
            response = self._client.post(
                URL,
                headers={"xi-api-key": self._key},
                data={"model_id": self.model, "language_code": lang},
                files={"file": (audio_path.name, audio, mime)},
            )
        except httpx.RequestError as exc:  # timeouts, connection errors
            raise ProviderError(f"Scribe request failed: {type(exc).__name__}: {exc}") from exc

        if response.status_code != 200:
            status = response.status_code
            retryable = status in _RETRYABLE_STATUS or status >= 500
            raise ProviderError(f"Scribe HTTP {status}: {response.text[:300]}", retryable=retryable)

        try:
            text = response.json()["text"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(f"Scribe returned no transcript: {response.text[:300]}") from exc
        if not isinstance(text, str):
            raise ProviderError(f"Scribe 'text' is not a string: {text!r:.300}")
        return text
