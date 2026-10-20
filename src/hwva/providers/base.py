"""Provider contracts. A provider declares what it can do; the pipeline picks a strategy from that.

Strategies (see README):
  A — video follows OUR audio (accepts_audio_input)        <- implemented in the PoC
  B — model speaks itself, STT verifies (native_speech)    <- described in the note
  C — silent video + separate lip-sync                     <- described in the note
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from hwva.schemas import JudgeVerdict, Lang, Script, VoicePersona
from hwva.voices import Voice


@dataclass(frozen=True)
class Capabilities:
    accepts_audio_input: bool = False
    native_speech: bool = False
    needs_image: bool = False  # e.g. Seedance 2.0 needs an image when audio is given
    max_duration_s: float = 10.0
    min_audio_s: float = 0.0
    aspect_ratios: tuple[str, ...] = ("9:16",)


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    duration_s: float
    aspect_ratio: str = "9:16"
    audio_path: Path | None = None  # strategy A: lips follow this audio
    reference_image: str | None = None  # path or URL of a frame for character consistency
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    last_frame: str | None = None  # path or URL, used as reference for the next shot
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ImageResult:
    path: Path
    cost_usd: float = 0.0


@dataclass(frozen=True)
class HostedMedia:
    url: str  # public https URL
    expires_at: datetime | None = None  # UTC; None = the host gives no expiry


@runtime_checkable
class LlmProvider(Protocol):
    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script: ...


@runtime_checkable
class TtsProvider(Protocol):
    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona,
        voice: Voice | None = None,
    ) -> Path:
        """Speak exactly `text`. `voice` is the mapping the stage resolved and pinned for this
        run (key/voice/style); when given, the adapter uses it and does NOT read
        config/voices.yaml. Without it, `persona` (Script.voice) picks the voice/style via
        config/voices.yaml. A style is a provider field, never part of the spoken text."""
        ...


@runtime_checkable
class SttProvider(Protocol):
    def transcribe(self, audio_path: Path, lang: Lang) -> str: ...


@runtime_checkable
class JudgeProvider(Protocol):
    """Soft quality signals for the final video. Never a gate (video-LLMs are insensitive to
    temporal errors, can't judge lip-sync). Raises / returns garbage -> quality_check records it."""

    def judge(self, video_path: Path, script: Script) -> JudgeVerdict: ...


@runtime_checkable
class VideoProvider(Protocol):
    capabilities: Capabilities

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        """Submit, poll until done, download to out_path. Raises ProviderError on failure."""
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """A still image from a text prompt (the character reference for video models that need an
    image with audio). Writes the image to `out_path` (PNG) and returns it with its price."""

    def generate(self, prompt: str, out_path: Path) -> ImageResult: ...


@runtime_checkable
class MediaHost(Protocol):
    """Puts a local file at a public https URL for APIs that only take URLs (OpenRouter rejects
    data URIs). The URL is a capability: anyone who has it can fetch the file until it expires.
    Free to call (no money is spent), so a retry may upload again."""

    def upload(self, path: Path) -> HostedMedia: ...


class ProviderError(RuntimeError):
    """A provider failed. `retryable` tells the pipeline whether a retry makes sense."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable
