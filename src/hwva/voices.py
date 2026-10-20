"""Voice persona -> concrete TTS voice, from `config/voices.yaml` (specs/voice.md, voice persona).

Layout: TTS provider name -> persona key `<gender>-<age>` -> `{voice: str, style: str | null}`.
The style is a delivery hint for the provider's style field — never spoken text.

Shared by the `voice` stage (which records the choice and any fallback) and the TTS adapters
(which turn the resolved persona into request fields), so both always agree on the key.
Fallback when a persona has no entry: same gender `adult`, then the first entry.

The stage resolves once per run and pins the result (specs/voice.md, pinned mapping); adapters
receive that `Voice` via `synthesize(..., voice=)` and then don't read this file at all."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from hwva.schemas import VoicePersona

VOICES_FILE = "voices.yaml"


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: str
    style: str | None = None


@dataclass(frozen=True)
class Voice:
    key: str  # the voices.yaml key used (differs from the persona's key after a fallback)
    voice: str  # provider voice name
    style: str | None  # provider style field; None -> the provider's default

    @property
    def persona(self) -> VoicePersona:
        """The persona this entry is configured for (keys are validated on load)."""
        return persona_of(self.key)


def voices_path() -> Path:
    """Next to HWVA_CONFIG (profiles.yaml) if set, else ./config/voices.yaml (repo root)."""
    config = os.environ.get("HWVA_CONFIG")
    if config:
        return Path(config).parent / VOICES_FILE
    return Path("config") / VOICES_FILE


def persona_key(persona: VoicePersona) -> str:
    return f"{persona.gender}-{persona.age}"


def persona_of(key: str) -> VoicePersona:
    gender, _, age = key.partition("-")
    return VoicePersona.model_validate({"gender": gender, "age": age})


def load(provider: str, path: Path | None = None) -> dict[str, Voice]:
    """Entries for one TTS provider, in file order. Missing provider -> {}.
    A missing file, unparseable YAML or a malformed entry (bad key, no voice) raises ValueError
    naming the file — never silently "no voices"."""
    path = path or voices_path()
    if not path.is_file():
        raise ValueError(f"{path}: file not found")
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of TTS provider -> personas")
    section: Any = data.get(provider) or {}
    if not isinstance(section, dict):
        raise ValueError(f"{path}: {provider!r} must map persona keys to voices")
    voices: dict[str, Voice] = {}
    for key, raw in section.items():
        try:
            persona_of(str(key))
            entry = _Entry.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: bad entry {provider}.{key}: {exc}") from exc
        voices[str(key)] = Voice(key=str(key), voice=entry.voice, style=entry.style)
    return voices


def resolve(provider: str, persona: VoicePersona, path: Path | None = None) -> Voice | None:
    """The voice for `persona`: exact key, else same gender adult, else the first entry.
    None when the provider has no entries at all."""
    voices = load(provider, path)
    for key in (persona_key(persona), f"{persona.gender}-adult"):
        if key in voices:
            return voices[key]
    return next(iter(voices.values()), None)
