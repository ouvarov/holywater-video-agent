"""Profile -> provider instances. The only place that knows concrete provider classes."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hwva.providers import (
    elevenlabs_tts,
    fakes,
    gemini,
    gemini_judge,
    gemini_tts,
    litterbox,
    openrouter_image,
    openrouter_video,
    scribe,
    seedance,
    uguu,
)
from hwva.providers.base import (
    ImageProvider,
    JudgeProvider,
    LlmProvider,
    MediaHost,
    SttProvider,
    TtsProvider,
    VideoProvider,
)
from hwva.schemas import ModelRef

REQUIRED_ROLES = ("llm", "tts", "stt", "video")
# image: required only when the video model needs an image (checked in `build`);
# media_host: required by video providers that take media only as URLs (openrouter-video)
OPTIONAL_ROLES = ("judge", "image", "media_host")
OPTIONS = ("max_cost_usd",)  # profile settings that are not providers

MAX_COST_ENV = "HWVA_MAX_COST_USD"
DEFAULT_MAX_COST_USD = 3.0


def profiles_path() -> Path:
    """HWVA_CONFIG, else ./config/profiles.yaml (run from the repo root, as README says)."""
    return Path(os.environ.get("HWVA_CONFIG", "config/profiles.yaml"))


@dataclass(frozen=True)
class Providers:
    llm: LlmProvider
    tts: TtsProvider
    stt: SttProvider
    video: VideoProvider
    refs: dict[str, ModelRef]
    judge: JudgeProvider | None = None
    image: ImageProvider | None = None
    media_host: MediaHost | None = None


# provider name -> factory(ModelRef). Real providers are added here by /new-provider.
Factory = Callable[[ModelRef], Any]
_FACTORIES: dict[str, dict[str, Factory]] = {
    "llm": {"fake": lambda ref: fakes.FakeLlm(), "gemini": gemini.GeminiLlm},
    "tts": {
        "fake": lambda ref: fakes.FakeTts(),
        "gemini-tts": gemini_tts.GeminiTts,
        "elevenlabs-tts": elevenlabs_tts.ElevenLabsTts,
    },
    "stt": {"fake": lambda ref: fakes.FakeStt(), "scribe": scribe.ScribeStt},
    "video": {
        "fake": lambda ref: fakes.FakeVideo(),
        "seedance": seedance.SeedanceVideo,
        "openrouter-video": openrouter_video.OpenRouterVideo,
    },
    "judge": {"gemini": gemini_judge.GeminiJudge},
    "image": {
        "fake": lambda ref: fakes.FakeImage(),
        "openrouter-image": openrouter_image.OpenRouterImage,
    },
    "media_host": {
        "fake": lambda ref: fakes.FakeMediaHost(),
        "litterbox": lambda ref: litterbox.Litterbox(),
        "uguu": lambda ref: uguu.Uguu(),
    },
}
# video providers built with the profile's media host (`media_host=`)
_TAKES_MEDIA_HOST = {"openrouter-video"}


def _load(name: str, path: Path | None) -> dict[str, Any]:
    path = path or profiles_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run from the repo root or set HWVA_CONFIG")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in data:
        raise KeyError(f"Unknown profile {name!r}. Available: {', '.join(data)}")
    profile: dict[str, Any] = data[name]
    return profile


def load_profile(name: str, path: Path | None = None) -> dict[str, ModelRef]:
    """Role -> pinned model. `media_host: litterbox` is short for a ref whose provider, model and
    version are all "litterbox". Options (`max_cost_usd`) are not roles: see their loaders."""
    profile = _load(name, path)
    refs: dict[str, ModelRef] = {}
    for role, ref in profile.items():
        if role in OPTIONS:
            continue
        if role == "media_host" and isinstance(ref, str):
            ref = {"provider": ref, "model": ref, "version": ref}
        refs[role] = ModelRef(**ref)
    missing = [r for r in REQUIRED_ROLES if r not in refs]
    unknown = [r for r in refs if r not in REQUIRED_ROLES + OPTIONAL_ROLES]
    if missing or unknown:
        raise ValueError(f"Profile {name!r}: missing roles {missing}, unknown roles {unknown}")
    return refs


def load_max_cost_usd(profile: str, path: Path | None = None) -> float:
    """Money cap per run: env HWVA_MAX_COST_USD, else the profile's `max_cost_usd`, else 3.0.
    Read at every start and resume (not pinned): raise it to continue a run that hit it."""
    env = os.environ.get(MAX_COST_ENV)
    if env:
        return _positive(env, MAX_COST_ENV)
    value = _load(profile, path).get("max_cost_usd")
    if value is None:
        return DEFAULT_MAX_COST_USD
    return _positive(value, f"max_cost_usd of profile {profile!r} (or set {MAX_COST_ENV})")


def _positive(value: object, what: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{what} must be a positive number of USD, got {str(value)[:20]!r}")
    return number


def build(profile: str, refs: dict[str, ModelRef] | None = None) -> Providers:
    """Build providers. On resume pass the refs pinned in the manifest, not the current YAML."""
    refs = refs or load_profile(profile)
    instances: dict[str, Any] = {}
    # the media host first: video providers that take media only as URLs get it
    for role in sorted(refs, key=lambda r: r != "media_host"):
        ref = refs[role]
        factory = _FACTORIES[role].get(ref.provider)
        if factory is None:
            raise NotImplementedError(
                f"Provider {ref.provider!r} for role {role!r} is not implemented yet. "
                f"Use --profile dev, or add it with /new-provider."
            )
        if role == "video" and ref.provider in _TAKES_MEDIA_HOST:
            host = instances.get("media_host")
            if host is None:
                raise ValueError(
                    f"Profile {profile!r}: video provider {ref.provider!r} takes media only as "
                    f"public URLs; add a media_host (e.g. `media_host: litterbox`)"
                )
            instances[role] = factory(ref, media_host=host)  # type: ignore[call-arg]
        else:
            instances[role] = factory(ref)
    video: VideoProvider = instances["video"]
    if video.capabilities.needs_image and "image" not in instances:
        raise ValueError(
            f"Profile {profile!r}: video model {refs['video'].model!r} needs a reference image "
            f"with audio; add an `image:` role (e.g. openrouter-image)"
        )
    return Providers(refs=refs, **instances)
