"""Free, offline providers for the `dev` profile and tests.

They produce real media files via ffmpeg, so assemble/final_check are exercised for real,
but no API is called."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hwva.providers.base import (
    Capabilities,
    HostedMedia,
    ImageResult,
    VideoRequest,
    VideoResult,
)
from hwva.schemas import Age, Gender, Lang, LineSource, Script, Shot, VoicePersona
from hwva.text import detect_lang, extract_quoted_line, line_gender
from hwva.voices import Voice

SECONDS_PER_WORD = 0.4
LAST_SPOKEN = "fake_tts_last.txt"  # what FakeTts said last in this dir


BASE_HZ: dict[Gender, int] = {"male": 140, "female": 220}  # adult
AGE_SHIFT_HZ: dict[Age, int] = {"adult": 0, "teen": 60, "child": 160, "elderly": -20}
NO_PERSONA_HZ = 440  # callers that pass no persona


def pitch_hz(persona: VoicePersona | None) -> int:
    if persona is None:
        return NO_PERSONA_HZ
    return BASE_HZ[persona.gender] + AGE_SHIFT_HZ[persona.age]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _sidecar(audio: Path) -> Path:
    return audio.with_name(audio.name + ".txt")  # line.wav -> line.wav.txt (no clash with .mp3)


class FakeLlm:
    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        quoted = extract_quoted_line(idea)
        if line is not None:
            final_line, source = line, LineSource.INPUT
        elif quoted is not None:
            final_line, source = quoted, LineSource.INPUT
        else:
            final_line, source = "Привіт, це тестова репліка", LineSource.GENERATED
        final_lang = lang or detect_lang(final_line)
        gender = line_gender(final_line, final_lang) or "male"
        return Script(
            title=idea[:40],
            line=final_line,
            line_source=source,
            lang=final_lang,
            character="a ginger cat with green eyes",
            shots=[
                Shot(visual_prompt=f"{idea} — speaking", duration_s=5, has_line=True),
                Shot(visual_prompt=f"{idea} — reaction", duration_s=4),
            ],
            voice=VoicePersona(gender=gender, age="adult"),
        )


class FakeTts:
    """Sine tone whose length depends on word count. Records the text next to the audio so FakeStt
    can "hear" it back. The pitch follows the persona so tests can tell voices apart."""

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: Voice | None = None,
    ) -> Path:
        duration = max(2.0, len(text.split()) * SECONDS_PER_WORD)
        # a pinned mapping wins: pitch follows the voices.yaml key the stage resolved
        hz = pitch_hz(voice.persona if voice is not None else persona)
        _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={hz}:duration={duration}", str(out_path))
        _sidecar(out_path).write_text(text, encoding="utf-8")
        (out_path.parent / LAST_SPOKEN).write_text(text, encoding="utf-8")
        return out_path


class FakeStt:
    """Returns what FakeTts said for this file, or — for audio extracted later (e.g. from
    final.mp4) — the last line FakeTts spoke in that directory.

    `mishear` simulates STT/TTS errors: {"головний": "найголовніший"} replaces words."""

    def __init__(self, mishear: dict[str, str] | None = None) -> None:
        self.mishear = mishear or {}

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        for candidate in (_sidecar(audio_path), audio_path.parent / LAST_SPOKEN):
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
                for wrong, heard in self.mishear.items():
                    text = text.replace(wrong, heard)
                return text
        return ""


class FakeImage:
    """Solid-colour 720x1280 PNG (9:16) rendered by ffmpeg; free. Records every prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, out_path: Path) -> ImageResult:
        self.prompts.append(prompt)
        _ffmpeg("-f", "lavfi", "-i", "color=c=orange:s=720x1280", "-frames:v", "1", str(out_path))
        return ImageResult(path=out_path)


class FakeMediaHost:
    """Pretends to host files: a new https URL per upload (as a real host gives), no network.
    Records every uploaded path."""

    def __init__(self) -> None:
        self.uploads: list[Path] = []

    def upload(self, path: Path) -> HostedMedia:
        self.uploads.append(path)
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        return HostedMedia(f"https://media.invalid/{len(self.uploads)}/{path.name}", expires_at)


class FakeVideo:
    capabilities = Capabilities(
        accepts_audio_input=True, max_duration_s=30, min_audio_s=2.0, aspect_ratios=("9:16",)
    )

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        _ffmpeg(
            "-f", "lavfi",
            "-i", f"testsrc=size=720x1280:rate=24:duration={request.duration_s}",
            "-pix_fmt", "yuv420p",
            str(out_path),
        )  # fmt: skip
        frame = out_path.with_suffix(".last.png")
        # decode the tail and keep overwriting one image -> the true last frame
        _ffmpeg("-sseof", "-1", "-i", str(out_path), "-update", "1", "-q:v", "2", str(frame))
        return VideoResult(video_path=out_path, last_frame=str(frame))
