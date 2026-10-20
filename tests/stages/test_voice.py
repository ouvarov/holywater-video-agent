"""Acceptance tests for the `voice` stage (specs/voice.md)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from conftest import (
    REPO,
    ScriptedStt,
    SpyTts,
    dominant_hz,
    duration,
    make_audio,
    needs_ffmpeg,
    probe,
)

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import fakes, registry
from hwva.providers.base import Capabilities, ProviderError, VideoRequest, VideoResult
from hwva.schemas import (
    Gender,
    Lang,
    LineSource,
    ModelRef,
    Script,
    Shot,
    StageStatus,
    VoiceChoice,
    VoiceData,
    VoicePersona,
)
from hwva.stages import voice
from hwva.text import diff_words
from hwva.voices import Voice

MakeCtx = Callable[..., Context]

LINE = "Я тут головний"
MISHEARD = "Я тут найголовніший"
MAX_ATTEMPTS = 4
ARTIFACTS = {"audio": "line.wav", "transcript": "transcript.txt"}
# Lead-in for the video model (specs/voice.md)
SHOT_AUDIO = "shot_audio.wav"
LEAD_ARTIFACTS = {**ARTIFACTS, "shot_audio": SHOT_AUDIO}
LEAD_IN_S = 0.7  # default HWVA_LEAD_IN_S


# --- helpers ------------------------------------------------------------------------------------


class NoCallVideo:
    """Video model with configurable limits; voice must never generate video."""

    def __init__(self, max_duration_s: float = 30.0, min_audio_s: float = 2.0) -> None:
        self.capabilities = Capabilities(
            accepts_audio_input=True, max_duration_s=max_duration_s, min_audio_s=min_audio_s
        )

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        raise AssertionError("voice must not call the video model")


class FixedTts:
    """TTS whose audio has a fixed length; FakeStt 'hears' the text via the sidecar."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.texts: list[str] = []

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: Voice | None = None,
    ) -> Path:
        self.texts.append(text)
        return make_audio(out_path, self.seconds, text)


def _shots(*durations: float) -> list[Shot]:
    return [
        Shot(visual_prompt=f"shot {i}", duration_s=d, has_line=i == 0)
        for i, d in enumerate(durations)
    ]


def _ctx(
    make_ctx: MakeCtx,
    line: str = LINE,
    *,
    lang: Lang = "uk",
    shots: list[Shot] | None = None,
    persona: VoicePersona | None = None,
    **overrides: object,
) -> Context:
    """Context as it is after `script`: params hold the line, script.json is on disk."""
    overrides.setdefault("video", NoCallVideo())
    params = {"line": line, "line_source": LineSource.INPUT.value, "lang": lang}
    ctx = make_ctx(idea=f'Кіт каже: "{line}"', params=params, **overrides)
    script = Script(
        title="test",
        line=line,
        line_source=LineSource.INPUT,
        lang=lang,
        character="a ginger cat with green eyes",
        shots=shots or _shots(5, 4),
        voice=persona or VoicePersona(gender="male", age="adult"),
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    record = mf.stage(ctx.manifest, "script")
    record.status, record.artifacts = StageStatus.DONE, {"script": "script.json"}
    return ctx


def _data(out: StageOutput) -> VoiceData:
    assert isinstance(out, StageOutput)
    return VoiceData.model_validate(out.data)


def _voice_stage() -> pipeline.Stage:
    return next(s for s in pipeline.STAGES if s.name == "voice")


# --- AC: TTS receives exactly the line ----------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [LINE, "М'ята — це «життя», друже!", "Hello there, Mr. Cat."],
    ids=["plain", "punctuation_apostrophe_quotes", "english"],
)
def test_tts_receives_exactly_the_line(make_ctx: MakeCtx, no_stdin: None, line: str) -> None:
    tts = SpyTts()
    lang: Lang = "en" if line.startswith("Hello") else "uk"
    ctx = _ctx(make_ctx, line, lang=lang, tts=tts)
    voice.run(ctx)
    assert tts.texts == [line]  # byte-exact, no wrapping, no normalization


def test_every_retry_synthesizes_exactly_the_line(make_ctx: MakeCtx, no_stdin: None) -> None:
    tts = SpyTts()
    ctx = _ctx(make_ctx, tts=tts, stt=ScriptedStt(MISHEARD, MISHEARD, LINE))
    voice.run(ctx)
    assert tts.texts == [LINE, LINE, LINE]


# --- AC: verbatim match passes on attempt 1 -----------------------------------------------------


def test_verbatim_match_passes_on_first_attempt(make_ctx: MakeCtx, no_stdin: None) -> None:
    tts = SpyTts()
    ctx = _ctx(make_ctx, tts=tts)  # FakeStt hears what FakeTts said
    out = voice.run(ctx)
    data = _data(out)
    assert data.attempts == 1
    assert data.diffs == []
    assert len(tts.texts) == 1
    assert out.artifacts == LEAD_ARTIFACTS
    audio = ctx.run_dir / ARTIFACTS["audio"]
    assert audio.exists()
    transcript = (ctx.run_dir / ARTIFACTS["transcript"]).read_text(encoding="utf-8")
    assert diff_words(LINE, transcript) == []


def test_durations_are_reported(make_ctx: MakeCtx, no_stdin: None) -> None:
    # 2.0 s of audio inside a 5 s speaking shot -> planned = 5 + 4
    ctx = _ctx(make_ctx, tts=FixedTts(2.0))
    data = _data(voice.run(ctx))
    assert data.audio_duration_s == pytest.approx(duration(ctx.run_dir / "line.wav"), abs=0.05)
    assert data.audio_duration_s == pytest.approx(2.0, abs=0.05)
    assert data.planned_total_s == pytest.approx(9.0, abs=0.05)


def test_audio_longer_than_speaking_shot_extends_plan(make_ctx: MakeCtx, no_stdin: None) -> None:
    # speaking shot = max(5, 0.7 + 6.5) = 7.2; + 4 = 11.2
    ctx = _ctx(make_ctx, tts=FixedTts(6.5))
    data = _data(voice.run(ctx))
    assert data.audio_duration_s == pytest.approx(6.5, abs=0.05)
    assert data.planned_total_s == pytest.approx(11.2, abs=0.05)


def test_stage_done_in_pipeline(make_ctx: MakeCtx, no_stdin: None, runs_root: Path) -> None:
    ctx = _ctx(make_ctx)
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    record = result.stages["voice"]
    assert record.status is StageStatus.DONE
    assert record.artifacts == LEAD_ARTIFACTS
    assert VoiceData.model_validate(record.data).attempts == 1


# --- AC: mishears once, then correct -> attempt 2, one diff --------------------------------------


def test_mishear_once_then_pass_on_second_attempt(make_ctx: MakeCtx, no_stdin: None) -> None:
    tts, stt = SpyTts(), ScriptedStt(MISHEARD, LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    out = voice.run(ctx)
    data = _data(out)
    assert data.attempts == 2
    assert len(data.diffs) == 1
    assert data.diffs[0] == diff_words(LINE, MISHEARD, "uk")
    assert data.diffs[0]  # a failed attempt has a non-empty diff
    assert len(tts.texts) == 2
    assert len(stt.calls) == 2
    transcript = (ctx.run_dir / ARTIFACTS["transcript"]).read_text(encoding="utf-8")
    assert transcript.strip() == LINE  # the accepted attempt, not the misheard one


def test_three_mishears_then_pass_on_last_attempt(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(MISHEARD, MISHEARD, MISHEARD, LINE))
    data = _data(voice.run(ctx))
    assert data.attempts == MAX_ATTEMPTS
    assert len(data.diffs) == MAX_ATTEMPTS - 1


# --- AC: 4 mismatches -> GateFailed containing the diff -----------------------------------------


def test_four_mismatches_fail_gate_with_diff(make_ctx: MakeCtx, no_stdin: None) -> None:
    tts, stt = SpyTts(), ScriptedStt(MISHEARD)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    for problem in diff_words(LINE, MISHEARD, "uk"):
        assert problem in str(exc.value)
    assert len(tts.texts) == MAX_ATTEMPTS
    assert len(stt.calls) == MAX_ATTEMPTS


def test_four_mismatches_fail_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt(MISHEARD))
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    record = result.stages["voice"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1  # the stage owns its retries; the pipeline does not add more
    assert record.error is not None and record.error.startswith("GateFailed")
    assert "найголовніший" in record.error


# --- AC: stress / apostrophes / case / punctuation / digits from STT are not mismatches ---------

TOLERATED = {
    "stress_marks": (LINE, "uk", "Я ту́т голо́вний"),
    "apostrophe_variants": ("Я п'ю м'ятний чай", "uk", "Я п’ю мʼятний чай"),
    "case": (LINE, "uk", "я ТУТ ГОЛОВНИЙ"),
    "punctuation": (LINE, "uk", "Я, тут — головний!!!"),
    "digits_uk": ("Мені п'ять років", "uk", "Мені 5 років"),
    "digits_en": ("I have five cats", "en", "I have 5 cats"),
}


@pytest.mark.parametrize("case", list(TOLERATED))
def test_tolerated_differences_pass_first_attempt(
    make_ctx: MakeCtx, no_stdin: None, case: str
) -> None:
    line, lang, heard = TOLERATED[case]
    tts, stt = SpyTts(), ScriptedStt(heard)
    ctx = _ctx(make_ctx, line, lang=lang, tts=tts, stt=stt)
    data = _data(voice.run(ctx))
    assert data.attempts == 1
    assert data.diffs == []
    assert tts.texts == [line]


# --- AC: truncated transcript fails --------------------------------------------------------------


def test_truncated_transcript_fails(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ScriptedStt("Я тут")
    ctx = _ctx(make_ctx, stt=stt)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    assert "головний" in str(exc.value)
    assert len(stt.calls) == MAX_ATTEMPTS


def test_truncated_once_then_full_records_missing_word(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _ctx(make_ctx, stt=ScriptedStt("Я тут", LINE))
    data = _data(voice.run(ctx))
    assert data.attempts == 2
    assert data.diffs == [diff_words(LINE, "Я тут", "uk")]


# --- AC: durations — min_audio_s, speaking shot vs max_duration_s, total <= 30 -------------------


def test_audio_shorter_than_min_fails_without_retry(make_ctx: MakeCtx, no_stdin: None) -> None:
    tts, stt = FixedTts(1.5), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt, video=NoCallVideo(min_audio_s=3.5))
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    msg = str(exc.value)
    assert "3.5" in msg and "2.2" in msg  # names the limit and the value (lead-in + line)
    assert len(tts.texts) == 1  # the line itself doesn't fit -> not retried
    assert stt.calls == []  # durations are checked before STT


def test_audio_plus_second_shot_over_30s_fails(make_ctx: MakeCtx, no_stdin: None) -> None:
    # speaking = max(5, 0.7 + 6.5) = 7.2; + 24 = 31.2 > 30
    tts, stt = FixedTts(6.5), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt, shots=_shots(5, 24))
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    msg = str(exc.value)
    assert "30" in msg and "31.2" in msg
    assert len(tts.texts) == 1
    assert stt.calls == []


def test_total_exactly_30s_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    # speaking = max(5, 0.7 + 5.3) = 6.0; + 24 = 30.0 -> inclusive bound
    ctx = _ctx(make_ctx, tts=FixedTts(5.3), shots=_shots(5, 24))
    data = _data(voice.run(ctx))
    assert data.planned_total_s == pytest.approx(30.0, abs=0.05)


def test_speaking_shot_over_model_max_fails(make_ctx: MakeCtx, no_stdin: None) -> None:
    # speaking = max(4, 0.7 + 6.5) = 7.2 > 4.5
    tts, stt = FixedTts(6.5), ScriptedStt(LINE)
    video = NoCallVideo(max_duration_s=4.5)
    ctx = _ctx(make_ctx, tts=tts, stt=stt, shots=_shots(4, 4), video=video)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    msg = str(exc.value)
    assert "4.5" in msg and "7.2" in msg
    assert len(tts.texts) == 1
    assert stt.calls == []


def test_duration_failure_is_final_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _ctx(make_ctx, tts=FixedTts(6.5), shots=_shots(5, 24))
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    record = result.stages["voice"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")


# === Voice persona (specs/voice.md) ============================================
# New criteria only; strict xfail until the stage implements them.

TEEN_GIRL = VoicePersona(gender="female", age="teen")
CHILD_BOY = VoicePersona(gender="male", age="child")
STYLE = "an excited teenage girl"
# A small voices.yaml for the `fake` TTS (the dev profile's provider name).
TEST_VOICES = (
    "fake:\n"
    "  male-adult: {voice: Puck, style: null}\n"
    '  female-adult: {voice: Aoede, style: "a calm woman"}\n'
    f'  female-teen: {{voice: Leda, style: "{STYLE}"}}\n'
)


# --- AC: TTS receives the persona from script.json; the style is never part of the text ----------


def test_tts_receives_persona_and_never_the_style(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml(TEST_VOICES)
    tts = SpyTts()
    ctx = _ctx(make_ctx, tts=tts, stt=ScriptedStt(MISHEARD, LINE), persona=TEEN_GIRL)
    voice.run(ctx)
    assert tts.texts == [LINE, LINE]  # text == line exactly, on every attempt
    assert all(STYLE not in text for text in tts.texts)
    assert tts.personas == [TEEN_GIRL, TEEN_GIRL]  # from script.json, on every attempt


# --- AC: female-adult and male-adult give different FakeTts pitch -------------------------------


@needs_ffmpeg
def test_female_and_male_adult_differ_in_fake_pitch(make_ctx: MakeCtx, no_stdin: None) -> None:
    # the repo's seeded config/voices.yaml: `fake` has all 8 personas
    pitch: dict[str, float] = {}
    genders: tuple[Gender, ...] = ("male", "female")
    for gender in genders:
        persona = VoicePersona(gender=gender, age="adult")
        ctx = _ctx(make_ctx, tts=SpyTts(), persona=persona)
        voice.run(ctx)
        pitch[gender] = dominant_hz(ctx.run_dir / ARTIFACTS["audio"])
    assert pitch["male"] == pytest.approx(140, abs=3)
    assert pitch["female"] == pytest.approx(220, abs=3)
    assert abs(pitch["female"] - pitch["male"]) > 50


# --- AC: missing persona -> fallback + assumption; empty provider entry -> GateFailed ------------


def test_missing_persona_falls_back_to_same_gender_adult(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml(TEST_VOICES)  # no male-child
    ctx = _ctx(make_ctx, tts=SpyTts(), persona=CHILD_BOY)
    data = _data(voice.run(ctx))
    assert data.voice == VoiceChoice(persona=CHILD_BOY, key="male-adult", voice="Puck", style=None)
    assert "no male-child voice for fake, used male-adult" in ctx.manifest.assumptions


def test_missing_persona_and_gender_falls_back_to_any_entry(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml('fake:\n  female-adult: {voice: Aoede, style: "a calm woman"}\n')
    elder = VoicePersona(gender="male", age="elderly")
    ctx = _ctx(make_ctx, tts=SpyTts(), persona=elder)
    data = _data(voice.run(ctx))
    assert data.voice == VoiceChoice(
        persona=elder, key="female-adult", voice="Aoede", style="a calm woman"
    )
    assert "no male-elderly voice for fake, used female-adult" in ctx.manifest.assumptions


@pytest.mark.parametrize(
    "content",
    ["fake: {}\ngemini-tts:\n  male-adult: {voice: Puck, style: null}\n",
     "gemini-tts:\n  male-adult: {voice: Puck, style: null}\n"],
    ids=["empty_entry", "provider_absent"],
)  # fmt: skip
def test_provider_without_voices_fails_before_tts(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path], content: str
) -> None:
    voices_yaml(content)
    tts, stt = SpyTts(), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    assert "fake" in str(exc.value)  # names the provider
    assert tts.texts == []  # before any TTS call
    assert stt.calls == []
    assert not (ctx.run_dir / ARTIFACTS["audio"]).exists()


# --- AC: data["voice"] names the key, voice and style used ---------------------------------------


def test_data_voice_names_key_voice_and_style(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path]
) -> None:
    voices_yaml(TEST_VOICES)
    ctx = _ctx(make_ctx, tts=SpyTts(), persona=TEEN_GIRL)
    out = voice.run(ctx)
    assert out.data["voice"] == {
        "persona": {"gender": "female", "age": "teen"},
        "key": "female-teen",
        "voice": "Leda",
        "style": STYLE,
    }
    assert _data(out).voice == VoiceChoice(
        persona=TEEN_GIRL, key="female-teen", voice="Leda", style=STYLE
    )
    assert not any("female-teen" in note for note in ctx.manifest.assumptions)  # no fallback


# === Voice mapping is pinned per run (specs/voice.md) ===========================
# New criteria only; strict xfail until the stage implements them.

# Same key (female-teen), a different voice and style: re-reading the config would change the voice.
CHANGED_VOICES = (
    "fake:\n"
    "  male-adult: {voice: Puck, style: null}\n"
    '  female-teen: {voice: Kore, style: "a bored teenage girl"}\n'
)
PINNED = {
    "persona": {"gender": "female", "age": "teen"},
    "key": "female-teen",
    "voice": "Leda",
    "style": STYLE,
}
PINNED_MAPPING = ("female-teen", "Leda", STYLE)


class PinSpyTts(fakes.FakeTts):
    """Real FakeTts that records the text and the `voice=` mapping the stage passes."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.voices: list[Voice | None] = []

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: Voice | None = None,
    ) -> Path:
        self.texts.append(text)
        self.voices.append(voice)
        return super().synthesize(text, lang, out_path, persona, voice=voice)


def _mapping(v: object) -> tuple[object, object, object] | None:
    """(key, voice, style) of whatever the stage passed as `voice=` (model, dataclass or dict)."""
    if v is None:
        return None
    if isinstance(v, dict):
        return (v.get("key"), v.get("voice"), v.get("style"))
    return (getattr(v, "key", None), getattr(v, "voice", None), getattr(v, "style", None))


# --- AC: 4 mismatches -> GateFailed, manifest record still has data["voice"] ----------------------


def test_failed_gate_still_records_voice_in_manifest(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path], runs_root: Path
) -> None:
    voices_yaml(TEST_VOICES)
    ctx = _ctx(make_ctx, stt=ScriptedStt(MISHEARD), persona=TEEN_GIRL)
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    record = result.stages["voice"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and record.error.startswith("GateFailed")
    assert record.data.get("voice") == PINNED
    saved = mf.load(ctx.manifest.run_id, runs_root)  # persisted, not only in memory
    assert saved.stages["voice"].data.get("voice") == PINNED


# --- AC: success, config changed, resume -> the originally resolved voice/style -----------------


def test_resume_after_success_reuses_pinned_voice(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path], runs_root: Path
) -> None:
    voices_yaml(TEST_VOICES)
    tts = PinSpyTts()
    ctx = _ctx(make_ctx, tts=tts, persona=TEEN_GIRL)
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    assert result.stages["voice"].status is StageStatus.DONE
    assert result.stages["voice"].data["voice"] == PINNED
    assert [_mapping(v) for v in tts.voices] == [PINNED_MAPPING]

    voices_yaml(CHANGED_VOICES)  # edited after the run
    out = voice.run(ctx)  # the stage runs again on the same manifest
    assert tts.texts == [LINE, LINE]
    assert _mapping(tts.voices[-1]) == PINNED_MAPPING  # not Kore from the edited file
    assert out.data["voice"] == PINNED
    assert ctx.manifest.stages["voice"].data["voice"] == PINNED


def test_resume_after_failed_gate_reuses_pinned_voice(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path], runs_root: Path
) -> None:
    voices_yaml(TEST_VOICES)
    tts = PinSpyTts()
    # 4 mishears fail the first run; the 5th transcript (first of the resumed run) is correct
    stt = ScriptedStt(*([MISHEARD] * MAX_ATTEMPTS), LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt, persona=TEEN_GIRL)
    first = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    assert first.stages["voice"].status is StageStatus.FAILED
    assert len(tts.texts) == MAX_ATTEMPTS

    voices_yaml(CHANGED_VOICES)  # edited between the failed run and the resume
    second = pipeline.run(ctx, [_voice_stage()], root=runs_root)  # FAILED is not done -> reruns
    record = second.stages["voice"]
    assert record.status is StageStatus.DONE
    assert len(tts.texts) == MAX_ATTEMPTS + 1
    assert [_mapping(v) for v in tts.voices] == [PINNED_MAPPING] * (MAX_ATTEMPTS + 1)
    assert record.data["voice"] == PINNED


# --- AC: malformed voices.yaml names the file; missing file says "file not found" ---------------


def test_malformed_voices_yaml_fails_naming_the_file(
    make_ctx: MakeCtx, no_stdin: None, voices_yaml: Callable[[str], Path]
) -> None:
    path = voices_yaml("fake:\n  male-adult: {voice: Puck, style: [unclosed\n")
    tts, stt = SpyTts(), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    assert path.name in str(exc.value)
    assert tts.texts == []
    assert stt.calls == []


def test_missing_voices_yaml_fails_with_file_not_found(
    make_ctx: MakeCtx, no_stdin: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "noyaml" / "config"
    cfg.mkdir(parents=True)
    shutil.copy(REPO / "config" / "profiles.yaml", cfg / "profiles.yaml")
    monkeypatch.setenv("HWVA_CONFIG", str(cfg / "profiles.yaml"))
    monkeypatch.chdir(cfg.parent)  # no ./config/voices.yaml either
    missing = cfg / "voices.yaml"
    tts, stt = SpyTts(), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    msg = str(exc.value)
    assert "file not found" in msg.lower()
    assert str(missing) in msg  # with the path
    assert "no voices for" not in msg.lower()
    assert tts.texts == []


# === Provider errors inside the attempt loop (specs/voice.md) ===================


class FlakyTts(SpyTts):
    """SpyTts that raises the given errors on its first calls, then speaks."""

    def __init__(self, *errors: ProviderError) -> None:
        super().__init__()
        self.errors = list(errors)

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: Voice | None = None,
    ) -> Path:
        path = super().synthesize(text, lang, out_path, persona, voice=voice)
        if self.errors:
            raise self.errors.pop(0)
        return path


class FlakyStt(ScriptedStt):
    """ScriptedStt that raises the given errors on its first calls (the last one repeats when
    `always`), then hears the line."""

    def __init__(self, *errors: ProviderError, always: bool = False) -> None:
        super().__init__(LINE)
        self.errors = list(errors)
        self.always = always

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        text = super().transcribe(audio_path, lang)
        if self.errors:
            raise self.errors[0] if self.always else self.errors.pop(0)
        return text


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff waits the stage asked for (not actually slept)."""
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", waits.append)
    return waits


RATE_LIMITED = ProviderError("TTS HTTP 429", retryable=True)
BAD_KEY = ProviderError("TTS HTTP 401 bad key", retryable=False)


def test_retryable_tts_error_is_retried_with_backoff(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    tts = FlakyTts(RATE_LIMITED)
    ctx = _ctx(make_ctx, tts=tts)
    data = _data(voice.run(ctx))
    assert data.attempts == 2  # the failed call counts as an attempt
    assert data.diffs == []  # no transcript, no diff
    assert tts.texts == [LINE, LINE]
    assert len(sleeps) == 1 and sleeps[0] > 0


def test_retryable_stt_error_once_then_pass(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    stt = FlakyStt(ProviderError("STT timed out", retryable=True))
    ctx = _ctx(make_ctx, stt=stt)
    data = _data(voice.run(ctx))
    assert data.attempts == 2
    assert len(stt.calls) == 2
    assert len(sleeps) == 1


def test_retryable_stt_error_on_every_attempt_raises_after_four(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    tts, stt = SpyTts(), FlakyStt(ProviderError("STT HTTP 503", retryable=True), always=True)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(ProviderError, match="503"):
        voice.run(ctx)
    assert len(tts.texts) == MAX_ATTEMPTS
    assert len(stt.calls) == MAX_ATTEMPTS
    assert len(sleeps) == MAX_ATTEMPTS - 1  # no wait after the last attempt
    assert sleeps == sorted(sleeps)  # backoff does not shrink


def test_non_retryable_tts_error_is_final(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    tts, stt = FlakyTts(BAD_KEY), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(ProviderError, match="bad key"):
        voice.run(ctx)
    assert tts.texts == [LINE]
    assert stt.calls == []
    assert sleeps == []


def test_retryable_error_then_mismatches_fail_gate_with_diff(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    # 1 provider error + 3 mismatches = 4 attempts; the last failure is a mismatch -> GateFailed
    tts = FlakyTts(RATE_LIMITED)
    ctx = _ctx(make_ctx, tts=tts, stt=ScriptedStt(MISHEARD))
    with pytest.raises(GateFailed, match="найголовніший"):
        voice.run(ctx)
    assert len(tts.texts) == MAX_ATTEMPTS


def test_retryable_errors_used_up_fail_stage_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, sleeps: list[float]
) -> None:
    stt = FlakyStt(ProviderError("STT HTTP 503", retryable=True), always=True)
    ctx = _ctx(make_ctx, stt=stt)
    result = pipeline.run(ctx, [_voice_stage()], root=runs_root)
    record = result.stages["voice"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1  # the stage owns its retries; the pipeline does not add more
    assert record.error is not None and record.error.startswith("ProviderError")
    assert len(stt.calls) == MAX_ATTEMPTS


class SequenceStt(ScriptedStt):
    """Each call returns the next transcript or raises the next error; the last one repeats."""

    def __init__(self, *outcomes: str | ProviderError) -> None:
        super().__init__(LINE)
        self.outcomes = list(outcomes)

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        self.calls.append(audio_path)
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def test_backoff_counts_errors_in_a_row_and_error_names_the_last_diff(
    make_ctx: MakeCtx, no_stdin: None, sleeps: list[float]
) -> None:
    unavailable = ProviderError("STT HTTP 503", retryable=True)
    stt = SequenceStt(MISHEARD, MISHEARD, unavailable)
    ctx = _ctx(make_ctx, stt=stt)
    with pytest.raises(ProviderError) as err:
        voice.run(ctx)
    assert len(stt.calls) == MAX_ATTEMPTS
    assert len(sleeps) == 1  # mismatches are re-synthesized at once; only the error waits
    assert sleeps[0] == voice.BACKOFF_S  # the first error in a row: the first wait
    assert "503" in str(err.value)
    assert "найголовніший" in str(err.value)  # the earlier mismatch is not lost
    assert err.value.retryable is True  # keeps the provider's flag


# --- ffprobe of the audio: timeout 30 s; timeout / non-zero exit -> GateFailed --------------------


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_ffprobe_failure_is_gate_failed(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    real_run = subprocess.run
    seen: list[object] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        if cmd[0] != "ffprobe":
            return real_run(cmd, *args, **kwargs)
        seen.append(kwargs.get("timeout"))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)
        raise subprocess.CalledProcessError(1, cmd, stderr="moov atom not found")

    monkeypatch.setattr(voice.subprocess, "run", fake_run)
    ctx = _ctx(make_ctx, tts=SpyTts(), stt=ScriptedStt(LINE))
    with pytest.raises(GateFailed):
        voice.run(ctx)
    assert seen == [30]


# --- elevenlabs-tts (default real TTS): no child voices -> adult fallback ------


@pytest.mark.parametrize(
    ("gender", "adult_key", "adult_id"),
    [
        ("male", "male-adult", "bIHbv24MWmeRgasZH58o"),
        ("female", "female-adult", "EXAVITQu4vr4xnSDxMaL"),
    ],
)
def test_elevenlabs_child_persona_uses_the_adult_voice_with_assumption(
    make_ctx: MakeCtx, no_stdin: None, gender: Gender, adult_key: str, adult_id: str
) -> None:
    # the repo's seeded config/voices.yaml; the run's TTS provider name is elevenlabs-tts
    refs = dict(registry.build("dev").refs)
    refs["tts"] = ModelRef(provider="elevenlabs-tts", model="eleven_v4", version="eleven_v4")
    child = VoicePersona(gender=gender, age="child")
    tts = SpyTts()
    ctx = _ctx(make_ctx, tts=tts, refs=refs, persona=child)
    data = _data(voice.run(ctx))
    assert data.voice == VoiceChoice(persona=child, key=adult_key, voice=adult_id, style=None)
    assert tts.voices == [Voice(key=adult_key, voice=adult_id, style=None)]
    note = f"no {gender}-child voice for elevenlabs-tts, used {adult_key}"
    assert note in ctx.manifest.assumptions


# === Lead-in for the video model (specs/voice.md) ===============================
# New criteria only; strict xfail until the stage implements them.

DECODE_RATE = 16000
SILENT_PEAK = 10 ** (-60 / 20)  # -60 dBFS


def _decoded(path: Path) -> npt.NDArray[np.float64]:
    """Mono 16 kHz float samples of `path` (ffmpeg)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(DECODE_RATE),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def _audio_stream(path: Path) -> dict[str, Any]:
    streams: list[dict[str, Any]] = probe(path)["streams"]
    return next(s for s in streams if s["codec_type"] == "audio")


# --- AC: default lead-in 0.7 s: shot_audio.wav = 0.7 s of silence + line.wav; STT on line.wav ----


def test_default_lead_in_makes_shot_audio(make_ctx: MakeCtx, no_stdin: None) -> None:
    stt = ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=FixedTts(2.0), stt=stt)
    out = voice.run(ctx)
    data = _data(out)
    assert out.data["lead_in_s"] == LEAD_IN_S
    assert data.lead_in_s == LEAD_IN_S
    assert out.artifacts == LEAD_ARTIFACTS
    line_wav, shot_audio = ctx.run_dir / ARTIFACTS["audio"], ctx.run_dir / SHOT_AUDIO
    assert duration(shot_audio) == pytest.approx(duration(line_wav) + LEAD_IN_S, abs=0.05)
    assert data.shot_audio_duration_s == pytest.approx(duration(shot_audio), abs=0.05)
    assert data.audio_duration_s == pytest.approx(duration(line_wav), abs=0.05)  # the line alone

    samples = _decoded(shot_audio)
    head = samples[: int((LEAD_IN_S - 0.02) * DECODE_RATE)]
    assert np.max(np.abs(head)) < SILENT_PEAK  # the first 0.7 s are digital silence
    tail = samples[int((LEAD_IN_S + 0.05) * DECODE_RATE) :]
    assert np.sqrt(np.mean(tail**2)) > 0.05  # then the TTS tone
    spectrum = np.abs(np.fft.rfft(tail))
    freqs = np.fft.rfftfreq(len(tail), 1 / DECODE_RATE)
    assert float(freqs[int(np.argmax(spectrum[1:])) + 1]) == pytest.approx(440, abs=3)

    a, b = _audio_stream(shot_audio), _audio_stream(line_wav)
    assert a["sample_rate"] == b["sample_rate"]  # same sample rate
    assert a["channels"] == 1  # mono
    assert stt.calls and all(Path(p).name == ARTIFACTS["audio"] for p in stt.calls)  # gate: line


def test_line_wav_is_unchanged_by_the_lead_in(make_ctx: MakeCtx, no_stdin: None) -> None:
    """line.wav stays the pure TTS (the audio of the final video): no silence in front."""
    ctx = _ctx(make_ctx, tts=FixedTts(2.0))
    voice.run(ctx)
    assert (ctx.run_dir / SHOT_AUDIO).is_file()
    line = _decoded(ctx.run_dir / ARTIFACTS["audio"])
    assert np.max(np.abs(line[: int(0.1 * DECODE_RATE)])) > 0.05  # the tone starts at once
    assert duration(ctx.run_dir / ARTIFACTS["audio"]) == pytest.approx(2.0, abs=0.05)


# --- AC: HWVA_LEAD_IN_S=0 -> same length; abc / -1 / 3 -> GateFailed naming it, no TTS call ------


@pytest.mark.parametrize(("value", "lead_in"), [("0", 0.0), ("0.25", 0.25), ("2", 2.0)])
def test_lead_in_from_env(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, value: str, lead_in: float
) -> None:
    monkeypatch.setenv("HWVA_LEAD_IN_S", value)
    ctx = _ctx(make_ctx, tts=FixedTts(2.0))
    out = voice.run(ctx)
    assert out.data["lead_in_s"] == pytest.approx(lead_in)
    line_s, shot_s = duration(ctx.run_dir / ARTIFACTS["audio"]), duration(ctx.run_dir / SHOT_AUDIO)
    assert shot_s == pytest.approx(line_s + lead_in, abs=0.05)


@pytest.mark.parametrize("value", ["abc", "-1", "3", "2.01", "nan"])
def test_bad_lead_in_fails_before_tts(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HWVA_LEAD_IN_S", value)
    tts, stt = SpyTts(), ScriptedStt(LINE)
    ctx = _ctx(make_ctx, tts=tts, stt=stt)
    with pytest.raises(GateFailed, match="HWVA_LEAD_IN_S"):
        voice.run(ctx)
    assert tts.texts == []  # before any TTS call
    assert stt.calls == []


# --- AC: fits max_duration_s alone but not with the lead-in -> GateFailed, before any video call


def test_line_that_fits_only_without_the_lead_in_fails(make_ctx: MakeCtx, no_stdin: None) -> None:
    # line 4.5 s <= 5.0 alone; sent = 0.7 + 4.5 = 5.2 > 5.0. NoCallVideo raises if called.
    tts, stt = FixedTts(4.5), ScriptedStt(LINE)
    video = NoCallVideo(max_duration_s=5.0)
    ctx = _ctx(make_ctx, tts=tts, stt=stt, shots=_shots(4, 4), video=video)
    with pytest.raises(GateFailed) as exc:
        voice.run(ctx)
    msg = str(exc.value)
    assert "5.2" in msg  # names the speaking shot (lead-in + line)
    assert len(tts.texts) == 1  # final: a retry won't make the line shorter
    assert stt.calls == []


def test_line_shorter_than_min_alone_passes_with_the_lead_in(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    # min_audio_s applies to the sent audio: 0.7 + 1.5 = 2.2 >= 2.0
    ctx = _ctx(make_ctx, tts=FixedTts(1.5), video=NoCallVideo(min_audio_s=2.0))
    data = _data(voice.run(ctx))
    assert data.audio_duration_s == pytest.approx(1.5, abs=0.05)
    assert data.planned_total_s == pytest.approx(9.0, abs=0.05)  # max(5, 2.2) + 4


def test_planned_total_includes_the_lead_in(make_ctx: MakeCtx, no_stdin: None) -> None:
    # speaking = max(5, 0.7 + 5.0) = 5.7; + 4 = 9.7 (without the lead-in it would be 9.0)
    ctx = _ctx(make_ctx, tts=FixedTts(5.0))
    data = _data(voice.run(ctx))
    assert data.planned_total_s == pytest.approx(9.7, abs=0.05)


# --- shot_audio.wav is made by ffmpeg with a 120 s timeout; a failure -> GateFailed ---------------


def _reads_line_wav(cmd: list[str]) -> bool:
    """An ffmpeg call that takes line.wav as an INPUT (the TTS fake only writes it)."""
    return Path(cmd[0]).stem.lower() == "ffmpeg" and any(
        a == "-i" and Path(b).name == ARTIFACTS["audio"] for a, b in pairwise(cmd)
    )


@pytest.mark.parametrize("failure", ["none", "timeout", "exit"])
def test_shot_audio_ffmpeg_has_timeout_and_failure_is_gate_failed(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    real_run = subprocess.run
    seen: list[object] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        argv = [str(c) for c in cmd]
        if not _reads_line_wav(argv):
            return real_run(cmd, *args, **kwargs)
        seen.append(kwargs.get("timeout"))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)
        if failure == "exit":
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd, stderr="Invalid data found")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Invalid data found")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ctx = _ctx(make_ctx, tts=FixedTts(2.0), stt=ScriptedStt(LINE))
    if failure == "none":
        voice.run(ctx)
    else:
        with pytest.raises(GateFailed):
            voice.run(ctx)
    assert seen and all(t == 120 for t in seen), seen


def test_shot_audio_ffmpeg_missing_is_gate_failed(
    make_ctx: MakeCtx, no_stdin: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg not on PATH (OSError from subprocess) while making shot_audio.wav -> GateFailed."""
    real_run = subprocess.run
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        argv = [str(c) for c in cmd]
        if not _reads_line_wav(argv):
            return real_run(cmd, *args, **kwargs)
        seen.append(argv)
        raise FileNotFoundError(2, "The system cannot find the file specified", argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    ctx = _ctx(make_ctx, tts=FixedTts(2.0), stt=ScriptedStt(LINE))
    with pytest.raises(GateFailed):
        voice.run(ctx)
    assert seen, "shot_audio.wav was never made"
