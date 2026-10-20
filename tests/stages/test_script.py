"""Acceptance tests for the `script` stage (specs/script.md)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import ParaphrasingLlm
from pydantic import ValidationError

from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import fakes
from hwva.providers.base import Capabilities, ProviderError, VideoRequest, VideoResult
from hwva.schemas import (
    Age,
    Gender,
    Lang,
    LineSource,
    Script,
    ScriptData,
    Shot,
    StageStatus,
    VoicePersona,
)
from hwva.stages import script

MakeCtx = Callable[..., Context]

KNOWN_LINE = "Я тут головний"
QUOTED_IDEA = f'Кіт каже: "{KNOWN_LINE}"'
NO_QUOTES_IDEA = "Кіт сидить на підвіконні і дивиться на дощ"
RESTORED = "LLM rewrote the line; restored the original"
GOOD_GENERATED = "Дощ знову, а я сухий і щасливий"
CHARACTER = "a ginger cat with green eyes"


# --- helpers ------------------------------------------------------------------------------------


def make_script(
    line: str = KNOWN_LINE,
    *,
    lang: Lang = "uk",
    source: LineSource = LineSource.INPUT,
    shots: list[Shot] | None = None,
) -> Script:
    return Script(
        title="test",
        line=line,
        line_source=source,
        lang=lang,
        character=CHARACTER,
        shots=shots
        or [
            Shot(visual_prompt="cat speaking", duration_s=5, has_line=True),
            Shot(visual_prompt="cat reaction", duration_s=4),
        ],
        voice=VoicePersona(gender="male", age="adult"),  # the LLM always answers it
    )


def generated(line: str, lang: Lang = "uk") -> Script:
    return make_script(line, lang=lang, source=LineSource.GENERATED)


class ScriptedLlm:
    """Returns the given scripts in order (last one repeats); an Exception item is raised.
    Records every call's (idea, line, lang)."""

    def __init__(self, *answers: Script | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str | None, Lang | None]] = []

    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        self.calls.append((idea, line, lang))
        answer = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer


class SpyLlm(fakes.FakeLlm):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, Lang | None]] = []

    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        self.calls.append((idea, line, lang))
        return super().write_script(idea, line, lang)


class ShortVideo(fakes.FakeVideo):
    """Video model that can't do more than 4 s -> at 0.4 s/word, at most 10 words."""

    capabilities = Capabilities(accepts_audio_input=True, max_duration_s=4.0)

    def generate(self, request: VideoRequest, out_path: Path) -> VideoResult:
        raise AssertionError("script must not generate video")


def _invalid_answer() -> ValidationError:
    try:
        Script.model_validate({"title": "x"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _data(out: StageOutput) -> ScriptData:
    assert isinstance(out, StageOutput)
    return ScriptData.model_validate(out.data)


def _written(ctx: Context, out: StageOutput) -> Script:
    assert out.artifacts["script"] == "script.json"
    path = ctx.run_dir / out.artifacts["script"]
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


def _known_ctx(make_ctx: MakeCtx, llm: object, line: str = KNOWN_LINE, **kw: object) -> Context:
    params = {"line": line, "line_source": LineSource.INPUT.value, "lang": "uk"}
    return make_ctx(idea=QUOTED_IDEA, params=params, llm=llm, **kw)


def _script_stage() -> pipeline.Stage:
    return next(s for s in pipeline.STAGES if s.name == "script")


# --- AC: writes script.json that parses as Script; known line is forwarded ---------------------


def test_writes_script_json_and_forwards_known_line(make_ctx: MakeCtx, no_stdin: None) -> None:
    llm = SpyLlm()
    ctx = _known_ctx(make_ctx, llm)
    out = script.run(ctx)
    written = _written(ctx, out)
    assert written.line == KNOWN_LINE
    assert llm.calls == [(QUOTED_IDEA, KNOWN_LINE, "uk")]
    assert ctx.manifest.params["line"] == KNOWN_LINE
    assert ctx.manifest.params["line_source"] == LineSource.INPUT.value
    data = _data(out)
    assert data.line_corrected is False
    assert data.line_source is LineSource.INPUT
    assert ctx.manifest.assumptions == []


def test_user_line_source_is_reported(make_ctx: MakeCtx, no_stdin: None) -> None:
    params = {"line": KNOWN_LINE, "line_source": LineSource.USER.value, "lang": "uk"}
    ctx = make_ctx(idea=NO_QUOTES_IDEA, params=params, llm=SpyLlm())
    out = script.run(ctx)
    assert _data(out).line_source is LineSource.USER
    assert ctx.manifest.params["line_source"] == LineSource.USER.value


def test_without_known_line_provider_gets_none(make_ctx: MakeCtx, no_stdin: None) -> None:
    llm = SpyLlm()
    ctx = make_ctx(idea=NO_QUOTES_IDEA, llm=llm)
    script.run(ctx)
    assert llm.calls == [(NO_QUOTES_IDEA, None, None)]


def test_schema_invalid_answer_is_retryable_provider_error(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _known_ctx(make_ctx, ScriptedLlm(_invalid_answer()))
    with pytest.raises(ProviderError) as exc:
        script.run(ctx)
    assert exc.value.retryable is True


# --- AC: ParaphrasingLlm -> original restored, assumption recorded, stage DONE -----------------


def test_paraphrased_line_is_restored(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _known_ctx(make_ctx, ParaphrasingLlm())
    out = script.run(ctx)
    assert _written(ctx, out).line == KNOWN_LINE  # byte-exact original
    assert ctx.manifest.params["line"] == KNOWN_LINE
    assert RESTORED in ctx.manifest.assumptions
    assert _data(out).line_corrected is True


def test_paraphrased_line_stage_is_done_in_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    ctx = _known_ctx(make_ctx, ParaphrasingLlm())
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 1
    assert record.data["line_corrected"] is True
    assert RESTORED in result.assumptions


def test_punctuation_only_difference_is_not_a_correction(make_ctx: MakeCtx, no_stdin: None) -> None:
    # normalize() equal -> not a rewrite; source of truth untouched
    ctx = _known_ctx(make_ctx, ScriptedLlm(make_script("я тут головний!")))
    out = script.run(ctx)
    assert _data(out).line_corrected is False
    assert RESTORED not in ctx.manifest.assumptions
    assert ctx.manifest.params["line"] == KNOWN_LINE


# --- AC: generated line copied into params with line_source=generated --------------------------


def test_generated_line_becomes_source_of_truth(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = make_ctx(idea=NO_QUOTES_IDEA, llm=ScriptedLlm(generated(GOOD_GENERATED)))
    out = script.run(ctx)
    params = ctx.manifest.params
    assert params["line"] == GOOD_GENERATED
    assert params["line_source"] == LineSource.GENERATED.value
    assert params["lang"] == "uk"
    assert _written(ctx, out).line == GOOD_GENERATED
    data = _data(out)
    assert data.line_source is LineSource.GENERATED
    assert data.line_corrected is False


def test_generated_line_keeps_known_lang(make_ctx: MakeCtx, no_stdin: None) -> None:
    llm = ScriptedLlm(generated("It is raining again and I am happy", lang="en"))
    ctx = make_ctx(idea=NO_QUOTES_IDEA, params={"lang": "en"}, llm=llm)
    script.run(ctx)
    assert llm.calls[0][1:] == (None, "en")
    assert ctx.manifest.params["lang"] == "en"
    assert ctx.manifest.params["line_source"] == LineSource.GENERATED.value


def test_generated_line_at_length_limit_passes(make_ctx: MakeCtx, no_stdin: None) -> None:
    ten_words = "раз два три чотири п'ять шість сім вісім дев'ять десять"  # 10 * 0.4 = 4.0 s
    ctx = make_ctx(idea=NO_QUOTES_IDEA, llm=ScriptedLlm(generated(ten_words)), video=ShortVideo())
    script.run(ctx)
    assert ctx.manifest.params["line"] == ten_words


# --- AC: generated line with digits / wrong language / too long -> retried; 3 bad -> FAILED ----

ELEVEN_WORDS = "раз два три чотири п'ять шість сім вісім дев'ять десять одинадцять"  # 4.4 s > 4

BAD_GENERATED = {
    "digits": ({}, generated("Мені 5 років")),
    "wrong_lang": ({"lang": "en"}, generated("Дощ знову, а я сухий", lang="en")),
    "too_long": ({}, generated(ELEVEN_WORDS)),
}


@pytest.mark.parametrize("case", list(BAD_GENERATED))
def test_bad_generated_line_is_retryable_provider_error(
    make_ctx: MakeCtx, no_stdin: None, case: str
) -> None:
    params, bad = BAD_GENERATED[case]
    ctx = make_ctx(idea=NO_QUOTES_IDEA, params=params, llm=ScriptedLlm(bad), video=ShortVideo())
    with pytest.raises(ProviderError) as exc:
        script.run(ctx)
    assert exc.value.retryable is True


@pytest.mark.parametrize("case", list(BAD_GENERATED))
def test_bad_generated_line_retried_then_good_one_accepted(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, case: str
) -> None:
    params, bad = BAD_GENERATED[case]
    lang: Lang = "en" if params.get("lang") == "en" else "uk"
    good = generated("It is raining again" if lang == "en" else GOOD_GENERATED, lang=lang)
    llm = ScriptedLlm(bad, bad, good)
    ctx = make_ctx(idea=NO_QUOTES_IDEA, params=params, llm=llm, video=ShortVideo())
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 3
    # a rejected generated line must not leak into params and turn into a "known" line
    assert [call[1] for call in llm.calls] == [None, None, None]
    assert result.params["line"] == good.line
    assert result.params["line_source"] == LineSource.GENERATED.value


@pytest.mark.parametrize("case", list(BAD_GENERATED))
def test_three_bad_generated_lines_fail_the_stage(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path, case: str
) -> None:
    params, bad = BAD_GENERATED[case]
    llm = ScriptedLlm(bad)
    ctx = make_ctx(idea=NO_QUOTES_IDEA, params=params, llm=llm, video=ShortVideo())
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 3
    assert len(llm.calls) == 3
    assert record.error is not None and record.error.startswith("ProviderError")
    # nothing left behind that a --resume would accept as a known line
    assert "line" not in result.params
    assert result.params.get("line_source") != LineSource.GENERATED.value


def test_schema_invalid_answers_are_retried_by_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    llm = ScriptedLlm(_invalid_answer(), _invalid_answer(), make_script())
    ctx = _known_ctx(make_ctx, llm)
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    assert result.stages["script"].status is StageStatus.DONE
    assert result.stages["script"].attempts == 3


# --- AC: speaking shot second -> reordered to first --------------------------------------------


def test_speaking_shot_second_is_reordered_first(make_ctx: MakeCtx, no_stdin: None) -> None:
    shots = [
        Shot(visual_prompt="cat reaction", duration_s=4),
        Shot(visual_prompt="cat speaking", duration_s=5, has_line=True),
    ]
    ctx = _known_ctx(make_ctx, ScriptedLlm(make_script(shots=shots)))
    out = script.run(ctx)
    written = _written(ctx, out)
    assert [s.visual_prompt for s in written.shots] == ["cat speaking", "cat reaction"]
    assert [s.has_line for s in written.shots] == [True, False]
    assert [s.duration_s for s in written.shots] == [5, 4]
    assert len(ctx.manifest.assumptions) == 1  # the reorder is recorded


def test_speaking_shot_first_is_not_recorded(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _known_ctx(make_ctx, ScriptedLlm(make_script()))
    out = script.run(ctx)
    assert [s.visual_prompt for s in _written(ctx, out).shots] == ["cat speaking", "cat reaction"]
    assert ctx.manifest.assumptions == []


# --- AC: zero / two speaking shots, total outside 8-30 s -> GateFailed -------------------------

BAD_STRUCTURE = {
    "zero_speaking": [Shot(visual_prompt="a", duration_s=5), Shot(visual_prompt="b", duration_s=4)],
    "two_speaking": [
        Shot(visual_prompt="a", duration_s=5, has_line=True),
        Shot(visual_prompt="b", duration_s=4, has_line=True),
    ],
    "total_7s": [
        Shot(visual_prompt="a", duration_s=4, has_line=True),
        Shot(visual_prompt="b", duration_s=3),
    ],
    "single_shot_5s": [Shot(visual_prompt="a", duration_s=5, has_line=True)],
    "total_35s": [
        Shot(visual_prompt="a", duration_s=20, has_line=True),
        Shot(visual_prompt="b", duration_s=15),
    ],
}


@pytest.mark.parametrize("case", list(BAD_STRUCTURE))
def test_bad_structure_fails_gate(make_ctx: MakeCtx, no_stdin: None, case: str) -> None:
    ctx = _known_ctx(make_ctx, ScriptedLlm(make_script(shots=BAD_STRUCTURE[case])))
    with pytest.raises(GateFailed):
        script.run(ctx)


@pytest.mark.parametrize("case", ["zero_speaking", "total_7s"])
def test_bad_structure_with_generated_line_fails_gate(
    make_ctx: MakeCtx, no_stdin: None, case: str
) -> None:
    bad = make_script(GOOD_GENERATED, source=LineSource.GENERATED, shots=BAD_STRUCTURE[case])
    ctx = make_ctx(idea=NO_QUOTES_IDEA, llm=ScriptedLlm(bad))
    with pytest.raises(GateFailed):
        script.run(ctx)


def test_bad_structure_is_not_retried_by_pipeline(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    llm = ScriptedLlm(make_script(shots=BAD_STRUCTURE["two_speaking"]))
    ctx = _known_ctx(make_ctx, llm)
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.FAILED
    assert record.attempts == 1
    assert record.error is not None and record.error.startswith("GateFailed")


@pytest.mark.parametrize(
    "durations", [(5.0, 3.0), (8.0,), (15.0, 15.0), (30.0,)], ids=["8s", "8s_one", "30s", "30s_one"]
)
def test_total_duration_bounds_are_inclusive(
    make_ctx: MakeCtx, no_stdin: None, durations: tuple[float, ...]
) -> None:
    shots = [
        Shot(visual_prompt=f"shot {i}", duration_s=d, has_line=i == 0)
        for i, d in enumerate(durations)
    ]
    ctx = _known_ctx(make_ctx, ScriptedLlm(make_script(shots=shots)))
    out = script.run(ctx)
    assert sum(s.duration_s for s in _written(ctx, out).shots) == sum(durations)


# --- Voice persona (specs/script.md) -----------------------------------------
# Each test is xfail until the implementer adds `Script.voice` and `text.line_gender`.
# `Script.voice` is read through model_dump() so these tests import cleanly before the field exists.

GENDER_CHANGED = "voice gender changed to match the line"
NEUTRAL_LINE = "Red Bull надає крила"
FEMALE_IDEA = 'Кішка каже: "Я тут головна"'
AGES: tuple[Age, ...] = ("child", "teen", "adult", "elderly")


def with_voice(answer: Script, gender: Gender, age: Age = "adult") -> Script:
    """The LLM's answer with the given persona (the stage re-validates it from model_dump())."""
    return Script.model_validate({**answer.model_dump(), "voice": {"gender": gender, "age": age}})


def _voice(written: Script) -> VoicePersona:
    return VoicePersona.model_validate(written.model_dump()["voice"])


def _gender_changed(assumptions: list[str]) -> bool:
    return any(GENDER_CHANGED in note for note in assumptions)


# --- AC: script.json has `voice` with gender and age --------------------------------------------


def test_script_json_has_voice_with_gender_and_age(make_ctx: MakeCtx, no_stdin: None) -> None:
    ctx = _known_ctx(make_ctx, SpyLlm())
    out = script.run(ctx)
    raw = json.loads((ctx.run_dir / out.artifacts["script"]).read_text(encoding="utf-8"))
    assert set(raw["voice"]) == {"gender", "age"}
    assert raw["voice"]["gender"] in ("male", "female")
    assert raw["voice"]["age"] in AGES


@pytest.mark.parametrize(
    ("idea", "params", "expected"),
    [
        (QUOTED_IDEA, {"line": KNOWN_LINE, "lang": "uk"}, ("male", "adult")),
        (NO_QUOTES_IDEA, {}, ("male", "adult")),  # FakeLlm's neutral generated line
        (FEMALE_IDEA, {"line": "Я тут головна", "lang": "uk"}, ("female", "adult")),
    ],
    ids=["known_male", "generated_neutral", "known_female"],
)
def test_fake_llm_persona_is_male_adult_unless_line_says_otherwise(
    make_ctx: MakeCtx,
    no_stdin: None,
    idea: str,
    params: dict[str, str],
    expected: tuple[Gender, Age],
) -> None:
    ctx = make_ctx(idea=idea, params=params, llm=fakes.FakeLlm())
    out = script.run(ctx)
    voice = _voice(_written(ctx, out))
    assert (voice.gender, voice.age) == expected


# --- AC: gendered line overrides the LLM's gender; neutral line keeps the LLM's choice ----------

PERSONA_CASES: dict[str, tuple[str, Gender, Gender]] = {
    # case: (line, gender the LLM answered, expected voice.gender)
    "golovnyi_llm_female": (KNOWN_LINE, "female", "male"),
    "golovna_llm_male": ("Я тут головна", "male", "female"),
    "sama_zrobyla_llm_male": ("Та ну, я сама все зробила!", "male", "female"),
    "sama_zrobyla_llm_female": ("Та ну, я сама все зробила!", "female", "female"),
    "neutral_llm_female": (NEUTRAL_LINE, "female", "female"),
    "neutral_llm_male": (NEUTRAL_LINE, "male", "male"),
}


@pytest.mark.parametrize("case", list(PERSONA_CASES))
def test_voice_gender_follows_the_line(make_ctx: MakeCtx, no_stdin: None, case: str) -> None:
    line, llm_gender, expected = PERSONA_CASES[case]
    answer = with_voice(make_script(line), llm_gender, age="teen")
    ctx = _known_ctx(make_ctx, ScriptedLlm(answer), line=line)
    out = script.run(ctx)
    written = _written(ctx, out)
    voice = _voice(written)
    assert voice.gender == expected
    assert voice.age == "teen"  # only the gender is ever corrected
    assert written.line == line  # the line is never changed to fit the persona
    assert ctx.manifest.params["line"] == line
    assert _gender_changed(ctx.manifest.assumptions) is (llm_gender != expected)


def test_generated_gendered_line_overrides_llm_gender(make_ctx: MakeCtx, no_stdin: None) -> None:
    answer = with_voice(generated("Я сама все зробила"), "male")
    ctx = make_ctx(idea=NO_QUOTES_IDEA, llm=ScriptedLlm(answer))
    out = script.run(ctx)
    assert _voice(_written(ctx, out)).gender == "female"
    assert _gender_changed(ctx.manifest.assumptions)


def test_english_line_keeps_llm_gender(make_ctx: MakeCtx, no_stdin: None) -> None:
    line = "I did it all by myself"
    answer = with_voice(make_script(line, lang="en"), "female")
    params = {"line": line, "line_source": LineSource.INPUT.value, "lang": "en"}
    ctx = make_ctx(idea=f'A cat says: "{line}"', params=params, llm=ScriptedLlm(answer))
    out = script.run(ctx)
    assert _voice(_written(ctx, out)).gender == "female"
    assert not _gender_changed(ctx.manifest.assumptions)


def test_gender_mismatch_never_fails_the_stage(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    llm = ScriptedLlm(with_voice(make_script(KNOWN_LINE), "female"))
    ctx = _known_ctx(make_ctx, llm)
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 1
    assert len(llm.calls) == 1  # corrected in place, not by asking the LLM again
    assert _gender_changed(result.assumptions)
    written = Script.model_validate_json((ctx.run_dir / "script.json").read_text(encoding="utf-8"))
    assert _voice(written).gender == "male"


# --- AC: `voice` is required; an answer without it is retryable, never a default persona -------
# (specs/script.md, "Voice persona")


def _raw_without_voice(line: str = KNOWN_LINE) -> dict[str, object]:
    raw = make_script(line).model_dump(mode="json")
    del raw["voice"]
    return raw


class NoVoiceLlm:
    """A real provider's path: parse the model's raw JSON into `Script`. The JSON has no `voice`.
    After `answers_without_voice` calls it answers with a voice. Records every call."""

    def __init__(self, answers_without_voice: int = 1_000) -> None:
        self.without = answers_without_voice
        self.calls: list[tuple[str, str | None, Lang | None]] = []

    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        self.calls.append((idea, line, lang))
        if len(self.calls) <= self.without:
            return Script.model_validate(_raw_without_voice())
        return with_voice(make_script(), "female", age="elderly")


def test_voice_is_required_in_llm_schema() -> None:
    assert "voice" in Script.model_json_schema()["required"]


def test_answer_without_voice_is_retryable_provider_error(
    make_ctx: MakeCtx, no_stdin: None
) -> None:
    ctx = _known_ctx(make_ctx, NoVoiceLlm())
    with pytest.raises(ProviderError) as exc:
        script.run(ctx)
    assert exc.value.retryable is True
    assert not (ctx.run_dir / "script.json").exists()  # no default persona written


def test_answers_without_voice_are_retried_then_real_voice_used(
    make_ctx: MakeCtx, no_stdin: None, runs_root: Path
) -> None:
    llm = NoVoiceLlm(answers_without_voice=2)
    ctx = _known_ctx(make_ctx, llm, line="Привіт, це тестова репліка")
    result = pipeline.run(ctx, [_script_stage()], root=runs_root)
    record = result.stages["script"]
    assert record.status is StageStatus.DONE
    assert record.attempts == 3
    assert len(llm.calls) == 3
    written = Script.model_validate_json((ctx.run_dir / "script.json").read_text(encoding="utf-8"))
    # the LLM's persona, not a default
    assert _voice(written) == VoicePersona(gender="female", age="elderly")


def test_fake_llm_always_answers_voice() -> None:
    answer = fakes.FakeLlm().write_script(QUOTED_IDEA, KNOWN_LINE, "uk")
    assert "voice" in answer.model_fields_set
