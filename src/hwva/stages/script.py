"""script — idea -> structured `Script` via the LLM, validated deterministically.

The LLM is not trusted with the line: a known line is restored byte-exact if the LLM touched it;
a generated line must pass the same rules a user line does before it becomes the source of truth.

Contract: specs/script.md."""

from __future__ import annotations

from pydantic import ValidationError

from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import Lang, LineSource, Script, ScriptData
from hwva.text import detect_lang, has_digits, line_gender, normalize

ARTIFACT = "script.json"
SECONDS_PER_WORD = 0.4
MIN_TOTAL_S = 8.0
MAX_TOTAL_S = 30.0
ASSUMPTION_RESTORED = "LLM rewrote the line; restored the original"
ASSUMPTION_REORDERED = "LLM put the speaking shot second; moved it to shots[0]"
ASSUMPTION_GENDER = "voice gender changed to match the line"


def run(ctx: Context) -> StageOutput:
    params = ctx.manifest.params
    known_line = params.get("line") or None
    known_lang: Lang | None = _as_lang(params.get("lang"))

    script = _ask_llm(ctx, known_line, known_lang)
    assumptions: list[str] = []
    corrected = False

    if known_line is not None:
        source = LineSource(params.get("line_source") or LineSource.INPUT.value)
        if normalize(script.line) != normalize(known_line):
            corrected = True
            assumptions.append(ASSUMPTION_RESTORED)
        # always the byte-exact original, even when only punctuation/case differed
        lang = known_lang or detect_lang(known_line)
        script = script.model_copy(update={"line": known_line, "line_source": source, "lang": lang})
    else:
        source = LineSource.GENERATED
        lang = known_lang or script.lang
        _check_generated_line(ctx, script.line, lang)
        script = script.model_copy(update={"line_source": source, "lang": lang})

    script, reordered = _check_structure(script)
    if reordered:
        assumptions.append(ASSUMPTION_REORDERED)

    script, regendered = _match_voice_to_line(script)
    if regendered:
        assumptions.append(ASSUMPTION_GENDER)

    # all checks passed — only now touch the manifest, so a rejected answer leaves nothing behind
    params["line"] = script.line
    params["line_source"] = source.value
    params["lang"] = script.lang
    for note in assumptions:
        if note not in ctx.manifest.assumptions:
            ctx.manifest.assumptions.append(note)

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.run_dir / ARTIFACT).write_text(script.model_dump_json(indent=2), encoding="utf-8")
    data = ScriptData(line_corrected=corrected, line_source=source)
    return StageOutput(artifacts={"script": ARTIFACT}, data=data.model_dump(mode="json"))


def _as_lang(value: str | None) -> Lang | None:
    if value == "uk":
        return "uk"
    if value == "en":
        return "en"
    return None


def _ask_llm(ctx: Context, line: str | None, lang: Lang | None) -> Script:
    """Structured output from the LLM, re-validated (a provider may hand back an unvalidated copy).
    A schema-invalid answer is the LLM's fault, not ours -> retryable.

    `Script.voice` has a default (old scripts and later-stage tests omit it), so a provider that
    parses the model's JSON gets a silent male/adult persona when the LLM left `voice` out. The
    LLM schema requires it, so an answer without it is schema-invalid too — never a default."""
    try:
        answer = ctx.providers.llm.write_script(ctx.manifest.idea, line, lang)
        if "voice" not in answer.model_fields_set:
            raise ProviderError("LLM returned a script without `voice`", retryable=True)
        return Script.model_validate(answer.model_dump())
    except ValidationError as exc:
        raise ProviderError(f"LLM returned a schema-invalid script: {exc}", retryable=True) from exc


def _check_generated_line(ctx: Context, line: str, lang: Lang) -> None:
    """Rules for a line the LLM invented. Failing any of them = a bad LLM answer -> retryable."""
    if has_digits(line):
        raise ProviderError(f"Generated line contains digits: {line!r}", retryable=True)
    detected = detect_lang(line)
    if detected != lang:
        raise ProviderError(
            f"Generated line is in {detected!r}, expected {lang!r}: {line!r}", retryable=True
        )
    words = len(normalize(line).split())
    estimate = words * SECONDS_PER_WORD
    limit = ctx.providers.video.capabilities.max_duration_s
    if estimate > limit:
        raise ProviderError(
            f"Generated line too long: {words} words ~ {estimate:.1f} s > {limit:.1f} s",
            retryable=True,
        )


def _match_voice_to_line(script: Script) -> tuple[Script, bool]:
    """The line's grammar wins over the LLM's persona: only `voice.gender` is corrected, never
    the line, and never by failing the stage. Returns the script and whether it changed."""
    gender = line_gender(script.line, script.lang)
    if gender is None or gender == script.voice.gender:
        return script, False
    voice = script.voice.model_copy(update={"gender": gender})
    return script.model_copy(update={"voice": voice}), True


def _check_structure(script: Script) -> tuple[Script, bool]:
    """Exactly one speaking shot, placed first; total duration within bounds. Returns the
    (possibly reordered) script and whether a reorder happened.

    "Same character for all shots" holds by construction: `Script.character` is a single field
    that every shot prompt is built with downstream."""
    speaking = [i for i, shot in enumerate(script.shots) if shot.has_line]
    if len(speaking) != 1:
        raise GateFailed(f"Script must have exactly one speaking shot, got {len(speaking)}")
    total = sum(shot.duration_s for shot in script.shots)
    if not MIN_TOTAL_S <= total <= MAX_TOTAL_S:
        raise GateFailed(f"Total duration {total:g} s is outside {MIN_TOTAL_S:g}-{MAX_TOTAL_S:g} s")
    if not script.character.strip():
        raise GateFailed("Script has no character description")
    idx = speaking[0]
    if idx == 0:
        return script, False
    shots = [script.shots[idx], *(s for i, s in enumerate(script.shots) if i != idx)]
    return script.model_copy(update={"shots": shots}), True
