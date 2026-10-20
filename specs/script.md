# script

**Purpose.** Turn the idea into a structured script via the LLM (structured output) and validate it deterministically in the same stage — the LLM is not trusted to keep the line intact. (Validation lives here, not in a separate stage, so a bad LLM answer is retried by the pipeline instead of failing forever on resume.)

**Inputs.** idea, `params["line"]` (may be unset), `params["lang"]` (may be unset). Providers: `llm.write_script`, `video.capabilities` (max duration).

**Outputs.** Artifact `script` = `script.json` (`Script` schema). Data: `{"line_corrected": bool, "line_source": str}`. Sets `params["line"]`, `params["line_source"]`, `params["lang"]` when the LLM wrote the line.

**Behaviour.**
1. Call the LLM with the known line (if any) and lang. Schema-invalid answer → `ProviderError(retryable=True)`. Stage `max_attempts=3`.
2. Known line: `normalize(script.line) != normalize(params.line)` → set `script.line = params.line`, record "LLM rewrote the line; restored the original" in `manifest.assumptions`, `data.line_corrected=True`. Never fails for this.
3. Generated line (no known line): it becomes the source of truth → `params["line"]`, `line_source=generated`. Then it must pass, else `ProviderError(retryable=True)` (the LLM gets another try):
   - no digits (`has_digits`);
   - `detect_lang(line) == lang` when lang is known;
   - estimated speech length `words * 0.4 s` ≤ `video.capabilities.max_duration_s`.
4. Structure (any line), else `GateFailed`: exactly one shot with `has_line`; **it is `shots[0]`** (if the LLM put it second, reorder and record); total duration 8–30 s; same `character` for all shots.

**Acceptance criteria.**
- Writes `script.json` that parses as `Script`; known line is forwarded to the provider.
- `ParaphrasingLlm` → original line restored, assumption recorded, stage DONE.
- Generated line copied into `params["line"]` with `line_source=generated`.
- Generated line with digits / wrong language / too long → retried; after 3 bad answers the stage is FAILED (pipeline retry, not a resume loop).
- Speaking shot second → reordered to first.
- Zero or two speaking shots, total outside 8–30 s → `GateFailed`.

**Voice persona.** The script also says who speaks, so the voice matches the character and the grammar of the line (a female voice saying «Я тут головний» is a bug).
- `Script.voice: VoicePersona` = `{gender: male|female, age: child|teen|adult|elderly}`, part of the LLM's structured output; the LLM prompt says: pick the persona from the character, and when the line is grammatically gendered (Ukrainian past tense «зробив/зробила», short adjectives «головний/головна», «сам/сама»), the gender MUST agree with the line — the line is never changed to fit the persona.
- `voice` is required in the schema: an LLM answer without it is schema-invalid → `ProviderError(retryable=True)` (behaviour 1), never a silent male/adult default.
- Deterministic check for Ukrainian, after the LLM: `text.line_gender(line, lang) -> "male" | "female" | None` from a small marker list (first-person «я … -в/-ла» past tense, «головний/головна», «сам/сама», «радий/рада», «готовий/готова»; None when no marker or both). The past-tense marker matches only a lowercase word right after «я», optionally with one adverb between («я вже зробив»); a capitalised word is never a verb («Я люблю Львів» → None). «радий/рада» count only right after «я» («я рада»; «Міська рада вирішила» → None). If it returns a gender and `voice.gender` differs → set `voice.gender` to the line's gender and record the assumption "voice gender changed to match the line". Never fails for this. English → always None.
- `FakeLlm` returns `{male, adult}` unless the line's gender says otherwise.

**Acceptance criteria (voice persona).**
- `script.json` has `voice` with gender and age.
- «Я тут головний» with an LLM that answered `female` → `voice.gender == "male"`, assumption recorded; «Я тут головна» → female.
- «Та ну, я сама все зробила!» → female; a neutral line («Red Bull надає крила») keeps the LLM's choice.
- `line_gender` unit tests: each marker both ways, no marker → None, both genders → None, English → None.
- `line_gender` false positives: «Я люблю Львів» → None (capitalised word after «я» is not a verb); «я вже зробив» → male (one adverb between); «Міська рада вирішила» → None; «я рада» → female.
- `voice` is in `Script.model_json_schema()["required"]`; an LLM answer without `voice` → `ProviderError(retryable=True)`, not a default persona.
