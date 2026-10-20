# hwva — idea → vertical video with a verbatim spoken line

PoC for the HOLYWATER AI Engineer test task. Agents write the code; this file is the contract they work under.

## The one requirement that matters
Task spec: *"у відео має звучати репліка дослівно — слово в слово так, як прописана в тексті на вході, без перефразувань з боку моделей."* Therefore:
- The line comes from the input (quotes / `--line` / intake answer). `manifest.params["line"]` is the single source of truth; an LLM never rewrites it.
- **No digits in the line** — numbers are written in words by the user (Ukrainian numerals inflect; we can't spell them right, the user can). The LLM writes numbers in words when it has to invent a line.
- Voice = TTS of the exact line. Verified by STT (word-level, `hwva.text.diff_words`) before video and again on the final file.
- The video model only animates lips to OUR audio (strategy A). The final audio track is always our TTS.

## Architecture (do not change without updating specs/ and README)
- `src/hwva/pipeline.py` — fixed stage order (`STAGES`), gates, retries, resume. The LLM never decides what runs next.
- `src/hwva/context.py` — `Context`, `StageOutput`, `GateFailed`. Stages import these from here.
- `src/hwva/strategy.py` — strategy A/B/C chosen from the video model's `Capabilities` at start (`preflight`), before any run dir or spend. Only A is implemented; B/C fail fast. Never add a silent fallback to another model.
- `src/hwva/stages/<name>.py` — one module per stage: `def run(ctx: Context) -> StageOutput`. Raise `GateFailed` when output fails its check, `ProviderError` for provider failures (`retryable=False` for moderation etc.).
- `src/hwva/providers/base.py` — Protocols + `Capabilities`. **Stages import only from `base`**; concrete providers come via `ctx.providers` (enforced by `tests/test_architecture.py`).
- `src/hwva/providers/registry.py` + `config/profiles.yaml` — the only place that maps names to classes. Model + version pinned per profile, copied into the manifest; resume uses the pinned refs.
- `src/hwva/manifest.py` — run state in `runs/<run_id>/manifest.json`. Facts go in `StageOutput.data`; things the agent decided by itself in `manifest.assumptions`.
- `tests/conftest.py` — fixtures for stage tests: `make_ctx`, `dev_ctx`, `ScriptedStt`, `SpyTts`, `ParaphrasingLlm`, `no_stdin`, `probe`, `make_audio`, `make_video`.

## Boundaries
- Work only in `src/`, `tests/`, `specs/`, `config/`. Don't touch `.claude/` or this file unless the task says so.
- Never weaken, skip or delete a test to make it pass. The only allowed test edit for an implementer is removing the spec-writer's `xfail` marker. If a test is wrong, say so and stop.
- Secrets only from env (`.env`, see `.env.example`). Never print or log keys.
- No new dependency without a one-line reason in the summary.
- Tests never call paid APIs. Real calls only in tests marked `@pytest.mark.live`.
- Default profile is `dev` (fakes, $0). Run commands from the repo root.

## Definition of done
`uv run check` is green (ffmpeg present, ruff, format, mypy strict, pytest) — the Stop hook enforces it — and every acceptance criterion in the stage's spec is covered by a test.

## How work is done
New stage: `/build-stage <name>` → spec-writer (xfail tests) → implementer → reviewer (see `.claude/skills/build-stage`).
New model: `/new-provider <role> <name>`.
Log every human intervention and every rule added to the harness in `AI-LOG.md`.
