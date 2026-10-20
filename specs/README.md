# Stage specs

One file per stage. A spec is the contract the agents build against (`/build-stage <name>`).
Order below IS the pipeline order (`STAGES` in `src/hwva/pipeline.py`) — enforced by
`tests/test_architecture.py::test_stages_follow_specs_order`.

| # | Stage | Gate (what must be true to continue) |
|---|---|---|
| 0 | [intake](intake.md) | the line is known (no digits) or explicitly left to the LLM |
| 1 | [script](script.md) | valid `Script`; line identical to the source of truth; no digits; language and length fit; speaking shot first |
| 2 | [voice](voice.md) | TTS audio within limits, total video ≤ 30 s, AND STT == line, word for word |
| 3 | [shot_1](shot_1.md) | speaking shot downloaded, last frame saved locally |
| 4 | [shot_2](shot_2.md) | second shot downloaded (or skipped), same character |
| 5 | [assemble](assemble.md) | final.mp4 with OUR audio track |
| 6 | [final_check](final_check.md) | format, signal (black / silence / loudness), audio == the placed TTS track (line.wav, or aligned_line.wav cut from it by exact sample copy), STT on final == line |
| 7 | [quality_check](quality_check.md) | soft — never blocks: Gemini judge + SyncNet on the final |

`quality_check` runs last on purpose: it judges the deliverable itself, can't influence it, and
doesn't spend judge quota on a run that already failed a hard gate.

Not a stage: [eval](eval.md) — `uv run eval`, runs the pipeline on a small idea set for Part 3.
Not a stage: [sync](sync.md) — lip-sync gate (SyncNet) and per-phrase alignment inside `shot_1`; the aligned track placed by `assemble`.

Shared rules:
- `manifest.params["line"]` is the single source of truth for the dialogue line.
- No digits in the line — numbers are written in words (user, or the LLM when it writes the line).
- Stage signature: `def run(ctx: Context) -> StageOutput` (from `hwva.context`).
- Artifacts are local files in the run dir. Provider URLs expire (24 h) — download immediately; never store a URL as the only copy (resume must work).
- Gates are deterministic and proven by tests on synthetic bad inputs. Soft scores record raw numbers and never block.

Spec template: Purpose · Inputs · Outputs · Behaviour · Acceptance criteria (each becomes a test) · Out of scope.
