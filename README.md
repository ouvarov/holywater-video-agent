# hwva — idea → vertical video with a verbatim spoken line

HOLYWATER AI Engineer test task. One command turns an idea (1–3 sentences) into an 8–30 s vertical video where a character says the dialogue line **word for word**.

> Status: all 8 stages and `eval` are built and tested (`uv run check` green). Script, voice, STT and the judge run on real models. Seedance via OpenRouter was checked with paid live clips (2.0-fast and 2.5, Ukrainian and English) and a full real run; the adapter and the lip-sync check were fixed from what they showed (see below). Seedance on BytePlus ModelArk direct is covered by mocked tests only (no ARK key). The design note (in Ukrainian) is `docs/NOTE.md`.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 itself), ffmpeg/ffprobe on PATH, the SyncNet weights (the lip-sync check, see "SyncNet") and, for real videos, three API keys in `.env`.

### Windows (built and run here)

```powershell
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
# open a NEW terminal, so uv and ffmpeg are on PATH
git clone <this repo's URL> holywater-video-agent
cd holywater-video-agent
uv sync                                                                   # deps incl. SyncNet + CPU PyTorch (~900 MB)
powershell -ExecutionPolicy Bypass -File scripts\get-syncnet-weights.ps1  # weights into weights\ (sha256-checked)
copy .env.example .env                                                    # then fill in the keys (below)
uv run check                                                              # optional: lint, types, tests (~7 min)
```

### macOS (same steps, not tested on a Mac)

```bash
brew install uv ffmpeg
git clone <this repo's URL> holywater-video-agent && cd holywater-video-agent
uv sync
mkdir -p weights
for f in syncnet_v2.model sfd_face.pth; do
  curl -L --fail -o "weights/$f" "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/$f"
done
shasum -a 256 weights/*   # expected prefixes: syncnet_v2.model 961e8696…, sfd_face.pth d54a87c2…
cp .env.example .env      # then fill in the keys (below)
```

Keys for a real video (`premium-openrouter`): `GEMINI_API_KEY` (script + judge), `ELEVENLABS_API_KEY` (voice + STT), `OPENROUTER_API_KEY` (Seedance video + character image). The `dev` profile needs none.

## Run

**A real video** (all real models, Seedance 2.5 via OpenRouter, about 1–2.5 USD, capped at 3 USD per run):

```bash
uv run video 'Мультяшна дівчина-бариста каже: "Понеділок мене не злякає, якщо є кава"' --profile premium-openrouter --no-interactive
```

Other profiles and commands:

```bash
uv run video 'Кіт сидить на підвіконні і каже: "Я тут головний"'                # dev profile (default): free, offline fakes
uv run video 'Кіт каже: "Я тут головний"' --profile real-no-video --no-interactive  # real models except video
uv run video 'Кіт каже: "Я тут головний"' --profile premium --no-interactive        # everything real, Seedance on BytePlus (needs ARK_API_KEY)
uv run video --resume <run_id>                                                    # continue a failed run (paid clips are never paid again)
```

The line is taken from quotes in the idea or from `--line`. No line? In a terminal the agent asks once *"Що каже персонаж?"*; with `--no-interactive` it never asks — the LLM writes the line and the manifest records that it was generated.

**Numbers in the line are written in words** ("дві гривні", not "2 гривні"): Ukrainian numerals inflect, and the line must sound exactly as written, so the user spells them, not a converter.

Output: `runs/<run_id>/final.mp4` + `manifest.json` (every stage, model + version, attempts, cost, assumptions).

## Examples

Two finished videos made by the pipeline, one command each (`docs/examples/`):
- `en-barista-seedance-2.5.mp4` — English, Seedance 2.5 with the character image: 9.1 s, 2 shots, 2.14 USD. `final_check` passed (STT heard the line word for word, the audio is our track: correlation 0.998); SyncNet 2.73 (a soft lip-sync warning); judge 4/5 on all five criteria.
- `uk-run1-seedance-2.0-fast.mp4` — Ukrainian, Seedance 2.0-fast, 0.86 USD: the first real run; lip-sync "not good" by eye (SyncNet 1.7), which is where the SyncNet gate and the per-phrase alignment came from.

## Profiles and keys

`config/profiles.yaml` pins model **and version** for each role (copied into every manifest):

| Profile | LLM / judge | TTS | STT | Video |
|---|---|---|---|---|
| `dev` | fake | fake (sine) | fake | fake (ffmpeg testsrc) |
| `real-no-video` | Gemini 3.8 Flash | ElevenLabs `eleven_v4` | ElevenLabs Scribe v2 | fake |
| `premium` | Gemini 3.8 Flash | ElevenLabs `eleven_v4` | ElevenLabs Scribe v2 | Seedance 2.5 (BytePlus ModelArk) |
| `premium-openrouter` | Gemini 3.8 Flash | ElevenLabs `eleven_v4` | ElevenLabs Scribe v2 | Seedance 2.5 via OpenRouter (`bytedance/seedance-2.5`) + character image (`google/gemini-2.5-flash-image`) |

**`premium-openrouter`, what the live tests showed (paid clips):**
- Seedance follows our audio only with the audio **and** an image both sent as `input_references`: 2.5 with audio alone **ignored it** (the mouth moved during silence), 2.0-fast refuses audio without an image. So `shot_1` first makes `character.png` (one image per run, ~0.04 USD, reused on resume) for both.
- 2.5 is the default: with the image it lip-synced an English line well (SyncNet confidence 8.6, rated "good" by eye) and a Ukrainian one worse (1.9; 3.0 after the per-phrase alignment, "clearly better"); 2.0-fast's Ukrainian clips scored 1.1–1.7. 2.0-fast stays one line away in `config/profiles.yaml`.
- The character image is always a **stylized 3D cartoon, not a real person**: a photorealistic face was rejected by moderation ("may contain real person"). Animals get a mouth instead of a beak (the beak lip-synced worse).
- OpenRouter takes media only as **public https URLs** (data URIs → 400). Audio and image are uploaded to [uguu.se](https://uguu.se) (`media_host: uguu`; [litterbox](https://litterbox.catbox.moe), the first choice, answered 500 to uploads for hours and stays one line away): free, no account, files deleted after ~3 h — but public to anyone with the random URL while they live. Fine for a test task; in production this would be our own bucket (Cloudflare R2 or S3) with short-lived signed URLs.
- Prices seen: 480p 4 s — 2.0-fast 0.17 USD, 2.5 0.42 USD; 480p 5 s — 2.5 0.52 USD; 720p 5 s — 2.0-fast 0.46 USD (2.5 at 720p not measured); the image 0.04 USD.

**Money cap:** every run stops before a paid image/video call once it has spent `HWVA_MAX_COST_USD` (default 3.0 USD; a profile may set `max_cost_usd`). The manifest's committed cost is what counts; one call can overshoot by its own price. Raise the cap and `--resume` to continue.

Video resolution for `premium-openrouter` is 720p; set `HWVA_VIDEO_RESOLUTION=480p` for cheaper test runs. Full pipeline, cheap:

```bash
HWVA_VIDEO_RESOLUTION=480p uv run video 'Кіт каже: "Я тут головний"' --profile premium-openrouter --no-interactive
```

A clip already paid for is never paid again: `runs/<run_id>/shot_N.task.json` keeps the job id, and `--resume` polls or re-downloads it. Resume with the same `HWVA_VIDEO_RESOLUTION`: with another one the video stage stops and says so rather than paying for a second clip (to switch resolution on purpose, delete `runs/<run_id>/shot_N.task.json`).

Gemini 3.8 Flash TTS is registered as an alternative TTS (one line in the profile). ElevenLabs is the default because its free tier allows ~10k characters/month, Gemini TTS free 10 requests/day. Voices per character persona (gender × age) are in `config/voices.yaml`.

Copy `.env.example` to `.env`:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | script (structured output), soft quality judge, alternative TTS |
| `ELEVENLABS_API_KEY` | TTS and Scribe STT — the verbatim gate |
| `ARK_API_KEY` | Seedance 2.5 on BytePlus ModelArk — video (`premium` only) |
| `OPENROUTER_API_KEY` | Seedance via OpenRouter — video, and the character image (`premium-openrouter` only) |
| `HWVA_VIDEO_RESOLUTION` | optional, `premium-openrouter` only: `480p` or `720p` (default `720p`) |
| `HWVA_MAX_COST_USD` | optional money cap per run in USD (default `3.0`), all profiles |

## Eval

```bash
uv run eval --profile real-no-video                 # 4 fixed ideas -> evals/results/<ts>.csv
uv run eval --compare old.csv new.csv               # accept: gate pass rate not lower, cost ≤ +25 %
```

Rows are written as each idea finishes; provider failures (moderation, timeout, quota) are counted apart from gate failures. Details: `specs/eval.md`.

## SyncNet (the lip-sync gate)

A required dependency: `shot_1` measures the talking shot with SyncNet, decides the verdict (`pass` / `correct` / `weak` / `no_sync` / `error`; confidence thresholds 5.0 and 3.0, provisional, `HWVA_SYNC_CONF` / `HWVA_SYNC_WEAK_CONF`) and picks the audio for shot 1: the whole line shifted by SyncNet's offset, or each phrase shifted on its own (our mouth-motion metric proposes the shifts) — whichever SyncNet scores higher. Never time-stretched. Soft by default; `HWVA_SYNC_REJECT=1` pays for one regeneration on `no_sync`. Details and the human-rated evidence: `specs/sync.md`. ~900 MB (CPU PyTorch, installed by `uv sync`), weights are not on PyPI:

```powershell
uv sync
powershell -ExecutionPolicy Bypass -File scripts\get-syncnet-weights.ps1   # Oxford VGG weights + sha256
```

Weights dir: `$HWVA_SYNCNET_WEIGHTS` (default `weights/`). Weights missing or SyncNet not importable → verdict `error` in `shot_1` (soft, with a warning; the line is placed unshifted) and `syncnet.error` in `quality_check`. No face in the clip (e.g. the dev fakes) → `error` in `shot_1`, skipped in `quality_check`.

Early spike on this laptop (Windows, CPU only), 5.3 s talking-head sample:

| Audio | AV offset (frames) | LSE-C | Time |
|---|---|---|---|
| original | 2 | 4.39 | 17 s |
| delayed 300 ms | −6 | 4.74 | 19 s |

The **offset** moved by 8 frames ≈ 320 ms — it sees the shift. **LSE-C went up** on the broken clip — confidence alone is not a quality signal. That was a single shift on one clip; on 5 human-rated Seedance clips confidence did separate good (8.6) from bad (1.7) lip-sync, so it became the gate (`specs/sync.md`).

## How it works

```
idea → intake → script → voice → shot_1 → shot_2 → assemble → final_check → quality_check
```

- Order is fixed in code; a stage starts only after the previous one passed its gate. The LLM works inside stages, never decides what runs next.
- Voice = TTS of the exact line in a voice matching the character's gender and age; an independent STT checks it word for word before video and again on the final file.
- The video model only animates lips to **our** audio; the final audio track is always our TTS (checked by cross-correlation).
- `final_check` is the hard gate on the finished file; `quality_check` (Gemini judge with timestamps, SyncNet) is soft and never fails a run.
- Models are swappable **within a strategy**. Each provider declares `Capabilities`; at start the run picks how the line gets into the video from them, not from model names (`src/hwva/strategy.py`):

  | Strategy | Video model can… | Verbatim | Status |
  |---|---|---|---|
  | A | take our audio (Seedance 2.5, Wan, OmniHuman) | by construction | **implemented** |
  | B | only speak by itself (Veo, Omni Flash) | probabilistic: STT check, every retry is a paid video | fails fast at start, nothing spent |
  | C | only make silent video (Kling, Runway) | needs a separate lip-sync model | fails fast at start |

  Preference A > B > C is the one hard rule — it orders strategies by how strongly they guarantee the line. No silent fallback to another model.
- Adding a model = one adapter in `src/hwva/providers/` + one line in `_FACTORIES` + one profile line.

## How it was built

The stages and adapters were written by Claude Code agents running headless under a harness (`CLAUDE.md`, `.claude/`); the human set the plan and the specs, reviewed the results and made the decisions:
- `specs/<stage>.md` → `spec-writer` writes tests marked `xfail(strict)` → `implementer` removes the marker and makes them green → `reviewer` checks the diff against the spec without the author's context (`/build-stage`, `/new-provider`).
- Hooks: `path_guard` blocks writes outside the repo and into the harness's own hooks/settings (added after an agent wrote a file to `C:\Users\Public`); lint (+ types for `src/`) after every edit; a Stop hook blocks "done" while `uv run check` is red, releasing as STUCK after 5 blocks; a crashing hook blocks too.
- Architecture rules are tests (`tests/test_architecture.py`), not prompt text.

See `AI-LOG.md` for every stage: review rounds, what the reviewer caught, where AI was wrong and what changed.

How the human led it, in the human's own messages: `docs/ai-chats/01-planner-highlights.md` (Ukrainian).

## Development

```bash
uv run check            # ruff + format + mypy strict + pytest — the definition of done
uv run pytest -m live   # optional live smoke tests against the real APIs (keys in .env)
```
