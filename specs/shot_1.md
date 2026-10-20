# shot_1

**Purpose.** The speaking shot: the character says OUR audio (strategy A).

**Inputs.** `script.json` (`shots[0]` is the speaking shot, `character`), `shot_audio.wav` (lead-in + `line.wav`), voice data `audio_duration_s`, `lead_in_s`. Providers: `video.generate`; `image.generate` only when the video model `needs_image`.

**Outputs.** Artifacts: `shot_1.mp4`, `shot_1_last_frame` = **local** `shot_1.last.png` (extracted from the local clip); `character_image` = `character.png` when the video model needs an image. `sync` = `sync.json`; `aligned_audio` = `aligned_line.wav` (the chosen shot-1 audio track, see "Lip-sync check"; absent on verdict `error`). Data: `duration_s`, `sync`. Cost via `StageOutput.cost_usd` (image + clip of this attempt).

**Behaviour.**
- Requires `video.capabilities.accepts_audio_input` (preflight already checked; re-check here for resume) else `GateFailed`.
- `capabilities.needs_image` (Seedance 2.5 / 2.0 / 2.0-fast via OpenRouter, which ignore or refuse audio without an image): the reference image is `character.png` in the run dir.
  - It is created once with `image.generate(prompt, run_dir/character.png)` when no non-empty `character.png` exists; an existing one (an earlier attempt, a resume) is reused: no second image is paid for. A zero-byte file is replaced.
  - The prompt is built from `script.json`'s `character` and ALWAYS asks for a stylized 3D animated cartoon look, clearly not a real person, head and shoulders, mouth closed, looking at the camera, vertical 9:16 (moderation rejects photoreal people: live, `InputImageSensitiveContentDetected.PrivacyInformation`). An animal character (keyword match, e.g. cat, dog, bird, parrot) is asked for as an anthropomorphic cartoon animal with an expressive, human-like mouth, no beak (a beak lip-synced worse live).
  - Its cost is committed to the manifest right after the call, before the video call.
  - No `image` provider in the profile → `GateFailed` naming the `image:` role, nothing paid. The image provider leaves no non-empty file → `GateFailed`, no video call.
  - The request's `reference_image` = local `character.png`; the video prompt says the character is the one in the reference image.
- Money cap (`ctx.max_cost_usd`: env `HWVA_MAX_COST_USD`, else the profile's `max_cost_usd`, else 3.0 USD per run): before every paid call (image, video) the committed `manifest.cost_usd` is checked; if it already reached the cap → `GateFailed` "cost cap reached" naming the cap value and `HWVA_MAX_COST_USD`, no call. Prices are known only after a call, so one call can overshoot the cap by at most its own price. A resume of a clip that is already paid also hits the cap: raise it to continue.
- Prompt = character + `shots[0].visual_prompt` + camera, 9:16, plus "No logos, brand names or readable text" (live: a coffee-shop background grew Starbucks-like bags). The character image prompt carries the same sentence.
- Duration = `max(shots[0].duration_s, lead_in_s + audio_duration_s)` (≤ max, guaranteed by `voice`; see "Lip-sync check").
- The provider downloads the clip into the run dir immediately (URLs expire in 24 h).
- The clip's cost is written to the manifest (stage record `cost_usd` + `manifest.cost_usd`, saved to disk) right after generation, before any frame step — a later failure never loses it. Counted once: success does not add it a second time.
- A missing or zero-byte `shot_1.mp4` after generation → `GateFailed` (not retried, no second paid generation).
- The last frame is extracted from the local `shot_1.mp4` with ffmpeg (`-sseof`, single frame) into `shot_1.last.png` — never re-downloaded or re-generated. The provider's `return_last_frame` is not required (resume must work tomorrow from local files only).
- Frame extraction fails (non-zero exit, missing or zero-byte PNG) → `GateFailed`: the paid clip exists, a retry would pay again.
- Every ffmpeg/ffprobe call has a timeout (120 s renders/analysis, 30 s probes); `subprocess.TimeoutExpired` → `GateFailed`.
- `ProviderError(retryable)` from `video.generate` → retried once (`max_attempts=2`). Moderation errors are `retryable=False`.
- Never pay twice for one clip (video provider, Seedance; `<clip>.task.json` next to the clip, written atomically: temp file + `os.replace`):
  - Before the create POST a `create_pending` marker (fingerprint + UTC time) is saved. A create whose outcome is unknown (the request was sent: read/write timeout, broken connection, 408, 5xx, non-JSON body, a 200 answer without a task id) → `ProviderError(retryable=False)`: the task may exist, check the ModelArk console; delete `<clip>.task.json` to allow a new paid task. Only a connect error/timeout (nothing sent) and HTTP 429 stay retryable.
  - A later `generate` that finds a pending marker without a task id for the same request raises the same non-retryable error and does not POST. A task id the human adds to that file is resumed.
  - A pending marker or task id saved for another request is kept under `abandoned` when the file is reused, so a possibly paid task never disappears without a trace.
  - A succeeded task without a video URL is paid for: non-retryable error, the task id is kept, so a rerun re-checks it instead of paying for a new task.
  - A `<clip>.task.json` that exists but cannot be parsed → `ProviderError(retryable=False)` naming the file; it is never read as "no task".
  - After a download the file records status `downloaded` with the clip's sha256 and size. A later `generate` of the same request whose local clip matches them returns it with no API call and `cost_usd=0`; only otherwise the clip is deleted and the task resumed (or a new one created).
  - A clip's cost is reported once: by the call that first downloads it. A resumed task that was already downloaded reports `cost_usd=0`.
  - Polls (GET) are idempotent: 429/5xx/timeouts/connection errors are retried in place with capped exponential backoff + jitter until the poll deadline, not raised.

**Lip-sync check (v2: SyncNet decides, phrases aligned; method, thresholds and evidence in `sync.md`).**
- The audio sent is voice's `shot_audio` artifact (lead-in + line). A run from before the lead-in (no `shot_audio` artifact) sends `line.wav` with `lead_in_s = 0`.
- Duration = `max(shots[0].duration_s, lead_in_s + audio_duration_s)` (= voice's `shot_audio_duration_s` when recorded).
- Before any paid call: `sync.box_from_env()` and `sync.thresholds_from_env()`; an invalid value → `GateFailed`.
- After the clip is downloaded and its last frame extracted, the raw SyncNet measurement `sync.syncnet_measure(clip, shot_audio)` gives the verdict (`pass` / `correct` / `weak` / `no_sync` / `error`, `sync.md` "Verdict"); a `SyncError` (incl. no face, SyncNet or weights missing) → verdict `error`, recorded, the stage still succeeds. The motion metric `sync.measure` is recorded as `motion` (diagnostic; its failure leaves `motion` null and changes nothing).
- `weak` / `no_sync` / `error` → `warning` in the record; the stage succeeds (soft, default).
- `HWVA_SYNC_REJECT=1` and verdict `no_sync` on take 1 → ONE regeneration: `ctx.check_cost_cap` first (cap reached → `GateFailed` "cost cap reached", no call), then the same request into a NEW clip path `shot_1.take2.mp4` (a new path = a new provider task file, so the provider cannot hand back the rejected clip). Its cost is committed right after the call. Its last frame is extracted to `shot_1.last.png` and it is measured; take 1's record goes to `rejected`. Take 2 not `no_sync` → artifacts `video = shot_1.take2.mp4`. Take 2 `no_sync` → `sync.json` is written with both takes, then `GateFailed` "no lip-sync after 2 takes" naming both SyncNet confidences. Each regeneration is a paid clip; at most one per stage run. `weak` is never rejected.
- Alignment (`sync.md` "Alignment in shot_1") runs on the KEPT take only, for every verdict except `error`: global candidate `sync_global.wav`, piecewise candidate `sync_piecewise.wav`, the chosen one copied to `aligned_line.wav` (artifact `aligned_audio`). Piecewise is kept only when its SyncNet confidence ≥ the global one. All of it is local (no paid call); any failure inside it falls back to global, and a failure of the global render itself → no `aligned_audio`, `method` null, a `warning`, the stage still succeeds (assemble then uses `lead_in_s + shift_s`).
- Resume of a stage that failed mid-take-2: take 1 is reused by the provider (no new payment), measured again (deterministic, same verdict), take 2 resumes its own task.
- `error` is never rejected, also with the switch on.

**Acceptance criteria.**
- Request carries the audio path (`shot_audio.wav`), 9:16, duration = max(shot, lead-in + audio).
- Lip-sync (SyncNet stubbed by monkeypatching `hwva.sync.syncnet_measure`, answers keyed by the audio file name; real ffmpeg on synthetic media): the stage writes `sync.json` and data `sync` with `verdict`, `syncnet_confidence`, `syncnet_offset_frames`, `shift_s`, `thresholds`, `method`, `confidence_global`, `confidence_piecewise`, `phrases`, `aligned_audio`; `sync.json` equals data `sync`.
- Raw confidence 8.6, offset −6 → verdict `correct`, `shift_s == -0.24`; `sync_global.wav` places the speech 0.24 s earlier than in `shot_audio.wav` (cross-correlation ±10 ms).
- Piecewise confidence ≥ global → `method == "piecewise"` and `aligned_line.wav` is byte-identical to `sync_piecewise.wav`; piecewise < global → `method == "global"`, `aligned_line.wav` identical to `sync_global.wav`; both confidences recorded either way. Equal → piecewise.
- Verdict `pass` (|offset| ≤ 1) or `no_sync` → `shift_s == 0` and the global confidence is the raw one: `syncnet_measure` is NOT called on `sync_global.wav`.
- The piecewise SyncNet run raising `SyncError` → `method == "global"`, `piecewise_error` set, DONE.
- Phrases of a `shot_audio.wav` with a 0.7 s lead-in and two pauses → three recorded phrases with `start_s`/`end_s` in clip time and a `shift_s` each; `aligned_line.wav` lasts the clip's duration (±0.05 s).
- `weak` (confidence 4.0) → DONE, `warning` set, shift applied (offset/25); `no_sync` (1.7) → DONE with a `warning`, `video.generate` called once, alignment still recorded.
- `syncnet_measure` raising `SyncError` / `NoFaceError` on the raw run → DONE, verdict `error`, `warning` set, `method` null, no `aligned_audio` artifact, also with `HWVA_SYNC_REJECT=1` (no second call).
- `HWVA_SYNC_CONF=abc` (or weak > sync) → `GateFailed` naming the variable, `video.generate` not called.
- `HWVA_SYNC_REJECT=1`, take 1 `no_sync`, take 2 `correct` → `video.generate` called twice, the second with out path `shot_1.take2.mp4`; artifact `video` is the take-2 clip; `sync.take == 2`, `sync.rejected.verdict == "no_sync"`; the alignment was computed on the take-2 clip; both clip costs in the manifest.
- `HWVA_SYNC_REJECT=1`, both takes `no_sync` → FAILED with `GateFailed` naming both SyncNet confidences, `video.generate` called twice, `sync.json` has both takes.
- `HWVA_SYNC_REJECT=1`, take 1 `weak` → one call, DONE (weak is never rejected).
- `HWVA_SYNC_REJECT=1`, take 1 `no_sync` and cost cap reached → `GateFailed` "cost cap reached", `video.generate` called once.
- Switch off (unset or `0`) and `no_sync` → one call, DONE.
- A run without a `shot_audio` artifact → the request carries `line.wav`, duration = max(shot, audio).
- Model without audio input → `GateFailed`; `needs_image` without an `image` provider → `GateFailed`, no image or video call.
- `needs_image`: `image.generate` is called once with a prompt that has the character, "stylized 3D", "cartoon", "not a real person", "head and shoulders", "mouth closed", "looking at the camera"; the video request's `reference_image` is the local `character.png`; artifact `character_image`; the image cost is in the stage cost and the saved manifest.
- Animal character → the image prompt asks for an anthropomorphic cartoon animal with an expressive mouth; a human character's prompt does not.
- `needs_image` with a `character.png` already in the run dir → no image call; a zero-byte one is regenerated.
- Image cost is committed before the video call: a video `ProviderError` after it still leaves the image cost in the saved manifest; the pipeline retry reuses `character.png` (image called once in total).
- Image provider leaves no file → `GateFailed`, video not called.
- Cost cap: committed cost ≥ cap → `GateFailed` naming the cap, no image or video call; below the cap → runs. The cap comes from `HWVA_MAX_COST_USD`, else profile `max_cost_usd`, else 3.0; an unparsable or non-positive value is a usage error.
- `needs_image` false (Seedance 2.5, fakes) → no image call, request has no reference image.
- Last frame artifact is an existing local file.
- Non-retryable provider error → FAILED after 1 attempt.
- Last frame is extracted by ffmpeg (`-sseof`, one frame) from the local `shot_1.mp4`; works when the provider returns no `last_frame`; `video.generate` is called once.
- Clip cost is in the saved manifest (stage record and total) when a later frame step fails; on success the total equals the clip cost (not doubled).
- Missing or zero-byte clip → FAILED after 1 attempt with `GateFailed`, `video.generate` called once.
- Frame extraction fails or yields a missing/zero-byte PNG → FAILED after 1 attempt with `GateFailed`, `video.generate` called once.
- ffmpeg/ffprobe calls pass `timeout` (120 s render, 30 s probe); `subprocess.TimeoutExpired` → `GateFailed`, `video.generate` called once.
- Seedance on a mocked HTTP transport: first attempt times out while polling, the pipeline retry resumes the task → exactly one POST.
- Seedance with a price: shot_1 fails on frame extraction after the cost commit, then the run is resumed → `manifest.cost_usd` equals one clip, one POST in total.
- Seedance create: ReadTimeout / WriteTimeout / RemoteProtocolError / 408 / 5xx / non-JSON / 200 without task id after send → non-retryable, message names the ModelArk console and `<clip>.task.json`; ConnectError / ConnectTimeout / 429 → retryable. A pending marker for the same request → non-retryable, no POST; with a task id added by hand → resumed, no POST. Another request's pending marker → kept under `abandoned`.
- Seedance: a succeeded task without a video URL → non-retryable; a rerun polls the same task, no POST.
- Seedance: unparsable `<clip>.task.json` → non-retryable error naming the file, no POST.
- Seedance: verified local clip after the video URL expired → no API call, same file, `cost_usd=0`; a local clip that doesn't match the recorded sha256 is replaced.
- Seedance poll: a transient error is retried in place (no new attempt, no POST).
- Seedance: `ARK_BASE_URL` that is not `https://` → `ProviderError` when the provider is built (the key must never go over plain HTTP; `uv run video` reports it as a usage error); a task's video URL that is not `https://` → non-retryable `ProviderError`, nothing downloaded.
