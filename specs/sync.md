# sync (not a stage): lip-sync gate (SyncNet) and phrase alignment

**Purpose.** The video model animates the mouth to OUR audio, but not exactly on time, and not
equally well per language. Decide whether the talking shot lip-syncs at all, and build the audio
track for shot 1 that sits best under the mouth. Not a pipeline stage: it runs inside `shot_1`
(the stage that owns the clip and its one paid regeneration); `assemble` places the chosen track;
`final_check`, `quality_check` and `eval` read the recorded numbers.
Code: `src/hwva/sync.py`. Tests: `tests/stages/test_sync.py` plus the touched stages' own test
files.

**v2.** v1 used our own motion metric as the gate and the lag. Replaced:
SyncNet decides and gives the global shift; our motion metric only proposes per-phrase shifts.

## Evidence (LIVE, human-rated clips; n is tiny, every threshold is PROVISIONAL)

| clip | SyncNet confidence | SyncNet offset (frames) | our motion peak r | human verdict |
|---|---|---|---|---|
| run 1, Ukrainian | 1.71 | −7 | 0.23 | not good |
| Seedance 2.5, Ukrainian, piecewise | 2.95 | −1 | 0.36 | clearly better |
| Seedance 2.5, English, raw | 8.63 | −6 | 0.35 | good |
| Seedance 2.5, English, piecewise | 8.86 | +1 | 0.46 | very good |
| model's own voice, Ukrainian | 5.40 | −1 | 0.45 | in sync |

Other Ukrainian clips (not rated): confidence 1.06, 1.07, 1.85. Ukrainian lip-syncs worse than
English with Seedance; piecewise alignment helps Ukrainian somewhat.

- **Why SyncNet is the gate.** On the good vs bad clip, SyncNet confidence was 8.6 vs 1.7; our
  motion peak r was 0.35 vs 0.23. Our metric separates good from bad far worse, so it has no
  verdict and no threshold any more (v1's `MIN_PEAK_R` / `PASS_LAG_S` / `MAX_LAG_S` are removed).
- **Why our metric still exists.** SyncNet returns ONE offset per clip (verified:
  `inference()` returns `offsets=[-6]`, `confs=[8.63]`, `dists=[one float]`, `max_conf`,
  `min_dist`, `faces` (str), `has_face`). Per-phrase timing needs a per-frame signal: our motion
  energy vs audio envelope gives one.
- **Sign of the global shift.** The English clip with offset −6 had the mouth LEADING the audio by
  ~0.25–0.29 s; moving our audio 0.27 s EARLIER changed the offset to +1. So
  `shift_s = offset_frames / 25` (SyncNet works at 25 fps), **negative = audio earlier**.
- Piecewise was approved by eye by the human on the clips above.

## SyncNet — `hwva.sync.syncnet_measure(video, audio) -> SyncNetResult`

- The ONE place that runs SyncNet (`shot_1` and `quality_check` both call it). Required dependency
  (`syncnet-python`, `torch`, `torchvision` in `[project] dependencies`, CPU index); weights in
  `$HWVA_SYNCNET_WEIGHTS` (default `weights/`): `syncnet_v2.model`, `sfd_face.pth`
  (`scripts/get-syncnet-weights.ps1`).
- `syncnet_python` is imported lazily inside the function (`importlib.import_module`), so tests
  replace it via `sys.modules`. `SyncNetPipeline({"s3fd_weights": …, "syncnet_weights": …},
  device="cpu").inference(video_path=…, audio_path=…)`. Its ffmpeg runs quiet and bounded and the
  `pyscenedetect` logger is silenced for the duration of the call only (the `_BoundedFfmpeg` /
  `_quiet_syncnet` mechanism moves here from `quality_check`, unchanged).
- `SyncNetResult(offset_frames: int, confidence: float, min_dist: float, seconds: float)`:
  the offset at the highest confidence in `confs`, `confidence = max_conf`, `min_dist`, wall time.
  Lists or numpy arrays/scalars are both read correctly.
- Every failure raises `SyncError` (a `RuntimeError`) with a readable message: import fails (any
  exception, e.g. a broken torch DLL), a weight file missing (message names the missing files and
  `scripts/get-syncnet-weights.ps1`), video or audio missing, the call raises, an odd result.
  No face (`has_face` false or no offsets) → `NoFaceError` (a subclass of `SyncError`).
- ~15 s per clip on CPU. Unit tests never run the real model: the stub goes in `sys.modules` or
  `hwva.sync.syncnet_measure` is monkeypatched.

## Verdict — `hwva.sync.verdict(confidence, offset_frames, thresholds=…) -> SyncVerdict`

Taken on the RAW measurement: SyncNet on (the clip, the audio sent to the model = `shot_audio.wav`).

| verdict | rule | global `shift_s` |
|---|---|---|
| `pass` | confidence ≥ `sync_conf` and \|offset\| ≤ 1 frame | 0.0 |
| `correct` | confidence ≥ `sync_conf` and \|offset\| > 1 frame | offset / 25 |
| `weak` | `weak_conf` ≤ confidence < `sync_conf` | offset / 25 (corrected, plus a warning) |
| `no_sync` | confidence < `weak_conf` | 0.0 (the offset of a clip without sync means nothing) |
| `error` | `syncnet_measure` raised (`SyncError`, incl. no face) | 0.0 |

- Defaults `SYNC_CONF = 5.0`, `WEAK_CONF = 3.0`, `SYNCNET_FPS = 25`, `PASS_OFFSET_FRAMES = 1`.
  **Provisional** (n=8 clips, 5 human-rated, table above). Configurable:
  `HWVA_SYNC_CONF` (default 5.0) and `HWVA_SYNC_WEAK_CONF` (default 3.0), read by
  `hwva.sync.thresholds_from_env() -> Thresholds(sync_conf, weak_conf)`; unset / empty /
  whitespace = default; not a number, negative, or `weak_conf > sync_conf` → `ValueError` naming
  the variable. In `shot_1` an invalid value → `GateFailed` before any paid call. The thresholds
  used are recorded.
- The verdict is on the raw (unrounded) confidence; the record keeps 3 decimals.
- `no_sync` is SOFT by default: recorded plus a warning, never a rejection. `error` is never a
  rejection (a local tool failure must not pay for a new clip).
- **Reject switch (default off).** Env `HWVA_SYNC_REJECT=1` makes `no_sync` a gate in `shot_1`:
  ONE regeneration (see `shot_1.md`); a second `no_sync` → `GateFailed`. Each regeneration is a new
  paid clip, preceded by `ctx.check_cost_cap`, so bounded by the run's cost cap. Any other value
  = off.

## Motion metric (proposes per-phrase shifts only, never decides)

Kept from v1, numpy + ffmpeg: mouth box crop (default `MOUTH_BOX` x 0.20–0.80, y 0.42–0.60, env
`HWVA_SYNC_MOUTH_BOX="x0,x1,y0,y1"`, invalid → `ValueError`, in `shot_1` → `GateFailed` before any
paid call), scaled to 96×48 gray at 24 fps; `motion[t]` = mean |frame t − frame t−1|
(`motion[0] = 0`); `envelope[t]` = RMS of the mono audio over hop [t/24, (t+1)/24) s; z-score of a
constant series = zeros. Every ffmpeg call has a 120 s timeout; timeout / non-zero exit / < 2
frames / empty audio → `SyncError`.

`hwva.sync.measure(video, audio, *, box) -> MotionResult(lag_s, peak_r, r_at_0)`: the v1
cross-correlation over L in −24..+24 frames (positive L = mouth after the audio, smallest |L| on a
tie), 3 decimals. **Diagnostic only** — recorded as `motion`, no verdict, no shift.

## Phrases — `hwva.sync.phrases(audio) -> list[Phrase]`

- `Phrase(start_s, end_s)`: the speech runs of the audio sent to the model (`shot_audio.wav`, so in
  clip time, lead-in included), split at real pauses: ffmpeg `silencedetect=n=-40dB:d=0.15`
  (constants `PHRASE_NOISE_DB = -40`, `PHRASE_MIN_PAUSE_S = 0.15`). A run = the complement of the
  silences inside [0, audio duration]. Every run is a phrase, however short (only runs < 5 ms,
  `MIN_RUN_S`, rounding artefacts, are dropped: found a 30 ms burst lost), in
  time order. Numbers in ffmpeg's log may be in exponent form (`1e-05`).
- Silent audio → `[]`. ffmpeg failure / timeout → `SyncError`.

## Per-phrase shifts — `hwva.sync.choose_shifts(motion, envelope, phrases, clip_s) -> list[float]`

Pure numpy (24 fps frames; `motion` and `envelope` as above, `motion` z-scored over the whole clip).
For each phrase, LEFT TO RIGHT:
- Phrase frames `[a, b)` with `a = floor(start_s·24)`, `b = ceil(end_s·24)`; `e` = z-score of
  `envelope[a:b]` (the phrase alone).
- Candidate shifts `k/24` s for k in −12..+12 (`MAX_PHRASE_SHIFT_S = 0.5`, step 1/24 s).
- Score(k) = `mean(e · z(motion)[a+k : b+k])`.
- Constraints: the shifted phrase stays inside the clip (`start_s + shift ≥ 0`,
  `end_s + shift ≤ clip_s`, and the frame window inside `motion`); order kept and the gap to the
  previous PLACED phrase ≥ `MIN_GAP_S = 0.12` s (`start_s + shift ≥ prev_end + prev_shift + 0.12`).
- Pick the feasible k with the highest score; a tie → the smallest |k|, then the negative one.
- No feasible k for a phrase → `SyncError("piecewise: no feasible shift for phrase i")` (the
  caller then keeps the global alignment).
- Shifts rounded to 3 decimals in the record (the render uses `k/24` exactly).

## Render — `hwva.sync.render(audio, phrases, shifts, duration_s, out) -> None`

- Reads `audio` at its own sample rate, mono (PCM 16-bit in, PCM 16-bit WAV out; samples copied,
  never resampled or time-stretched).
- Each phrase is cut with `PAD_S = 0.08` s of padding each side (clamped to the file), a
  `FADE_S = 0.015` s linear fade-in at the cut start and fade-out at the cut end, only over the
  padding actually present (none where the cut meets the file edge), so the speech run's own
  samples are never faded, and ADDED (mixed) into a silent
  track of `duration_s` at `cut start + shift`. Parts falling outside [0, duration_s] are dropped
  (speech never does: `choose_shifts` keeps it inside).
- No time-stretch ever: every phrase's speech samples in `out` equal the source samples exactly.
- The global alignment is the same render with every phrase shifted by the global `shift_s`
  (`render_global`: clamped so the first speech starts ≥ 0 and the last ends ≤ `duration_s`, the
  start winning; the applied value is the recorded `shift_s`, never `-0.0`). Speech that cannot
  fit even then → `SyncError` (never a silent cut of the verbatim line: `shot_1` then records no
  aligned track and `assemble`'s own gate names the problem).
- ffmpeg / IO failure → `SyncError`.

## Alignment in `shot_1` (after the take is kept)

1. Raw: `syncnet_measure(clip, shot_audio)` → verdict. `error` → stop here: no alignment, no
   aligned audio, `assemble` falls back to `lead_in_s` (shift 0).
2. Global candidate: render `sync_global.wav` (duration = the clip's VIDEO-stream duration, what
   `assemble` checks speech against) with the global `shift_s`. If `shift_s == 0` the candidate
   is `shot_audio.wav` itself as one block (unchanged samples, padded to the clip), so its
   confidence really is the raw one (no second run); else the phrases are moved by the shift and
   `syncnet_measure(clip, sync_global.wav)` scores it. No phrases found → the line as one block
   from the lead-in (so a negative shift can still use the lead-in), the reason recorded.
3. Piecewise candidate: `phrases(shot_audio)`, motion + envelope of (clip, shot_audio) over the WHOLE clip (`motion_series`: the envelope padded with silence to the clip; cut to the shorter audio, the last phrase had no room on live Ukrainian clips),
   `choose_shifts`, render `sync_piecewise.wav`, `syncnet_measure(clip, sync_piecewise.wav)`
   (skipped when `confidence_global` is null: nothing to compare with, ~15 s saved).
   `choose_shifts` is greedy left to right; an early phrase moved late can leave no room for the
   next one → infeasible → global (a lookahead would be the next step).
4. Keep piecewise ONLY if both confidences are numbers and `confidence_piecewise ≥
   confidence_global`; otherwise global. Any `SyncError` in step 3 (no phrases, infeasible,
   render, SyncNet) → global, the reason in `piecewise_error`. A `SyncError` in step 2's SyncNet
   run → global with `confidence_global = null`, piecewise not chosen, reason recorded.
5. The chosen file is copied to `aligned_line.wav` (artifact `aligned_audio`), covering the
   shot_1 duration; `assemble` uses it for shot 1.

Runs for `pass`, `correct`, `weak` and `no_sync` (piecewise may still help a weak clip). With the
reject switch, only the kept take is aligned. Deterministic given the same SyncNet answers; in
practice a DONE stage is never recomputed on resume. SyncNet is repeatable on the same files (three
clips re-run gave identical scores), but the same clip muxed into an AAC preview scored 8.63 vs
8.38 on the raw files, so a piecewise win by a few tenths may be within that encoding noise; the `>=` rule is kept because the human preferred piecewise by eye on every rated clip.
`shift_s` in the record is the GLOBAL candidate's shift: applied only when `method == "global"`
(or without an aligned track); piecewise uses the per-phrase shifts.

## Record

`sync.json` in the run dir and `shot_1` data `sync` (same content, `hwva.schemas.SyncReport`):
`{verdict, syncnet_confidence, syncnet_offset_frames, syncnet_min_dist, shift_s, thresholds:
{sync_conf, weak_conf}, method, confidence_global, confidence_piecewise, piecewise_error,
phrases: [{start_s, end_s, shift_s}], aligned_audio, motion: {lag_s, peak_r, r_at_0}, box, take,
error, warning, rejected}`
- `syncnet_*`: the raw measurement (null on `error`). `shift_s`: the global shift applied (see
  the Verdict table; 0 on `pass` / `no_sync` / `error`).
- `method`: `"global"` | `"piecewise"`, null when no alignment ran (`error`, or a rejected take).
- `phrases`: the piecewise proposal (`shift_s` per phrase), `[]` when not computed.
- `aligned_audio`: `"aligned_line.wav"` or null. `motion`: diagnostic, null if it failed.
- `take`: 1, or 2 after a regeneration; `rejected`: the earlier take's record (without
  `rejected`) or null. `error`: the `SyncError` text on verdict `error`, else null.
- `warning`: one line for `weak`, `no_sync`, `error`, else null; ONE builder,
  `hwva.sync.warning_text(verdict, confidence, thresholds, error=None)`, e.g.
  "weak lip-sync (SyncNet confidence 3.4 < 5)", "lip-sync not detected (SyncNet confidence 1.71 <
  3)", "lip-sync measurement error: <text>".
- All new fields have defaults: a v1 record (`lag_s`, `peak_r`, `r_at_0`, `shift_s`, `box`,
  `verdict` in pass/correct/no_sync/error) still validates (resume of older runs).
- The cli prints `warning: <text>` to stderr once at the end of a run whose `shot_1` record has a
  `sync.warning` (unchanged).

## Consumers

- `assemble`: shot 1's audio = `aligned_line.wav` when recorded, else `line.wav` at
  `lead_in_s + shift_s` (older runs, `error`). See `assemble.md`.
- `final_check`: audio match against the audio actually placed; leading-silence allowance from the
  first speech position. See `final_check.md`.
- `quality_check`: copies the numbers into `quality.json["sync"]` (see `quality_check.md`).
- `eval`: CSV columns (see `eval.md`).

## Acceptance criteria (`tests/stages/test_sync.py`; synthetic media, real ffmpeg, SyncNet stubbed)

- `verdict` boundaries (pure): confidence 5.0 / 4.999 / 3.0 / 2.999 with the defaults; offset
  ±1 → `pass`, ±2 → `correct`; custom thresholds are honoured. Global `shift_s` = offset/25 for
  `correct`/`weak` (offset −6 → −0.24), 0 otherwise.
- `thresholds_from_env`: defaults; valid overrides; `abc`, `-1`, weak > sync → `ValueError` naming
  the variable; empty/whitespace = default.
- `syncnet_measure` with a stub `syncnet_python`: the 7-tuple (lists and numpy) → the offset at the
  max confidence, `max_conf`, `min_dist`; no face → `NoFaceError`; import failure, missing weights
  (message names the files and the script), a raising `inference`, an odd result → `SyncError`.
- Motion metric (`measure`): mouth lagging by 0.5 s → `lag_s` 0.5 ± 0.05; leading → ≈ −0.5;
  in sync → |lag| < 0.08; the crop is really used; static video / silent audio → no exception;
  `MotionResult` has no verdict. ffmpeg failure / timeout / unreadable file → `SyncError`; every
  ffmpeg call passes `timeout`. `HWVA_SYNC_MOUTH_BOX` parsed, invalid → `ValueError`.
- `phrases`: tone runs separated by 0.3 s and 0.2 s pauses (plus a 0.7 s silent lead-in) → three
  phrases at the right times (±0.03 s); a 0.1 s gap is not a split; silence → `[]`.
- `choose_shifts`: a motion series following each phrase's envelope delayed by its own amount
  (e.g. +0.25 s, −0.17 s, 0) → those shifts (±1/24); order kept and gap ≥ 0.12 s even when the
  best unconstrained shifts would overlap; shifts within ±0.5; a phrase near the clip end is never
  pushed past it; flat motion → all 0.
- `render`: speech samples of every phrase are identical to the source at the new position (no
  stretch); 80 ms padding each side and 15 ms linear fades on the cut edges; output length =
  `duration_s` at the source sample rate, mono; nothing outside [0, duration_s].
- Test isolation: an autouse fixture in `tests/conftest.py` makes the real SyncNet unreachable in
  every test (e.g. `syncnet_python` → `None` in `sys.modules`) unless the test is marked
  `@pytest.mark.syncnet`; the dev-profile pipeline tests therefore record verdict `error` (or a
  stub answer) and never spend ~15 s per run.
- ONE local test marked `@pytest.mark.syncnet` (no network, never `live`) runs the real
  `syncnet_measure` on a real clip + audio from disk, given by env `HWVA_SYNCNET_TEST_CLIP` and
  `HWVA_SYNCNET_TEST_AUDIO`; it SKIPS when they are unset, the files or the weights
  are absent (clips are never committed) and asserts an `int` offset, a finite confidence and
  `verdict(...)` in the verdict set. It runs in `uv run check` only where the files exist.
