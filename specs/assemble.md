# assemble

**Purpose.** Glue shots and put OUR verified TTS on top.

**Inputs.** `shot_1.mp4`, optional `shot_2.mp4`, `line.wav`, shot_1's `aligned_line.wav` when recorded.

**Outputs.** `final.mp4`; artifact `audio` = the audio file placed. Data: `duration_s`, `audio_offset_s`, `audio_offset_wanted_s`, `audio_source`, `speech_start_s`.

**Behaviour.**
- Concatenate shots in order (re-encode to a common format: H.264, yuv420p, 24 fps, 720x1280, AAC).
- Drop any audio the video model produced; the only audio track is our TTS — shot_1's `aligned_line.wav` when recorded (see "Aligned audio"), else `line.wav` — starting at `audio_offset_s` (see "Lip-sync correction"; 0 for older runs), padded with silence to the video length. Speech never runs past the video end (`voice` + `shot_1` guarantee shot 1 ≥ audio); if it does, `GateFailed`. Trailing silence past the video end may be cut.
- Never time-stretch video or audio.

**Lip-sync correction (see `sync.md`).** The line no longer starts at 0: it starts at `audio_offset_s = max(0, lead_in_s + shift_s)` (`lead_in_s` from voice data, default 0 for older runs; `shift_s` from shot_1 data `sync`, 0 when absent). That is the place of the line in the audio the model saw, plus the measured mouth lag. Silence is added at the start (`adelay`); the end is padded or trimmed to the video length as before. Speech end in the final = `audio_offset_s + speech end of line.wav` must be ≤ the video end. If the wanted offset would push speech past the video end, the offset is REDUCED to `video_s − speech end` (the line is never cut, nothing is paid again; a re-run of shot_1 would likely measure the same lag). Only speech that doesn't fit even at offset 0 → `GateFailed` (as before). Data adds `audio_offset_s` (applied, rounded to ms) and `audio_offset_wanted_s` (`lead_in_s + shift_s`, unclamped); `final_check` reads `audio_offset_s` from here. Whenever applied ≠ wanted (reduced to fit, or a negative shift clamped at 0), `manifest.assumptions` gets one line naming both values and the reason ("shot 1 too short" / "negative shift larger than the lead-in").

**Aligned audio (v2, see `sync.md`).** When shot_1 recorded an `aligned_audio` artifact (`aligned_line.wav`: the chosen global or piecewise track, already in clip time with the lead-in and every phrase shift baked in, lasting the shot_1 duration), it is THE audio of the final: placed at `audio_offset_s = 0`, padded with silence to the video length (shot 2 gets silence), no `adelay` of `lead_in_s + shift_s` on top (the shift is already in the file). Without the artifact (older runs, verdict `error`, a failed render) the correction above applies unchanged, with `shift_s` from the record (negative = earlier).
- The line is never cut: the speech end of the aligned track (silencedetect `n=-40dB d=0.3`) must be ≤ the video end. If not, the track is moved EARLIER by the overflow — its leading samples are trimmed, `audio_offset_s = −overflow` — as long as the trimmed part is silence (overflow ≤ the first speech start in the track); an assumption names both values and the reason ("shot 1 too short"). Overflow larger than the first speech start → `GateFailed`. Never stretched, never re-ordered.
- Data adds `audio_source` (`"aligned"` | `"line"`), `speech_start_s` (first speech in the final = first speech start in the placed file + `audio_offset_s`, rounded to ms; the line path: `audio_offset_s` + first speech start of `line.wav`, i.e. `audio_offset_s` for a TTS without leading silence). `audio_offset_s` may be negative only for `"aligned"`. Artifacts add `audio` = the file actually placed (`aligned_line.wav` or `line.wav`); `final_check` reads it.
- `audio_offset_wanted_s` = 0 for `"aligned"`.

**Acceptance criteria (aligned audio).**
- shot_1 records `aligned_audio` = a synthetic track with two tone phrases at 1.0–1.8 s and 2.3–3.0 s → DONE, `audio_source == "aligned"`, `audio_offset_s == 0`, artifact `audio == "aligned_line.wav"`; in `final.mp4` the audio cross-correlates with `aligned_line.wav` at 0 ± 50 ms (peak ≥ 0.9) and the first 1.0 s are silent; `speech_start_s == 1.0` (±0.05); duration = shot total (±0.2 s); no `lead_in_s + shift_s` delay added (a record with lead-in 0.7 and shift −0.24 changes nothing).
- An aligned track whose speech ends 0.3 s past the video end with ≥ 0.3 s of leading silence → DONE, `audio_offset_s == -0.3` (±1 ms), assumption names it, speech complete in the final, `final_check` passes; overflow larger than the leading silence → `GateFailed`.
- No `aligned_audio` artifact → the line path, same output as before; `audio_source == "line"`, artifact `audio == "line.wav"`.
- The aligned final passes `final_check` (end to end, real ffmpeg, fakes).

**Acceptance criteria (correction).**
- lead-in 0.7 + shift 0.5 → `data["audio_offset_s"] == 1.2`; in `final.mp4` the first 1.2 s are silent (silencedetect) and the TTS tone follows (cross-correlation of the final audio with `line.wav` peaks at 1.2 s ± 50 ms); the duration still equals the shot total (±0.2 s).
- No voice `lead_in_s` and no `sync` record (an older run) → offset 0, the same output as before.
- Negative shift larger than the lead-in (lead-in 0.7, shift −0.9) → offset 0, never a trimmed line; `audio_offset_wanted_s == -0.2`; an assumption names it.
- Speech that fits at offset 0 but runs past the video end after the wanted offset → DONE, `audio_offset_s == video_s − speech end` (±1 ms), `audio_offset_wanted_s` unchanged, an assumption naming both values; the speech is complete in `final.mp4` and `final_check` passes on it.
- Speech that does not fit even at offset 0 → `GateFailed` (unchanged).
- Offset applied = wanted → no assumption added.
- The shifted final passes `final_check` (end to end, real ffmpeg, fakes).

**Acceptance criteria** (real ffmpeg; fixtures from `tests/conftest.py`).
- Exactly one video and one audio stream.
- Duration = sum of shot durations (±0.2 s).
- The audio comes from `line.wav`, not from the shots: build shots WITH their own audio at a different tone (`make_video(..., audio=True, tone=880)`, TTS fixture at 440 Hz) and assert the final audio's dominant frequency is ~440 Hz.
- The audio is never trimmed, but what is compared with the video length is the END OF SPEECH in `line.wav` (end of the last non-silent segment, ffmpeg `silencedetect n=-40dB d=0.3`), not the file length. Trailing silence past the video end may be cut: trailing silence that runs past the video end → OK and the final duration still equals the shot total (±0.2 s); speech that runs past the video end by any amount → `GateFailed` (no tolerance).
- A malformed `script.json` → `GateFailed`.
- Missing ffmpeg binary → `GateFailed`.
- Every ffmpeg/ffprobe subprocess call has a timeout (120 s for renders/analysis, 30 s for probes); `subprocess.TimeoutExpired` → `GateFailed`. Tool output is decoded as UTF-8 with `errors="replace"` (a non-UTF-8 byte in ffmpeg's stderr never crashes the stage).
- The number of shots comes from `script.json` (`len(Script.shots)`); a two-shot script with `shot_2` missing (no record, no video artifact, or no file) → `GateFailed`, never a one-shot render.
- A corrupt shot (zero-byte, garbage, truncated) → `GateFailed`, no `final.mp4` left behind.
- The rendered `final.mp4` is re-probed: not exactly 1 video + 1 audio stream, or duration off the shot total by more than 0.2 s → `GateFailed`, no `final.mp4` left behind.
