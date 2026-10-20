# final_check

**Purpose.** Prove the deliverable is right **without anyone watching it**. Only deterministic, provable checks — each threshold is proven by a test on a synthetic bad file.

**Inputs.** `final.mp4`, the placed audio (assemble artifact `audio`, default `line.wav`), `params["line"]`, lang, number of shots. Provider: `stt`.

**Outputs.** `final_report.json`: every check with its raw value, threshold and pass/fail; transcript; verdict. Data: same summary.

**Hard gates** (any failure → `GateFailed`, message points to the report):
| Check | How | Threshold |
|---|---|---|
| format | ffprobe | 8 ≤ duration ≤ 30 s; height > width; exactly 1 video + 1 audio stream |
| no black frames | `blackdetect=d=0.1:pix_th=0.10` | total black ≤ 1.0 s |
| speech starts at once | `silencedetect=n=-40dB:d=0.4` | leading silence ≤ 0.5 s + `speech_start_s` |
| audio survived muxing | `ebur128` integrated loudness | ≥ −35 LUFS |
| **audio is OUR TTS** | decode final audio and the placed reference (assemble artifact `audio`: `aligned_line.wav` or `line.wav`) to mono 16 kHz, normalized cross-correlation (numpy) | peak ≥ 0.9 at \|offset − `audio_offset_s`\| ≤ 50 ms |
| line survived | extract audio → `stt` → `diff_words` vs line | no diff (digits from STT match only a run of number words with the same value, see voice.md "Numbers heard as digits"); on mismatch retry STT once (video is already paid, STT is not deterministic) |

`audio_offset_s` (see `sync.md`) = where assemble placed the line: assemble's data `audio_offset_s` (lead-in + recorded lip-sync shift), 0 when absent (older runs). Recorded in both checks' values (`allowed_s` / `expected_offset_ms`). The allowance is exactly that offset, no more: a file whose speech starts later than offset + 0.5 s still fails, and audio placed at 0 when the record says 1.2 s fails the match.

**Aligned audio (v2, see `sync.md`, `assemble.md`).** The audio match compares against the audio actually placed: assemble's artifact `audio` (`aligned_line.wav` — phrases moved one by one — or `line.wav`; absent → `line.wav`) at assemble's `audio_offset_s` (may be negative for an aligned track trimmed to fit: the reference is then compared from its trimmed start). The leading-silence allowance follows the first phrase's position: `speech_start_s` from assemble's data (absent → `audio_offset_s`, older runs). Both values recorded in the checks (`reference`, `allowed_s`, `expected_offset_ms`). The STT verbatim gate is unchanged and still proves the words survived the alignment.
- Acceptance: an aligned track with phrases at 1.0–1.8 s and 2.3–3.0 s recorded (artifact `audio`, `audio_offset_s` 0, `speech_start_s` 1.0) and a final made from it → `leading_silence` (1.0 s ≤ 1.5) and `audio_match` pass; the same record and a final made from the contiguous `line.wav` placed at 1.0 s (phrases NOT moved; the test track's second phrase is moved by ≥ 0.4 s so the shapes differ) → `audio_match` fails; the same record and a final with 1.7 s of leading silence → `leading_silence` fails; a record without `audio` artifact → compared with `line.wav` exactly as before (the existing synthetic failing files still fail).
- Acceptance: a record with `audio_offset_s == -0.3` and a final built from the aligned track with its first 0.3 s trimmed → `audio_match` passes.


**Soft signals** (recorded, never block): `freezedetect` (talking heads are nearly static → false positives), true peak, `scdet` cut count vs number of shots.

**Acceptance criteria.**
- Valid file (fakes) passes and writes the report with raw values.
- Each gate is proven by a synthetic failing file: too short / horizontal / no audio; 2 s of black; 1 s of leading silence; silent audio; audio from a different source (other tone); `ScriptedStt` mishearing twice. Each → `GateFailed`, and the report names the failed check.
- With no `audio_offset_s` recorded, the synthetic failing files above fail exactly as before (offset 0).
- `audio_offset_s = 1.2` and a final whose line starts at 1.2 s → `leading_silence` and `audio_match` pass; the same final with no offset recorded → both fail; offset 1.2 recorded and a final with 2.0 s of leading silence → `leading_silence` fails; offset 1.2 recorded and the line placed at 0 → `audio_match` fails.
- The match is searched up to the recorded offset even when the final is shorter than offset + `line.wav` (assemble cut `line.wav`'s trailing silence): the final is zero-padded for the search. Test: `line.wav` = 2.5 s speech + 0.6 s trailing silence, offset 1.2, a 4.0 s final whose line starts at 1.2 s → `audio_match` passes (peak ≥ 0.9, offset 1200 ± 50 ms).
- STT mismatch once then match → passes (retry).
- STT hearing a different number as digits («5 гривні» for «дві гривні») twice → `line` check failed (changed: digits used to match any number).
- Soft signals present in the report but never fail the stage.
- STT raising counts as a failed attempt (max 2 STT calls in total, raise or mishear). Raise then match → passes. Raises twice → `line` check failed with the error text in its value; `final_report.json` is still written with every other measurement, verdict `fail` → `GateFailed`.
- Missing or malformed `script.json` does not stop the stage before the report: only `scene_cuts` (the one check that needs the shot count) records `shots: null` + the error; being soft, it does not change the verdict.
- The extracted audio STT heard (`final_audio.wav`) is listed in the stage's artifacts next to the report: `{"report": "final_report.json", "stt_audio": "final_audio.wav"}`.
- Missing or unknown lang (`params["lang"]` not `uk`/`en`) is an error, never a silent default to Ukrainian: STT is not called, the `line` check records the error, the report is still written, verdict `fail` → `GateFailed`.
- Every ffmpeg/ffprobe call has a timeout (120 s for renders/analysis, 30 s for probes). `subprocess.TimeoutExpired` → the affected check records the error, the report is still written, verdict `fail` → `GateFailed`.

**Dependency.** numpy (cross-correlation) — reason: the "audio is our TTS" gate.
