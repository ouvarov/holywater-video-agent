# voice

**Purpose.** Voice the exact line and prove it's verbatim — before any video money is spent. One stage: synthesize → check durations → STT gate, with retries. Independent of which TTS produced the audio.

**Inputs.** `params["line"]` (source of truth, no digits), `params["lang"]`, `script.json` (shot durations). Providers: `tts.synthesize`, `stt.transcribe`, `video.capabilities`.

**Outputs.** Artifacts: `line.wav`, `transcript.txt`. Data: `audio_duration_s`, `planned_total_s`, `attempts`, `diffs` (one list per failed attempt).

**Behaviour.**
1. Synthesize exactly `params["line"]` — no prompt wrapping that could be spoken.
2. Durations (ffprobe), else `GateFailed` naming the limit and the value (not retried — the line itself doesn't fit):
   - `audio >= video.capabilities.min_audio_s`;
   - speaking shot `= max(shots[0].duration_s, audio)` must be `<= video.capabilities.max_duration_s`;
   - `planned_total_s = speaking shot + other shots` must be `<= 30`.
3. `diff_words(line, transcript, lang)`; empty → pass.
4. Mismatch → re-synthesize and re-check, up to 4 attempts total. Each attempt's diff goes to `data["diffs"]`.
5. Still mismatching → `GateFailed` with the last diff.

**Acceptance criteria.**
- TTS receives exactly the line (`SpyTts`).
- Verbatim match passes on attempt 1; `data["attempts"] == 1`.
- `ScriptedStt` mishears once, then hears correctly → passes on attempt 2, `data["diffs"]` has 1 entry.
- 4 mismatches → `GateFailed` whose message contains the diff.
- Stress marks / apostrophe variants / case / punctuation / STT writing numbers as digits are not mismatches.
- Truncated transcript ("Я тут" for "Я тут головний") fails.

**Numbers heard as digits.**
- STT may write a number we spoke as digits. Only whole tokens that are known numeral forms count as number words: explicit uk lists with inflections (`один/одна/одне/одного…`, `два/дві/двох…`, `п'ять/п'яти…`, `тисяча/тисячі…`) and en `zero`..`billion`, `hundred`, `and` inside a number. No stem/prefix matching: «одразу», «двері», «стою», «сім'я» are ordinary words.
- A run of number words is parsed to its integer value; a run of digit tokens too (thousands groups "25 000" join; digits followed by scale words — "2 тисячі", "5 million" — are one number). The two runs match only if the values are equal ("двадцять п'ять" = "25"). Otherwise the mismatch is reported as `expected 'дві', heard '5'`.
- Only decimal digits count as digits; "²", "①" are ordinary words (a mismatch, never a crash).
- Known limits (fail safe: a mismatch, never a false match): fractions («півтори» heard as "1,5"), ordinals («двадцять п'ятому» heard as "2025"), years read in pairs ("nineteen eighty four" as "1984"). Two separate numbers heard as one digit group ("2 100" for «два, сто») are a mismatch too.

**Acceptance criteria (numbers).**
- «дві гривні» vs «2 гривні», «двадцять п'ять» vs «25», «twenty five cats» vs «25 cats» → no diff.
- «дві гривні» vs «5 гривні», «сім хвилин» vs «5 хвилин» → mismatch naming both («дві»/«5»).
- «Тримай двері» vs «3 двері» → mismatch (an ordinary word is never swallowed into a number); same for «одразу», «стою», «сім'я»; en «tent»/«tennis»/«android».
- «дві тисячі гривень» vs «2 тисячі гривень», «five million» vs «5 million» → no diff; «дві тисячі» vs «3 тисячі» → mismatch.
- Each known limit above → a mismatch.

**Provider errors inside the attempt loop.**
- A retryable `ProviderError` (429 / 5xx / timeout) from TTS or STT counts as one attempt and is retried with backoff (2 s, doubling per provider error in a row), within the same 4 attempts in total. A non-retryable `ProviderError` (moderation, bad key) is raised at once. The last attempt decides what is raised: a mismatch → `GateFailed` with its diff; a provider error → `ProviderError` whose message also carries the last STT diff, if an earlier attempt had one (the pipeline does not retry `voice` again).

**Acceptance criteria (provider errors).**
- TTS raises a retryable `ProviderError` once, then succeeds → passes on attempt 2.
- STT raises a retryable `ProviderError` on every attempt → `ProviderError` after 4 TTS calls.
- A non-retryable `ProviderError` from TTS → raised after 1 call, no retry.
- Mismatch, mismatch, then retryable errors → the waits restart at 2 s (errors in a row), and the final `ProviderError` names the last diff.
- ffprobe of the audio has a 30 s timeout; a timeout or a non-zero exit → `GateFailed`.
- Audio shorter than `min_audio_s` → `GateFailed`; audio + second shot > 30 s → `GateFailed` (before any video call).

**Lead-in for the video model (see `sync.md`).**
- `lead_in_s` = env `HWVA_LEAD_IN_S` (seconds, 0 ≤ x ≤ 2), default 0.7; an unparsable or out-of-range value → `GateFailed` naming the variable, before any TTS call. Recorded in data `lead_in_s`; downstream stages read the recorded value, never the env (resume keeps it).
- The STT verbatim gate is unchanged: it runs on `line.wav` (pure TTS, no lead-in).
- After the gate passes, `shot_audio.wav` = `lead_in_s` of digital silence + `line.wav` (ffmpeg, same sample rate, mono, 120 s timeout; failure → `GateFailed`). Artifact `shot_audio`; data `shot_audio_duration_s` (ffprobe). This is the audio sent to the video model; `line.wav` stays the audio of the final video.
- Duration math uses the SENT audio, `sent = lead_in_s + audio_duration_s`: `sent >= min_audio_s`; speaking shot `= max(shots[0].duration_s, sent) <= max_duration_s`; `planned_total_s = speaking shot + other shots <= 30`.

**Acceptance criteria (lead-in).**
- Default: `data["lead_in_s"] == 0.7`; `shot_audio.wav` is ~0.7 s longer than `line.wav` (±0.05 s), its first 0.7 s are silent (peak < −60 dBFS) and the rest has the TTS tone; STT was called on `line.wav`.
- `HWVA_LEAD_IN_S=0` → `shot_audio.wav` has the length of `line.wav`; `HWVA_LEAD_IN_S=abc` / `-1` / `3` → `GateFailed` naming `HWVA_LEAD_IN_S`, no TTS call.
- A line whose audio fits `max_duration_s` alone but not with the lead-in → `GateFailed` (speaking shot over the maximum), before any video call; `planned_total_s` includes the lead-in.

**Voice persona.**
- `TtsProvider.synthesize(text, lang, out_path, persona: VoicePersona)` — the persona comes from `script.json` (`Script.voice`).
- Which concrete voice/style a persona maps to is config, not code: `config/voices.yaml`, keyed by TTS provider name, then persona key `<gender>-<age>` → `{voice: str, style: str | null}`. Example for `gemini-tts`: `male-adult: {voice: Puck, style: "a cheerful man in his thirties"}`, `female-child: {voice: Leda, style: "an excited 7-year-old girl"}`. The style goes to the provider's style field (Gemini: `speech_metadata.style`), NEVER into the spoken text.
- Fallback when a persona is missing for a provider: same gender `adult`, then any entry; the stage records the assumption "no <persona> voice for <provider>, used <key>". A provider with no entries at all → `GateFailed` before synthesis.
- Data adds `voice: {persona, key, voice, style}` (what was actually used).
- `FakeTts` maps gender/age to the sine frequency (male-adult 140 Hz, female-adult 220 Hz, teen +60 Hz, child +160 Hz, elderly −20 Hz) so a test can tell voices apart by pitch.
- Seed `config/voices.yaml` with: `fake` (all 8 personas), `gemini-tts` (Puck/Aoede adults, Fenrir male-teen, Leda female-teen and female-child, Puck male-child with child style, Charon male-elderly, Aoede female-elderly with elderly styles) — voices verified by a listening test. A default TTS of the real profiles (`premium`, `real-no-video`) is `elevenlabs-tts` (premade voices, `style: null`, no child/elderly entries → same-gender adult fallback); `gemini-tts` stays as an alternative.

**Acceptance criteria (voice persona).**
- The TTS receives the persona from `script.json`; the style is never part of `text` (SpyTts records text == line exactly).
- female-adult and male-adult give different FakeTts pitch (dominant frequency check).
- Missing persona → fallback used and assumption recorded; empty provider entry → `GateFailed` before any TTS call.
- `data["voice"]` names the key, voice and style used.

**Voice mapping is pinned per run.**
- The persona is resolved before the first attempt; `data["voice"]` is written on every exit, including the `GateFailed` path after all attempts (the failed manifest record still has `persona`, `key`, `voice`, `style`).
- On success the resolved mapping (`key`, `voice`, `style`) is stored in the stage data; on resume the stage reuses it instead of reading `config/voices.yaml` again, so editing the config later can't change the voice of an existing run. A `voice` record that already has `data["voice"]` (from a success or from the failed path above) is the pin.
- So the pin reaches the provider, the stage passes the resolved mapping to the TTS (`synthesize(text, lang, out_path, persona, voice=<resolved key/voice/style>)`); adapters use it and don't re-read `config/voices.yaml` when it's given.
- `config/voices.yaml` that can't be parsed (bad YAML / wrong shape) → `GateFailed` naming the file; a missing file → `GateFailed` saying "file not found" (with the path), not "no voices for provider".

**Acceptance criteria (pinned mapping).**
- 4 mismatches → `GateFailed`, and the manifest record for `voice` still has `data["voice"]` with `persona`, `key`, `voice`, `style`.
- Success, then `voices.yaml` changed (different voice for the same key), then the stage runs again on the same manifest (resume) → TTS receives the originally resolved voice/style; `data["voice"]` unchanged. Same for a run whose `voice` stage failed after 4 attempts.
- Malformed `voices.yaml` → `GateFailed` whose message names the file; missing `voices.yaml` → `GateFailed` containing "file not found", not "no voices for".
