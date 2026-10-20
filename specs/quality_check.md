# quality_check

**Purpose.** Soft quality signals on the finished video. **Never blocks.** Runs last. Video-LLMs are largely insensitive to temporal errors, so the judge is not trusted on lip-sync, verbatim or technical facts — those are `final_check`'s job.

**Inputs.** `final.mp4`, `script.json` (character, shot prompts), number of shots. Optional providers: `ctx.providers.judge` (Gemini in `premium`, `None` in `dev`). SyncNet on the final (see below).

**Outputs.** `quality.json`. Data: same.

**Judge.** `JudgeProvider.judge(video_path, script) -> JudgeVerdict` (pydantic, add to `schemas.py`):
- `face_present`, `mouth_moves_when_speaking`, `matches_character`, `same_character_across_cut`, `matches_prompt`: each `{"verdict": "pass"|"fail"|"unsure", "score": 1-5, "evidence": str, "timestamp": "MM:SS" | null}`;
- `defects`: list of `{"kind": str, "timestamp": "MM:SS", "note": str}`.
- Real provider: video sent inline at fps=2 with audio, request timeout 300 s (a live 9 s final took 95 s and once exceeded the old 120 s), temperature 0, a timestamp asked for on every criterion and defect, JSON schema enforced; the rubric text with anchors for 1/3/5 lives in the provider.
- Post-processing (here, deterministic): defects whose timestamp is beyond the video duration are dropped and counted (`hallucinated_defects`).

**SyncNet on the final (required dependency, weights in `$HWVA_SYNCNET_WEIGHTS`, default `weights/` — `syncnet_v2.model`, `sfd_face.pth`, via `scripts/get-syncnet-weights.ps1`).** Runs through the shared runner `hwva.sync.syncnet_measure(final.mp4, audio)` (see `sync.md`; it imports `syncnet_python` lazily and owns the quiet/bounded ffmpeg mechanism). Records `av_offset_frames`, `confidence` (LSE-C), `min_dist` (LSE-D) and seconds taken. Never a gate here (the gate is in `shot_1`). ~15 s per clip on CPU. No face (`NoFaceError`, e.g. the dev fakes) → `{"skipped": reason}`. SyncNet not importable or weights missing is now an **error** (`syncnet.error`, naming what is missing), not the normal skipped path: it is a required dependency.

**Lip-sync numbers (v2, see `sync.md`).** `quality.json` has `sync`: shot_1's recorded `sync`, copied, never re-measured: `syncnet_confidence`, `syncnet_offset_frames`, `verdict`, `method` (`global` | `piecewise` | null), `shift_s`, `phrases` (`[{start_s, end_s, shift_s}]`), `confidence_global`, `confidence_piecewise`, `take`; or `{"skipped": "no sync measurement"}` when shot_1 has none. A v1 record (only `lag_s`/`peak_r`) → the new fields null, never an error. Never blocks. SyncNet on the final gets the audio as it is in `final.mp4` (final_check's `stt_audio` artifact when present, else assemble's artifact `audio`, else `line.wav`), so the line's placement doesn't show up as a lip-sync error.

**Acceptance criteria (sync).** shot_1 data with a v2 `sync` record → `quality.json["sync"]` has the same `syncnet_confidence`/`syncnet_offset_frames`/`verdict`/`method`/`phrases` shifts/both candidate confidences; a v1 record → copied without error; none → `skipped`; a stub SyncNet receives the `stt_audio` path when final_check recorded one, else assemble's `audio` artifact.

**Behaviour.** No judge → `{"judge": {"skipped": "no judge in profile"}}`. Judge raises or returns invalid JSON → error recorded, stage still DONE.

**Acceptance criteria** (fakes in `tests/conftest.py`: `FakeJudge`, `CrashingJudge`, `InvalidJudge`, `HallucinatingJudge`).
- Never raises `GateFailed`; always writes `quality.json`.
- Missing judge → skipped record; crashing / invalid judge → error recorded, DONE.
- Out-of-range defect timestamps are dropped and counted.
- SyncNet not importable → `syncnet.error` (changed: required dependency); weights missing → `syncnet.error` naming the files; no face → skipped record.
- SyncNet never fails the stage: any exception while importing it (`ImportError`, `OSError`, `RuntimeError` — e.g. a broken torch DLL on Windows), while running it, or while reading its result → `syncnet.error`, stage DONE. A result in numpy arrays instead of lists is either read into a correct metric or recorded as `syncnet.error` — never a stage failure.
- Video length unreadable (ffprobe fails and assemble recorded none) → defects kept as returned and `judge.filter_skipped` says why the hallucinated-timestamp filter did not run.
- Exactly one of `skipped` / `error` / `verdict` is set in `judge`; exactly one of `skipped` / `error` / the metric (`av_offset_frames`…) in `syncnet`.
- Every ffmpeg/ffprobe subprocess has a timeout: 30 s for probes, 120 s for renders/analysis. `subprocess.TimeoutExpired` is recorded as an error in the affected record (judge length → `filter_skipped`, SyncNet → `syncnet.error`); the stage still never raises.
- Console stays clean: SyncNet's ffmpeg runs with `-loglevel error`; the `pyscenedetect` logger is set to CRITICAL (it logs deprecations at ERROR) only for the duration of the SyncNet call (restored afterwards). A stage run with a stub SyncNet writes nothing to stdout/stderr.
