# shot_2

**Purpose.** Non-speaking second shot with the same character.

**Inputs.** `script.json` (`shots[1]`, if any), local `shot_1.last.png`.

**Outputs.** `shot_2.mp4`. Data: `duration_s`, or `{"skipped": "one-shot script"}`.

**Behaviour.**
- One-shot script → no artifacts, DONE, data says skipped. A stale `shot_2.mp4` from an earlier run is removed, so `assemble` never picks it up.
- `reference_image` = shot 1's local last frame (also when shot_1 used `character.png`: the last frame keeps the cut continuous). No audio. Duration = `shots[1].duration_s`.
- Money cap as in `shot_1`: committed `manifest.cost_usd` ≥ `ctx.max_cost_usd` before the video call → `GateFailed` "cost cap reached" naming the cap, no call. A one-shot script is skipped before the check (it spends nothing).
- Reference image rejected by the provider's face filter (moderation, `retryable=False`) → `GateFailed` whose message names the fallback (one-shot video); never the raw `ProviderError`.
- The clip's cost is written to the manifest (stage record `cost_usd` + `manifest.cost_usd`, saved to disk) right after generation, before any check that can still fail. Counted once: success does not add it a second time.
- A missing or zero-byte `shot_2.mp4` after generation → `GateFailed` with a clear message (not retried, no second paid generation).
- The stage makes no ffmpeg/ffprobe calls. Any added later must have a timeout (120 s renders/analysis, 30 s probes); `subprocess.TimeoutExpired` → `GateFailed`.
- Same retry rules as `shot_1`, including the video provider's "never pay twice for one clip" rules (`<clip>.task.json` = `shot_2.task.json`).

**Acceptance criteria.**
- One-shot script → skipped, DONE.
- Request carries shot 1's last frame as reference and no audio.
- The prompt says "No logos, brand names or readable text" (same rule as shot_1).
- Missing last-frame file → `GateFailed` (not a crash).
- One-shot script with a stale `shot_2.mp4` in the run dir → skipped, DONE, the file is gone.
- Moderation rejection → `GateFailed` (type asserted) whose message mentions the one-shot fallback; FAILED after 1 attempt.
- Missing or zero-byte clip → FAILED after 1 attempt with `GateFailed`, `video.generate` called once.
- Clip cost is in the saved manifest (stage record and total) when the clip check fails; on success the total equals the clip cost (not doubled).
- The stage calls no subprocess (a `subprocess.run` that raises `TimeoutExpired` does not affect it).
- Cost cap reached → `GateFailed` naming the cap, `video.generate` not called; one-shot script with the cap reached → still skipped, DONE.
