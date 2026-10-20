"""shot_2 — non-speaking second shot with the same character, referenced on shot 1's last frame.

No audio goes to the video model here: the line is spoken in shot 1 only, and the final audio
track is always our TTS (assemble). A one-shot script skips this stage without spending.

Contract: specs/shot_2.md."""

from __future__ import annotations

import shutil
from pathlib import Path

from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError, VideoRequest
from hwva.schemas import SHOT_2_SKIPPED, Script, ShotData

SCRIPT = "script.json"
VIDEO = "shot_2.mp4"
LAST_FRAME = "shot_1.last.png"  # default name; the shot_1 record's artifact wins if present
LAST_FRAME_KEY = "shot_1_last_frame"
ASPECT = "9:16"
# Words by which a non-retryable provider error is recognised as a moderation rejection.
MODERATION_MARKERS = ("moderation", "face filter", "content policy", "safety", "nsfw")
FALLBACK_NOTE = (
    "Fallback: re-run with a one-shot script (speaking shot only) or a less realistic character."
)


def run(ctx: Context) -> StageOutput:
    script = Script.model_validate_json((ctx.run_dir / SCRIPT).read_text(encoding="utf-8"))
    if len(script.shots) < 2:
        # a stale clip from an earlier two-shot run must not be picked up by assemble
        (ctx.run_dir / VIDEO).unlink(missing_ok=True)
        return StageOutput(artifacts={}, data=dict(SHOT_2_SKIPPED), cost_usd=0.0)

    last_frame = _last_frame_path(ctx)
    if not last_frame.is_file():
        raise GateFailed(
            f"Shot 1 last frame not found at {last_frame}; shot_2 needs it as reference image. "
            "Re-run shot_1."
        )

    shot = script.shots[1]
    request = VideoRequest(
        prompt=_prompt(script.character, shot.visual_prompt, shot.camera),
        duration_s=shot.duration_s,
        aspect_ratio=ASPECT,
        audio_path=None,
        reference_image=str(last_frame),
    )
    out_path = ctx.run_dir / VIDEO
    # a stale clip from an earlier attempt must never pass the clip check below
    out_path.unlink(missing_ok=True)
    ctx.check_cost_cap("the shot_2 video call")
    try:
        result = ctx.providers.video.generate(request, out_path)
    except ProviderError as exc:
        if not exc.retryable and _is_moderation(str(exc)):
            raise GateFailed(
                f"Video provider rejected shot 1's last frame as reference (moderation): {exc}. "
                f"{FALLBACK_NOTE}"
            ) from exc
        raise
    # the clip is paid for: persist its cost before any check that can still fail
    ctx.commit_cost(result.cost_usd)

    source = Path(result.video_path)
    if not source.is_file():
        raise GateFailed(
            f"Video provider clip is missing: returned video_path={result.video_path!r} "
            f"(expected clip for {out_path})"
        )
    if source.stat().st_size == 0:
        raise GateFailed(
            f"Video provider clip is 0 bytes: returned video_path={result.video_path!r} "
            f"(expected clip for {out_path})"
        )
    if source.resolve() != out_path.resolve():
        shutil.copyfile(source, out_path)
    if not (out_path.is_file() and out_path.stat().st_size > 0):
        raise GateFailed(f"Copy of {result.video_path!r} to {out_path} is missing or 0 bytes")

    return StageOutput(
        artifacts={"video": VIDEO},
        data=ShotData(duration_s=shot.duration_s).model_dump(mode="json"),
        cost_usd=result.cost_usd,
    )


def _last_frame_path(ctx: Context) -> Path:
    record = ctx.manifest.stages.get("shot_1")
    name = record.artifacts.get(LAST_FRAME_KEY, LAST_FRAME) if record is not None else LAST_FRAME
    return ctx.run_dir / name


def _is_moderation(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in MODERATION_MARKERS)


# same sentence as shot_1.NO_BRANDS (stages don't import each other)
NO_BRANDS = "No logos, brand names or readable text."


def _prompt(character: str, visual: str, camera: str) -> str:
    return (
        f"{character}. {visual}. Camera: {camera}. "
        f"Vertical {ASPECT} frame. Same character as in the reference image. No speech. "
        f"{NO_BRANDS}"
    )
