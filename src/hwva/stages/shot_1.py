"""shot_1 — the speaking shot: the video model animates the character to OUR audio (strategy A).

The provider downloads the clip into the run dir (its URLs expire); the last frame is extracted
from that local clip with ffmpeg, so resume (and shot_2's reference image) works tomorrow from
local files only.

A video model that needs an image with audio (Seedance 2.0 / 2.0-fast) gets `character.png`,
made once by the image provider as a stylized cartoon (moderation rejects photoreal people) and
reused on retry/resume. Every paid call is preceded by the run's money cap check.

Lip-sync (specs/sync.md, v2): SyncNet measures the clip against the audio that was sent
(`shot_audio.wav`, lead-in + line) and decides; the kept take is then aligned: the whole line moved
by SyncNet's offset (`sync_global.wav`) or each phrase moved on its own (`sync_piecewise.wav`),
whichever SyncNet scores higher, copied to `aligned_line.wav`. The record goes to `sync.json` and
data `sync`. With `HWVA_SYNC_REJECT=1` a `no_sync` take 1 is regenerated ONCE into
`shot_1.take2.mp4` (a new paid clip).

Contract: specs/shot_1.md."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from hwva import manifest as mf
from hwva import sync
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import VideoRequest
from hwva.schemas import (
    Script,
    ShotData,
    SyncMotion,
    SyncPhrase,
    SyncReport,
    SyncThresholds,
    VoiceData,
)

SCRIPT = "script.json"
AUDIO = "line.wav"
VIDEO = "shot_1.mp4"
VIDEO_TAKE_2 = "shot_1.take2.mp4"  # the one lip-sync regeneration: a new path = a new task
SYNC_JSON = "sync.json"
SYNC_GLOBAL = "sync_global.wav"
SYNC_PIECEWISE = "sync_piecewise.wav"
ALIGNED = "aligned_line.wav"
LAST_FRAME = "shot_1.last.png"
CHARACTER_IMAGE = "character.png"
ASPECT = "9:16"
# script.json's `character` is English (the script prompt asks for it)
_ANIMAL = re.compile(
    r"\b(animal|cat|kitten|kitty|dog|puppy|bird|parrot|owl|crow|raven|duck|goose|chicken|"
    r"rooster|hen|penguin|eagle|pigeon|fox|wolf|bear|rabbit|bunny|hare|hamster|mouse|rat|"
    r"squirrel|raccoon|horse|pony|cow|bull|pig|goat|sheep|lamb|frog|toad|monkey|ape|gorilla|"
    r"lion|tiger|panda|koala|hedgehog|otter|beaver|capybara|deer|elephant|giraffe|zebra|"
    r"camel|kangaroo|sloth|turtle|lizard|snake|dinosaur|dragon|fish|shark|dolphin|whale)"
    r"(s|es)?\b",
    re.IGNORECASE,
)
FFMPEG_TIMEOUT_S = 120  # renders/analysis (ffprobe calls would get 30 s; this stage makes none)
TAIL_S = 1.0  # seek this far before the end, then decode through to the last frame
STDERR_TAIL_LINES = 3


def run(ctx: Context) -> StageOutput:
    caps = ctx.providers.video.capabilities
    # preflight checked this already; re-check because resume may run with another provider
    if not caps.accepts_audio_input:
        raise GateFailed("Video model does not accept audio input (strategy A needs it)")
    if caps.needs_image and ctx.providers.image is None:
        raise GateFailed(
            "Video model needs a reference image with audio, but the profile has no image "
            "provider (add an `image:` role)"
        )

    script = Script.model_validate_json((ctx.run_dir / SCRIPT).read_text(encoding="utf-8"))
    voice_record = ctx.manifest.stages["voice"]
    voice = VoiceData.model_validate(voice_record.data)
    shot = script.shots[0]
    shot_audio = voice_record.artifacts.get("shot_audio")
    if shot_audio:
        audio_path = ctx.run_dir / shot_audio
        sent_s = voice.shot_audio_duration_s or voice.lead_in_s + voice.audio_duration_s
    else:  # a run from before the lead-in: line.wav was (and is) the audio sent
        audio_path, sent_s = ctx.run_dir / AUDIO, voice.audio_duration_s
    duration_s = max(shot.duration_s, sent_s)
    try:
        box = sync.box_from_env()  # before any paid call: a typo must not cost a clip
    except ValueError as exc:
        raise GateFailed(f"Invalid lip-sync mouth box: {exc}") from exc
    try:
        thresholds = sync.thresholds_from_env()
    except ValueError as exc:
        raise GateFailed(f"Invalid lip-sync threshold: {exc}") from exc
    reject = os.environ.get(sync.REJECT_ENV) == "1"

    artifacts = {"video": VIDEO, "shot_1_last_frame": LAST_FRAME, "sync": SYNC_JSON}
    image_cost = 0.0
    reference_image: str | None = None
    if caps.needs_image:
        image_cost = _character_image(ctx, script.character)
        reference_image = str(ctx.run_dir / CHARACTER_IMAGE)
        artifacts["character_image"] = CHARACTER_IMAGE

    request = VideoRequest(
        prompt=_prompt(script.character, shot.visual_prompt, shot.camera, reference_image),
        duration_s=duration_s,
        aspect_ratio=ASPECT,
        audio_path=audio_path,
        reference_image=reference_image,
    )
    clip_cost = _take(ctx, request, VIDEO, "the shot_1 video call")
    kept = ctx.run_dir / VIDEO
    report, raw = _measure(ctx, kept, audio_path, box, thresholds, take=1)

    if reject and report.verdict == "no_sync":
        first = report
        # each regeneration is a new paid clip: bounded by the run's money cap
        clip_cost += _take(ctx, request, VIDEO_TAKE_2, "the shot_1 lip-sync regeneration")
        kept = ctx.run_dir / VIDEO_TAKE_2
        report, raw = _measure(ctx, kept, audio_path, box, thresholds, take=2, rejected=first)
        if report.verdict == "no_sync":
            raise GateFailed(
                f"no lip-sync after 2 takes: SyncNet confidence {_conf(first)} (take 1) and "
                f"{_conf(report)} (take 2), below {thresholds.weak_conf:g} ({sync.REJECT_ENV}=1)"
            )
        artifacts["video"] = VIDEO_TAKE_2

    if raw is not None:  # verdict `error` -> no alignment
        lead_in_s = voice.lead_in_s if shot_audio else 0.0
        report = _align(ctx, kept, audio_path, lead_in_s, box, report, raw)
        if report.aligned_audio:
            artifacts["aligned_audio"] = ALIGNED

    return StageOutput(
        artifacts=artifacts,
        data=ShotData(duration_s=duration_s, sync=report).model_dump(mode="json"),
        cost_usd=image_cost + clip_cost,
    )


def _take(ctx: Context, request: VideoRequest, name: str, call: str) -> float:
    """One paid clip into `name` in the run dir, its cost committed at once, its last frame
    extracted to `shot_1.last.png`. Returns the clip's cost."""
    out_path = ctx.run_dir / name
    ctx.check_cost_cap(call)
    result = ctx.providers.video.generate(request, out_path)
    # the clip is paid for: persist its cost before any step that can still fail
    ctx.commit_cost(result.cost_usd)

    source = Path(result.video_path)
    if source.is_file() and source.resolve() != out_path.resolve():
        shutil.copyfile(source, out_path)
    if not _non_empty(out_path):
        raise GateFailed(f"Video provider left no clip at {out_path} (missing or zero bytes)")
    # the provider's last_frame (path or URL) is ignored: the frame comes from OUR local clip
    _extract_last_frame(out_path, ctx.run_dir / LAST_FRAME)
    return result.cost_usd


def _measure(
    ctx: Context,
    clip: Path,
    audio: Path,
    box: sync.Box,
    thresholds: sync.Thresholds,
    *,
    take: int,
    rejected: SyncReport | None = None,
) -> tuple[SyncReport, sync.SyncNetResult | None]:
    """The raw SyncNet measurement of one take -> its verdict, recorded at once (sync.json + the
    stage record's data, so a failure later in the stage keeps it). A SyncError is verdict
    `error` (result None): recorded, never a rejection. Our motion metric goes along as `motion`
    (diagnostic; its failure leaves it null)."""
    limits = SyncThresholds(sync_conf=thresholds.sync_conf, weak_conf=thresholds.weak_conf)
    motion: SyncMotion | None = None
    try:
        found = sync.measure(clip, audio, box=box)
        motion = SyncMotion(lag_s=found.lag_s, peak_r=found.peak_r, r_at_0=found.r_at_0)
    except sync.SyncError:
        pass
    base = SyncReport(
        verdict="error", thresholds=limits, motion=motion, box=list(box), take=take,
        rejected=rejected,
    )  # fmt: skip
    result: sync.SyncNetResult | None
    try:
        result = sync.syncnet_measure(clip, audio)
    except sync.SyncError as exc:
        result = None
        text = _one_line(exc)
        report = base.model_copy(
            update={
                "error": text,
                "warning": sync.warning_text("error", None, thresholds, error=text),
            }
        )
    else:
        decided = sync.verdict(result.confidence, result.offset_frames, thresholds=thresholds)
        report = base.model_copy(
            update={
                "verdict": decided,
                "syncnet_confidence": round(result.confidence, 3),
                "syncnet_offset_frames": result.offset_frames,
                "syncnet_min_dist": round(result.min_dist, 3),
                "shift_s": round(sync.global_shift(decided, result.offset_frames), 3),
                "warning": sync.warning_text(decided, result.confidence, thresholds),
            }
        )
    _record(ctx, report)
    return report, result


def _align(
    ctx: Context,
    clip: Path,
    audio: Path,
    lead_in_s: float,
    box: sync.Box,
    report: SyncReport,
    raw: sync.SyncNetResult,
) -> SyncReport:
    """sync.md "Alignment in shot_1": the global and the piecewise candidate; the one SyncNet
    scores higher (piecewise on a tie) is copied to aligned_line.wav. All local, never a paid
    call; any piecewise failure falls back to global; a failed global render leaves no aligned
    audio (assemble then places line.wav at lead-in + shift)."""
    run_dir = ctx.run_dir
    errors: list[str] = []
    try:
        clip_s = sync.media_duration(clip, video=True)  # what assemble checks speech against
        audio_s = sync.media_duration(audio)
        try:
            found = sync.phrases(audio)
        except sync.SyncError as exc:
            found = []
            errors.append(_one_line(exc))
        if report.shift_s == 0 or not found:
            # no shift: the global candidate is the audio the model saw, unchanged (padded to
            # the clip), so the raw SyncNet score is really its score. No phrases: the line as
            # one block after the lead-in, so a shift can still use the lead-in.
            start_s = 0.0 if report.shift_s == 0 else min(lead_in_s, audio_s)
            blocks = [sync.Phrase(start_s=start_s, end_s=audio_s)]
        else:
            blocks = found
        applied = sync.render_global(audio, blocks, report.shift_s, clip_s, run_dir / SYNC_GLOBAL)
    except sync.SyncError as exc:
        note = f"lip-sync alignment failed ({_one_line(exc)}); line placed at lead-in + shift"
        warning = f"{report.warning}; {note}" if report.warning else note
        report = report.model_copy(update={"warning": warning})
        _record(ctx, report)
        return report

    confidence_global: float | None = None
    if report.shift_s == 0 and applied == 0:
        confidence_global = raw.confidence  # the raw placement: no second SyncNet run
    else:
        try:
            confidence_global = sync.syncnet_measure(clip, run_dir / SYNC_GLOBAL).confidence
        except sync.SyncError as exc:
            errors.append(f"global SyncNet: {_one_line(exc)}")

    confidence_piecewise: float | None = None
    phrase_records: list[SyncPhrase] = []
    try:
        if not found:
            raise sync.SyncError("piecewise: no phrases found in the audio")
        if confidence_global is None:  # nothing to compare with: save the ~15 s SyncNet run
            raise sync.SyncError("piecewise skipped: no global confidence to compare with")
        motion, envelope = sync.motion_series(clip, audio, box=box)
        shifts = sync.choose_shifts(motion, envelope, found, clip_s)
        phrase_records = [
            SyncPhrase(start_s=p.start_s, end_s=p.end_s, shift_s=round(k, 3))
            for p, k in zip(found, shifts, strict=True)
        ]
        sync.render(audio, found, shifts, clip_s, run_dir / SYNC_PIECEWISE)
        confidence_piecewise = sync.syncnet_measure(clip, run_dir / SYNC_PIECEWISE).confidence
    except sync.SyncError as exc:
        errors.append(_one_line(exc))

    piecewise = (
        confidence_global is not None
        and confidence_piecewise is not None
        and confidence_piecewise >= confidence_global
    )
    shutil.copyfile(run_dir / (SYNC_PIECEWISE if piecewise else SYNC_GLOBAL), run_dir / ALIGNED)
    report = report.model_copy(
        update={
            "shift_s": round(applied, 3),
            "method": "piecewise" if piecewise else "global",
            "confidence_global": _round(confidence_global),
            "confidence_piecewise": _round(confidence_piecewise),
            "piecewise_error": "; ".join(errors) or None,
            "phrases": phrase_records,
            "aligned_audio": ALIGNED,
        }
    )
    _record(ctx, report)
    return report


def _record(ctx: Context, report: SyncReport) -> None:
    (ctx.run_dir / SYNC_JSON).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    mf.stage(ctx.manifest, "shot_1").data["sync"] = report.model_dump(mode="json")


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split()) or type(exc).__name__


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _conf(report: SyncReport) -> str:
    return "n/a" if report.syncnet_confidence is None else f"{report.syncnet_confidence:g}"


def _character_image(ctx: Context, character: str) -> float:
    """`character.png` for a video model that needs an image: made once (paid), reused by a
    retry or resume. Returns what this call paid (0 when reused)."""
    image = ctx.providers.image
    assert image is not None  # checked in run()
    path = ctx.run_dir / CHARACTER_IMAGE
    if _non_empty(path):
        return 0.0
    path.unlink(missing_ok=True)
    ctx.check_cost_cap("the character image call")
    result = image.generate(_image_prompt(character), path)
    # paid: on disk before the video call, which can still fail
    ctx.commit_cost(result.cost_usd)
    if not _non_empty(path):
        raise GateFailed(f"Image provider left no character image at {path} (missing or 0 bytes)")
    return result.cost_usd


# live: a coffee-shop background grew Starbucks-like coffee bags
NO_BRANDS = "No logos, brand names or readable text."


def _image_prompt(character: str) -> str:
    """Always a stylized cartoon: moderation rejects photoreal people (live,
    "may contain real person"). Animals get a mouth: a beak lip-synced worse live."""
    animal = (
        " Make it an anthropomorphic cartoon animal with an expressive mouth with lips "
        "(no beak), able to speak."
        if _ANIMAL.search(character)
        else ""
    )
    return (
        f"Character reference image of {character}. Stylized 3D animated cartoon look, "
        f"clearly not a real person, not photorealistic.{animal} Head and shoulders, centered, "
        f"facing and looking at the camera, mouth closed, neutral expression, plain soft "
        f"background, even lighting. Vertical {ASPECT}. {NO_BRANDS}"
    )


def _prompt(character: str, visual: str, camera: str, reference_image: str | None) -> str:
    same = " The character is the one in the reference image." if reference_image else ""
    return (
        f"{character}. {visual}. Camera: {camera}.{same} "
        f"Vertical {ASPECT} frame. Lips move in sync with the provided audio. {NO_BRANDS}"
    )


def _non_empty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _extract_last_frame(clip: Path, frame: Path) -> None:
    """Decode the clip's tail, overwriting one PNG per frame: what remains is the last frame.
    Every failure is GateFailed — the paid clip exists, a pipeline retry would pay again."""
    frame.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-sseof", f"-{TAIL_S}", "-i", str(clip), "-an", "-update", "1", str(frame)]  # fmt: skip
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=FFMPEG_TIMEOUT_S, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired as err:
        raise GateFailed(f"Last frame extraction timed out after {FFMPEG_TIMEOUT_S} s") from err
    except OSError as err:
        raise GateFailed(f"Last frame extraction could not start ffmpeg: {err}") from err
    if out.returncode != 0:
        lines = [ln for ln in (out.stderr or "").strip().splitlines() if ln.strip()]
        tail = " | ".join(lines[-STDERR_TAIL_LINES:]) or f"exit code {out.returncode}"
        raise GateFailed(f"Last frame extraction failed: {tail}")
    if not _non_empty(frame):
        raise GateFailed(f"Last frame extraction produced no image at {frame}")
