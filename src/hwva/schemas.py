"""Contracts between stages. Every artifact a stage writes is one of these models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Lang = Literal["uk", "en"]


class LineSource(StrEnum):
    INPUT = "input"  # taken verbatim from the idea or --line
    USER = "user"  # answered in intake
    GENERATED = "generated"  # written by the LLM (no line in the idea, non-interactive)


class IntakeData(BaseModel):
    """`StageOutput.data` of the `intake` stage."""

    asked: bool  # True if ctx.ask was called at least once


class Shot(BaseModel):
    visual_prompt: str = Field(min_length=1)
    duration_s: float = Field(gt=0, le=30)
    camera: str = "medium close-up"
    has_line: bool = False


Gender = Literal["male", "female"]
Age = Literal["child", "teen", "adult", "elderly"]


class VoicePersona(BaseModel):
    """Who speaks the line (`Script.voice`). Gender must agree with a grammatically gendered line;
    the line is never changed to fit the persona."""

    gender: Gender
    age: Age


def _require_voice(schema: dict[str, Any]) -> None:
    """`voice` has a default (old scripts and tests omit it) but the LLM must always answer it."""
    required = schema.setdefault("required", [])
    if "voice" not in required:
        required.append("voice")


class Script(BaseModel):
    """Output of the `script` stage (LLM structured output, validated by `validate_script`)."""

    model_config = ConfigDict(json_schema_extra=_require_voice)

    title: str
    line: str = Field(min_length=1, description="Exact dialogue line as the user wrote it")
    line_source: LineSource
    lang: Lang
    character: str = Field(min_length=1, description="Appearance, reused in every shot")
    shots: list[Shot] = Field(min_length=1, max_length=2)
    voice: VoicePersona = Field(
        default_factory=lambda: VoicePersona(gender="male", age="adult"),
        description="Who speaks the line; gender must agree with a grammatically gendered line",
    )


class ScriptData(BaseModel):
    """`StageOutput.data` of the `script` stage."""

    line_corrected: bool  # True if the LLM rewrote a known line and we restored the original
    line_source: LineSource  # = manifest.params["line_source"] after the stage


class VoiceChoice(BaseModel):
    """`VoiceData.voice`: the voice actually used, resolved from config/voices.yaml
    (TTS provider name -> "<gender>-<age>" -> {voice, style})."""

    persona: VoicePersona  # Script.voice, as the script asked
    key: str  # the voices.yaml key used; differs from the persona's key after a fallback
    voice: str  # provider voice name, e.g. "Puck"
    style: str | None = None  # provider style field (never spoken text)


class VoiceData(BaseModel):
    """`StageOutput.data` of the `voice` stage. Artifacts: {"audio": "line.wav",
    "transcript": "transcript.txt", "shot_audio": "shot_audio.wav"}."""

    audio_duration_s: float  # ffprobe of the accepted line.wav
    # max(shots[0].duration_s, lead_in_s + audio) + other shots, <= 30
    planned_total_s: float
    attempts: int = Field(ge=1, le=4)  # TTS->STT attempts used, 1 = verbatim on the first try
    diffs: list[list[str]] = Field(default_factory=list)  # diff_words() of each failed attempt
    voice: VoiceChoice | None = None  # what was used; None only in runs from before personas
    # Lead-in (specs/voice.md): silence before the line in shot_audio.wav, the
    # audio sent to the video model (artifact "shot_audio"). 0.0 / None in runs from before it.
    lead_in_s: float = Field(default=0.0, ge=0, le=2)  # HWVA_LEAD_IN_S, default 0.7
    shot_audio_duration_s: float | None = None  # ffprobe of shot_audio.wav


# v2 adds "weak"; v1 records only ever hold pass / correct / no_sync / error
SyncVerdict = Literal["pass", "correct", "weak", "no_sync", "error"]
SyncMethod = Literal["global", "piecewise"]


class SyncThresholds(BaseModel):
    """SyncNet confidence thresholds used for the verdict (specs/sync.md, "Verdict"):
    `HWVA_SYNC_CONF` (default 5.0) and `HWVA_SYNC_WEAK_CONF` (default 3.0)."""

    sync_conf: float
    weak_conf: float


class SyncPhrase(BaseModel):
    """One speech run of `shot_audio.wav` (clip time, lead-in included) and the per-phrase shift
    the piecewise alignment proposed for it (3 decimals; negative = audio earlier)."""

    start_s: float
    end_s: float
    shift_s: float = 0.0


class SyncMotion(BaseModel):
    """Our motion metric (`hwva.sync.measure`): diagnostic only, no verdict, no shift."""

    lag_s: float  # positive = the mouth moves AFTER the audio
    peak_r: float
    r_at_0: float


class SyncReport(BaseModel):
    """Lip-sync record of the talking shot (specs/sync.md, "Record"): `sync.json` in the run dir
    (artifact "sync" of shot_1) and shot_1 data `sync` — same content.

    v2: SyncNet decides (`syncnet_*`, `verdict`, global `shift_s`), the piecewise
    alignment proposes per-phrase shifts; the chosen track is `aligned_audio`. Every v2 field has a
    default so a v1 record (`lag_s`, `peak_r`, `r_at_0`, `shift_s`, `box`, `verdict`) validates."""

    verdict: SyncVerdict
    # raw SyncNet measurement on (clip, shot_audio.wav); None on `error` (and in v1 records)
    syncnet_confidence: float | None = None  # max_conf, 3 decimals (the verdict uses the raw one)
    syncnet_offset_frames: int | None = None  # offset at the max confidence, 25 fps frames
    syncnet_min_dist: float | None = None
    # the GLOBAL candidate's shift (negative = audio earlier): offset/25 for correct / weak, else
    # 0.0, clamped to fit the clip. Applied in the final only when method == "global" or when
    # there is no aligned_audio (then assemble adds it to lead_in_s); piecewise uses `phrases`.
    shift_s: float = 0.0
    thresholds: SyncThresholds | None = None  # the thresholds used; None in v1 records
    method: SyncMethod | None = None  # None when no alignment ran (error, rejected take, v1)
    confidence_global: float | None = None  # SyncNet on (clip, sync_global.wav), or the raw one
    confidence_piecewise: float | None = None  # SyncNet on (clip, sync_piecewise.wav)
    piecewise_error: str | None = None  # why the piecewise candidate was not usable
    phrases: list[SyncPhrase] = Field(default_factory=list)  # [] when not computed
    aligned_audio: str | None = None  # "aligned_line.wav" or None
    motion: SyncMotion | None = None  # diagnostic; None if it failed (and in v1 records)
    box: list[float] = Field(min_length=4, max_length=4)  # [x0, x1, y0, y1], frame fractions
    take: int = Field(default=1, ge=1, le=2)  # 2 after the one regeneration (HWVA_SYNC_REJECT=1)
    error: str | None = None  # SyncError text (verdict `error`), else None
    warning: str | None = None  # one line for `weak` / `no_sync` / `error`, else None
    rejected: SyncReport | None = None  # the earlier take's record (its own `rejected` is None)
    # v1 motion numbers at the top level; kept only so older records still read.
    # A v2 record leaves them None (the motion metric lives in `motion`).
    lag_s: float | None = None
    peak_r: float | None = None
    r_at_0: float | None = None


class ShotData(BaseModel):
    """`StageOutput.data` of the `shot_1` stage. Artifacts: {"video": "shot_1.mp4",
    "shot_1_last_frame": "shot_1.last.png", "sync": "sync.json", "aligned_audio":
    "aligned_line.wav"} — local files in the run dir (video = "shot_1.take2.mp4" after a lip-sync
    regeneration; no "aligned_audio" when no alignment ran, e.g. verdict `error`).

    Also the data of `shot_2` when it runs (artifacts: {"video": "shot_2.mp4"}); a one-shot
    script makes shot_2 return `SHOT_2_SKIPPED` instead and no artifacts."""

    # shot_1: max(shots[0].duration_s, lead_in_s + audio); shot_2: shots[1]
    duration_s: float = Field(gt=0)
    sync: SyncReport | None = None  # shot_1 only; None for shot_2 and runs from before sync


SHOT_2_SKIPPED: dict[str, str] = {"skipped": "one-shot script"}  # shot_2 data, one-shot script


AudioSource = Literal["aligned", "line"]


class AssembleData(BaseModel):
    """`StageOutput.data` of the `assemble` stage. Artifacts: {"video": "final.mp4", "audio":
    <the file placed: "aligned_line.wav" or "line.wav">} (older runs: no "audio").

    final.mp4: shots concatenated in order, re-encoded (H.264, yuv420p, 24 fps, 720x1280, AAC);
    the only audio track is our TTS — shot_1's aligned_line.wav at 0 when recorded, else line.wav
    from t=audio_offset_s — padded with silence to the video length."""

    duration_s: float = Field(gt=0)  # ffprobe of final.mp4 = sum of shot durations
    # where the placed file starts, rounded to ms. "line": max(0, voice lead_in_s + shot_1
    # sync.shift_s) (0.0 in runs from before the correction). "aligned": 0.0, or negative when the
    # track's leading silence was trimmed so its speech ends by the video end (specs/assemble.md)
    audio_offset_s: float = 0.0
    # "line": lead_in_s + shift_s, unclamped (may be negative); "aligned": 0.0. Applied != wanted
    # -> a manifest assumption. None in older runs.
    audio_offset_wanted_s: float | None = None
    # v2: which file was placed; None in older runs (= "line")
    audio_source: AudioSource | None = None
    # first speech in the final (first speech start in the placed file + audio_offset_s), ms;
    # None in older runs (final_check then uses audio_offset_s)
    speech_start_s: float | None = None


# final_check: names of the checks in final_report.json, and the raw-value keys each one records.
FINAL_HARD_CHECKS: tuple[str, ...] = (
    "format",  # value: duration_s, width, height, video_streams, audio_streams
    "black_frames",  # value: black_s (total of blackdetect=d=0.1:pix_th=0.10)
    "leading_silence",  # value: leading_silence_s (silencedetect=n=-40dB:d=0.4; 0.0 if none)
    "loudness",  # value: integrated_lufs (ebur128)
    "audio_match",  # value: peak (normalized cross-correlation), offset_ms (lag of the peak)
    "line",  # value: attempts (STT calls, 1..2), diffs (diff_words() of each attempt)
)
FINAL_SOFT_CHECKS: tuple[str, ...] = (
    "freeze",  # value: freeze_s (freezedetect total)
    "true_peak",  # value: true_peak_dbtp
    "scene_cuts",  # value: cuts (scdet count), shots (len(Script.shots))
)


class FinalCheck(BaseModel):
    """One row of final_report.json. Soft checks (`hard=False`) never affect the verdict."""

    name: str
    hard: bool
    passed: bool
    value: dict[str, Any] = Field(default_factory=dict)  # raw measurements, keys as listed above
    threshold: str  # human-readable threshold, e.g. ">= -35 LUFS"


class FinalReport(BaseModel):
    """Content of final_report.json (artifact "report") AND `StageOutput.data` of `final_check`.

    Written on success and on failure (before `GateFailed` is raised)."""

    verdict: Literal["pass", "fail"]
    failed: list[str] = Field(default_factory=list)  # names of failed HARD checks, report order
    transcript: str = ""  # last STT transcript of the audio extracted from final.mp4
    checks: list[FinalCheck] = Field(default_factory=list)


MMSS = r"^\d{2,}:[0-5]\d$"  # "MM:SS" timestamp in the judge's answer


class JudgeCriterion(BaseModel):
    """One rubric item of the judge (rubric text with 1/3/5 anchors lives in the provider)."""

    verdict: Literal["pass", "fail", "unsure"]
    score: int = Field(ge=1, le=5)
    evidence: str
    timestamp: str | None = Field(default=None, pattern=MMSS)


class JudgeDefect(BaseModel):
    kind: str
    timestamp: str = Field(pattern=MMSS)
    note: str


class JudgeVerdict(BaseModel):
    """What `JudgeProvider.judge(video_path, script)` returns. Soft signal, never a gate."""

    face_present: JudgeCriterion
    mouth_moves_when_speaking: JudgeCriterion
    matches_character: JudgeCriterion
    same_character_across_cut: JudgeCriterion
    matches_prompt: JudgeCriterion
    defects: list[JudgeDefect] = Field(default_factory=list)


class JudgeOutcome(BaseModel):
    """quality.json["judge"]. Exactly one of `skipped` / `error` / `verdict` is set.

    `verdict.defects` keeps only defects inside the video; the dropped ones are counted in
    `hallucinated_defects`. If the video length could not be read (ffprobe failed / timed out and
    assemble recorded none), defects are kept as returned and `filter_skipped` says why."""

    skipped: str | None = None  # e.g. "no judge in profile"
    error: str | None = None  # judge raised or returned something that isn't a JudgeVerdict
    verdict: JudgeVerdict | None = None
    hallucinated_defects: int = Field(default=0, ge=0)
    filter_skipped: str | None = None  # why the hallucinated-timestamp filter did not run


class SyncNetOutcome(BaseModel):
    """quality.json["syncnet"]. Either `skipped`/`error`, or the measurement (offset first: it is
    the meaningful signal; LSE-C is context only). Measured by `hwva.sync.syncnet_measure`."""

    skipped: str | None = None  # no face (NoFaceError)
    # import / weights missing (required dependency) / run / result reading
    # failed (incl. TimeoutExpired)
    error: str | None = None
    av_offset_frames: int | None = None
    confidence: float | None = None  # LSE-C
    min_dist: float | None = None  # LSE-D
    seconds: float | None = None  # wall time of the measurement


class QualitySync(BaseModel):
    """quality.json["sync"] (specs/quality_check.md, "Lip-sync numbers"): shot_1's recorded
    `SyncReport`, copied (never re-measured), or `{"skipped": "no sync measurement"}`.
    A v1 record leaves the v2 fields None."""

    skipped: str | None = None
    syncnet_confidence: float | None = None
    syncnet_offset_frames: int | None = None
    verdict: SyncVerdict | None = None
    method: SyncMethod | None = None
    shift_s: float | None = None
    phrases: list[SyncPhrase] = Field(default_factory=list)
    confidence_global: float | None = None
    confidence_piecewise: float | None = None
    take: int | None = None
    # v1 motion numbers; kept only so quality.json files from v1 still read
    lag_s: float | None = None
    peak_r: float | None = None
    r_at_0: float | None = None


class QualityReport(BaseModel):
    """Content of quality.json AND `StageOutput.data` of `quality_check`. Never blocks."""

    judge: JudgeOutcome
    syncnet: SyncNetOutcome
    sync: QualitySync | None = None  # None only in quality.json from before the lip-sync check


EvalKind = Literal["uk_quoted", "en_line", "no_line", "long_line"]


class EvalIdea(BaseModel):
    """One entry of `evals/ideas.yaml` (`ideas: [...]`), see specs/eval.md. Not a stage contract:
    `uv run eval` turns it into a run like `uv run video <idea> [--line] [--lang] --no-interactive`.
    Digits are NOT rejected here — the intake gate is what the eval measures."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)  # unique within the file; becomes the CSV `idea_id`
    kind: EvalKind
    idea: str = Field(min_length=1)
    line: str | None = None  # -> params["line"] (like --line)
    lang: Lang | None = None  # -> params["lang"] (like --lang)

    @field_validator("line")
    @classmethod
    def _line_not_blank(cls, value: str | None) -> str | None:
        # `line: ""` is a config error, never "no line": to let the LLM write it, leave it out
        if value is not None and not value.strip():
            raise ValueError("`line` is empty; leave it out to let the LLM write the line")
        return value


class StageStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> path relative to run dir
    data: dict[str, Any] = Field(default_factory=dict)  # stage facts: durations, diffs, scores
    cost_usd: float = 0.0
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ModelRef(BaseModel):
    provider: str
    model: str
    version: str


class Manifest(BaseModel):
    """State of one run. Lives in runs/<run_id>/manifest.json; the only source of truth."""

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    idea: str
    profile: str
    models: dict[str, ModelRef] = Field(default_factory=dict)  # role -> model, pinned at start
    strategy: str = "A"  # chosen from the video model's capabilities at start (see strategy.py)
    # cli args + intake answers. params["line"] is THE source of truth for the dialogue line.
    params: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)  # what the agent filled in by itself
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    cost_usd: float = 0.0
