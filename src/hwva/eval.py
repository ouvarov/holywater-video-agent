"""uv run eval [--profile dev|premium] [--ideas evals/ideas.yaml] [--out evals/results/<ts>.csv]
uv run eval --compare A.csv B.csv

Not a pipeline stage (specs/eval.md): runs the pipeline on a small idea set and writes one CSV row
per idea; `--compare` tells whether a new version is better or worse, with numbers."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hwva import manifest as mf
from hwva import pipeline
from hwva.cli import load_env
from hwva.context import Context
from hwva.providers import registry
from hwva.providers.base import ProviderError
from hwva.schemas import (
    EvalIdea,
    FinalReport,
    Manifest,
    QualityReport,
    Script,
    StageRecord,
    StageStatus,
    SyncReport,
    VoiceData,
)
from hwva.strategy import preflight

EXIT_OK, EXIT_REJECT, EXIT_USAGE = 0, 1, 2
EXIT_INTERRUPTED = 130  # 128 + SIGINT, as a shell reports Ctrl-C
COST_TOLERANCE = 1.25  # a new version may cost at most +25 % (specs/eval.md)
EPS = 1e-9

STAGE_NAMES = tuple(st.name for st in pipeline.STAGES)
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?")
CHECKS = ("format", "black_frames", "leading_silence", "loudness", "audio_match", "line",
          "freeze", "true_peak", "scene_cuts")  # fmt: skip
# raw final_check value columns: column -> (check name, key in FinalCheck.value)
CHECK_VALUES: dict[str, tuple[str, str]] = {
    "fc_duration_s": ("format", "duration_s"),
    "fc_width": ("format", "width"),
    "fc_height": ("format", "height"),
    "fc_video_streams": ("format", "video_streams"),
    "fc_audio_streams": ("format", "audio_streams"),
    "fc_black_s": ("black_frames", "black_s"),
    "fc_leading_silence_s": ("leading_silence", "leading_silence_s"),
    "fc_integrated_lufs": ("loudness", "integrated_lufs"),
    "fc_audio_peak": ("audio_match", "peak"),
    "fc_audio_offset_ms": ("audio_match", "offset_ms"),
    "fc_line_attempts": ("line", "attempts"),
    "fc_freeze_s": ("freeze", "freeze_s"),
    "fc_true_peak_dbtp": ("true_peak", "true_peak_dbtp"),
    "fc_cuts": ("scene_cuts", "cuts"),
}
JUDGE = ("face_present", "mouth_moves_when_speaking", "matches_character",
         "same_character_across_cut", "matches_prompt")  # fmt: skip
MODEL_ROLES = ("llm", "tts", "stt", "video", "judge")
HUMAN = ("human_stress", "human_emotion", "human_overall")
# shot_1's lip-sync record (specs/eval.md, v2): column -> SyncReport field
# ("sync_shifts" is built from `phrases`)
SYNC_COLUMNS: dict[str, str] = {
    "syncnet_confidence": "syncnet_confidence",
    "syncnet_offset_frames": "syncnet_offset_frames",
    "sync_verdict": "verdict",
    "sync_method": "method",
    "sync_conf_global": "confidence_global",
    "sync_conf_piecewise": "confidence_piecewise",
    "sync_shifts": "phrases",
}
SYNC_JSON = "sync.json"

COLUMNS: list[str] = [
    "idea_id", "kind", "run_id", "profile",
    *[f"model_{r}" for r in MODEL_ROLES],
    "strategy", "status", "failed_stage", "error", "provider_failure_rate",
    "line", "line_source", "voice_persona", "voice_key",
    *[f"attempts_{s}" for s in STAGE_NAMES],
    "voice_tts_attempts",
    "final_verdict",
    *[f"fc_{c}" for c in CHECKS],
    *CHECK_VALUES,
    *SYNC_COLUMNS,
    *[f"judge_{j}" for j in JUDGE], "judge_defects",
    "cost_usd", "seconds",
    *HUMAN,
]  # fmt: skip


# --- ideas --------------------------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def load_ideas(path: Path) -> list[EvalIdea]:
    """`ideas: [...]` from the YAML file, validated. Every problem with the file (unreadable, not
    UTF-8, bad YAML, invalid entry, duplicate id) -> ValueError, one line, naming the file."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    except OSError as exc:
        raise ValueError(f"{path}: cannot read: {_one_line(exc.strerror or str(exc))}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML syntax error: {_one_line(str(exc))}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("ideas"), list):
        raise ValueError(f"{path}: expected a mapping with an `ideas:` list")
    ideas: list[EvalIdea] = []
    for i, entry in enumerate(data["ideas"]):
        try:
            ideas.append(EvalIdea.model_validate(entry))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'entry'}: {e['msg']}" for e in exc.errors()
            )
            raise ValueError(f"{path}: idea #{i + 1} is invalid: {_one_line(problems)}") from exc
    seen: set[str] = set()
    for idea in ideas:
        if idea.id in seen:
            raise ValueError(f"{path}: duplicate idea id {idea.id!r}")
        seen.add(idea.id)
    return ideas


# --- one row ------------------------------------------------------------------------------------


def csv_cell(value: str) -> str:
    """A cell Excel won't run as a formula: a leading = + - @ tab or CR gets a `'` (the line and
    error texts are user/provider input). A plain number like -23.5 is written as is."""
    if value.startswith(_FORMULA_START) and not _PLAIN_NUMBER.fullmatch(value):
        return "'" + value
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _done_data(manifest: Manifest, name: str) -> dict[str, Any] | None:
    record = manifest.stages.get(name)
    if record is None or record.status is not StageStatus.DONE or not record.data:
        return None
    return record.data


def _final_report(manifest: Manifest, run_dir: Path) -> FinalReport | None:
    data = _done_data(manifest, "final_check")
    try:
        if data is not None:
            return FinalReport.model_validate(data)
        path = run_dir / "final_report.json"  # written before GateFailed; stage keeps no data then
        if path.exists():
            return FinalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError):
        return None
    return None


def _sync_report(manifest: Manifest, run_dir: Path) -> SyncReport | None:
    """shot_1's lip-sync record in any status: its data `sync`, else `sync.json` in the run dir
    (a failed reject: written before GateFailed). None when shot_1 never measured."""
    record = manifest.stages.get("shot_1")
    if record is None:
        return None
    try:
        raw = record.data.get("sync")
        if raw is not None:
            return SyncReport.model_validate(raw)
        path = run_dir / SYNC_JSON
        if path.exists():
            return SyncReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError):
        return None
    return None


def _voice(manifest: Manifest, run_dir: Path) -> tuple[str, str, str]:
    """(voice_persona, voice_key, voice_tts_attempts)."""
    data = _done_data(manifest, "voice")
    if data is not None:
        try:
            voice = VoiceData.model_validate(data)
        except ValidationError:
            voice = None
        if voice is not None:
            attempts = str(voice.attempts)
            if voice.voice is not None:
                persona = voice.voice.persona
                return f"{persona.gender}-{persona.age}", voice.voice.key, attempts
            return _script_persona(run_dir), "", attempts
    return _script_persona(run_dir), "", ""


def _script_persona(run_dir: Path) -> str:
    path = run_dir / "script.json"
    if not path.exists():
        return ""
    try:
        script = Script.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError):
        return ""
    return f"{script.voice.gender}-{script.voice.age}"


def _status(manifest: Manifest) -> tuple[str, str, str]:
    """(status, failed_stage, error)."""
    for name, record in manifest.stages.items():
        if record.status is StageStatus.FAILED:
            return "failed", name, record.error or ""
    records: list[StageRecord] = list(manifest.stages.values())
    if records and all(r.status is StageStatus.DONE for r in records):
        return "done", "", ""
    return "pending", "", ""


def is_provider_failure(failed_stage: str, error: str) -> bool:
    """A run that ended on a ProviderError, or on shot_2's moderation GateFailed (the stage wraps
    the provider rejection to add the one-shot fallback note). `Unexpected ...` is ours: a gate
    fail, so a bug never makes a version look better (specs/eval.md, Outcome of a run)."""
    if error.startswith("ProviderError"):
        return True
    return failed_stage == "shot_2" and error.startswith("GateFailed") and "(moderation)" in error


def row_from_manifest(
    manifest: Manifest, idea: EvalIdea, seconds: float, run_dir: Path
) -> dict[str, str]:
    """One CSV row (column -> text) for a finished or failed run."""
    row = dict.fromkeys(COLUMNS, "")
    status, failed_stage, error = _status(manifest)
    row |= {
        "idea_id": idea.id,
        "kind": idea.kind,
        "run_id": manifest.run_id,
        "profile": manifest.profile,
        "strategy": manifest.strategy,
        "status": status,
        "failed_stage": failed_stage,
        "error": error,
        "provider_failure_rate": "1" if is_provider_failure(failed_stage, error) else "0",
        "line": manifest.params.get("line", ""),
        "line_source": manifest.params.get("line_source", ""),
        "cost_usd": _text(round(manifest.cost_usd, 6)),
        "seconds": _text(seconds),
    }
    for role in MODEL_ROLES:
        ref = manifest.models.get(role)
        if ref is not None:
            row[f"model_{role}"] = f"{ref.provider}/{ref.model}@{ref.version}"
    for name in STAGE_NAMES:
        record = manifest.stages.get(name)
        if record is not None:
            row[f"attempts_{name}"] = str(record.attempts)

    row["voice_persona"], row["voice_key"], row["voice_tts_attempts"] = _voice(manifest, run_dir)

    report = _final_report(manifest, run_dir)
    if report is not None:
        row["final_verdict"] = report.verdict
        by_name = {c.name: c for c in report.checks}
        for check in CHECKS:
            if check in by_name:
                row[f"fc_{check}"] = "pass" if by_name[check].passed else "fail"
        for col, (check, key) in CHECK_VALUES.items():
            if check in by_name:
                row[col] = _text(by_name[check].value.get(key))

    sync = _sync_report(manifest, run_dir)
    if sync is not None:
        for col, key in SYNC_COLUMNS.items():
            if key == "phrases":
                row[col] = ";".join(f"{p.shift_s:g}" for p in sync.phrases)
            else:
                row[col] = _text(getattr(sync, key))

    quality = _done_data(manifest, "quality_check")
    if quality is not None:
        try:
            verdict = QualityReport.model_validate(quality).judge.verdict
        except ValidationError:
            verdict = None
        if verdict is not None:
            for j in JUDGE:
                row[f"judge_{j}"] = str(getattr(verdict, j).score)
            row["judge_defects"] = str(len(verdict.defects))
    return row


# --- running ------------------------------------------------------------------------------------


def run_idea(idea: EvalIdea, profile: str) -> dict[str, str]:
    """Like `uv run video <idea> [--line] [--lang] --no-interactive --profile <profile>`."""
    started = time.perf_counter()
    providers = registry.build(profile)
    strategy = preflight(providers.video.capabilities, providers.refs["video"].provider)
    params = {k: v for k, v in {"line": idea.line, "lang": idea.lang}.items() if v}
    manifest = mf.new_run(idea.idea, profile, providers.refs, params)
    manifest.strategy = strategy
    mf.save(manifest)
    run_dir = mf.run_dir(manifest.run_id)
    ctx = Context(
        manifest=manifest,
        providers=providers,
        run_dir=run_dir,
        interactive=False,
        max_cost_usd=registry.load_max_cost_usd(profile),  # per run, like `uv run video`
    )
    result = pipeline.run(ctx)
    return row_from_manifest(result, idea, time.perf_counter() - started, run_dir)


def _run_eval(profile: str, ideas_path: Path, out: Path | None) -> int:
    try:
        ideas = load_ideas(ideas_path)
        providers = registry.build(profile)  # fail on a bad profile before any run dir
        registry.load_max_cost_usd(profile)  # and on a bad money cap
        preflight(providers.video.capabilities, providers.refs["video"].provider)
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError, ProviderError) as exc:
        print(f"error: {_one_line(str(exc))}", file=sys.stderr)
        return EXIT_USAGE
    out = out or Path("evals/results") / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # header first, then one flushed row per finished idea: a crash or Ctrl-C keeps what is done
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: opens right in Excel
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        f.flush()
        try:
            for idea in ideas:
                row = run_idea(idea, profile)
                writer.writerow({k: csv_cell(v) for k, v in row.items()})
                f.flush()
                written += 1
                print(f"{idea.id}: {row['status']} {row['failed_stage']} run {row['run_id']}"
                      .rstrip())  # fmt: skip
        except KeyboardInterrupt:
            print(f"interrupted: {written} rows in {out}", file=sys.stderr)
            return EXIT_INTERRUPTED
    print(f"wrote {out}")
    return EXIT_OK


# --- compare ------------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _floats(rows: list[dict[str, str]], col: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            out.append(float(row.get(col) or ""))
        except ValueError:
            continue
    return out


def _pass_rate(rows: list[dict[str, str]], col: str) -> float | None:
    values = [row.get(col, "") for row in rows if row.get(col, "") in {"pass", "fail"}]
    return sum(v == "pass" for v in values) / len(values) if values else None


def _provider_failed(row: dict[str, str]) -> bool:
    flag = row.get("provider_failure_rate") or ""
    if flag:
        return flag == "1"
    return is_provider_failure(row.get("failed_stage") or "", row.get("error") or "")  # old CSV


def _gate_rate(rows: list[dict[str, str]]) -> float | None:
    """done / (done + gate fail); provider failures are out of both. None = no verdict (n/a)."""
    verdicts = [r.get("status") == "done" for r in rows if not _provider_failed(r)]
    return sum(verdicts) / len(verdicts) if verdicts else None


def _rate_line(name: str, a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return f"{name}: {_na(a)} -> {_na(b)}"
    return f"{name}: {a:.2f} -> {b:.2f} ({b - a:+.2f})"


def _na(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.2f}"


def match_ideas(
    a_rows: list[dict[str, str]], b_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]], str | None]:
    """Rows of the ideas common to A and B (in A's order) and a warning when the sets differ.
    A duplicate idea_id or no common idea -> ValueError (specs/eval.md, Compare details)."""
    by_id: list[dict[str, dict[str, str]]] = []
    for side, rows in (("A", a_rows), ("B", b_rows)):
        ids: dict[str, dict[str, str]] = {}
        for row in rows:
            idea_id = row.get("idea_id") or ""
            if idea_id in ids:
                raise ValueError(f"duplicate idea_id {idea_id!r} in {side}")
            ids[idea_id] = row
        by_id.append(ids)
    a_ids, b_ids = by_id
    common = [i for i in a_ids if i in b_ids]
    if not common:
        raise ValueError("A and B have no idea_id in common")
    only_a = [i for i in a_ids if i not in b_ids]
    only_b = [i for i in b_ids if i not in a_ids]
    warning = None
    if only_a or only_b:
        warning = (f"warning: idea sets differ, comparing {len(common)} common ideas only; "
                   f"only in A: {', '.join(only_a) or '-'}; "
                   f"only in B: {', '.join(only_b) or '-'}")  # fmt: skip
    return [a_ids[i] for i in common], [b_ids[i] for i in common], warning


def compare(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]]) -> tuple[list[str], bool]:
    """Over the rows given (already matched to common ideas by `match_ideas`)."""
    lines: list[str] = []
    rate_a, rate_b = _gate_rate(a_rows), _gate_rate(b_rows)
    lines.append(_rate_line("gate pass rate", rate_a, rate_b))
    pfr_a = _mean([1.0 if _provider_failed(r) else 0.0 for r in a_rows])
    pfr_b = _mean([1.0 if _provider_failed(r) else 0.0 for r in b_rows])
    lines.append(_rate_line("provider failure rate", pfr_a, pfr_b))

    cost_a, cost_b = _mean(_floats(a_rows, "cost_usd")), _mean(_floats(b_rows, "cost_usd"))
    pct = f"{(cost_b - cost_a) / cost_a * 100:+.1f}%" if cost_a > 0 else "n/a"
    lines.append(f"mean cost_usd: {cost_a:.4f} -> {cost_b:.4f} ({pct})")

    sec_a, sec_b = _mean(_floats(a_rows, "seconds")), _mean(_floats(b_rows, "seconds"))
    lines.append(f"mean seconds: {sec_a:.1f} -> {sec_b:.1f} ({sec_b - sec_a:+.1f})")

    for check in CHECKS:
        col = f"fc_{check}"
        pa, pb = _pass_rate(a_rows, col), _pass_rate(b_rows, col)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            lines.append(f"{col}: {_opt(pa)} -> {_opt(pb)}")
        else:
            lines.append(_rate_line(col, pa, pb))

    for j in (*JUDGE, "defects"):
        col = f"judge_{j}"
        ja, jb = _floats(a_rows, col), _floats(b_rows, col)
        if ja and jb:
            ma, mb = _mean(ja), _mean(jb)
            lines.append(f"mean {col}: {ma:.2f} -> {mb:.2f} ({mb - ma:+.2f})")

    sa, sb = _floats(a_rows, "syncnet_confidence"), _floats(b_rows, "syncnet_confidence")
    if sa and sb:  # reported, not part of the accept rule (thresholds provisional, n=8)
        ma, mb = _mean(sa), _mean(sb)
        lines.append(f"mean syncnet_confidence: {ma:.2f} -> {mb:.2f} ({mb - ma:+.2f})")

    reasons = []
    if rate_b is None:
        reasons.append("no run reached a gate verdict")
    elif rate_a is not None and rate_b < rate_a - EPS:  # only A n/a: the gate part passes
        reasons.append("gate pass rate dropped")
    if cost_b > COST_TOLERANCE * cost_a + EPS:
        reasons.append("cost grew by more than 25%")
    accepted = not reasons
    lines.append("verdict: ACCEPT" if accepted else f"verdict: REJECT ({'; '.join(reasons)})")
    return lines, accepted


def _opt(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.2f}"


def _compare(a: Path, b: Path) -> int:
    try:
        a_rows, b_rows = read_rows(a), read_rows(b)
        a_rows, b_rows, warning = match_ideas(a_rows, b_rows)
    except (OSError, ValueError) as exc:  # ValueError covers UnicodeDecodeError too
        print(f"error: {_one_line(str(exc))}", file=sys.stderr)
        return EXIT_USAGE
    lines, accepted = compare(a_rows, b_rows)
    print(f"{a} -> {b}")
    if warning is not None:
        print(warning)
    for line in lines:
        print(line)
    return EXIT_OK if accepted else EXIT_REJECT


# --- cli ----------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval", description=__doc__)
    p.add_argument("--profile", default="dev", help="config/profiles.yaml (default: dev, free)")
    p.add_argument("--ideas", type=Path, default=Path("evals/ideas.yaml"))
    p.add_argument("--out", type=Path, help="default: evals/results/<timestamp>.csv")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A.csv", "B.csv"),
                   help="compare two result files; exit 0 = ACCEPT B, 1 = REJECT")  # fmt: skip
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv)
    if args.compare:
        return _compare(*args.compare)
    return _run_eval(args.profile, args.ideas, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
