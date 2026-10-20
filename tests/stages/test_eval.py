"""Acceptance tests for `uv run eval` (specs/eval.md). Not a pipeline stage: the code lives in
`src/hwva/eval.py` (`main(argv) -> int`, console script `eval = "hwva.eval:main"`), nothing is
registered in `pipeline.STAGES`. Offline and free: profile `dev`, runs dir redirected to tmp_path.

Contract pinned here (the spec leaves the details open):
- `load_ideas(path) -> list[EvalIdea]` (schemas.EvalIdea); duplicate ids / bad entry ->
  ValueError.
- `row_from_manifest(manifest, idea, seconds, run_dir) -> dict[str, str]`: one CSV row, every
  value is text, "" when unknown. Raw final_check values come from the stage data, or from
  `final_report.json` in the run dir when final_check failed (the pipeline keeps no data then).
  `voice_persona` comes from the voice data, else from `script.json`.
- CSV columns = `COLUMNS` below, in that order; read back with encoding "utf-8-sig".
- Model columns: "<provider>/<model>@<version>", "" when the profile has no such role.
- Check pass columns ("fc_<check>"): "pass" | "fail" | "".
- Each idea runs like `uv run video <idea> [--line] [--lang] --no-interactive` (never asks), via
  `pipeline.run(ctx)` with the module-level `pipeline.STAGES` (tests monkeypatch it). A failed
  idea is a row, not a crash; `main` returns 0 once the CSV is written. Unknown profile -> 2.
- `--compare A.csv B.csv` prints the lines pinned in `test_compare_prints_rates_costs_times_and_
  check_deltas` and `verdict: ACCEPT` (exit 0) or `verdict: REJECT` (exit 1); missing file -> 2.
  Gate pass rate = done / (done + gate fail); provider failures (ProviderError, or shot_2's
  "(moderation)" GateFailed) are left out of both, an `Unexpected ...` crash is a gate fail;
  `n/a` when no row reached a verdict. Numbers are over the idea_ids common to A and B.
  Accepted iff pass rate B >= A AND mean cost B <= 1.25 * mean cost A (so 0 -> anything > 0 is
  rejected); B n/a -> REJECT, only A n/a -> the gate part passes.
- `provider_failure_rate` column ("1" | "0") right after `error`.
- The CSV is written incrementally (header first, one flushed row per finished idea); Ctrl-C ->
  exit 130 and one stderr line `interrupted: <n> rows in <path>`.
- A broken ideas file -> exit 2, one stderr line `error: ...` naming the file, nothing written."""

from __future__ import annotations

import csv
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest
from conftest import judge_verdict, needs_ffmpeg

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.schemas import (
    EvalIdea,
    FinalCheck,
    FinalReport,
    JudgeOutcome,
    LineSource,
    Manifest,
    ModelRef,
    QualityReport,
    Script,
    Shot,
    ShotData,
    StageRecord,
    StageStatus,
    SyncNetOutcome,
    SyncReport,
    VoiceChoice,
    VoiceData,
    VoicePersona,
)
from hwva.text import detect_lang, extract_quoted_line, has_digits, line_gender, normalize

REPO = Path(__file__).resolve().parents[2]
IDEAS = REPO / "evals" / "ideas.yaml"
SECONDS_PER_WORD = 0.4  # specs/script.md: estimated speech length = words * 0.4 s
TOTAL_LIMIT_S = 30.0  # specs/voice.md: planned total <= 30 s (also max_duration_s of dev/premium)

STAGE_NAMES = ["intake", "script", "voice", "shot_1", "shot_2", "assemble", "final_check",
               "quality_check"]  # fmt: skip
CHECKS = ["format", "black_frames", "leading_silence", "loudness", "audio_match", "line",
          "freeze", "true_peak", "scene_cuts"]  # fmt: skip
JUDGE = ["face_present", "mouth_moves_when_speaking", "matches_character",
         "same_character_across_cut", "matches_prompt"]  # fmt: skip
HUMAN = ["human_stress", "human_emotion", "human_overall"]

COLUMNS = [
    "idea_id", "kind", "run_id", "profile",
    "model_llm", "model_tts", "model_stt", "model_video", "model_judge",
    "strategy", "status", "failed_stage", "error", "provider_failure_rate",
    "line", "line_source", "voice_persona", "voice_key",
    *[f"attempts_{s}" for s in STAGE_NAMES],
    "voice_tts_attempts",
    "final_verdict",
    *[f"fc_{c}" for c in CHECKS],
    "fc_duration_s", "fc_width", "fc_height", "fc_video_streams", "fc_audio_streams",
    "fc_black_s", "fc_leading_silence_s", "fc_integrated_lufs", "fc_audio_peak",
    "fc_audio_offset_ms", "fc_line_attempts", "fc_freeze_s", "fc_true_peak_dbtp", "fc_cuts",
    # v2 (specs/eval.md "Lip-sync columns"; replace v1's sync_lag_s, sync_peak_r)
    "syncnet_confidence", "syncnet_offset_frames", "sync_verdict", "sync_method",
    "sync_conf_global", "sync_conf_piecewise", "sync_shifts",
    *[f"judge_{j}" for j in JUDGE], "judge_defects",
    "cost_usd", "seconds",
    *HUMAN,
]  # fmt: skip


# Every test that pins the whole header waits for the v2 lip-sync columns.

DEV_MODELS = {
    "model_llm": "fake/fake-llm@1",
    "model_tts": "fake/sine@1",
    "model_stt": "fake/sidecar@1",
    "model_video": "fake/testsrc@1",
    "model_judge": "",  # dev has no judge
}


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    monkeypatch.setenv("HWVA_RUNS", str(runs))
    # SyncNet takes ~15-35 s per call and the regression test must not depend on the machine:
    # unimportable here (conftest's autouse `no_real_syncnet` does the same for every test).
    monkeypatch.setitem(sys.modules, "syncnet_python", None)
    return runs


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_ideas(path: Path, *ideas: dict[str, str]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"ideas": list(ideas)}, allow_unicode=True), encoding="utf-8")
    return path


def known_line(idea: EvalIdea) -> str | None:
    return idea.line if idea.line is not None else extract_quoted_line(idea.idea)


Recorded = list[dict[str, Any]]


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """Replaces the pipeline with one cheap stage that records what each run got; an idea that
    contains "FAIL" fails its gate."""
    seen: Recorded = []

    def touch(c: Context) -> StageOutput:
        seen.append({"idea": c.manifest.idea, "params": dict(c.manifest.params),
                     "interactive": c.interactive, "run_dir": c.run_dir})  # fmt: skip
        if "FAIL" in c.manifest.idea:
            raise GateFailed("synthetic gate failure")
        return StageOutput(cost_usd=0.25)

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("touch", touch)])
    return seen


# --- the idea set -------------------------------------------------------------------------------


def test_idea_set_has_four_ideas_one_of_each_kind() -> None:
    from hwva import eval as ev

    ideas = ev.load_ideas(IDEAS)
    assert len(ideas) == 4
    assert sorted(i.kind for i in ideas) == ["en_line", "long_line", "no_line", "uk_quoted"]
    assert len({i.id for i in ideas}) == 4


def test_idea_set_has_no_digits() -> None:
    from hwva import eval as ev

    for idea in ev.load_ideas(IDEAS):
        assert not has_digits(idea.idea), idea.id
        assert not has_digits(known_line(idea) or ""), idea.id


def test_idea_set_line_sources_match_kinds() -> None:
    from hwva import eval as ev

    by_kind = {i.kind: i for i in ev.load_ideas(IDEAS)}
    uk = by_kind["uk_quoted"]
    assert uk.line is None
    assert extract_quoted_line(uk.idea) is not None
    assert detect_lang(extract_quoted_line(uk.idea) or "") == "uk"
    en = by_kind["en_line"]
    assert en.line is not None and detect_lang(en.line) == "en"
    no = by_kind["no_line"]
    assert no.line is None and extract_quoted_line(no.idea) is None
    assert known_line(by_kind["long_line"]) is not None


def test_ukrainian_lines_are_gendered_both_ways() -> None:
    """The voice persona must be exercised: every Ukrainian line has a grammatical gender, and
    the set covers both (male is FakeLlm's default, so female proves the persona is followed)."""
    from hwva import eval as ev

    genders = set()
    for idea in ev.load_ideas(IDEAS):
        line = known_line(idea)
        if line is None or detect_lang(line) != "uk":
            continue
        gender = line_gender(line, "uk")
        assert gender is not None, f"{idea.id}: line has no gender marker"
        genders.add(gender)
    assert genders == {"male", "female"}


def test_long_line_is_near_but_within_the_duration_limit() -> None:
    """Close to the limit (>= 18 s estimated) but leaves room for a 4 s second shot in 30 s."""
    from hwva import eval as ev

    long = next(i for i in ev.load_ideas(IDEAS) if i.kind == "long_line")
    line = known_line(long)
    assert line is not None
    estimate = len(normalize(line).split()) * SECONDS_PER_WORD
    assert 18.0 <= estimate <= TOTAL_LIMIT_S - 4.0, estimate


@pytest.mark.parametrize(
    "entries",
    [
        [{"id": "a", "kind": "no_line", "idea": "x"}, {"id": "a", "kind": "no_line", "idea": "y"}],
        [{"id": "a", "kind": "haiku", "idea": "x"}],
        [{"id": "a", "kind": "no_line", "idea": "x", "voice": "Puck"}],
        [{"id": "a", "kind": "no_line", "idea": ""}],
    ],
    ids=["duplicate_id", "unknown_kind", "unknown_key", "empty_idea"],
)
def test_load_ideas_rejects_bad_files(tmp_path: Path, entries: list[dict[str, str]]) -> None:
    from hwva import eval as ev

    with pytest.raises(ValueError):
        ev.load_ideas(write_ideas(tmp_path / "ideas.yaml", *entries))


@pytest.mark.parametrize("line", ["", "   "], ids=["empty", "whitespace"])
def test_load_ideas_rejects_empty_line(tmp_path: Path, line: str) -> None:
    """`line: ""` is never read as "no line" (specs/eval.md, Ideas file errors)."""
    from hwva import eval as ev

    path = write_ideas(tmp_path / "ideas.yaml",
                       {"id": "a", "kind": "en_line", "idea": "A dog.", "line": line})  # fmt: skip
    with pytest.raises(ValueError):
        ev.load_ideas(path)


BROKEN_NAME = "broken-ideas.yaml"


def _broken_yaml_syntax(path: Path) -> None:
    path.write_text("ideas: [\n  {id: a, kind: no_line, idea: x\n", encoding="utf-8")


def _broken_not_utf8(path: Path) -> None:
    path.write_bytes(b"ideas:\n  - id: a\n    kind: no_line\n    idea: \xff\xfe\xfa\n")


def _broken_top_level_list(path: Path) -> None:
    path.write_text("- id: a\n  kind: no_line\n  idea: x\n", encoding="utf-8")


def _broken_invalid_entry(path: Path) -> None:
    # two errors in one entry: pydantic's message spans several lines
    write_ideas(path, {"id": "a", "kind": "haiku", "idea": ""})


def _broken_empty_line(path: Path) -> None:
    write_ideas(path, {"id": "a", "kind": "en_line", "idea": "A dog.", "line": ""})


def _broken_blank_line(path: Path) -> None:
    write_ideas(path, {"id": "a", "kind": "en_line", "idea": "A dog.", "line": "  \t "})


def _broken_missing(path: Path) -> None:
    pass


def _broken_directory(path: Path) -> None:
    path.mkdir()  # exists but cannot be read as a file


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_broken_yaml_syntax, id="yaml_syntax"),
        pytest.param(_broken_not_utf8, id="not_utf8"),
        pytest.param(_broken_top_level_list, id="top_level_list"),
        pytest.param(_broken_invalid_entry, id="invalid_entry"),
        pytest.param(_broken_empty_line, id="empty_line"),
        pytest.param(_broken_blank_line, id="whitespace_line"),
        pytest.param(_broken_missing, id="missing"),
        pytest.param(_broken_directory, id="unreadable"),
    ],
)
def test_broken_ideas_file_is_a_one_line_usage_error(
    tmp_path: Path, isolated_runs: Path, fake_stages: Recorded,
    capsys: pytest.CaptureFixture[str], make: Any,
) -> None:  # fmt: skip
    from hwva import eval as ev

    path = tmp_path / BROKEN_NAME
    make(path)
    out = tmp_path / "r.csv"
    rc = ev.main(["--profile", "dev", "--ideas", str(path), "--out", str(out)])
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert rc == 2
    assert len(lines) == 1, err
    assert lines[0].startswith("error:"), err
    assert BROKEN_NAME in lines[0], err
    assert "Traceback" not in err
    assert not out.exists()
    assert fake_stages == []
    assert not isolated_runs.exists() or not any(isolated_runs.iterdir())


# --- running ideas and writing the CSV ----------------------------------------------------------


def test_runs_all_ideas_and_writes_csv_with_pinned_columns(
    tmp_path: Path, isolated_runs: Path, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    out = tmp_path / "out" / "result.csv"
    assert ev.main(["--profile", "dev", "--ideas", str(IDEAS), "--out", str(out)]) == 0
    header, rows = read_csv(out)
    assert header == COLUMNS
    assert [r["idea_id"] for r in rows] == [i.id for i in ev.load_ideas(IDEAS)]
    assert [r["kind"] for r in rows] == [i.kind for i in ev.load_ideas(IDEAS)]
    assert len(fake_stages) == 4
    for row in rows:
        assert row["profile"] == "dev"
        assert {k: row[k] for k in DEV_MODELS} == DEV_MODELS
        assert row["strategy"] == "A"
        assert row["status"] == "done"
        assert row["failed_stage"] == "" and row["error"] == ""
        assert float(row["cost_usd"]) == pytest.approx(0.25)
        assert float(row["seconds"]) > 0
        assert all(row[h] == "" for h in HUMAN)
        assert (isolated_runs / row["run_id"] / "manifest.json").exists()
    assert len({r["run_id"] for r in rows}) == 4


def test_ideas_run_non_interactively_with_their_line_and_lang(
    tmp_path: Path, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    ideas = write_ideas(
        tmp_path / "ideas.yaml",
        {"id": "q", "kind": "uk_quoted", "idea": 'Кіт каже: "Я тут головний"'},
        {"id": "e", "kind": "en_line", "idea": "A dog.", "line": "Hello there", "lang": "en"},
    )
    assert ev.main(["--ideas", str(ideas), "--out", str(tmp_path / "r.csv")]) == 0
    assert [s["interactive"] for s in fake_stages] == [False, False]
    assert fake_stages[0]["idea"] == 'Кіт каже: "Я тут головний"'
    assert "line" not in fake_stages[0]["params"]  # intake takes the quoted line, not eval
    assert fake_stages[1]["params"] == {"line": "Hello there", "lang": "en"}


def test_failed_idea_is_a_row_and_does_not_stop_the_eval(
    tmp_path: Path, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    ideas = write_ideas(
        tmp_path / "ideas.yaml",
        {"id": "bad", "kind": "no_line", "idea": "FAIL here"},
        {"id": "good", "kind": "no_line", "idea": "A cat on a roof."},
    )
    out = tmp_path / "r.csv"
    assert ev.main(["--ideas", str(ideas), "--out", str(out)]) == 0
    _, rows = read_csv(out)
    assert [r["idea_id"] for r in rows] == ["bad", "good"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["failed_stage"] == "touch"
    assert rows[0]["error"].startswith("GateFailed")
    assert rows[1]["status"] == "done" and rows[1]["failed_stage"] == ""


FORMULA_LINES = ['=HYPERLINK("https://evil.example")', "+так", "-ні", "@ось"]


def test_formula_like_cells_are_quoted(
    tmp_path: Path, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    ideas = write_ideas(
        tmp_path / "ideas.yaml",
        *[{"id": f"f{i}", "kind": "en_line", "idea": "A dog.", "line": line, "lang": "en"}
          for i, line in enumerate(FORMULA_LINES)],
    )  # fmt: skip
    out = tmp_path / "r.csv"
    assert ev.main(["--ideas", str(ideas), "--out", str(out)]) == 0
    _, rows = read_csv(out)
    assert [r["line"] for r in rows] == ["'" + line for line in FORMULA_LINES]


def test_provider_error_at_build_is_a_usage_error_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from hwva import eval as ev
    from hwva.providers import registry
    from hwva.providers.base import ProviderError

    def refuse(*args: object, **kwargs: object) -> registry.Providers:
        raise ProviderError("ARK_BASE_URL must be an https:// URL", retryable=False)

    monkeypatch.setattr(registry, "build", refuse)
    out = tmp_path / "r.csv"
    assert ev.main(["--ideas", str(IDEAS), "--out", str(out)]) == ev.EXIT_USAGE
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("error:") and "ARK_BASE_URL" in err[0]
    assert not out.exists()


@pytest.mark.parametrize(
    ("value", "written"),
    [
        ("=1+1", "'=1+1"),
        ("+так", "'+так"),
        ("-ні", "'-ні"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tтаб", "'\tтаб"),
        ("\rcr", "'\rcr"),
        ("-23.5", "-23.5"),  # a plain number is not a formula: compare and Excel read it as one
        ("-1e-05", "-1e-05"),
        ("-2+3", "'-2+3"),
        ("GateFailed: x", "GateFailed: x"),
        ("", ""),
    ],
)
def test_csv_cell_neutralises_formulas_but_keeps_numbers(value: str, written: str) -> None:
    from hwva import eval as ev

    assert ev.csv_cell(value) == written


def test_csv_cell_non_ascii_digits_are_not_a_plain_number() -> None:
    from hwva import eval as ev

    assert ev.csv_cell("-٣") == "'-٣"


def test_stage_columns_follow_pipeline_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from hwva import eval as ev

    extra = pipeline.Stage("extra", lambda c: StageOutput())
    monkeypatch.setattr(pipeline, "STAGES", [*pipeline.STAGES, extra])
    try:
        importlib.reload(ev)
        assert ev.STAGE_NAMES[-1] == "extra"
        assert "attempts_extra" in ev.COLUMNS
    finally:
        monkeypatch.undo()
        importlib.reload(ev)
    assert tuple(ev.STAGE_NAMES) == tuple(s.name for s in pipeline.STAGES)


def test_default_out_is_evals_results_timestamp_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stages: Recorded
) -> None:
    from hwva import eval as ev

    ideas = write_ideas(tmp_path / "ideas.yaml", {"id": "a", "kind": "no_line", "idea": "A cat."})
    monkeypatch.setenv("HWVA_CONFIG", str(REPO / "config" / "profiles.yaml"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert ev.main(["--ideas", str(ideas)]) == 0
    written = list((work / "evals" / "results").glob("*.csv"))
    assert len(written) == 1
    assert re.search(r"\d{8}-?\d{6}", written[0].stem), written[0].name
    assert read_csv(written[0])[0] == COLUMNS


def test_unknown_profile_is_a_usage_error_and_writes_nothing(
    tmp_path: Path, fake_stages: Recorded
) -> None:
    from hwva import eval as ev

    out = tmp_path / "r.csv"
    assert ev.main(["--profile", "nope", "--ideas", str(IDEAS), "--out", str(out)]) == 2
    assert not out.exists()
    assert fake_stages == []


# --- incremental CSV, crash and Ctrl-C ----------------------------------------------------------


def _two_ideas(tmp_path: Path, second: str = "A dog on a roof.") -> Path:
    return write_ideas(
        tmp_path / "ideas.yaml",
        {"id": "first", "kind": "no_line", "idea": "A cat on a roof."},
        {"id": "second", "kind": "no_line", "idea": second},
    )


def _peek_csv(path: Path) -> tuple[list[str], list[dict[str, str]]] | None:
    return read_csv(path) if path.exists() else None


def test_header_is_on_disk_before_the_first_idea_and_each_row_when_its_idea_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_stdin: None
) -> None:
    """A stage reads the CSV while each idea runs: what it sees is what a crash would leave."""
    from hwva import eval as ev

    out = tmp_path / "r.csv"
    seen: list[tuple[list[str], list[dict[str, str]]] | None] = []

    def spy(c: Context) -> StageOutput:
        seen.append(_peek_csv(out))
        return StageOutput()

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("spy", spy)])
    assert ev.main(["--ideas", str(_two_ideas(tmp_path)), "--out", str(out)]) == 0
    assert len(seen) == 2
    assert seen[0] is not None, "no CSV while the first idea ran"
    assert seen[0] == (COLUMNS, [])
    assert seen[1] is not None
    assert seen[1][0] == COLUMNS
    assert [r["idea_id"] for r in seen[1][1]] == ["first"]
    assert [r["idea_id"] for r in read_csv(out)[1]] == ["first", "second"]


def _crash_on_call(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """`pipeline.run` raises outside the pipeline's own catch on its n-th call."""
    original = pipeline.run
    calls = {"n": 0}

    def run(ctx: Context, *args: Any, **kwargs: Any) -> Manifest:
        calls["n"] += 1
        if calls["n"] == n:
            raise RuntimeError("crash outside the pipeline")
        return original(ctx, *args, **kwargs)

    monkeypatch.setattr(pipeline, "run", run)


def test_crash_on_second_idea_propagates_and_keeps_the_first_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    _crash_on_call(monkeypatch, 2)
    out = tmp_path / "r.csv"
    with pytest.raises(RuntimeError, match="crash outside the pipeline"):
        ev.main(["--ideas", str(_two_ideas(tmp_path)), "--out", str(out)])
    header, rows = read_csv(out)
    assert header == COLUMNS
    assert [r["idea_id"] for r in rows] == ["first"]
    assert rows[0]["status"] == "done"


def test_crash_on_first_idea_leaves_a_header_only_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_stages: Recorded, no_stdin: None
) -> None:
    from hwva import eval as ev

    _crash_on_call(monkeypatch, 1)
    out = tmp_path / "r.csv"
    with pytest.raises(RuntimeError, match="crash outside the pipeline"):
        ev.main(["--ideas", str(_two_ideas(tmp_path)), "--out", str(out)])
    assert read_csv(out) == (COLUMNS, [])


def test_ctrl_c_during_second_idea_exits_130_and_keeps_the_first_row(
    tmp_path: Path, isolated_runs: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], no_stdin: None,
) -> None:  # fmt: skip
    from hwva import eval as ev

    def stage(c: Context) -> StageOutput:
        if "INTERRUPT" in c.manifest.idea:
            raise KeyboardInterrupt  # BaseException: the pipeline does not catch it
        return StageOutput(cost_usd=0.1)

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("touch", stage)])
    out = tmp_path / "r.csv"
    ideas = _two_ideas(tmp_path, second="INTERRUPT here")
    try:
        rc = ev.main(["--ideas", str(ideas), "--out", str(out)])
    except KeyboardInterrupt:  # must not abort the pytest session
        pytest.fail("KeyboardInterrupt escaped main")
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert rc == 130
    assert len(lines) == 1, err
    assert lines[0].startswith("interrupted: 1 rows in "), err
    assert out.name in lines[0]
    header, rows = read_csv(out)
    assert header == COLUMNS
    assert [r["idea_id"] for r in rows] == ["first"]  # the running idea gets no row
    manifests = list(isolated_runs.glob("*/manifest.json"))
    assert len(manifests) == 2  # ... but its run dir stays for --resume


# --- one row from a manifest --------------------------------------------------------------------


def _manifest(tmp_path: Path, stages: dict[str, StageRecord]) -> tuple[Manifest, Path]:
    root = tmp_path / "rows"
    models = {
        "llm": ModelRef(provider="fake", model="fake-llm", version="1"),
        "tts": ModelRef(provider="fake", model="sine", version="1"),
        "stt": ModelRef(provider="fake", model="sidecar", version="1"),
        "video": ModelRef(provider="fake", model="testsrc", version="1"),
        "judge": ModelRef(provider="gemini", model="gemini-3.8-flash", version="g-3.8"),
    }
    params = {"line": "Та я сама все зробила!", "line_source": "input", "lang": "uk"}
    m = mf.new_run("Бабуся каже", "dev", models, params, root=root)
    m.stages = stages
    m.cost_usd = 1.2345
    mf.save(m, root)
    return m, mf.run_dir(m.run_id, root)


def _done(attempts: int = 1, data: dict[str, Any] | None = None) -> StageRecord:
    return StageRecord(status=StageStatus.DONE, attempts=attempts, data=data or {})


def _report(*failed: str) -> FinalReport:
    values: dict[str, dict[str, Any]] = {
        "format": {"duration_s": 9.0, "width": 720, "height": 1280, "video_streams": 1,
                   "audio_streams": 1},
        "black_frames": {"black_s": 0.0},
        "leading_silence": {"leading_silence_s": 0.12},
        "loudness": {"integrated_lufs": -50.0 if "loudness" in failed else -20.5},
        "audio_match": {"peak": 0.98, "offset_ms": 0.0},
        "line": {"attempts": 1, "diffs": []},
        "freeze": {"freeze_s": 9.0},
        "true_peak": {"true_peak_dbtp": -1.5},
        "scene_cuts": {"cuts": 1},
    }  # fmt: skip
    soft = {"freeze", "true_peak", "scene_cuts"}
    checks = [
        FinalCheck(name=n, hard=n not in soft, passed=n not in failed and n != "freeze",
                   value=v, threshold="t")
        for n, v in values.items()
    ]  # fmt: skip
    hard_failed = [n for n in failed if n not in soft]
    return FinalReport(verdict="fail" if hard_failed else "pass", failed=hard_failed,
                       transcript="та я сама все зробила", checks=checks)  # fmt: skip


def _script(persona: VoicePersona) -> Script:
    return Script(title="t", line="Та я сама все зробила!", line_source=LineSource.INPUT,
                  lang="uk", character="an old woman",
                  shots=[Shot(visual_prompt="p", duration_s=5, has_line=True)],
                  voice=persona)  # fmt: skip


IDEA = EvalIdea(id="uk-quoted", kind="uk_quoted", idea='Бабуся каже: "Та я сама все зробила!"')


def test_row_from_done_manifest(tmp_path: Path) -> None:
    from hwva import eval as ev

    voice = VoiceData(
        audio_duration_s=2.0, planned_total_s=9.0, attempts=3, diffs=[["-x"], ["-y"]],
        voice=VoiceChoice(persona=VoicePersona(gender="female", age="elderly"),
                          key="female-adult", voice="sine-220hz"),
    )  # fmt: skip
    quality = QualityReport(
        judge=JudgeOutcome(verdict=judge_verdict("00:02", score=4)),
        syncnet=SyncNetOutcome(skipped="extra not installed"),
    )
    stages = {
        "intake": _done(), "script": _done(attempts=2),
        "voice": _done(data=voice.model_dump(mode="json")),
        "shot_1": _done(), "shot_2": _done(), "assemble": _done(),
        "final_check": _done(data=_report().model_dump(mode="json")),
        "quality_check": _done(data=quality.model_dump(mode="json")),
    }  # fmt: skip
    m, run_dir = _manifest(tmp_path, stages)
    row = ev.row_from_manifest(m, IDEA, 42.5, run_dir)

    assert list(row) == COLUMNS
    assert all(isinstance(v, str) for v in row.values())
    assert row["idea_id"] == "uk-quoted" and row["kind"] == "uk_quoted"
    assert row["run_id"] == m.run_id and row["profile"] == "dev" and row["strategy"] == "A"
    assert row["model_llm"] == "fake/fake-llm@1"
    assert row["model_judge"] == "gemini/gemini-3.8-flash@g-3.8"
    assert row["status"] == "done" and row["failed_stage"] == "" and row["error"] == ""
    assert row["line"] == "Та я сама все зробила!" and row["line_source"] == "input"
    assert row["voice_persona"] == "female-elderly"  # what the script asked
    assert row["voice_key"] == "female-adult"  # what was used (fallback)
    assert row["attempts_script"] == "2" and row["attempts_voice"] == "1"
    assert row["attempts_quality_check"] == "1"
    assert row["voice_tts_attempts"] == "3"
    assert row["final_verdict"] == "pass"
    assert row["fc_format"] == "pass" and row["fc_line"] == "pass"
    assert row["fc_freeze"] == "fail"  # soft check: recorded, not a gate
    assert float(row["fc_duration_s"]) == pytest.approx(9.0)
    assert row["fc_width"] == "720" and row["fc_height"] == "1280"
    assert float(row["fc_leading_silence_s"]) == pytest.approx(0.12)
    assert float(row["fc_integrated_lufs"]) == pytest.approx(-20.5)
    assert float(row["fc_audio_peak"]) == pytest.approx(0.98)
    assert float(row["fc_audio_offset_ms"]) == pytest.approx(0.0)
    assert row["fc_line_attempts"] == "1"
    assert float(row["fc_true_peak_dbtp"]) == pytest.approx(-1.5)
    assert row["fc_cuts"] == "1"
    assert all(row[f"judge_{j}"] == "4" for j in JUDGE)
    assert row["judge_defects"] == "1"
    assert float(row["cost_usd"]) == pytest.approx(1.2345)
    assert float(row["seconds"]) == pytest.approx(42.5)
    assert all(row[h] == "" for h in HUMAN)


def test_row_from_manifest_failed_at_voice(tmp_path: Path) -> None:
    from hwva import eval as ev

    stages = {
        "intake": _done(), "script": _done(),
        "voice": StageRecord(status=StageStatus.FAILED, attempts=1,
                             error="GateFailed: line not verbatim: -головний"),
    }  # fmt: skip
    m, run_dir = _manifest(tmp_path, stages)
    (run_dir / "script.json").write_text(
        _script(VoicePersona(gender="male", age="teen")).model_dump_json(), encoding="utf-8"
    )
    row = ev.row_from_manifest(m, IDEA, 3.0, run_dir)

    assert row["status"] == "failed"
    assert row["failed_stage"] == "voice"
    assert row["error"] == "GateFailed: line not verbatim: -головний"
    assert row["voice_persona"] == "male-teen"  # from script.json: voice kept no data
    assert row["voice_key"] == "" and row["voice_tts_attempts"] == ""
    assert row["attempts_voice"] == "1" and row["attempts_shot_1"] == ""
    assert row["final_verdict"] == ""
    assert all(row[f"fc_{c}"] == "" for c in CHECKS)
    assert row["fc_integrated_lufs"] == ""
    assert all(row[f"judge_{j}"] == "" for j in JUDGE) and row["judge_defects"] == ""


def test_row_from_manifest_failed_at_final_check_reads_final_report(tmp_path: Path) -> None:
    """The pipeline keeps no data for a failed stage; final_report.json in the run dir has it."""
    from hwva import eval as ev

    stages = {name: _done() for name in STAGE_NAMES[:6]}
    stages["final_check"] = StageRecord(status=StageStatus.FAILED, attempts=1,
                                        error="GateFailed: loudness")  # fmt: skip
    m, run_dir = _manifest(tmp_path, stages)
    (run_dir / "final_report.json").write_text(
        _report("loudness").model_dump_json(), encoding="utf-8"
    )
    row = ev.row_from_manifest(m, IDEA, 30.0, run_dir)

    assert row["status"] == "failed" and row["failed_stage"] == "final_check"
    assert row["final_verdict"] == "fail"
    assert row["fc_loudness"] == "fail" and row["fc_format"] == "pass"
    assert float(row["fc_integrated_lufs"]) == pytest.approx(-50.0)
    assert row["attempts_quality_check"] == ""


def _shot_2_moderation_error() -> str:
    """What the pipeline records when shot_2 wraps a moderation rejection (stages/shot_2.py)."""
    from hwva.stages.shot_2 import FALLBACK_NOTE

    return ("GateFailed: Video provider rejected shot 1's last frame as reference (moderation): "
            f"face filter: content rejected. {FALLBACK_NOTE}")  # fmt: skip


OUTCOMES: dict[str, tuple[str | None, str | None, str]] = {
    # case: (failed stage, recorded error, provider_failure_rate)
    "done": (None, None, "0"),
    "gate_failed": ("voice", "GateFailed: line not verbatim: -головний", "0"),
    "unexpected_crash": ("assemble", "Unexpected RuntimeError: ffmpeg exploded", "0"),
    "provider_error": ("shot_1", "ProviderError: moderation: face filter rejected the prompt",
                       "1"),
    "provider_timeout": ("shot_1", "ProviderError: task timed out after 600 s", "1"),
    "shot_2_moderation_gate": ("shot_2", None, "1"),  # error filled by _shot_2_moderation_error
}  # fmt: skip


@pytest.mark.parametrize("case", list(OUTCOMES))
def test_row_provider_failure_rate_classifies_the_outcome(tmp_path: Path, case: str) -> None:
    from hwva import eval as ev

    failed_stage, error, expected = OUTCOMES[case]
    if case == "shot_2_moderation_gate":
        error = _shot_2_moderation_error()
    stages: dict[str, StageRecord] = {}
    for name in STAGE_NAMES:
        if name == failed_stage:
            stages[name] = StageRecord(status=StageStatus.FAILED, attempts=1, error=error)
            break
        stages[name] = _done()
    m, run_dir = _manifest(tmp_path, stages)
    row = ev.row_from_manifest(m, IDEA, 1.0, run_dir)

    assert list(row) == COLUMNS
    assert list(row).index("provider_failure_rate") == list(row).index("error") + 1
    assert row["status"] == ("done" if failed_stage is None else "failed")
    assert row["provider_failure_rate"] == expected


# --- compare ------------------------------------------------------------------------------------


def write_results(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({c: "" for c in COLUMNS} | {"idea_id": f"i{i}"} | row)
    return path


def runs(statuses: list[str], costs: list[float], seconds: list[float] | None = None,
         **checks: list[str]) -> list[dict[str, str]]:  # fmt: skip
    secs = seconds or [10.0] * len(statuses)
    rows = [
        {"status": s, "cost_usd": str(c), "seconds": str(t)}
        for s, c, t in zip(statuses, costs, secs, strict=True)
    ]
    for col, values in checks.items():
        for row, v in zip(rows, values, strict=True):
            row[col] = v
    return rows


def test_compare_prints_rates_costs_times_and_check_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hwva import eval as ev

    a = write_results(tmp_path / "a.csv", runs(["done", "failed"], [0.3, 0.5], [8.0, 12.0],
                                               fc_loudness=["pass", "pass"]))  # fmt: skip
    b = write_results(tmp_path / "b.csv", runs(["done", "done"], [0.5, 0.5], [12.0, 12.0],
                                               fc_loudness=["pass", "fail"]))  # fmt: skip
    rc = ev.main(["--compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert "gate pass rate: 0.50 -> 1.00 (+0.50)" in out
    assert "mean cost_usd: 0.4000 -> 0.5000 (+25.0%)" in out
    assert "mean seconds: 10.0 -> 12.0 (+2.0)" in out
    assert "fc_loudness: 1.00 -> 0.50 (-0.50)" in out
    assert "verdict: ACCEPT" in out
    assert rc == 0


VERDICTS: dict[str, tuple[list[dict[str, str]], list[dict[str, str]], bool]] = {
    "same": (runs(["done", "done"], [1.0, 1.0]), runs(["done", "done"], [1.0, 1.0]), True),
    "cost_exactly_plus_25pct": (
        runs(["done", "done"], [1.0, 1.0]), runs(["done", "done"], [1.25, 1.25]), True),
    "cost_plus_26pct": (
        runs(["done", "done"], [1.0, 1.0]), runs(["done", "done"], [1.26, 1.26]), False),
    "cheaper_same_pass_rate": (
        runs(["done", "failed"], [1.0, 1.0]), runs(["done", "failed"], [0.5, 0.5]), True),
    "cheaper_but_lower_pass_rate": (
        runs(["done", "done"], [1.0, 1.0]), runs(["done", "failed"], [0.5, 0.5]), False),
    "higher_pass_rate_but_plus_30pct": (
        runs(["done", "failed"], [1.0, 1.0]), runs(["done", "done"], [1.3, 1.3]), False),
    "free_to_free": (runs(["done"], [0.0]), runs(["done"], [0.0]), True),
    "free_to_paid": (runs(["done"], [0.0]), runs(["done"], [0.01]), False),
}  # fmt: skip


@pytest.mark.parametrize("case", list(VERDICTS))
def test_compare_verdict(tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str) -> None:
    from hwva import eval as ev

    old, new, accepted = VERDICTS[case]
    a = write_results(tmp_path / "a.csv", old)
    b = write_results(tmp_path / "b.csv", new)
    rc = ev.main(["--compare", str(a), str(b)])
    out = capsys.readouterr().out
    assert ("verdict: ACCEPT" in out) is accepted
    assert ("verdict: REJECT" in out) is not accepted
    assert rc == (0 if accepted else 1)


def test_compare_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    from hwva import eval as ev

    a = write_results(tmp_path / "a.csv", runs(["done"], [0.0]))
    assert ev.main(["--compare", str(a), str(tmp_path / "missing.csv")]) == 2


# outcome -> (status, failed_stage, error, provider_failure_rate), as `uv run eval` writes them
_OUTCOME_COLS: dict[str, tuple[str, str, str, str]] = {
    "pass": ("done", "", "", "0"),
    "gate": ("failed", "voice", "GateFailed: line not verbatim: -x", "0"),
    "crash": ("failed", "assemble", "Unexpected RuntimeError: boom", "0"),
    "provider": ("failed", "shot_1", "ProviderError: moderation: rejected", "1"),
}


def outcome(kind: str, cost: float = 1.0, seconds: float = 10.0, idea_id: str | None = None,
            **extra: str) -> dict[str, str]:  # fmt: skip
    status, stage, error, pfr = _OUTCOME_COLS[kind]
    row = {"status": status, "failed_stage": stage, "error": error, "provider_failure_rate": pfr,
           "cost_usd": str(cost), "seconds": str(seconds)}  # fmt: skip
    if idea_id is not None:
        row["idea_id"] = idea_id
    return row | extra


def _compare_out(tmp_path: Path, capsys: pytest.CaptureFixture[str], a: list[dict[str, str]],
                 b: list[dict[str, str]]) -> tuple[int, list[str], list[str]]:  # fmt: skip
    from hwva import eval as ev

    rc = ev.main(["--compare", str(write_results(tmp_path / "a.csv", a)),
                  str(write_results(tmp_path / "b.csv", b))])  # fmt: skip
    captured = capsys.readouterr()
    return rc, captured.out.splitlines(), [x for x in captured.err.splitlines() if x.strip()]


def test_compare_gate_rate_leaves_provider_failures_out_and_prints_their_rate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("pass"), outcome("gate"), outcome("provider")]
    b = [outcome("pass"), outcome("pass"), outcome("provider")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "gate pass rate: 0.50 -> 1.00 (+0.50)" in out  # 1/2 -> 2/2, not 1/3 -> 2/3
    assert "provider failure rate: 0.33 -> 0.33 (+0.00)" in out
    assert "verdict: ACCEPT" in out and rc == 0


def test_compare_more_provider_failures_alone_do_not_reject(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept rule unchanged: provider failure rate is printed, not judged."""
    a = [outcome("pass"), outcome("gate"), outcome("pass"), outcome("gate")]
    b = [outcome("pass"), outcome("gate"), outcome("provider"), outcome("provider")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "gate pass rate: 0.50 -> 0.50 (+0.00)" in out
    assert "provider failure rate: 0.00 -> 0.50 (+0.50)" in out
    assert "verdict: ACCEPT" in out and rc == 0


def test_compare_rejects_when_b_has_no_gate_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("pass"), outcome("gate")]
    b = [outcome("provider"), outcome("provider")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "gate pass rate: 0.50 -> n/a" in out  # no delta when a side is n/a
    assert "provider failure rate: 0.00 -> 1.00 (+1.00)" in out
    verdict = [line for line in out if line.startswith("verdict:")]
    assert len(verdict) == 1 and verdict[0].startswith("verdict: REJECT")
    assert "no run reached a gate verdict" in verdict[0]
    assert rc == 1


def test_compare_gate_part_passes_when_only_a_has_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("provider"), outcome("provider")]
    b = [outcome("gate"), outcome("gate")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "gate pass rate: n/a -> 0.00" in out
    assert "verdict: ACCEPT" in out and rc == 0


def test_compare_unexpected_crash_is_a_gate_fail_not_a_provider_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("pass"), outcome("pass")]
    b = [outcome("pass"), outcome("crash")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "gate pass rate: 1.00 -> 0.50 (-0.50)" in out
    assert "provider failure rate: 0.00 -> 0.00 (+0.00)" in out
    assert "verdict: REJECT (gate pass rate dropped)" in out and rc == 1


def test_compare_mean_cost_and_time_include_provider_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The money was spent: provider-failure rows count in mean cost and time."""
    a = [outcome("pass", cost=1.0, seconds=10.0), outcome("provider", cost=3.0, seconds=30.0)]
    b = [outcome("pass", cost=1.0, seconds=10.0), outcome("provider", cost=1.0, seconds=10.0)]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "mean cost_usd: 2.0000 -> 1.0000 (-50.0%)" in out
    assert "mean seconds: 20.0 -> 10.0 (-10.0)" in out
    assert "provider failure rate: 0.50 -> 0.50 (+0.00)" in out
    assert "verdict: ACCEPT" in out and rc == 0


def test_compare_different_idea_sets_warns_and_uses_common_ideas_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every number below would differ if the rows only in A / only in B were counted."""
    a = [
        outcome("pass", idea_id="x", fc_loudness="pass", judge_matches_prompt="4"),
        outcome("gate", idea_id="y", fc_loudness="pass", judge_matches_prompt="4"),
        outcome("pass", cost=10.0, seconds=100.0, idea_id="only-in-a", fc_loudness="fail",
                judge_matches_prompt="1"),
    ]  # fmt: skip
    b = [
        outcome("provider", cost=10.0, seconds=100.0, idea_id="only-in-b", fc_loudness="fail",
                judge_matches_prompt="1"),
        outcome("pass", idea_id="y", fc_loudness="fail", judge_matches_prompt="3"),
        outcome("pass", idea_id="x", fc_loudness="pass", judge_matches_prompt="5"),
    ]  # fmt: skip
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    warnings = [line for line in out if line.startswith("warning:")]
    assert len(warnings) == 1, out
    assert "only-in-a" in warnings[0] and "only-in-b" in warnings[0]
    assert warnings[0].index("only-in-a") < warnings[0].index("only-in-b")  # A side, then B
    assert "gate pass rate: 0.50 -> 1.00 (+0.50)" in out
    assert "provider failure rate: 0.00 -> 0.00 (+0.00)" in out
    assert "mean cost_usd: 1.0000 -> 1.0000 (+0.0%)" in out
    assert "mean seconds: 10.0 -> 10.0 (+0.0)" in out
    assert "fc_loudness: 1.00 -> 0.50 (-0.50)" in out
    assert "mean judge_matches_prompt: 4.00 -> 4.00 (+0.00)" in out
    assert "verdict: ACCEPT" in out and rc == 0


def test_compare_same_idea_sets_prints_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, _ = _compare_out(tmp_path, capsys, [outcome("pass")], [outcome("pass")])
    assert not any(line.startswith("warning:") for line in out)
    assert rc == 0


@pytest.mark.parametrize(
    ("a_ids", "b_ids"),
    [(["a1", "a2"], ["b1"]), (["x", "x"], ["x"]), (["x"], ["x", "x"])],
    ids=["no_common_ideas", "duplicate_in_a", "duplicate_in_b"],
)
def test_compare_bad_idea_ids_are_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], a_ids: list[str], b_ids: list[str]
) -> None:
    rc, out, err = _compare_out(tmp_path, capsys, [outcome("pass", idea_id=i) for i in a_ids],
                                [outcome("pass", idea_id=i) for i in b_ids])  # fmt: skip
    assert rc == 2
    assert len(err) == 1, err
    assert not any(line.startswith("verdict:") for line in out)


def test_compare_prints_judge_score_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("pass", judge_face_present="4", judge_defects="1"),
         outcome("pass", judge_face_present="5", judge_defects="0")]  # fmt: skip
    b = [outcome("pass", judge_face_present="5", judge_defects="2"),
         outcome("pass", judge_face_present="5", judge_defects="2")]  # fmt: skip
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "mean judge_face_present: 4.50 -> 5.00 (+0.50)" in out
    assert "mean judge_defects: 0.50 -> 2.00 (+1.50)" in out
    assert not any(line.startswith("mean judge_matches_prompt") for line in out)  # no values
    assert rc == 0


# === Lip-sync columns (specs/eval.md "Lip-sync columns", v2; see specs/sync.md) =======
# Replaces v1's sync_lag_s / sync_peak_r columns and the `mean sync_peak_r` compare line.

BOX = [0.2, 0.8, 0.42, 0.6]
SYNC_COLUMNS = ["syncnet_confidence", "syncnet_offset_frames", "sync_verdict", "sync_method",
                "sync_conf_global", "sync_conf_piecewise", "sync_shifts"]  # fmt: skip


def _sync_v2(**kw: Any) -> SyncReport:
    fields: dict[str, Any] = {
        "verdict": "correct", "syncnet_confidence": 8.63, "syncnet_offset_frames": -6,
        "syncnet_min_dist": 7.1, "shift_s": -0.24, "method": "piecewise",
        "confidence_global": 8.71, "confidence_piecewise": 8.86,
        "phrases": [{"start_s": 0.7, "end_s": 1.5, "shift_s": -0.083},
                    {"start_s": 1.8, "end_s": 2.4, "shift_s": 0.042},
                    {"start_s": 2.6, "end_s": 3.3, "shift_s": 0.0}],
        "aligned_audio": "aligned_line.wav", "box": BOX,
    }  # fmt: skip
    return SyncReport.model_validate(fields | kw)


def _sync_stages(shot_1: StageRecord) -> dict[str, StageRecord]:
    stages = {name: _done() for name in STAGE_NAMES[:3]}
    stages["shot_1"] = shot_1
    return stages


def _sync_cells(row: dict[str, str]) -> list[str]:
    return [row[c] for c in SYNC_COLUMNS]


def test_sync_columns_come_right_after_the_fc_columns() -> None:
    from hwva import eval as ev

    assert ev.COLUMNS == COLUMNS
    i = ev.COLUMNS.index("fc_cuts")
    assert ev.COLUMNS[i + 1 : i + 1 + len(SYNC_COLUMNS)] == SYNC_COLUMNS
    assert "sync_lag_s" not in ev.COLUMNS and "sync_peak_r" not in ev.COLUMNS


def test_row_has_shot_1_sync_numbers(tmp_path: Path) -> None:
    from hwva import eval as ev

    shot = ShotData(duration_s=5.0, sync=_sync_v2())
    m, run_dir = _manifest(tmp_path, _sync_stages(_done(data=shot.model_dump(mode="json"))))
    row = ev.row_from_manifest(m, IDEA, 1.0, run_dir)
    assert float(row["syncnet_confidence"]) == pytest.approx(8.63)
    assert row["syncnet_offset_frames"] == "-6"
    assert row["sync_verdict"] == "correct"
    assert row["sync_method"] == "piecewise"
    assert float(row["sync_conf_global"]) == pytest.approx(8.71)
    assert float(row["sync_conf_piecewise"]) == pytest.approx(8.86)
    assert row["sync_shifts"] == "-0.083;0.042;0"  # the spec's own example


def test_row_of_a_failed_reject_reads_sync_json(tmp_path: Path) -> None:
    """HWVA_SYNC_REJECT=1 and both takes without sync: shot_1 FAILED (the pipeline keeps no data
    for a failed stage), sync.json in the run dir has the record (any status counts)."""
    from hwva import eval as ev

    error = "GateFailed: no lip-sync after 2 takes (SyncNet confidence 1.7, 1.2)"
    failed = StageRecord(status=StageStatus.FAILED, attempts=1, error=error)
    m, run_dir = _manifest(tmp_path, _sync_stages(failed))
    take_1 = SyncReport(
        verdict="no_sync", syncnet_confidence=1.7, syncnet_offset_frames=-7, box=BOX
    )
    take_2 = SyncReport(
        verdict="no_sync", syncnet_confidence=1.2, syncnet_offset_frames=3, box=BOX, take=2,
        rejected=take_1,
    )  # fmt: skip
    (run_dir / "sync.json").write_text(take_2.model_dump_json(), encoding="utf-8")
    row = ev.row_from_manifest(m, IDEA, 1.0, run_dir)
    assert row["failed_stage"] == "shot_1"
    assert _sync_cells(row) == ["1.2", "3", "no_sync", "", "", "", ""]


@pytest.mark.parametrize("case", ["error", "v1_record", "no_record", "no_shot_1"])
def test_row_sync_columns_empty_when_absent(tmp_path: Path, case: str) -> None:
    from hwva import eval as ev

    if case == "error":
        record = SyncReport(verdict="error", error="no face found", box=BOX)
        shot = ShotData(duration_s=5.0, sync=record)
        stages = _sync_stages(_done(data=shot.model_dump(mode="json")))
        expected = ["", "", "error", "", "", "", ""]
    elif case == "v1_record":  # a v1 record fills only sync_verdict
        v1 = {"lag_s": 0.5, "peak_r": 0.71, "r_at_0": 0.16, "verdict": "correct",
              "shift_s": 0.5, "box": BOX}  # fmt: skip
        stages = _sync_stages(_done(data={"duration_s": 5.0, "sync": v1}))
        expected = ["", "", "correct", "", "", "", ""]
    elif case == "no_record":  # a run from before the measurement
        stages = _sync_stages(_done(data={"duration_s": 5.0}))
        expected = [""] * 7
    else:
        stages = {name: _done() for name in STAGE_NAMES[:3]}
        expected = [""] * 7
    m, run_dir = _manifest(tmp_path, stages)
    row = ev.row_from_manifest(m, IDEA, 1.0, run_dir)
    assert list(row) == COLUMNS
    assert _sync_cells(row) == expected


def test_compare_prints_mean_syncnet_confidence_but_does_not_judge_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [outcome("pass", syncnet_confidence="8.6"), outcome("pass", syncnet_confidence="7.4")]
    b = [outcome("pass", syncnet_confidence="2.0"), outcome("pass", syncnet_confidence="1.0")]
    rc, out, _ = _compare_out(tmp_path, capsys, a, b)
    assert "mean syncnet_confidence: 8.00 -> 1.50 (-6.50)" in out
    assert not any("sync_peak_r" in line for line in out)
    assert "verdict: ACCEPT" in out and rc == 0  # not part of the accept rule

    rc, out, _ = _compare_out(tmp_path, capsys, [outcome("pass", syncnet_confidence="8.6")],
                              [outcome("pass")])  # fmt: skip
    assert not any(line.startswith("mean syncnet_confidence") for line in out)  # one side empty
    assert rc == 0


def test_compare_with_an_old_csv_without_the_column_omits_the_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hwva import eval as ev

    old_columns = [c for c in COLUMNS if c not in SYNC_COLUMNS]
    old = tmp_path / "old.csv"
    with old.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_columns)
        writer.writeheader()
        writer.writerow({c: "" for c in old_columns} | {"idea_id": "i0"} | outcome("pass"))
    new = write_results(tmp_path / "new.csv", [outcome("pass", syncnet_confidence="8.6")])
    assert ev.main(["--compare", str(new), str(new)]) == 0  # control: both sides have values
    assert "mean syncnet_confidence: 8.60 -> 8.60 (+0.00)" in capsys.readouterr().out
    for a, b in ((old, new), (new, old)):
        rc = ev.main(["--compare", str(a), str(b)])
        captured = capsys.readouterr()
        assert rc == 0, captured.err
        assert "mean syncnet_confidence" not in captured.out
        assert "verdict: ACCEPT" in captured.out


# --- regression: one real dev idea through the whole pipeline -----------------------------------

DEV_EVAL_MAX_S = 20.0  # a dev run is a few seconds of ffmpeg; generous for slow Windows CI


@needs_ffmpeg
def test_one_idea_dev_eval_end_to_end_is_fast(
    tmp_path: Path, isolated_runs: Path, no_stdin: None
) -> None:
    from hwva import eval as ev

    assert [s.name for s in pipeline.STAGES] == STAGE_NAMES  # the real pipeline, not a fake
    ideas = write_ideas(
        tmp_path / "ideas.yaml",
        {"id": "uk", "kind": "uk_quoted", "idea": 'Бабуся каже: "Та я сама все зробила!"'},
    )
    out = tmp_path / "dev.csv"
    started = time.perf_counter()
    rc = ev.main(["--profile", "dev", "--ideas", str(ideas), "--out", str(out)])
    elapsed = time.perf_counter() - started
    assert rc == 0
    header, rows = read_csv(out)
    assert header == COLUMNS and len(rows) == 1
    row = rows[0]
    assert row["status"] == "done", row["error"]
    assert row["final_verdict"] == "pass"
    assert all(row[f"fc_{c}"] == "pass" for c in CHECKS[:6])  # every hard check
    assert row["line"] == "Та я сама все зробила!" and row["line_source"] == "input"
    assert row["voice_persona"] == "female-adult" and row["voice_key"] == "female-adult"
    assert all(row[f"attempts_{s}"] == "1" for s in STAGE_NAMES)
    assert row["voice_tts_attempts"] == "1"
    assert row["fc_width"] == "720" and row["fc_height"] == "1280"
    assert {k: row[k] for k in DEV_MODELS} == DEV_MODELS
    assert all(row[f"judge_{j}"] == "" for j in JUDGE)  # no judge on dev
    # dev: SyncNet is unreachable in tests (conftest) -> verdict `error`, no alignment, no columns
    assert row["sync_verdict"] == "error"
    assert row["syncnet_confidence"] == "" and row["sync_method"] == ""
    assert float(row["cost_usd"]) == 0.0
    assert 0 < float(row["seconds"]) <= elapsed
    assert (isolated_runs / row["run_id"] / "final.mp4").exists()
    assert elapsed < DEV_EVAL_MAX_S, f"dev eval took {elapsed:.1f}s"


# --- wiring -------------------------------------------------------------------------------------


def test_console_script_is_registered() -> None:
    from hwva import eval as ev

    scripts = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts.get("eval") == "hwva.eval:main"
    assert callable(ev.main)


def test_results_dir_is_gitignored() -> None:
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert any(line.strip().rstrip("/") in {"evals/results", "/evals/results"} for line in lines)
