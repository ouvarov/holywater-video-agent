import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hwva import cli, pipeline
from hwva import manifest as mf
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers import registry
from hwva.providers.base import ProviderError
from hwva.schemas import StageStatus


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runs = tmp_path / "runs"
    monkeypatch.setenv("HWVA_RUNS", str(runs))
    return runs


def test_no_stages_is_not_reported_as_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "STAGES", [])
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_NOTHING_TO_RUN


def test_unknown_resume_id_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--resume", "nope"]) == cli.EXIT_USAGE
    assert "No run 'nope'" in capsys.readouterr().err


def test_unknown_profile_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["idea", "--profile", "nope"]) == cli.EXIT_USAGE
    assert "Unknown profile" in capsys.readouterr().err


def test_premium_without_real_providers_is_a_clean_error() -> None:
    assert cli.main(["idea", "--profile", "premium"]) == cli.EXIT_USAGE


def test_run_with_a_stage(monkeypatch: pytest.MonkeyPatch, isolated_runs: Path) -> None:
    def touch(c: Context) -> StageOutput:
        (c.run_dir / "x.txt").write_text("ok")
        return StageOutput(artifacts={"x": "x.txt"})

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("touch", touch)])
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_OK
    assert len(list(isolated_runs.glob("*/x.txt"))) == 1


def test_profile_validation_rejects_missing_roles(tmp_path: Path) -> None:
    cfg = tmp_path / "p.yaml"
    cfg.write_text("broken:\n  llm: {provider: fake, model: m, version: '1'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing roles"):
        registry.load_profile("broken", cfg)


def test_resume_uses_pinned_models_not_current_yaml() -> None:
    refs = registry.load_profile("dev")
    providers = registry.build("dev", refs=refs)
    assert providers.refs is refs


def test_provider_error_at_build_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # e.g. seedance with a plain-HTTP ARK_BASE_URL
    def refuse(*args: object, **kwargs: object) -> registry.Providers:
        raise ProviderError("ARK_BASE_URL must be https://", retryable=False)

    monkeypatch.setattr(registry, "build", refuse)
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_USAGE
    assert "ARK_BASE_URL" in capsys.readouterr().err


# --- resume from disk: earlier stages are not re-run, the voice pin and the cost stay ------------

VOICE_COST = 0.5


def test_cli_resume_from_disk_skips_done_stages_and_keeps_pin_and_cost(
    monkeypatch: pytest.MonkeyPatch, isolated_runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "syncnet_python", None)  # optional, slow; never a gate
    calls: dict[str, int] = {}
    broken = {"shot_1": True}

    def counted(st: pipeline.Stage) -> pipeline.Stage:
        def run(ctx: Context) -> StageOutput:
            calls[st.name] = calls.get(st.name, 0) + 1
            if broken.get(st.name):
                raise GateFailed("synthetic failure, fixed before the resume")
            out = st.run(ctx)
            if st.name == "voice":  # paid stage: its cost must be counted exactly once
                out = replace(out, cost_usd=VOICE_COST)
            return out

        return replace(st, run=run)

    monkeypatch.setattr(pipeline, "STAGES", [counted(st) for st in pipeline.STAGES])
    idea = 'Кіт каже: "Я тут головний"'
    assert cli.main([idea, "--no-interactive"]) == cli.EXIT_FAILED
    (run_dir,) = list(isolated_runs.iterdir())
    run_id = run_dir.name
    first = mf.load(run_id, isolated_runs)
    assert first.stages["shot_1"].status is StageStatus.FAILED
    pin = first.stages["voice"].data["voice"]
    assert pin and first.cost_usd == pytest.approx(VOICE_COST)
    assert calls == {"intake": 1, "script": 1, "voice": 1, "shot_1": 1}

    # fresh state: nothing from the first run is kept in memory, only the run dir on disk
    broken.clear()
    calls.clear()
    capsys.readouterr()
    assert cli.main(["--resume", run_id]) == cli.EXIT_OK, capsys.readouterr().err
    assert "intake" not in calls and "script" not in calls and "voice" not in calls
    assert calls["shot_1"] == 1
    resumed = mf.load(run_id, isolated_runs)
    assert all(r.status is StageStatus.DONE for r in resumed.stages.values())
    assert resumed.stages["voice"].data["voice"] == pin
    assert resumed.stages["voice"].cost_usd == pytest.approx(VOICE_COST)
    assert resumed.cost_usd == pytest.approx(VOICE_COST)  # fakes cost 0: voice counted once
    assert resumed.models == first.models


# --- image role, media host, money cap (live findings) --------------------------------

BASE = """
  llm:   {provider: fake, model: fake-llm, version: "1"}
  tts:   {provider: fake, model: sine, version: "1"}
  stt:   {provider: fake, model: sidecar, version: "1"}
"""
OR_20F = (
    "  video: {provider: openrouter-video, model: bytedance/seedance-2.0-fast,"
    ' version: "2.0-fast"}\n'
)
FAKE_IMAGE = '  image: {provider: fake, model: solid, version: "1"}\n'


def _profile(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "p.yaml"
    cfg.write_text("p:" + BASE + body, encoding="utf-8")
    return cfg


def test_dev_profile_has_a_free_image_provider() -> None:
    providers = registry.build("dev")
    assert providers.refs["image"].provider == "fake"
    assert providers.image is not None


def test_media_host_short_form_is_pinned_as_a_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    cfg = _profile(tmp_path, OR_20F + FAKE_IMAGE + "  media_host: fake\n")
    refs = registry.load_profile("p", cfg)
    assert refs["media_host"].provider == "fake"
    providers = registry.build("p", refs=refs)
    assert providers.media_host is not None
    assert providers.video.media_host is providers.media_host  # type: ignore[attr-defined]


def test_video_that_needs_an_image_without_an_image_role_is_refused_at_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    cfg = _profile(tmp_path, OR_20F + "  media_host: fake\n")
    with pytest.raises(ValueError, match="image"):
        registry.build("p", refs=registry.load_profile("p", cfg))


def test_openrouter_video_without_a_media_host_is_refused_at_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    cfg = _profile(tmp_path, OR_20F + FAKE_IMAGE)
    with pytest.raises(ValueError, match="media_host"):
        registry.build("p", refs=registry.load_profile("p", cfg))


def test_video_without_needs_image_needs_no_image_role(tmp_path: Path) -> None:
    cfg = _profile(tmp_path, '  video: {provider: fake, model: testsrc, version: "1"}\n')
    providers = registry.build("p", refs=registry.load_profile("p", cfg))
    assert "image" not in providers.refs


def test_max_cost_comes_from_env_then_profile_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HWVA_MAX_COST_USD", raising=False)
    plain = _profile(tmp_path, '  video: {provider: fake, model: testsrc, version: "1"}\n')
    assert registry.load_max_cost_usd("p", plain) == 3.0
    capped = tmp_path / "capped.yaml"
    capped.write_text(plain.read_text(encoding="utf-8") + "  max_cost_usd: 1.5\n", "utf-8")
    assert registry.load_max_cost_usd("p", capped) == 1.5
    assert "max_cost_usd" not in registry.load_profile("p", capped)  # an option, not a role
    monkeypatch.setenv("HWVA_MAX_COST_USD", "0.75")
    assert registry.load_max_cost_usd("p", capped) == 0.75


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "nan", "inf"])
def test_bad_max_cost_is_a_value_error_naming_the_variable(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _profile(tmp_path, '  video: {provider: fake, model: testsrc, version: "1"}\n')
    monkeypatch.setenv("HWVA_MAX_COST_USD", bad)
    with pytest.raises(ValueError, match="HWVA_MAX_COST_USD"):
        registry.load_max_cost_usd("p", cfg)


def test_cli_gives_the_cap_to_the_stages_on_a_new_run_and_on_resume(
    monkeypatch: pytest.MonkeyPatch, isolated_runs: Path
) -> None:
    seen: list[float] = []
    fail = {"once": True}

    def spy(ctx: Context) -> StageOutput:
        seen.append(ctx.max_cost_usd)
        if fail.pop("once", False):
            raise GateFailed("first run fails")
        return StageOutput()

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("spy", spy)])
    monkeypatch.setenv("HWVA_MAX_COST_USD", "1.5")
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_FAILED
    (run_dir,) = list(isolated_runs.iterdir())
    monkeypatch.setenv("HWVA_MAX_COST_USD", "2.5")  # raised to continue
    assert cli.main(["--resume", run_dir.name]) == cli.EXIT_OK
    assert seen == [1.5, 2.5]


def test_cli_resume_of_a_profile_gone_from_the_yaml_uses_the_default_cap(
    monkeypatch: pytest.MonkeyPatch, isolated_runs: Path, tmp_path: Path
) -> None:
    """Resume runs on the refs pinned in the manifest; a renamed profile must not block it."""
    seen: list[float] = []
    fail = {"once": True}

    def spy(ctx: Context) -> StageOutput:
        seen.append(ctx.max_cost_usd)
        if fail.pop("once", False):
            raise GateFailed("first run fails")
        return StageOutput()

    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("spy", spy)])
    monkeypatch.delenv("HWVA_MAX_COST_USD", raising=False)
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_FAILED
    (run_dir,) = list(isolated_runs.iterdir())
    other = tmp_path / "profiles.yaml"
    other.write_text("renamed: {}\n", encoding="utf-8")
    monkeypatch.setenv("HWVA_CONFIG", str(other))
    assert cli.main(["--resume", run_dir.name]) == cli.EXIT_OK
    assert seen == [3.0, 3.0]


def test_cli_bad_cap_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], isolated_runs: Path
) -> None:
    monkeypatch.setenv("HWVA_MAX_COST_USD", "lots")
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_USAGE
    assert "HWVA_MAX_COST_USD" in capsys.readouterr().err
    assert not isolated_runs.exists() or not list(isolated_runs.iterdir())  # nothing created


# --- lip-sync warning (specs/sync.md "Record") ----------------------------------

SYNC_WARNING = "lip-sync not detected (peak r 0.43 < 0.6); audio not shifted"


def _shot_1_with_sync(warning: str | None) -> pipeline.Stage:
    sync = {"lag_s": 0.25, "peak_r": 0.43, "r_at_0": 0.1, "verdict": "no_sync", "shift_s": 0.0,
            "box": [0.2, 0.8, 0.42, 0.6], "take": 1, "error": None, "warning": warning,
            "rejected": None}  # fmt: skip

    def run(ctx: Context) -> StageOutput:
        return StageOutput(data={"duration_s": 5.0, "sync": sync})

    return pipeline.Stage("shot_1", run)


def test_cli_prints_the_sync_warning_once_and_the_run_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def later(ctx: Context) -> StageOutput:
        return StageOutput()

    stages = [_shot_1_with_sync(SYNC_WARNING), pipeline.Stage("later", later)]
    monkeypatch.setattr(pipeline, "STAGES", stages)
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("warning:")]
    assert lines == [f"warning: {SYNC_WARNING}"]
    assert "warning:" not in captured.out
    assert captured.err.rstrip().endswith(f"warning: {SYNC_WARNING}")  # at the end of the run


def test_cli_prints_no_sync_warning_without_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression guard (not xfail): no `sync.warning` -> no warning line."""
    monkeypatch.setattr(pipeline, "STAGES", [_shot_1_with_sync(None)])
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_OK
    assert "warning:" not in capsys.readouterr().err
