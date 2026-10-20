from pathlib import Path

import pytest

from hwva import cli, pipeline
from hwva.context import StageOutput
from hwva.providers import fakes
from hwva.providers.base import Capabilities
from hwva.strategy import Strategy, StrategyNotImplemented, choose_strategy, preflight


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (Capabilities(accepts_audio_input=True, native_speech=True), Strategy.A),  # A wins
        (Capabilities(accepts_audio_input=True), Strategy.A),
        (Capabilities(native_speech=True), Strategy.B),  # e.g. Veo
        (Capabilities(), Strategy.C),  # silent video, e.g. Kling
    ],
)
def test_choose_strategy_from_capabilities(cap: Capabilities, expected: Strategy) -> None:
    assert choose_strategy(cap) == expected


def test_preflight_passes_strategy_a() -> None:
    assert preflight(Capabilities(accepts_audio_input=True), "seedance") is Strategy.A


@pytest.mark.parametrize("cap", [Capabilities(native_speech=True), Capabilities()])
def test_preflight_fails_fast_for_unimplemented(cap: Capabilities) -> None:
    with pytest.raises(StrategyNotImplemented, match="Nothing was generated or paid for"):
        preflight(cap, "veo")


class SpeakingVideo(fakes.FakeVideo):
    capabilities = Capabilities(native_speech=True)  # like Veo: can't take our audio


def test_cli_stops_before_creating_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("HWVA_RUNS", str(runs))
    monkeypatch.setitem(cli.registry._FACTORIES["video"], "fake", lambda ref: SpeakingVideo())
    called: list[str] = []
    monkeypatch.setattr(
        pipeline, "STAGES", [pipeline.Stage("x", lambda c: called.append("x") or StageOutput())]
    )

    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_USAGE
    assert "strategy B" in capsys.readouterr().err
    assert called == []  # no stage ran (no TTS, no video)
    assert not runs.exists()  # no run dir created


def test_strategy_recorded_in_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hwva import manifest as mf

    monkeypatch.setenv("HWVA_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(pipeline, "STAGES", [pipeline.Stage("noop", lambda c: StageOutput())])
    assert cli.main(["idea", "--no-interactive"]) == cli.EXIT_OK
    run = next(mf.runs_root().iterdir())
    assert mf.load(run.name).strategy == "A"
