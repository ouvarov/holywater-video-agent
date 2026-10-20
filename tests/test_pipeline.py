from collections.abc import Callable
from pathlib import Path

import pytest

from hwva import manifest as mf
from hwva import pipeline
from hwva.context import Context, GateFailed, StageOutput
from hwva.providers.base import ProviderError
from hwva.schemas import StageStatus


def ok(name: str, log: list[str]) -> Callable[[Context], StageOutput]:
    def _run(c: Context) -> StageOutput:
        log.append(name)
        return StageOutput(artifacts={name: f"{name}.txt"}, data={"n": 1}, cost_usd=0.5)

    return _run


def test_runs_stages_in_order_and_records_output(dev_ctx: Context, runs_root: Path) -> None:
    order: list[str] = []
    stages = [pipeline.Stage("a", ok("a", order)), pipeline.Stage("b", ok("b", order))]
    result = pipeline.run(dev_ctx, stages, root=runs_root)
    assert order == ["a", "b"]
    assert result.stages["a"].artifacts == {"a": "a.txt"}
    assert result.stages["a"].data == {"n": 1}
    assert result.cost_usd == pytest.approx(1.0)
    saved = mf.load(result.run_id, runs_root)
    assert saved.stages["b"].status is StageStatus.DONE
    assert saved.stages["b"].finished_at is not None


def test_gate_failure_stops_pipeline(dev_ctx: Context, runs_root: Path) -> None:
    def bad(c: Context) -> StageOutput:
        raise GateFailed("line not verbatim")

    ran_after: list[str] = []
    stages = [pipeline.Stage("gate", bad), pipeline.Stage("next", ok("next", ran_after))]
    result = pipeline.run(dev_ctx, stages, root=runs_root)
    assert result.stages["gate"].status is StageStatus.FAILED
    assert result.stages["gate"].error == "GateFailed: line not verbatim"
    assert ran_after == []


def test_retryable_provider_error_is_retried(dev_ctx: Context, runs_root: Path) -> None:
    calls = {"n": 0}

    def flaky(c: Context) -> StageOutput:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("timeout")
        return StageOutput()

    result = pipeline.run(dev_ctx, [pipeline.Stage("flaky", flaky, max_attempts=2)], runs_root)
    assert result.stages["flaky"].status is StageStatus.DONE
    assert result.stages["flaky"].attempts == 2


def test_retries_stop_at_max_attempts(dev_ctx: Context, runs_root: Path) -> None:
    calls = {"n": 0}

    def always(c: Context) -> StageOutput:
        calls["n"] += 1
        raise ProviderError("timeout")

    result = pipeline.run(dev_ctx, [pipeline.Stage("s", always, max_attempts=3)], runs_root)
    assert calls["n"] == 3
    assert result.stages["s"].status is StageStatus.FAILED


def test_non_retryable_provider_error_is_not_retried(dev_ctx: Context, runs_root: Path) -> None:
    calls = {"n": 0}

    def moderated(c: Context) -> StageOutput:
        calls["n"] += 1
        raise ProviderError("moderation", retryable=False)

    pipeline.run(dev_ctx, [pipeline.Stage("s", moderated, max_attempts=3)], runs_root)
    assert calls["n"] == 1


def test_unexpected_exception_fails_stage_with_readable_error(
    dev_ctx: Context, runs_root: Path
) -> None:
    def crash(c: Context) -> StageOutput:
        raise KeyError("missing")

    result = pipeline.run(dev_ctx, [pipeline.Stage("s", crash)], runs_root)
    record = mf.load(result.run_id, runs_root).stages["s"]
    assert record.status is StageStatus.FAILED
    assert record.error is not None and record.error.startswith("Unexpected KeyError")
    assert record.finished_at is not None


def test_resume_skips_done_stages_and_resets_attempts(dev_ctx: Context, runs_root: Path) -> None:
    calls: list[str] = []
    fail = {"times": 2}

    def second(c: Context) -> StageOutput:
        calls.append("second")
        if fail["times"]:
            fail["times"] -= 1
            raise ProviderError("timeout")
        return StageOutput()

    stages = [pipeline.Stage("first", ok("first", calls)),
              pipeline.Stage("second", second, max_attempts=2)]  # fmt: skip
    first_run = pipeline.run(dev_ctx, stages, root=runs_root)
    assert first_run.stages["second"].status is StageStatus.FAILED

    resumed = pipeline.run(dev_ctx, stages, root=runs_root)  # what --resume does
    assert resumed.stages["second"].status is StageStatus.DONE
    assert resumed.stages["second"].attempts == 1  # fresh budget on resume
    assert calls.count("first") == 1


def test_resume_that_commits_nothing_new_keeps_the_committed_stage_cost(
    dev_ctx: Context, runs_root: Path
) -> None:
    """Attempt 1 commits paid calls then fails a gate; the resume reuses them (commits 0.0):
    the stage record keeps what was paid, it is not overwritten with 0."""
    runs = {"n": 0}

    def paid(c: Context) -> StageOutput:
        runs["n"] += 1
        if runs["n"] == 1:
            c.commit_cost(0.04)  # image
            c.commit_cost(0.17)  # clip
            raise GateFailed("last frame failed")
        c.commit_cost(0.0)  # everything reused
        return StageOutput(cost_usd=0.0)

    stages = [pipeline.Stage("s", paid)]
    assert pipeline.run(dev_ctx, stages, root=runs_root).stages["s"].status is StageStatus.FAILED
    resumed = pipeline.run(dev_ctx, stages, root=runs_root)
    saved = mf.load(resumed.run_id, runs_root)
    assert saved.stages["s"].status is StageStatus.DONE
    assert saved.stages["s"].cost_usd == pytest.approx(0.21)
    assert saved.cost_usd == pytest.approx(0.21)
