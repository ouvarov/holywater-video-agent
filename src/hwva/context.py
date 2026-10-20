"""What a stage receives and returns. Kept apart from pipeline.py so stages can import it without
a cycle (pipeline imports the stages)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hwva.providers.registry import DEFAULT_MAX_COST_USD, MAX_COST_ENV, Providers
from hwva.schemas import Manifest

COST_EPS = 1e-9  # float sums of prices: 0.1 + 0.2 must count as 0.3 at a 0.3 cap


def _ask_stdin(prompt: str) -> str:
    return input(prompt)


def _no_commit(usd: float) -> None:
    """Default outside the pipeline (a stage called directly, e.g. in tests): nothing to save."""


@dataclass
class Context:
    manifest: Manifest
    providers: Providers
    run_dir: Path
    interactive: bool
    ask: Callable[[str], str] = _ask_stdin  # injected in tests; only intake may call it
    # Persist money already spent NOW (stage record + manifest total, saved to disk) so a later
    # failure in the same stage never loses it. Set by the pipeline per stage. The amount stays
    # part of StageOutput.cost_usd; on success the pipeline adds only the uncommitted rest.
    commit_cost: Callable[[float], None] = _no_commit
    # Money cap per run (USD): a stage refuses a paid call once manifest.cost_usd reached it.
    # Set by the cli from HWVA_MAX_COST_USD, else the profile's max_cost_usd, else this default.
    max_cost_usd: float = DEFAULT_MAX_COST_USD

    def check_cost_cap(self, call: str) -> None:
        """Call right before a paid `call`: GateFailed once the committed cost reached the cap.
        Prices are known only after a call, so one call can overshoot the cap by its own price."""
        spent = self.manifest.cost_usd
        if spent + COST_EPS >= self.max_cost_usd:
            raise GateFailed(
                f"cost cap reached: {spent:.2f} USD committed in this run, cap "
                f"{self.max_cost_usd:.2f} USD ({MAX_COST_ENV}); {call} not started. Raise "
                f"{MAX_COST_ENV} and --resume to continue."
            )


@dataclass
class StageOutput:
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path relative to run dir
    data: dict[str, Any] = field(default_factory=dict)  # durations, diffs, scores...
    cost_usd: float = 0.0


class GateFailed(RuntimeError):
    """A stage ran but its output did not pass the check. Not retried by the pipeline — the stage
    owns its own retries (e.g. TTS up to N times) before raising this."""
