"""How the dialogue gets into the video depends on what the video model can do — decided from the
capabilities the adapter declares, never from model names.

  A  model accepts audio      -> our TTS drives the lips           verbatim by construction
  B  model speaks by itself   -> line in prompt, STT verifies     verbatim is probabilistic,
                                                                  every retry costs a video
  C  model makes silent video -> our TTS + separate lip-sync model

Preference A > B > C is the one hard-coded rule: it orders strategies by how strongly they
guarantee the verbatim line. The PoC implements A; for B/C the run fails at the start, before
anything is spent, instead of silently falling back to another model."""

from __future__ import annotations

from enum import StrEnum

from hwva.providers.base import Capabilities


class Strategy(StrEnum):
    A = "A"
    B = "B"
    C = "C"


IMPLEMENTED = {Strategy.A}

_WHY = {
    Strategy.B: "speaks by itself from the prompt (strategy B: STT-verified model speech)",
    Strategy.C: "produces silent video (strategy C: separate lip-sync model)",
}


class StrategyNotImplemented(NotImplementedError):
    pass


def choose_strategy(cap: Capabilities) -> Strategy:
    if cap.accepts_audio_input:
        return Strategy.A
    if cap.native_speech:
        return Strategy.B
    return Strategy.C


def preflight(cap: Capabilities, model: str) -> Strategy:
    """Pick the strategy; fail fast if the PoC doesn't implement it."""
    strategy = choose_strategy(cap)
    if strategy not in IMPLEMENTED:
        raise StrategyNotImplemented(
            f"Video model {model!r} {_WHY[strategy]}, which is not implemented in the PoC. "
            f"Use a profile whose video model accepts audio input (e.g. seedance 2.5 in "
            f"config/profiles.yaml). Nothing was generated or paid for."
        )
    return strategy
