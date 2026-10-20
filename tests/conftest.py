"""Shared fixtures for stage tests. Everything here is offline and free."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from hwva import manifest as mf
from hwva.context import Context
from hwva.providers import fakes, registry
from hwva.providers.base import ProviderError
from hwva.schemas import JudgeCriterion, JudgeDefect, JudgeVerdict, Lang, Script, VoicePersona
from hwva.voices import Voice

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(autouse=True)
def no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cli.main() loads .env; tests must not pick up the developer's real keys from the repo."""
    monkeypatch.setenv("HWVA_DOTENV", str(tmp_path / "no.env"))


@pytest.fixture(autouse=True)
def no_sync_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lip-sync / lead-in knobs (specs/sync.md, voice.md) start unset in every test: defaults are
    what the tests pin, a developer's shell must not change them. Tests set them explicitly."""
    for name in (
        "HWVA_LEAD_IN_S",
        "HWVA_SYNC_REJECT",
        "HWVA_SYNC_MOUTH_BOX",
        "HWVA_SYNC_CONF",  # v2
        "HWVA_SYNC_WEAK_CONF",  # v2
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_real_syncnet(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """specs/sync.md "Test isolation": the real SyncNet (~15 s per clip, a required dependency)
    is unreachable in every test — `import syncnet_python` raises ImportError —
    unless the test is marked `@pytest.mark.syncnet`. Tests that need SyncNet install a stub
    module (`install_syncnet`) or monkeypatch `hwva.sync.syncnet_measure`
    (`install_syncnet_measure`)."""
    if request.node.get_closest_marker("syncnet") is None:
        monkeypatch.setitem(sys.modules, "syncnet_python", None)


# --- configurable fakes -------------------------------------------------------------------------


class ScriptedStt:
    """Returns the given transcripts in order (last one repeats). Simulates mishearing."""

    def __init__(self, *transcripts: str) -> None:
        self.transcripts = list(transcripts)
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, lang: Lang) -> str:
        self.calls.append(audio_path)
        idx = min(len(self.calls) - 1, len(self.transcripts) - 1)
        return self.transcripts[idx]


class SpyTts(fakes.FakeTts):
    """Real FakeTts that records what it was asked to say, for which persona, and with which
    pinned voice mapping."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.personas: list[VoicePersona | None] = []
        self.voices: list[Voice | None] = []

    def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        persona: VoicePersona | None = None,
        voice: Voice | None = None,
    ) -> Path:
        self.texts.append(text)
        self.personas.append(persona)
        self.voices.append(voice)
        if persona is None and voice is None:  # callers from before TtsProvider took a persona
            return super().synthesize(text, lang, out_path)
        return super().synthesize(text, lang, out_path, persona, voice=voice)


class ParaphrasingLlm(fakes.FakeLlm):
    """An LLM that 'improves' the line — validate_script must undo it."""

    def write_script(self, idea: str, line: str | None, lang: Lang | None) -> Script:
        script = super().write_script(idea, line, lang)
        return script.model_copy(update={"line": script.line + " і все тут"})


def judge_verdict(*defect_timestamps: str, verdict: str = "pass", score: int = 5) -> JudgeVerdict:
    """A schema-valid verdict; every criterion gets `verdict`/`score`, one defect per timestamp."""
    criterion = JudgeCriterion.model_validate(
        {"verdict": verdict, "score": score, "evidence": "seen in the clip", "timestamp": "00:01"}
    )
    defects = [JudgeDefect(kind="artifact", timestamp=t, note=f"at {t}") for t in defect_timestamps]
    return JudgeVerdict(
        face_present=criterion,
        mouth_moves_when_speaking=criterion,
        matches_character=criterion,
        same_character_across_cut=criterion,
        matches_prompt=criterion,
        defects=defects,
    )


class FakeJudge:
    """Returns a fixed valid verdict; records (video_path, script) of every call."""

    def __init__(self, verdict: JudgeVerdict | None = None) -> None:
        self.verdict = verdict or judge_verdict()
        self.calls: list[tuple[Path, Script]] = []

    def judge(self, video_path: Path, script: Script) -> JudgeVerdict:
        self.calls.append((video_path, script))
        return self.verdict


class HallucinatingJudge(FakeJudge):
    """Reports defects at the given timestamps — some of them beyond the end of the video."""

    def __init__(self, *defect_timestamps: str) -> None:
        super().__init__(judge_verdict(*defect_timestamps))


class CrashingJudge:
    """Raises (quota, network, moderation...)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or ProviderError("judge quota exceeded", retryable=False)
        self.calls = 0

    def judge(self, video_path: Path, script: Script) -> JudgeVerdict:
        self.calls += 1
        raise self.exc


class InvalidJudge:
    """Returns what a model may send instead of the schema: raw text or a non-validating dict."""

    def __init__(self, raw: object = "Sure! The video looks great.") -> None:
        self.raw = raw
        self.calls = 0

    def judge(self, video_path: Path, script: Script) -> JudgeVerdict:
        self.calls += 1
        return self.raw  # type: ignore[return-value]


# --- context ------------------------------------------------------------------------------------


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def make_ctx(runs_root: Path) -> Callable[..., Context]:
    """make_ctx(idea=..., params=..., interactive=..., answers=[...], **provider_overrides)"""

    def _make(
        idea: str = 'Кіт каже: "Я тут головний"',
        params: dict[str, str] | None = None,
        interactive: bool = False,
        answers: list[str] | None = None,
        **overrides: Any,
    ) -> Context:
        providers = replace(registry.build("dev"), **overrides)
        m = mf.new_run(idea, "dev", providers.refs, params or {}, root=runs_root)
        queue = list(answers or [])
        asked: list[str] = []

        def ask(prompt: str) -> str:
            asked.append(prompt)
            if not queue:
                raise AssertionError(f"unexpected question: {prompt!r}")
            return queue.pop(0)

        ctx = Context(m, providers, mf.run_dir(m.run_id, runs_root), interactive, ask=ask)
        ctx.asked = asked  # type: ignore[attr-defined]  # test-only: inspect questions asked
        return ctx

    return _make


@pytest.fixture
def dev_ctx(make_ctx: Callable[..., Context]) -> Context:
    return make_ctx()


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def voices_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """voices_yaml(text): writes `config/voices.yaml` (next to a copy of profiles.yaml) and runs
    the test from there, so it is found both relative to the cwd and next to HWVA_CONFIG."""

    def _write(text: str) -> Path:
        cfg = tmp_path / "repo" / "config"
        cfg.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "config" / "profiles.yaml", cfg / "profiles.yaml")
        path = cfg / "voices.yaml"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("HWVA_CONFIG", str(cfg / "profiles.yaml"))
        monkeypatch.chdir(cfg.parent)
        return path

    return _write


@pytest.fixture
def no_stdin(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if anything reads real stdin."""

    def boom(*_: object) -> str:
        raise AssertionError("stdin must not be read")

    monkeypatch.setattr("builtins.input", boom)
    yield


# --- media helpers ------------------------------------------------------------------------------


def probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    data: dict[str, Any] = json.loads(out.stdout)
    return data


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def dominant_hz(path: Path) -> float:
    """Strongest frequency of the audio (decoded by ffmpeg to 16 kHz mono s16le)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / 16000)
    return float(freqs[int(np.argmax(spectrum[1:])) + 1])


def make_audio(path: Path, seconds: float, text: str = "") -> Path:
    """Audio file; if text is given, FakeStt will 'hear' it."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}", str(path)],
        check=True,
    )  # fmt: skip
    if text:
        path.with_name(path.name + ".txt").write_text(text, encoding="utf-8")
    return path


def make_video(
    path: Path, seconds: float, size: str = "720x1280", audio: bool = False, tone: int = 440
) -> Path:
    """Test-pattern video; with audio=True it carries its own sine track at `tone` Hz."""
    args = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={tone}:duration={seconds}", "-shortest"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def speech_wav(
    path: Path,
    runs: Sequence[tuple[float, float]],
    total_s: float,
    *,
    rate: int = 24000,
    amp: float = 0.5,
    floor: int = 0,
) -> Path:
    """Mono PCM 16-bit WAV of `total_s` at `rate` with synthetic 'speech' in each (start_s, end_s)
    run: a chirp whose start pitch differs per run (tonal, so AAC keeps the waveform; not periodic,
    so a cross-correlation has one sharp peak). Outside the runs every sample is `floor` (0 =
    digital silence; a small constant makes cut padding visible). Times are rounded to samples."""
    n = round(total_s * rate)
    samples = np.full(n, float(floor))
    for i, (start_s, end_s) in enumerate(runs):
        a, b = round(start_s * rate), round(end_s * rate)
        t = np.arange(b - a) / rate
        f0 = 220.0 + 90.0 * i
        samples[a:b] = amp * 32767 * np.sin(2 * np.pi * (f0 * t + 150.0 * t * t))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.round(samples).astype("<i2").tobytes())
    return path


def read_wav(path: Path) -> tuple[npt.NDArray[np.int16], int, int]:
    """(samples as int16, sample rate, channels) of a PCM 16-bit WAV."""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, f"{path.name}: not 16-bit PCM"
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").copy(), w.getframerate(), w.getnchannels()


# --- SyncNet stubs (specs/sync.md: unit tests never run the real model) --------------------------

SYNCNET_WEIGHT_FILES = ("syncnet_v2.model", "sfd_face.pth")
SYNCNET_WEIGHTS_SCRIPT = "get-syncnet-weights.ps1"
NO_FACE_RESULT: tuple[Any, ...] = ([], [], [], 0.0, 0.0, "", False)  # inference() without a face


@pytest.fixture
def syncnet_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """$HWVA_SYNCNET_WEIGHTS pointing to a dir with both (fake) weight files present."""
    d = tmp_path / "syncnet_weights"
    d.mkdir()
    for name in SYNCNET_WEIGHT_FILES:
        (d / name).write_bytes(b"fake weights")
    monkeypatch.setenv("HWVA_SYNCNET_WEIGHTS", str(d))
    return d


def syncnet_result(offset: int, confidence: float, min_dist: float = 7.0) -> tuple[Any, ...]:
    """What `SyncNetPipeline.inference()` returns for one face track:
    (offsets, confs, dists, max_conf, min_dist, faces, has_face)."""
    return ([offset], [confidence], [min_dist], confidence, min_dist, "{}", True)


@dataclass
class SyncNetModuleCalls:
    """What the stub `syncnet_python` module saw: (args, kwargs) of every constructor and
    `inference` call."""

    init: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    inference: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)


def install_syncnet(
    monkeypatch: pytest.MonkeyPatch,
    result: object = NO_FACE_RESULT,
    exc: BaseException | None = None,
    during: Callable[[], None] | None = None,
) -> SyncNetModuleCalls:
    """Put a stub `syncnet_python` (with `SyncNetPipeline`) into sys.modules: `inference` calls
    `during()` (to look at the process state mid-call), then raises `exc` or returns `result`."""
    calls = SyncNetModuleCalls()

    class SyncNetPipeline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.init.append((args, kwargs))

        def inference(self, *args: Any, **kwargs: Any) -> object:
            calls.inference.append((args, kwargs))
            if during is not None:
                during()
            if exc is not None:
                raise exc
            return result

    module = types.ModuleType("syncnet_python")
    module.SyncNetPipeline = SyncNetPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "syncnet_python", module)
    return calls


# (offset_frames, confidence) or (offset_frames, confidence, min_dist), or an exception to raise
SyncNetAnswer = tuple[int, float] | tuple[int, float, float] | BaseException


@dataclass
class StubSyncNetMeasure:
    """Stands in for `hwva.sync.syncnet_measure(video, audio)`. `answers` is keyed by the audio
    file name ("shot_audio.wav", "sync_global.wav", "sync_piecewise.wav", ...) or, to tell takes
    apart, by "<video name>|<audio name>" (checked first). A call without an answer fails the
    test. Returns `hwva.sync.SyncNetResult` (looked up at call time)."""

    answers: Mapping[str, SyncNetAnswer]
    calls: list[tuple[Path, Path]] = field(default_factory=list)

    def __call__(self, video: Path, audio: Path) -> Any:
        video, audio = Path(video), Path(audio)
        self.calls.append((video, audio))
        key = f"{video.name}|{audio.name}"
        answer = self.answers.get(key, self.answers.get(audio.name))
        if answer is None:
            raise AssertionError(f"unexpected syncnet_measure call: {key}")
        if isinstance(answer, BaseException):
            raise answer
        from hwva import sync

        offset, confidence, *rest = answer
        result_type: Any = getattr(sync, "SyncNetResult")  # noqa: B009 - v2 API, absent in v1
        return result_type(
            offset_frames=offset,
            confidence=confidence,
            min_dist=rest[0] if rest else 7.0,
            seconds=0.01,
        )

    def audio_names(self) -> list[str]:
        return [a.name for _, a in self.calls]

    def video_names(self) -> list[str]:
        return [v.name for v, _ in self.calls]


def install_syncnet_measure(
    monkeypatch: pytest.MonkeyPatch, answers: Mapping[str, SyncNetAnswer]
) -> StubSyncNetMeasure:
    """Monkeypatch `hwva.sync.syncnet_measure` (stages must look it up at call time:
    `from hwva import sync` ... `sync.syncnet_measure(...)`)."""
    from hwva import sync

    stub = StubSyncNetMeasure(answers)
    monkeypatch.setattr(sync, "syncnet_measure", stub, raising=False)
    return stub
