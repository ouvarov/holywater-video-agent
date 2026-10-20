"""Deterministic architecture gates (same idea as the regex gates in AI Hub's builder):
rules a prompt can't enforce, a test can."""

import ast
import re
from pathlib import Path

from hwva import pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hwva"
PROTOCOLS = "hwva.providers.base"


def _concrete(name: str) -> bool:
    """Any provider module except the protocols: fakes, registry, gemini, future adapters."""
    return name.startswith("hwva.providers.") and not (
        name == PROTOCOLS or name.startswith(PROTOCOLS + ".")
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_stages_depend_only_on_provider_protocols() -> None:
    offenders = {
        str(p.relative_to(SRC)): sorted(i for i in _imports(p) if _concrete(i))
        for p in (SRC / "stages").glob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def _spec_order() -> list[str]:
    table = (ROOT / "specs" / "README.md").read_text(encoding="utf-8")
    return re.findall(r"^\| \d+ \| \[(\w+)\]", table, flags=re.MULTILINE)


def test_stages_follow_specs_order() -> None:
    """Built stages are registered, in spec order, with no gaps: STAGES must be a prefix of the
    table in specs/README.md. Deleting a registration or reordering stages turns this red."""
    built = sorted(p.stem for p in (SRC / "stages").glob("*.py") if p.stem != "__init__")
    registered = [s.name for s in pipeline.STAGES]
    order = _spec_order()
    assert order, "could not parse the stage table in specs/README.md"
    assert registered == order[: len(registered)], f"STAGES {registered} vs specs {order}"
    assert sorted(registered) == built, f"built but not registered: {set(built) - set(registered)}"


# Key VALUES, not header names: `xi-api-key` is the name ElevenLabs requires; only a literal
# value next to it is a leak. Google keys come as `AIza...` or `AQ.` + token.
_KEY_VALUE = re.compile(
    r"AIza[0-9A-Za-z_\-]{20,}"
    r"|AQ\.[0-9A-Za-z_\-]{20,}"
    r"|\bsk[-_][0-9A-Za-z]{16,}"
    r"|[\"']xi-api-key[\"']\s*:\s*[\"'][^\"']+[\"']"
    r"|Bearer [0-9A-Za-z_\-.]{20,}"
)


def test_no_api_keys_in_source() -> None:
    hits = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if _KEY_VALUE.search(p.read_text(encoding="utf-8"))
    ]
    assert hits == []


def test_key_value_pattern_catches_literals_not_header_names() -> None:
    leaks = [
        'KEY = "AIzaSyA1234567890abcdefghijklmnop"',
        'KEY = "AQ.Ab1234567890abcdefghijklmnop"',
        'KEY = "sk_0123456789abcdef0123"',
        'headers={"xi-api-key": "abc123"}',
        'auth = "Bearer abcdefghijklmnopqrstuvwxyz"',
    ]
    fine = [
        'headers={"xi-api-key": self._key}',
        'API_KEY_ENV = "ELEVENLABS_API_KEY"',
        'f"Bearer {key}"',
    ]
    assert all(_KEY_VALUE.search(s) for s in leaks)
    assert not any(_KEY_VALUE.search(s) for s in fine)
