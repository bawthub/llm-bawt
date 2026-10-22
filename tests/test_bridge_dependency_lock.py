"""TASK-1269: the bridge image and the test venv must not drift apart silently.

The bridge image installs from ``docker/agent-bridge-constraints.txt``; the
tests run against ``uv.lock``. Code that reads an SDK's private layout
(``claude_code_bridge.run_teardown``) is only proven by the tests if both
resolve the same version, so those packages must match exactly. Every other
package the two files share either matches or sits in ``KNOWN_DRIFT`` with
both versions. That makes any new or changed drift a deliberate edit here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "docker" / "agent-bridge-constraints.txt"
LOCK = ROOT / "uv.lock"

# Bridge code depends on these packages' private internals.
MUST_MATCH = frozenset({"claude-agent-sdk"})

# Pre-existing drift as of TASK-1269: name -> (bridge constraint, uv.lock).
# Shrink this as the app lock catches up. Never add an SDK from MUST_MATCH.
KNOWN_DRIFT = {
    "annotated-doc": ("0.0.5", "0.0.4"),
    "annotated-types": ("0.8.0", "0.7.0"),
    "anyio": ("4.15.1", "4.12.1"),
    "attrs": ("26.1.0", "25.4.0"),
    "certifi": ("2026.7.22", "2026.2.25"),
    "cffi": ("2.1.1", "2.0.0"),
    "click": ("8.5.0", "8.3.1"),
    "cryptography": ("50.0.1", "46.0.5"),
    "fastapi": ("0.141.1", "0.135.1"),
    "hiredis": ("3.4.2", "3.3.0"),
    "idna": ("3.20", "3.11"),
    "jiter": ("0.17.0", "0.13.0"),
    "mcp": ("2.2.0", "1.26.0"),
    "openai": ("3.19.2", "2.24.0"),
    "packaging": ("26.3", "26.0"),
    "pillow": ("12.3.0", "12.1.1"),
    "psycopg2-binary": ("2.9.13", "2.9.11"),
    "pydantic": ("2.13.5", "2.12.5"),
    "pydantic-core": ("2.46.5", "2.41.5"),
    "pyjwt": ("2.15.0", "2.11.0"),
    "python-multipart": ("0.0.32", "0.0.22"),
    "redis": ("8.1.0", "7.2.1"),
    "rpds-py": ("2026.6.3", "0.30.0"),
    "sqlalchemy": ("2.1.1", "2.0.48"),
    "sse-starlette": ("3.4.11", "3.3.2"),
    "starlette": ("1.7.0", "0.52.1"),
    "typing-inspection": ("0.4.4", "0.4.2"),
    "typing-extensions": ("4.16.0", "4.15.0"),
    "uvicorn": ("0.54.0", "0.41.0"),
    "websockets": ("17.1", "16.0"),
}


def _name(raw: str) -> str:
    return re.sub(r"[-_.]+", "-", raw).lower()


def constraint_pins(text: str) -> dict[str, str]:
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        assert sep, f"constraint is not an exact pin: {line!r}"
        pins[_name(name)] = version.strip()
    return pins


def lock_versions(text: str) -> dict[str, set[str]]:
    versions: dict[str, set[str]] = {}
    for package in tomllib.loads(text)["package"]:
        versions.setdefault(_name(package["name"]), set()).add(package["version"])
    return versions


def drift_problems(constraints: str, lock: str) -> list[str]:
    pins, locked = constraint_pins(constraints), lock_versions(lock)
    problems = []
    for name in sorted(MUST_MATCH):
        if pins.get(name) is None or {pins.get(name)} != locked.get(name):
            problems.append(f"{name}: bridge {pins.get(name)} != lock {sorted(locked.get(name, ()))}")
    for name, pin in sorted(pins.items()):
        if name in MUST_MATCH:
            continue
        if name not in locked:
            if name in KNOWN_DRIFT:
                problems.append(f"{name}: in KNOWN_DRIFT but no longer in uv.lock")
            continue
        lock_version = ",".join(sorted(locked[name]))
        if lock_version == pin:
            if name in KNOWN_DRIFT:
                problems.append(f"{name}: now agrees at {pin}; drop it from KNOWN_DRIFT")
        elif KNOWN_DRIFT.get(name) != (pin, lock_version):
            problems.append(f"{name}: bridge {pin} != lock {lock_version}, not in KNOWN_DRIFT")
    for name in sorted(KNOWN_DRIFT.keys() - pins.keys()):
        problems.append(f"{name}: in KNOWN_DRIFT but no longer a bridge constraint")
    return problems


def test_bridge_constraints_and_uv_lock_agree():
    assert drift_problems(CONSTRAINTS.read_text(), LOCK.read_text()) == []


def test_an_sdk_is_never_parked_as_known_drift():
    assert not MUST_MATCH & KNOWN_DRIFT.keys()


def _lock(**packages: str) -> str:
    return "\n".join(f'[[package]]\nname = "{n}"\nversion = "{v}"\n' for n, v in packages.items())


def test_detector_catches_each_kind_of_drift():
    lock = _lock(**{"claude-agent-sdk": "0.2.160", "anyio": "4.12.1", "redis": "8.1.0"})
    pins = "claude-agent-sdk==0.2.160\nanyio==4.15.1\nredis==8.1.0\n"
    known = drift_problems(pins, lock)
    assert all("anyio" not in p for p in known)  # recorded drift is accepted
    assert any("annotated-doc" in p and "no longer" in p for p in known)

    assert any(p.startswith("claude-agent-sdk:") for p in drift_problems(
        pins.replace("0.2.160", "0.2.172"), lock))
    assert any(p.startswith("redis:") and "KNOWN_DRIFT" in p for p in drift_problems(
        pins.replace("redis==8.1.0", "redis==8.2.0"), lock))
    assert any(p.startswith("anyio:") for p in drift_problems(
        pins.replace("anyio==4.15.1", "anyio==4.16.0"), lock))
    # A recorded package that leaves the lock is stale too.
    assert any(p.startswith("anyio:") and "uv.lock" in p for p in drift_problems(
        pins, _lock(**{"claude-agent-sdk": "0.2.160", "redis": "8.1.0"})))
    # Drift that disappears must be removed from the table.
    assert any(p.startswith("redis:") and "drop it" in p for p in drift_problems(
        pins.replace("redis==8.1.0", "redis==7.2.1"), lock.replace("8.1.0", "7.2.1")))
