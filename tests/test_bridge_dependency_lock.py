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
# TASK-2305: each entry says why it is here. DIRECT entries are imported by
# bridge code and are bumped in uv.lock next; the rest are transitive or not
# loaded in the bridge and stay as named exceptions until their parent moves.
# Classified against the running bridge on 2026-11-14 (import every
# claude_code_bridge module, then map loaded modules to distributions).
KNOWN_DRIFT = {
    "annotated-doc": ("0.0.5", "0.0.4"),  # transitive: fastapi
    "annotated-types": ("0.8.0", "0.7.0"),  # transitive: pydantic
    "attrs": ("26.1.0", "25.4.0"),  # transitive: jsonschema/referencing (mcp)
    "fastapi": ("0.141.1", "0.135.1"),  # DIRECT: imported by bridge code
    "packaging": ("26.3", "26.0"),  # transitive: openai-codex; not loaded by bridge code
    "psycopg2-binary": ("2.9.13", "2.9.11"),  # llm-bawt requirement; not loaded by bridge code
    "pydantic": ("2.13.5", "2.12.5"),  # transitive: fastapi/mcp/openai
    "pydantic-core": ("2.46.5", "2.41.5"),  # transitive: pydantic
    "rpds-py": ("2026.6.3", "0.30.0"),  # transitive: jsonschema/referencing (mcp)
    "sse-starlette": ("3.4.11", "3.3.2"),  # transitive: mcp
    "typing-inspection": ("0.4.4", "0.4.2"),  # transitive: pydantic/fastapi/mcp
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
    lock = _lock(**{"claude-agent-sdk": "0.2.160", "fastapi": "0.135.1", "redis": "8.1.0"})
    pins = "claude-agent-sdk==0.2.160\nfastapi==0.141.1\nredis==8.1.0\n"
    known = drift_problems(pins, lock)
    assert all("fastapi" not in p for p in known)  # recorded drift is accepted
    assert any("annotated-doc" in p and "no longer" in p for p in known)

    assert any(p.startswith("claude-agent-sdk:") for p in drift_problems(
        pins.replace("0.2.160", "0.2.172"), lock))
    assert any(p.startswith("redis:") and "KNOWN_DRIFT" in p for p in drift_problems(
        pins.replace("redis==8.1.0", "redis==8.2.0"), lock))
    assert any(p.startswith("fastapi:") for p in drift_problems(
        pins.replace("fastapi==0.141.1", "fastapi==0.142.0"), lock))
    # A recorded package that leaves the lock is stale too.
    assert any(p.startswith("fastapi:") and "uv.lock" in p for p in drift_problems(
        pins, _lock(**{"claude-agent-sdk": "0.2.160", "redis": "8.1.0"})))
    # Drift that disappears must be removed from the table.
    assert any(p.startswith("fastapi:") and "drop it" in p for p in drift_problems(
        pins, lock.replace("0.135.1", "0.141.1")))


def test_aligned_packages_cannot_drift_again():
    for name, version in {
        "anyio": "4.15.1",
        "uvicorn": "0.54.0",
        "typing-extensions": "4.16.0",
        "httptools": "0.9.0",
        "jiter": "0.17.0",
        "redis": "8.1.0",
        "hiredis": "3.4.2",
        "sqlalchemy": "2.1.1",
        "websockets": "17.1",
        "cryptography": "50.0.1",
        "cffi": "2.1.1",
    }.items():
        assert name not in KNOWN_DRIFT
        pins = f"claude-agent-sdk==0.2.160\n{name}=={version}\n"
        lock = _lock(**{"claude-agent-sdk": "0.2.160", name: version})
        assert not any(p.startswith(f"{name}:") for p in drift_problems(pins, lock))
        assert any(p.startswith(f"{name}:") and "not in KNOWN_DRIFT" in p
                   for p in drift_problems(pins, lock.replace(version, "0.0.0")))


def test_openai_major_is_aligned_with_no_exception():
    assert "openai" not in KNOWN_DRIFT
    pins = "claude-agent-sdk==0.2.160\nopenai==3.19.2\n"
    lock = _lock(**{"claude-agent-sdk": "0.2.160", "openai": "3.19.2"})
    assert not any(p.startswith("openai:") for p in drift_problems(pins, lock))
    assert any(p.startswith("openai:") and "not in KNOWN_DRIFT" in p
               for p in drift_problems(pins, lock.replace("3.19.2", "2.24.0")))
