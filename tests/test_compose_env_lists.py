"""Each service gets a closed ``environment:`` list, not the whole ``.env`` (TASK-1476).

``env_file: ./.env`` hands a container every line of the deployment's ``.env``,
including credentials that moved into encrypted records and operator-only
names. The compose files instead pass each service the names it reads.

* No service loads ``.env`` through ``env_file`` unless :data:`EXEMPT` names
  it with the task that removes it. A stale exemption fails too.
* A retired name (adopted by :mod:`llm_bawt.env_import`, or never meant for a
  container) is neither passed nor interpolated into a service's environment,
  and no env template assigns an adopted one.
* No service receives an OpenClaw name: TASK-1439 deleted ``openclaw-bridge``.
* Every name ``.env.docker`` assigns still reaches a container: passed or
  interpolated.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from llm_bawt.env_import import SPECS

ROOT = Path(__file__).resolve().parents[1]

#: (compose file, service) -> the task that gives it an explicit list.
EXEMPT: dict[tuple[str, str], str] = {
    ("docker-compose.prod.yml", "app"): "TASK-1617: standalone stack gets explicit lists",
    ("docker-compose.prod.yml", "claude-code-bridge"): "TASK-1617: standalone stack gets explicit lists",
}

#: Adopted into records by env_import: the env line is deleted after the window.
ADOPTED = frozenset(name for spec in SPECS for name in spec.env_names)
#: Never for a container: the Bitwarden unlock, Compose's own file list, and
#: the Crawl4AI root token (bridges get a derived token file instead).
NOT_FOR_CONTAINERS = frozenset({"BW_PASSWORD", "COMPOSE_FILE", "CRAWL4AI_API_TOKEN"})
RETIRED = ADOPTED | NOT_FOR_CONTAINERS

_OPENCLAW = re.compile(r"^(LLM_BAWT_)?OPENCLAW_")

_INTERPOLATION = re.compile(r"\$(?:\$|\{([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))")
_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=")


def _tracked(pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", pattern], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def _compose_files() -> list[Path]:
    return _tracked("docker-compose*.yml")


def _env_templates() -> list[Path]:
    return [p for p in _tracked(".env*") if p.name != ".env"]


def _interpolated(text: str) -> set[str]:
    return {m.group(1) or m.group(2) for m in _INTERPOLATION.finditer(text) if m.group(1) or m.group(2)}


def _services(path: Path) -> dict[str, dict]:
    return (yaml.safe_load(path.read_text()) or {}).get("services") or {}


def _environment(service: dict) -> dict[str, str]:
    """``{name: raw value}``; a bare name (passed through from ``.env``) maps to ``""``."""
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k): "" if v is None else str(v) for k, v in env.items()}
    out: dict[str, str] = {}
    for item in env:
        name, _, value = str(item).partition("=")
        out[name] = value
    return out


def _env_files(service: dict) -> list[str]:
    files = service.get("env_file") or []
    if isinstance(files, (str, dict)):
        files = [files]
    return [f["path"] if isinstance(f, dict) else str(f) for f in files]


def _loads_dotenv(service: dict) -> bool:
    return any(Path(f).name == ".env" for f in _env_files(service))


def _reached(service: dict) -> set[str]:
    """Names a service's environment passes or interpolates."""
    env = _environment(service)
    return set(env) | {n for v in env.values() for n in _interpolated(v)}


def _template_names(path: Path) -> set[str]:
    return {m.group(1) for line in path.read_text().splitlines() if (m := _ASSIGNMENT.match(line))}


def test_the_guard_scans_the_shipped_files():
    names = {p.name for p in _compose_files()}
    assert {"docker-compose.yml", "docker-compose.ops.yml"} <= names
    assert ".env.docker" in {p.name for p in _env_templates()}
    assert ADOPTED, "env_import.SPECS is empty: the retired set would be vacuous"


def test_no_service_loads_dotenv_unless_exempt():
    loads = {
        (path.name, name)
        for path in _compose_files()
        for name, service in _services(path).items()
        if _loads_dotenv(service)
    }
    unexpected = sorted(loads - EXEMPT.keys())
    assert not unexpected, (
        f"env_file: .env on {unexpected}. Pass the names the service reads in an explicit "
        "environment: list instead."
    )
    stale = sorted(EXEMPT.keys() - loads)
    assert not stale, f"These no longer load .env; delete their EXEMPT entries: {stale}"


def test_no_retired_name_reaches_a_container():
    found = {
        f"{path.name}:{name}": sorted(_reached(service) & RETIRED)
        for path in _compose_files()
        for name, service in _services(path).items()
        if _reached(service) & RETIRED
    }
    assert not found, f"Retired names in a service environment: {found}"


def test_no_template_assigns_an_adopted_name():
    found = {p.name: sorted(_template_names(p) & ADOPTED) for p in _env_templates() if _template_names(p) & ADOPTED}
    assert not found, f"Env templates still assign credentials that live in records now: {found}"


def test_no_service_receives_an_openclaw_name():
    found = {
        f"{path.name}:{name}": sorted(n for n in _reached(service) if _OPENCLAW.match(n))
        for path in _compose_files()
        for name, service in _services(path).items()
        if any(_OPENCLAW.match(n) for n in _reached(service))
    }
    assert not found, f"OpenClaw names in a service environment: {found}"


def test_every_template_name_still_reaches_a_container():
    dev_files = [p for p in _compose_files() if p.name != "docker-compose.prod.yml"]
    reached = {n for p in dev_files for s in _services(p).values() for n in _reached(s)}
    interpolated = {n for p in dev_files for n in _interpolated(p.read_text())}
    exempt_wholesale = {service for (file, service) in EXEMPT if file != "docker-compose.prod.yml"}
    assert not exempt_wholesale, "a dev service loads all of .env again: exempt its names below"
    template = ROOT / ".env.docker"
    missing = sorted(
        n for n in _template_names(template)
        if n not in reached and n not in interpolated
    )
    assert not missing, (
        f".env.docker assigns {missing}, but no service receives them. Pass each to the "
        "service that reads it, or drop it from the template."
    )


@pytest.mark.parametrize(("service", "files"), [
    ({"env_file": "./.env"}, True),
    ({"env_file": [{"path": "./.env", "required": False}]}, True),
    ({"env_file": ["./other.env"]}, False),
    ({}, False),
])
def test_dotenv_detection(service, files):
    assert _loads_dotenv(service) is files


def test_reached_includes_interpolated_values():
    service = {"environment": ["A", "B=${ZAI_API_KEY:-}", "C=literal"]}
    assert _reached(service) == {"A", "B", "C", "ZAI_API_KEY"}
    assert _reached({"environment": {"X": None, "Y": "$BW_PASSWORD"}}) == {"X", "Y", "BW_PASSWORD"}
