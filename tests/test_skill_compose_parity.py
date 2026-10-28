"""TASK-1309: the app reads skills from the tree the agent bridge loads.

The skill catalog, the injected-skills section and its rendered ``base_dir``
run in ``app``; the skills themselves are loaded by ``claude-code-bridge``.
Both must name the same path, and that path must come from the same host or
volume source in both containers, read-only in the app. Source guards; the
live mount is checked at activation.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ENV = "CLAUDE_CODE_SKILLS_PATH"


def _env(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {k: str(v) for k, v in env.items()}
    # A bare NAME passes the deployment .env value through; it has no literal here.
    return {name: value for name, _, value in (item.partition("=") for item in env)}


def _split(volume: str) -> list[str]:
    """Split a short-syntax volume on ':' outside ``${...}`` substitutions."""
    parts, current, depth = [], "", 0
    for char in volume:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == ":" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [*parts, current]


def _mounts(service: dict) -> list[tuple[str, PurePosixPath, bool]]:
    out = []
    for volume in service.get("volumes") or []:
        source, target, *rest = _split(volume)
        out.append((source, PurePosixPath(target), "ro" in (rest[0].split(",") if rest else [])))
    return out


def _backing(service: dict, path: str) -> tuple[str, bool]:
    """The source that serves ``path`` in this container and whether it is read-only."""
    wanted = PurePosixPath(path)
    covering = [m for m in _mounts(service) if wanted == m[1] or wanted.is_relative_to(m[1])]
    assert covering, f"nothing mounted at {path}"
    source, target, read_only = max(covering, key=lambda m: len(m[1].parts))
    rest = wanted.relative_to(target)
    return (source if rest == PurePosixPath(".") else f"{source}/{rest}"), read_only


@pytest.mark.parametrize("compose", ["docker-compose.yml", "docker-compose.prod.yml"])
def test_app_reads_the_bridge_skill_tree_at_the_bridge_path_read_only(compose):
    services = yaml.safe_load((ROOT / compose).read_text())["services"]
    app, bridge = services["app"], services["claude-code-bridge"]

    path = _env(bridge)[SKILLS_ENV]
    assert _env(app).get(SKILLS_ENV) == path
    app_source, app_read_only = _backing(app, path)
    assert app_source == _backing(bridge, path)[0]
    assert app_read_only
