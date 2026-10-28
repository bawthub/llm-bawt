"""TASK-1604: the app's exported bridge tokens reach every reader in Compose.

The app derives the broker, git-credential and Crawl4AI bearers and writes
each to a configured export file (``bridge_gate``, ``crawl4ai_token_export``).
A reader finds its token only if its ``*_FILE`` names the same path, served by
the same named volume, read-only in the reader. Enforcing the broker gate
before every bridge sends the derived token locks the bridges out, so the
wiring is guarded here; the live counts are checked at activation.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import yaml

from agent_bridge import bridge_token, crawl4ai_token
from llm_bawt.service import bridge_gate, crawl4ai_token_export

ROOT = Path(__file__).resolve().parents[1]
SERVICES = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
VOLUMES = set(yaml.safe_load((ROOT / "docker-compose.yml").read_text())["volumes"])

_CRAWL4AI_FOR_BRIDGES = next(t for t in crawl4ai_token_export.TARGETS if t.slug == "bridges")
_CRAWL4AI_FOR_CRAWL4AI = next(t for t in crawl4ai_token_export.TARGETS if t.slug == "crawl4ai")

#: (app export env, reader env) for each token a bridge reads.
BROKER = (bridge_gate.BROKER.export_env, bridge_token.BROKER.file_env)
GIT_CREDENTIAL = (bridge_gate.GIT_CREDENTIAL.export_env, bridge_token.GIT_CREDENTIAL.file_env)
CRAWL4AI = (_CRAWL4AI_FOR_BRIDGES.path_env, crawl4ai_token.FILE_ENV)

#: Every bridge that calls a gated endpoint, with the tokens its code reads
#: (bridge_token / crawl4ai_token callers under src/<bridge package>/).
READERS = {
    "claude-code-bridge": (BROKER, GIT_CREDENTIAL, CRAWL4AI),
    "codex-bridge": (BROKER, GIT_CREDENTIAL, CRAWL4AI),
}


def _env(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {k: str(v) for k, v in env.items()}
    return dict(item.split("=", 1) for item in env)


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


def _backing(service: dict, path: str) -> tuple[str, bool]:
    """The named volume serving ``path`` in this container, and whether it is read-only."""
    wanted = PurePosixPath(path)
    best = None
    for volume in service.get("volumes") or []:
        source, target, *rest = _split(volume)
        target = PurePosixPath(target)
        if wanted.is_relative_to(target) and (best is None or len(target.parts) > len(best[1].parts)):
            best = (source, target, "ro" in (rest[0].split(",") if rest else []))
    assert best, f"nothing mounted at {path}"
    assert best[0] in VOLUMES, f"{path} is served by {best[0]!r}, not a named volume"
    return best[0], best[2]


@pytest.mark.parametrize(("bridge", "pair"), [(b, p) for b, pairs in READERS.items() for p in pairs])
def test_every_bridge_reads_the_file_the_app_exports(bridge, pair):
    export_env, file_env = pair
    app, reader = SERVICES["app"], SERVICES[bridge]
    path = _env(app)[export_env]

    assert _env(reader).get(file_env) == path
    assert _backing(app, path) == (_backing(reader, path)[0], False)
    assert _backing(reader, path)[1], f"{bridge} must mount the token volume read-only"


def test_bridge_copies_are_owned_by_the_bridge_user():
    # 0600 files: without the uid they stay root-owned and the bridges can't read them.
    assert bridge_gate.EXPORT_UID_ENV == _CRAWL4AI_FOR_BRIDGES.uid_env
    assert _env(SERVICES["app"])[bridge_gate.EXPORT_UID_ENV]


def test_the_app_reads_its_own_crawl4ai_copy():
    app = _env(SERVICES["app"])
    assert app[crawl4ai_token.FILE_ENV] == app[_CRAWL4AI_FOR_BRIDGES.path_env]


def test_crawl4ai_reads_the_exported_token_where_its_entrypoint_looks():
    app, crawl = SERVICES["app"], SERVICES["crawl4ai"]
    path = PurePosixPath(_env(app)[_CRAWL4AI_FOR_CRAWL4AI.path_env])
    assert _env(app)[_CRAWL4AI_FOR_CRAWL4AI.uid_env]

    volume, read_only = _backing(app, str(path))
    assert not read_only
    # The image's entrypoint reads /run/secrets/api_token when the env is unset.
    assert path.name == "api_token"
    assert f"{volume}:/run/secrets:ro" in crawl["volumes"]
    assert crawl4ai_token.LEGACY_ENV not in _env(crawl)
    assert crawl["depends_on"]["app"]["condition"] == "service_healthy"
    assert "/run/secrets/api_token" in "".join(crawl["entrypoint"]), "must refuse to start without it"


def test_claude_bridge_starts_after_the_tokens_exist():
    # bootstrap writes the Crawl4AI MCP header once, at startup.
    assert SERVICES["claude-code-bridge"]["depends_on"]["app"]["condition"] == "service_healthy"
