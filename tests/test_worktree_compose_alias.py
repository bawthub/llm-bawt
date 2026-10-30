"""TASK-1275: both bridge worktree views must name the same host dev root."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_codex_keeps_its_existing_mount_and_adds_same_path_alias() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    codex = services["codex-bridge"]
    claude = services["claude-code-bridge"]
    existing_mount = "${HOME}/dev:/home/bridge/dev"
    same_path_mount = "${HOME}/dev:${HOME}/dev"

    for bridge in (codex, claude):
        assert existing_mount in bridge["volumes"]
        assert same_path_mount in bridge["volumes"]
    assert "CODEX_BRIDGE_CWD=${CODEX_BRIDGE_CWD:-/home/bridge/dev}" in codex[
        "environment"
    ]
    assert "CODEX_DEV_ROOT=${CODEX_DEV_ROOT:-/home/bridge/dev}" in codex[
        "environment"
    ]
