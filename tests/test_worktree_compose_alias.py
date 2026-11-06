"""Bridge worktree views name the host root; the retired mansion stays absent."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_shared_bridge_keeps_existing_mount_and_same_path_alias() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    claude = services["claude-code-bridge"]
    existing_mount = "${HOME}/dev:/home/bridge/dev"
    same_path_mount = "${HOME}/dev:${HOME}/dev"

    assert existing_mount in claude["volumes"]
    assert same_path_mount in claude["volumes"]


def test_retired_codex_mansion_is_absent_from_base_and_dev() -> None:
    for name in ("docker-compose.yml", "docker-compose.dev.yml"):
        services = yaml.safe_load((ROOT / name).read_text())["services"]
        assert "codex-bridge" not in services
        assert "claude-code-bridge" in services
