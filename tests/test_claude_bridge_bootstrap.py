from __future__ import annotations

import json

from claude_code_bridge.bootstrap import bootstrap_claude_home


def test_bootstrap_creates_skills_symlink_and_settings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    skills = tmp_path / "agent-skills"
    skills.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.delenv("CLAUDE_CODE_BAWTHUB_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", raising=False)

    bootstrap_claude_home()

    claude_dir = home / ".claude"
    skills_link = claude_dir / "skills"
    assert skills_link.is_symlink()
    assert skills_link.resolve() == skills.resolve()

    settings = json.loads((claude_dir / "settings.json").read_text())
    assert settings["mcpServers"]["bawthub"]["url"] == "http://app:8001/mcp"
    assert settings["mcpServers"]["playwright"]["url"] == "http://playwright-mcp:8931/mcp"


def test_bootstrap_migrates_legacy_mcp_name_and_preserves_other_settings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    skills = tmp_path / "agent-skills"
    skills.mkdir(parents=True)
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(*)"]},
                "mcpServers": {
                    "llm-bawt-memory": {"type": "url", "url": "http://172.18.0.1:8001/mcp"},
                    "custom": {"type": "url", "url": "http://example.test/mcp"},
                },
            }
        )
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.delenv("CLAUDE_CODE_BAWTHUB_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", raising=False)

    bootstrap_claude_home()

    settings = json.loads((claude_dir / "settings.json").read_text())
    assert "llm-bawt-memory" not in settings["mcpServers"]
    assert settings["mcpServers"]["bawthub"]["url"] == "http://172.18.0.1:8001/mcp"
    assert settings["mcpServers"]["custom"]["url"] == "http://example.test/mcp"
    assert settings["permissions"]["allow"] == ["Bash(*)"]


def test_bootstrap_preserves_rea_stdio_and_is_idempotent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    skills = tmp_path / "skills"
    skills.mkdir()
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    rea = {
        "type": "stdio",
        "command": "/home/bridge/dev/llm-bawt/scripts/rea/rea",
        "args": ["mcp"],
    }
    settings = {
        "mcpServers": {
            "rea": rea,
            "bawthub": {"type": "http", "url": "http://app:8001/mcp"},
            "playwright": {"type": "http", "url": "http://playwright-mcp:8931/mcp"},
            "crawl4ai": {"type": "sse", "url": "http://crawl4ai:11235/mcp/sse"},
            "safari": {"type": "stdio", "command": "/test/safari", "args": []},
        },
        "permissions": {"allow": ["Read"]},
        "hooks": {"Stop": []},
        "env": {"ENABLE_TOOL_SEARCH": "true", "TEST_SECRET": "do-not-change"},
        "attribution": {"commit": "", "pr": ""},
    }
    path = claude_dir / "settings.json"
    path.write_text(json.dumps(settings))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.delenv("CLAUDE_CODE_BAWTHUB_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_PLAYWRIGHT_MCP_URL", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CRAWL4AI_MCP_URL", raising=False)
    monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
    bootstrap_claude_home()
    actual = json.loads(path.read_text())
    assert actual["mcpServers"]["rea"] == rea
    assert set(actual["mcpServers"]) == set(settings["mcpServers"])
    for key in ("permissions", "hooks", "env", "attribution"):
        assert actual[key] == settings[key]
    for key in ("safari", "crawl4ai"):
        assert actual["mcpServers"][key] == settings["mcpServers"][key]
    bootstrap_claude_home()
    assert json.loads(path.read_text()) == actual
