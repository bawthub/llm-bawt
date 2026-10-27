"""The derived Crawl4AI bearer token: export and the one reader rule (TASK-1476)."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from pathlib import Path

import httpx
import pytest

from agent_bridge import crawl4ai_token
from agent_bridge.crawl4ai_token import FILE_ENV, LEGACY_ENV, api_token, auth_headers
from llm_bawt.service import crawl4ai_token_export
from llm_bawt.service.crawl4ai_token_export import LABEL, export_crawl4ai_token

SERVER = "CRAWL4AI_API_TOKEN_EXPORT_FILE"
SERVER_UID = "CRAWL4AI_API_TOKEN_EXPORT_UID"
BRIDGES = "CRAWL4AI_API_TOKEN_BRIDGE_EXPORT_FILE"
BRIDGES_UID = "BRIDGE_TOKEN_EXPORT_UID"
DERIVED = "d" * 64


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(crawl4ai_token, "_warned", set())
    monkeypatch.delenv(FILE_ENV, raising=False)
    monkeypatch.delenv(LEGACY_ENV, raising=False)


def _file(tmp_path: Path, content: str, name: str = "token") -> str:
    path = tmp_path / name
    path.write_text(content)
    return str(path)


# --- reader -----------------------------------------------------------
def test_file_wins_then_legacy_env_then_nothing(tmp_path):
    path = _file(tmp_path, "from-file\n")
    assert api_token({FILE_ENV: path, LEGACY_ENV: "from-env"}) == "from-file"
    assert api_token({LEGACY_ENV: " from-env "}) == "from-env"
    assert api_token({}) is None
    assert auth_headers({FILE_ENV: path}) == {"Authorization": "Bearer from-file"}
    assert auth_headers({}) == {}


@pytest.mark.parametrize("content", ["", "  \n"])
def test_empty_file_falls_back_to_env(tmp_path, content):
    assert api_token({FILE_ENV: _file(tmp_path, content), LEGACY_ENV: "env"}) == "env"


def test_missing_or_undecodable_file_falls_back_to_env(tmp_path):
    assert api_token({FILE_ENV: str(tmp_path / "absent"), LEGACY_ENV: "env"}) == "env"
    bad = tmp_path / "bad"
    bad.write_bytes(b"\xff\xfe")
    assert api_token({FILE_ENV: str(bad), LEGACY_ENV: "env"}) == "env"
    assert api_token({FILE_ENV: str(bad)}) is None


def test_file_is_reread_after_rotation(tmp_path):
    path = _file(tmp_path, "one")
    assert api_token({FILE_ENV: path}) == "one"
    Path(path).write_text("two")
    assert api_token({FILE_ENV: path}) == "two"


def test_legacy_env_warns_once_by_name_never_value(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=crawl4ai_token.__name__)
    for _ in range(3):
        assert api_token({LEGACY_ENV: "legacy-secret-value"}) == "legacy-secret-value"
    warnings = [r for r in caplog.records if LEGACY_ENV in r.getMessage()]
    assert len(warnings) == 1
    assert "legacy-secret-value" not in caplog.text


def test_file_token_logs_nothing(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=crawl4ai_token.__name__)
    api_token({FILE_ENV: _file(tmp_path, "quiet-value")})
    assert caplog.records == []


def test_reader_defaults_to_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv(FILE_ENV, _file(tmp_path, "process"))
    assert api_token() == "process"


# --- export -----------------------------------------------------------
def test_export_disabled_without_paths_never_derives():
    def derive():
        raise AssertionError("must not derive with no target")

    assert export_crawl4ai_token({}, derive=derive) == {"crawl4ai": "disabled", "bridges": "disabled"}


def test_export_writes_each_target_0600_to_its_reader(tmp_path, monkeypatch):
    chowned: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: chowned.append((Path(path).parent.name, uid)))
    server = tmp_path / "crawl4ai" / "token"
    bridges = tmp_path / "bridges" / "crawl4ai-token"
    env = {SERVER: str(server), SERVER_UID: "999", BRIDGES: str(bridges), BRIDGES_UID: "1000"}

    assert export_crawl4ai_token(env, derive=lambda: DERIVED) == {"crawl4ai": "written", "bridges": "written"}

    for path in (server, bridges):
        assert path.read_text().strip() == DERIVED
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(chowned) == [("bridges", 1000), ("crawl4ai", 999)]
    # What the export writes is exactly what a consumer sends.
    assert api_token({FILE_ENV: str(bridges), LEGACY_ENV: "stale"}) == DERIVED


def test_export_one_target_only(tmp_path):
    bridges = tmp_path / "t"
    assert export_crawl4ai_token({BRIDGES: str(bridges)}, derive=lambda: DERIVED) == {
        "crawl4ai": "disabled", "bridges": "written"}


def test_underivable_root_removes_stale_files(tmp_path, caplog):
    server = Path(_file(tmp_path, "stale", "server"))
    bridges = Path(_file(tmp_path, "stale", "bridges"))

    def derive():
        raise RuntimeError("no root")

    result = export_crawl4ai_token({SERVER: str(server), BRIDGES: str(bridges)}, derive=derive)
    assert result == {"crawl4ai": "removed", "bridges": "removed"}
    assert not server.exists() and not bridges.exists()
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("uid", ["bad-uid", str(2**65)])
def test_underivable_root_removes_stale_token_even_with_a_bad_uid(tmp_path, uid):
    stale = Path(_file(tmp_path, "old-value"))

    def underivable():
        raise RuntimeError("root unavailable")

    result = export_crawl4ai_token({SERVER: str(stale), SERVER_UID: uid}, derive=underivable)
    assert result["crawl4ai"] == "removed"
    assert not stale.exists()


@pytest.mark.parametrize("uid", [str(2**32 - 1), str(2**65)])
def test_out_of_range_uid_fails_the_target_without_raising(tmp_path, uid):
    target = tmp_path / "token"
    result = export_crawl4ai_token({SERVER: str(target), SERVER_UID: uid}, derive=lambda: DERIVED)
    assert result["crawl4ai"] == "failed"
    assert not target.exists()


def test_a_failed_target_does_not_stop_the_other(tmp_path):
    bridges = tmp_path / "b"
    env = {SERVER: str(tmp_path / "s"), SERVER_UID: "not-a-uid", BRIDGES: str(bridges)}
    assert export_crawl4ai_token(env, derive=lambda: DERIVED) == {"crawl4ai": "failed", "bridges": "written"}
    assert bridges.read_text().strip() == DERIVED


def test_export_never_logs_the_token(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    export_crawl4ai_token({SERVER: str(tmp_path / "s"), SERVER_UID: "x", BRIDGES: str(tmp_path / "b")},
                          derive=lambda: DERIVED)
    assert DERIVED not in caplog.text


def test_default_derivation_is_versioned_hex_and_follows_the_root(monkeypatch):
    from cryptography.fernet import Fernet

    from llm_bawt.service.providers import crypto

    labels: list[str] = []
    real = crypto.derive_key
    monkeypatch.setattr(crypto, "derive_key", lambda label: labels.append(label) or real(label))
    keys = iter([Fernet.generate_key(), Fernet.generate_key()])
    monkeypatch.setattr(crypto, "_load_or_create_key", lambda **_: next(keys))
    first = crawl4ai_token_export._derive()
    second = crawl4ai_token_export._derive()
    assert labels == [LABEL, LABEL] and LABEL.endswith("/v1")
    assert re.fullmatch(r"[0-9a-f]{64}", first) and first != second


# --- consumers --------------------------------------------------------
def test_web_fetch_sends_the_file_token(tmp_path, monkeypatch):
    from llm_bawt.integrations.web_fetch.client import WebFetchClient

    monkeypatch.setenv(FILE_ENV, _file(tmp_path, "file-token"))
    monkeypatch.setenv(LEGACY_ENV, "legacy-token")
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"results": [{"success": True, "markdown": "hi"}]})

    client = WebFetchClient(base_url="http://crawl4ai.test")
    client._client = httpx.Client(base_url="http://crawl4ai.test", transport=httpx.MockTransport(handler))
    client.fetch("https://example.com")
    assert seen == ["Bearer file-token"]


def test_web_fetch_sends_no_header_without_a_token(monkeypatch):
    from llm_bawt.integrations.web_fetch.client import WebFetchClient

    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"results": [{"success": True, "markdown": "hi"}]})

    client = WebFetchClient(base_url="http://crawl4ai.test")
    client._client = httpx.Client(base_url="http://crawl4ai.test", transport=httpx.MockTransport(handler))
    client.fetch("https://example.com")
    assert seen == [None]


def test_claude_bridge_registers_crawl4ai_with_the_file_token(tmp_path, monkeypatch):
    from claude_code_bridge.bootstrap import bootstrap_claude_home

    home = tmp_path / "home"
    skills = tmp_path / "agent-skills"
    skills.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(skills))
    monkeypatch.setenv(FILE_ENV, _file(tmp_path, "file-token"))
    monkeypatch.setenv(LEGACY_ENV, "legacy-token")

    bootstrap_claude_home()

    settings = json.loads((home / ".claude" / "settings.json").read_text())
    assert settings["mcpServers"]["crawl4ai"]["headers"] == {"Authorization": "Bearer file-token"}


def test_codex_bridge_registers_crawl4ai_with_the_file_token(tmp_path, monkeypatch):
    from codex_bridge.local_plugins import ensure_crawl4ai_mcp_config

    monkeypatch.setenv(FILE_ENV, _file(tmp_path, "file-token"))
    monkeypatch.setenv(LEGACY_ENV, "legacy-token")
    codex_home = tmp_path / ".codex"

    ensure_crawl4ai_mcp_config(logger=logging.getLogger("test"), codex_home=codex_home)

    contents = (codex_home / "config.toml").read_text()
    assert 'headers = { Authorization = "Bearer file-token" }' in contents
    assert "legacy-token" not in contents
