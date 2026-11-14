"""Offline regressions for the Claude login persistence boundary (TASK-2226)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_bawt.service.providers import claude_sub
from llm_bawt.service.providers.claude import ClaudeAdapter


@pytest.mark.parametrize("adapter_type", [ClaudeAdapter, claude_sub.ClaudeSubAdapter])
def test_completed_login_preserves_recovery_until_save_succeeds(monkeypatch, tmp_path, caplog, adapter_type):
    config_dir = tmp_path / "claude-login-offline"
    config_dir.mkdir()
    creds_path = config_dir / ".credentials.json"
    bundle = {
        "accessToken": "offline-access-token",
        "refreshToken": "offline-refresh-token",
        "scopes": ["user:inference", "user:profile"],
    }
    contents = json.dumps({"claudeAiOauth": bundle})
    creds_path.write_text(contents)
    session = claude_sub._LoginSession("offline-session", -1, -1, config_dir)
    closed = []
    submitted = []
    monkeypatch.setattr(session, "pump", lambda *_: None)
    monkeypatch.setattr(session, "text", lambda: "Login successful")
    monkeypatch.setattr(session, "close", lambda: closed.append(True))
    monkeypatch.setattr(claude_sub, "_SESSIONS", {session.session_id: session})
    monkeypatch.setattr(claude_sub.os, "write", lambda *args: submitted.append(args))
    adapter = object.__new__(adapter_type)
    saved = []
    fail_save = True

    def save(record):
        # The durable store, not _build_record, is the commit point for cleanup.
        assert creds_path.read_text() == contents
        if fail_save:
            raise RuntimeError("DB failure containing offline-refresh-token")
        saved.append(record)

    adapter.store = SimpleNamespace(save=save)
    result = adapter.complete_cli_login(session.session_id, "offline-code")
    assert result.status == "error" and result.record is None
    assert result.detail == "Could not save Claude credential. Retry completing this login."
    assert "offline-refresh-token" not in caplog.text
    assert creds_path.read_text() == contents
    assert session.session_id in claude_sub._SESSIONS
    assert not closed and not saved

    fail_save = False
    result = adapter.complete_cli_login(session.session_id, "offline-code")
    assert result.status == "connected" and result.record is saved[0]
    assert len(submitted) == 1  # no second exchange of the single-use code
    assert closed == [True]
    assert session.session_id not in claude_sub._SESSIONS
    if adapter_type is ClaudeAdapter:
        assert saved[0].secret == {"claudeAiOauth": bundle}
        assert not config_dir.exists()
    else:
        assert creds_path.read_text() == contents  # shared legacy directory


@pytest.mark.parametrize("adapter_type", [ClaudeAdapter, claude_sub.ClaudeSubAdapter])
def test_expired_failed_save_cleans_only_temporary_bundle(monkeypatch, tmp_path, adapter_type):
    config_dir = tmp_path / "claude-login-expired"
    config_dir.mkdir()
    creds_path = config_dir / ".credentials.json"
    creds_path.write_text("offline recovery bundle")
    adapter = object.__new__(adapter_type)
    session = claude_sub._LoginSession("expired", -1, -1, config_dir)
    session.cleanup = adapter._cleanup_login_dir
    session.record = claude_sub.ConnectionRecord(provider=adapter.id)
    closed = []
    monkeypatch.setattr(session, "close", lambda: closed.append(True))
    monkeypatch.setattr(claude_sub, "_SESSIONS", {session.session_id: session})
    monkeypatch.setattr(claude_sub.time, "monotonic",
                        lambda: session.created + claude_sub._SESSION_TTL_S + 1)
    claude_sub._reap()
    assert closed == [True]
    assert not claude_sub._SESSIONS
    if adapter_type is ClaudeAdapter:
        assert not config_dir.exists()
    else:
        assert creds_path.read_text() == "offline recovery bundle"
