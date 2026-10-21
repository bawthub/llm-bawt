"""Native Codex must use the app-owned credential after a reconnect."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from codex_bridge import auth_broker


def test_bootstrap_replaces_existing_auth_file_with_broker_bundle(monkeypatch, tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"refresh_token": "old"}}))
    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app:8642")
    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: SimpleNamespace(
        is_error=False, json=lambda: {"tokens": {"refresh_token": "new"}},
    ))

    assert auth_broker.ensure_auth_file(auth_file)
    assert json.loads(auth_file.read_text())["tokens"]["refresh_token"] == "new"
    assert auth_file.stat().st_mode & 0o777 == 0o600


def test_bootstrap_preserves_existing_file_if_broker_unreachable(monkeypatch, tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"refresh_token": "old"}}))
    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app:8642")
    monkeypatch.setenv("CODEX_AUTH_BROKER_ATTEMPTS", "1")
    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        httpx.ConnectError("app unavailable"),
    ))

    assert auth_broker.ensure_auth_file(auth_file)
    assert json.loads(auth_file.read_text())["tokens"]["refresh_token"] == "old"


def test_bootstrap_refuses_missing_file_when_broker_unreachable(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LLM_BAWT_API_URL", raising=False)
    assert not auth_broker.ensure_auth_file(tmp_path / "auth.json")
