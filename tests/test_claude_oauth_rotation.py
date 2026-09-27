"""Hermetic app-owned rotation tests; no credential DB or upstream calls."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.usage import claude_oauth as oauth
from llm_bawt.service.routes.providers import router


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


@pytest.fixture
def owner(monkeypatch):
    state = {"accessToken": "a", "refreshToken": "refresh-a", "expiresAt": int(time.time() * 1000) + 3600000}
    rotations = []
    monkeypatch.setattr(oauth, "_load", lambda: ({}, dict(state)))
    monkeypatch.setattr(oauth, "_save", lambda _raw, bundle: state.update(bundle))
    monkeypatch.setattr(oauth, "_record_refresh_outcome", lambda _error: None)

    def refresh(bundle):
        rotations.append(bundle["accessToken"])
        return {**bundle, "accessToken": "b", "refreshToken": "refresh-b"}

    monkeypatch.setattr(oauth, "_refresh_upstream", refresh)
    return state, rotations


def test_rejected_old_generation_reuses_current_token(owner):
    state, rotations = owner
    state["accessToken"] = "b"
    result = oauth.get_access_token(force_refresh=True, rejected_token_sha256=digest("a"))
    assert result.token == "b"
    assert rotations == []


def test_concurrent_401_recovery_rotates_only_once(owner):
    state, rotations = owner
    barrier = threading.Barrier(8)

    def recover(_):
        barrier.wait(timeout=5)
        return oauth.get_access_token(force_refresh=True, rejected_token_sha256=digest("a")).token

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(recover, range(8)))
    assert results == ["b"] * 8
    assert rotations == ["a"]
    assert state["refreshToken"] == "refresh-b"


def test_new_generation_still_refreshes_if_near_expiry(owner):
    state, rotations = owner
    state.update(accessToken="b", expiresAt=1)
    oauth.get_access_token(force_refresh=True, rejected_token_sha256=digest("a"))
    assert rotations == ["b"]


def test_legacy_force_remains_supported(owner):
    assert oauth.get_access_token(force_refresh=True).token == "b"
    assert owner[1] == ["a"]


def test_broker_route_forwards_validated_digest(owner, monkeypatch):
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        for _ in range(2):
            resp = client.get("/v1/providers/claude/token", params={
                "force": "true", "rejected_token_sha256": digest("a"),
            })
            assert resp.status_code == 200
            assert resp.json()["access_token"] == "b"
        assert client.get("/v1/providers/claude/token?rejected_token_sha256=invalid").status_code == 422
    assert owner[1] == ["a"]
