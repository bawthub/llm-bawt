"""A rotated OAuth bundle survives a failed credential save (TASK-2267).

Hermetic: fake credential store and single-use fake upstream; no DB, network
or live credentials.
"""

from __future__ import annotations

import time

import pytest

from llm_bawt.service.usage import claude_oauth, codex_oauth


class Store:
    def __init__(self, bundle):
        self.bundle = bundle
        self.fail = 0
        self.saves = 0

    def load(self):
        return {}, (dict(self.bundle) if self.bundle else None)

    def save(self, _raw, bundle):
        self.saves += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("db unavailable")
        self.bundle = dict(bundle)


@pytest.fixture
def outcomes():
    return []


# --- Claude -----------------------------------------------------------------


def _claude_bundle(n, expires_ms):
    return {"accessToken": f"a{n}", "refreshToken": f"r{n}", "expiresAt": expires_ms}


@pytest.fixture
def claude(monkeypatch, outcomes):
    store = Store(_claude_bundle(0, 1))  # already expired
    consumed: set[str] = set()
    counter = {"n": 0}

    def upstream(bundle):
        token = bundle["refreshToken"]
        if token in consumed:
            raise RuntimeError("invalid_grant")
        consumed.add(token)
        counter["n"] += 1
        return _claude_bundle(counter["n"], int(time.time() * 1000) + 3_600_000)

    monkeypatch.setattr(claude_oauth, "_pending", None)
    monkeypatch.setattr(claude_oauth, "_load_db", store.load)
    monkeypatch.setattr(claude_oauth, "_save", store.save)
    monkeypatch.setattr(claude_oauth, "_refresh_upstream", upstream)
    monkeypatch.setattr(claude_oauth, "_record_refresh_outcome", outcomes.append)
    return store


def test_claude_failed_save_keeps_rotated_bundle_and_reports_degraded(claude, outcomes):
    claude.fail = 1

    token = claude_oauth.get_access_token()

    assert token.token == "a1"
    assert claude.bundle["refreshToken"] == "r0"  # DB still old
    assert outcomes and outcomes[-1] and "not yet saved" in outcomes[-1]
    # Readers keep getting the unsaved rotation, never the consumed r0.
    assert claude_oauth.get_access_token().token == "a1"


def test_claude_pending_save_retries_and_recovers(claude, outcomes):
    claude.fail = 2
    claude_oauth.get_access_token()

    assert claude_oauth._retry_pending_save() is False  # store still failing
    assert claude_oauth._retry_pending_save() is True

    assert claude.bundle["refreshToken"] == "r1"
    assert outcomes[-1] is None
    assert claude_oauth._pending is None


def test_claude_next_refresh_uses_unsaved_rotation(claude, outcomes):
    claude.fail = 10
    claude_oauth.get_access_token()

    token = claude_oauth.get_access_token(force_refresh=True)

    assert token.token == "a2"  # rotated from r1, not the consumed r0
    claude.fail = 0
    assert claude_oauth._retry_pending_save() is True
    assert claude.bundle["refreshToken"] == "r2"


def test_claude_reconnect_replaces_pending_bundle(claude):
    claude.fail = 1
    claude_oauth.get_access_token()
    claude.bundle = {"accessToken": "fresh", "refreshToken": "reconnected",
                     "expiresAt": int(time.time() * 1000) + 3_600_000}

    assert claude_oauth.get_access_token().token == "fresh"
    assert claude_oauth._retry_pending_save() is True
    assert claude.bundle["refreshToken"] == "reconnected"


def test_claude_disconnect_drops_pending_bundle(claude):
    claude.fail = 1
    claude_oauth.get_access_token()
    claude.bundle = None

    assert claude_oauth.get_access_token().token is None
    assert claude_oauth._retry_pending_save() is True
    assert claude.bundle is None


# --- ChatGPT ----------------------------------------------------------------


def _codex_bundle(n):
    return {"tokens": {"access_token": f"a{n}", "refresh_token": f"r{n}", "id_token": "id"}}


@pytest.fixture
def codex(monkeypatch, outcomes):
    store = Store(_codex_bundle(0))
    consumed: set[str] = set()
    counter = {"n": 0}

    def upstream(bundle):
        token = bundle["tokens"]["refresh_token"]
        if token in consumed:
            raise RuntimeError("invalid_grant")
        consumed.add(token)
        counter["n"] += 1
        # Mirror the real helper: rotate in place.
        bundle["tokens"].update(_codex_bundle(counter["n"])["tokens"])
        return bundle

    monkeypatch.setattr(codex_oauth, "_pending", None)
    monkeypatch.setattr(codex_oauth, "_load_db", store.load)
    monkeypatch.setattr(codex_oauth, "_save", store.save)
    monkeypatch.setattr(codex_oauth, "_refresh_upstream", upstream)
    monkeypatch.setattr(codex_oauth, "_record_refresh_outcome", outcomes.append)
    return store


def test_codex_failed_save_keeps_rotated_bundle_and_recovers(codex, outcomes):
    codex.fail = 1

    bundle = codex_oauth._refresh_serialized(buffer_s=0, force=True)

    assert bundle is not None and bundle["tokens"]["refresh_token"] == "r1"
    assert codex.bundle["tokens"]["refresh_token"] == "r0"
    assert outcomes[-1] and "not yet saved" in outcomes[-1]
    _, loaded = codex_oauth._load()
    assert loaded is not None and loaded["tokens"]["access_token"] == "a1"

    assert codex_oauth._retry_pending_save() is True
    assert codex.bundle["tokens"]["refresh_token"] == "r1"
    assert outcomes[-1] is None


def test_codex_next_refresh_uses_unsaved_rotation(codex):
    codex.fail = 10
    codex_oauth._refresh_serialized(buffer_s=0, force=True)

    bundle = codex_oauth._refresh_serialized(buffer_s=0, force=True)

    assert bundle is not None and bundle["tokens"]["refresh_token"] == "r2"
    codex.fail = 0
    assert codex_oauth._retry_pending_save() is True
    assert codex.bundle["tokens"]["refresh_token"] == "r2"


def test_codex_reconnect_replaces_pending_bundle(codex):
    codex.fail = 1
    codex_oauth._refresh_serialized(buffer_s=0, force=True)
    codex.bundle = {"tokens": {"access_token": "fresh", "refresh_token": "reconnected"}}

    _, loaded = codex_oauth._load()

    assert loaded is not None and loaded["tokens"]["access_token"] == "fresh"
    assert codex_oauth._retry_pending_save() is True
    assert codex.bundle["tokens"]["refresh_token"] == "reconnected"
