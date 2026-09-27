from __future__ import annotations

from claude_code_bridge import _bridge_helpers as helpers


def test_direct_turn_reads_broker_each_time_and_observes_rotation(monkeypatch):
    tokens = iter([("token-a", 1000), ("token-b", 2000)])
    calls: list[bool] = []

    def fake_fetch(*, force: bool = False):
        calls.append(force)
        return next(tokens)

    monkeypatch.setattr(helpers, "_fetch_broker_token", fake_fetch)

    assert helpers._get_fresh_oauth_token() == "token-a"
    assert helpers._get_fresh_oauth_token() == "token-b"
    assert calls == [False, False]


def test_confirmed_401_force_fetches_broker(monkeypatch):
    calls: list[bool] = []

    def fake_fetch(*, force: bool = False):
        calls.append(force)
        return "token-b", 2000

    monkeypatch.setattr(helpers, "_fetch_broker_token", fake_fetch)

    assert helpers._get_fresh_oauth_token(force_refresh=True) == "token-b"
    assert calls == [True]


def test_conditional_recovery_sends_digest_not_bearer(monkeypatch):
    import hashlib
    import httpx

    digest = hashlib.sha256(b"rejected-bearer").hexdigest()
    captured = []

    def get(url, **kwargs):
        captured.append((url, kwargs))
        return httpx.Response(200, json={"access_token": "current", "expires_at": 2000})

    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app-fixture")
    monkeypatch.setenv("BRIDGE_CLAUDE_TOKEN_SECRET", "guard-fixture")
    monkeypatch.setattr(helpers.httpx, "get", get)
    assert helpers._fetch_broker_token(force=True, rejected_token_sha256=digest) == ("current", 2000)
    url, kwargs = captured[0]
    assert url == "http://app-fixture/v1/providers/claude/token"
    assert kwargs["params"] == {"force": "true", "rejected_token_sha256": digest}
    assert kwargs["headers"] == {"X-Bridge-Token": "guard-fixture"}
    assert "rejected-bearer" not in str(captured)
