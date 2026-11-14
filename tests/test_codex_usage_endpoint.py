"""Codex usage refresh uses the official non-inference endpoint."""

from __future__ import annotations

import asyncio
import time

from llm_bawt.service.usage import codex_oauth
from llm_bawt.service.usage.adapters import openai_chatgpt as mod


class _Tok:
    def __init__(self, token: str | None, account_id: str | None = None) -> None:
        self.token = token
        self.account_id = account_id


def _patched_adapter(monkeypatch, fetch_results: list[tuple[dict | None, int | None]]):
    adapter = mod.OpenAIChatGPTUsageAdapter()
    seq = list(fetch_results)

    async def fake_fetch_once(result):  # noqa: ANN001
        fake_fetch_once.tokens.append(result.token)
        return seq.pop(0)

    fake_fetch_once.tokens = []
    monkeypatch.setattr(adapter, "_fetch_usage_once", fake_fetch_once)
    return adapter, fake_fetch_once


def test_official_payload_maps_to_canonical_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(mod.time, "time", lambda: 1_800_000_000)
    snap = mod.OpenAIChatGPTUsageAdapter._snapshot_from_payload(
        {
            "plan_type": "plus",
            "rate_limit_reached_type": None,
            "rate_limit": {
                "primary_window": {
                    "used_percent": 61,
                    "limit_window_seconds": 18_000,
                    "reset_after_seconds": 900,
                    "reset_at": 1_800_000_900,
                },
                "secondary_window": {
                    "used_percent": 67,
                    "limit_window_seconds": 604_800,
                    "reset_after_seconds": 9_000,
                    "reset_at": 1_800_009_000,
                },
            },
            "credits": {"has_credits": False, "balance": "0"},
        }
    )

    assert snap == {
        "captured_at": 1_800_000_000,
        "plan_type": "plus",
        "active_limit": None,
        "primary": {
            "used_percent": 61,
            "window_minutes": 300,
            "reset_at": 1_800_000_900,
            "reset_after_seconds": 900,
        },
        "secondary": {
            "used_percent": 67,
            "window_minutes": 10_080,
            "reset_at": 1_800_009_000,
            "reset_after_seconds": 9_000,
        },
        "credits": {"has_credits": False, "balance": "0"},
    }


def test_fetch_once_gets_official_usage_endpoint_without_model_request(monkeypatch) -> None:
    calls: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 10,
                        "limit_window_seconds": 18_000,
                        "reset_at": 1_800_000_900,
                    }
                },
            }

    class FakeClient:
        def __init__(self, *, timeout):  # noqa: ANN001
            calls["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):  # noqa: ANN001
            calls["url"] = url
            calls["headers"] = headers
            return FakeResponse()

    adapter = mod.OpenAIChatGPTUsageAdapter()

    async def write_snapshot(snap):  # noqa: ANN001
        calls["snapshot"] = snap

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(adapter, "_write_snapshot", write_snapshot)

    snap, status = asyncio.run(adapter._fetch_usage_once(_Tok("secret", "acct-1")))

    assert status == 200
    assert snap is calls["snapshot"]
    assert calls["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls["headers"]["ChatGPT-Account-Id"] == "acct-1"
    assert calls["headers"]["Authorization"] == "Bearer secret"
    assert set(calls) == {"timeout", "url", "headers", "snapshot"}


def test_fetch_401_force_refreshes_and_retries(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_get(*, force_refresh: bool = False) -> _Tok:
        calls.append(force_refresh)
        return _Tok("fresh" if force_refresh else "dead")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    snap = {"captured_at": 123}
    adapter, fetch_once = _patched_adapter(monkeypatch, [(None, 401), (snap, 200)])

    out = asyncio.run(adapter._fetch_live_snapshot())

    assert out is snap
    assert calls == [False, True]
    assert fetch_once.tokens == ["dead", "fresh"]


def test_fetch_non_401_failure_does_not_retry(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_get(*, force_refresh: bool = False) -> _Tok:
        calls.append(force_refresh)
        return _Tok("tok")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    adapter, fetch_once = _patched_adapter(monkeypatch, [(None, 500)])

    out = asyncio.run(adapter._fetch_live_snapshot())

    assert out is None
    assert calls == [False]
    assert fetch_once.tokens == ["tok"]


def test_fetch_401_gives_up_when_refresh_yields_no_token(monkeypatch) -> None:
    def fake_get(*, force_refresh: bool = False) -> _Tok:
        return _Tok(None if force_refresh else "dead")

    monkeypatch.setattr(codex_oauth, "get_access_token", fake_get)
    adapter, fetch_once = _patched_adapter(monkeypatch, [(None, 401)])

    out = asyncio.run(adapter._fetch_live_snapshot())

    assert out is None
    assert fetch_once.tokens == ["dead"]


def test_fetch_uses_stale_passive_snapshot_without_claiming_bad_credential(monkeypatch) -> None:
    adapter = mod.OpenAIChatGPTUsageAdapter()
    monkeypatch.setattr(adapter, "report_identity", lambda: mod.account_scope("acct-fixture"))
    passive = {
        "account_scope": mod.account_scope("acct-fixture"),
        "captured_at": int(time.time()) - 120,
        "plan_type": "plus",
        "primary": {"used_percent": 40, "window_minutes": 300, "reset_at": None},
        "secondary": {"used_percent": 60, "window_minutes": 10_080, "reset_at": None},
    }

    async def read_snapshot():
        return passive

    async def fetch_live():
        return None

    monkeypatch.setattr(adapter, "_read_snapshot", read_snapshot)
    monkeypatch.setattr(adapter, "_fetch_live_snapshot", fetch_live)

    result = asyncio.run(adapter.fetch())

    assert result.available is True
    assert result.status == "usage_stale"
    assert result.cached is True
    assert "Credential health is checked separately" in (result.error or "")
    assert [limit.used_pct for limit in result.limits] == [40.0, 60.0]


def test_bridge_scope_contract_matches_app_without_importing_app():
    from pathlib import Path
    import subprocess
    import sys
    from claude_code_bridge.proxy import usage_scope
    from llm_bawt.shared import chatgpt_usage

    assert Path(usage_scope.__file__).read_bytes() == Path(chatgpt_usage.__file__).read_bytes()
    for account in (None, "", " ", "A", " B ", "åccount"):
        scope = usage_scope.account_scope(account)
        assert scope == chatgpt_usage.account_scope(account)
        if scope:
            assert usage_scope.snapshot_key(scope) == chatgpt_usage.snapshot_key(scope)
    # A fresh interpreter is essential: the ordinary test process already has
    # llm_bawt loaded and its application dependencies installed.
    source = Path(__file__).resolve().parents[1] / "src"
    probe = '''
import sys
sys.path.insert(0, sys.argv[1])
class RejectApp:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "llm_bawt" or fullname.startswith("llm_bawt."):
            raise ImportError("Application package unavailable in bridge")
sys.meta_path.insert(0, RejectApp())
from claude_code_bridge.proxy.usage_capture import schedule_capture, account_scope
assert callable(schedule_capture) and account_scope("fixture")
assert "llm_bawt" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", probe, str(source)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_account_switch_rejects_old_passive_and_process_cache(monkeypatch):
    from unittest.mock import AsyncMock
    from llm_bawt.service import usage

    adapter = usage.get_adapter("openai_chatgpt")
    scope_a, scope_b = mod.account_scope("account-A"), mod.account_scope("account-B")
    monkeypatch.setattr(adapter, "report_identity", lambda: scope_b)
    monkeypatch.setattr(adapter, "_read_snapshot", AsyncMock(return_value={
        "account_scope": scope_a, "captured_at": int(time.time()),
        "primary": {"used_percent": 99, "window_minutes": 300},
    }))
    monkeypatch.setattr(adapter, "_fetch_live_snapshot", AsyncMock(return_value=None))
    old = adapter._base(available=True, status="ok")
    monkeypatch.setitem(usage._cache, adapter.provider, (time.monotonic(), old))
    monkeypatch.setitem(usage._cache_identities, adapter.provider, scope_a)
    assert usage.get_cached(adapter.provider) is None
    result = asyncio.run(adapter.fetch())
    assert not result.available and not result.limits


def test_passive_snapshots_are_separate_and_late_old_writer_cannot_clobber_new(monkeypatch):
    import json
    import httpx
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from claude_code_bridge.proxy import usage_capture

    rows = {}
    async def put(key, value, **_kwargs):
        rows[key] = value
    client = SimpleNamespace(set=AsyncMock(side_effect=put),
                             get=AsyncMock(side_effect=lambda key: rows.get(key)), aclose=AsyncMock())
    monkeypatch.setattr(usage_capture, "_client", lambda: client)
    monkeypatch.setattr("redis.asyncio.from_url", lambda *_a, **_kw: client)
    adapter = mod.OpenAIChatGPTUsageAdapter()
    scope_b = mod.account_scope("account-B")
    monkeypatch.setattr(adapter, "report_identity", lambda: scope_b)

    async def run():
        for account, percent in [("account-B", "20"), ("account-A", "99")]:
            response = httpx.Response(200, headers={"x-codex-primary-used-percent": percent},
                request=httpx.Request("POST", "https://fixture.invalid/responses",
                                      headers={"ChatGPT-Account-Id": account}))
            usage_capture.schedule_capture(SimpleNamespace(response=response))
            await asyncio.gather(*usage_capture._tasks)
        snap = await adapter._read_snapshot()
        assert snap["account_scope"] == scope_b and snap["primary"]["used_percent"] == 20
        assert len(rows) == 2
        assert all("account-A" not in key and "account-B" not in key for key in rows)
        # Legacy snapshots and a mismatched payload are never trusted.
        rows[mod.snapshot_key(scope_b)] = json.dumps({"account_scope": mod.account_scope("account-A")})
        assert await adapter._read_snapshot() is None
    asyncio.run(run())


def test_account_changed_during_live_fetch_is_not_published(monkeypatch):
    from unittest.mock import AsyncMock
    adapter = mod.OpenAIChatGPTUsageAdapter()
    identities = iter([mod.account_scope("A"), mod.account_scope("B")])
    monkeypatch.setattr(adapter, "report_identity", lambda: next(identities))
    monkeypatch.setattr(adapter, "_read_snapshot", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_fetch_live_snapshot", AsyncMock(return_value={
        "account_scope": mod.account_scope("A"), "captured_at": int(time.time()),
        "primary": {"used_percent": 30, "window_minutes": 300},
    }))
    assert not asyncio.run(adapter.fetch()).available


def test_save_with_connected_at_does_not_crash(monkeypatch) -> None:
    """Regression: duplicate connected_at kwarg crashed token persistence."""
    saved: dict = {}

    class FakeStore:
        available = True

        def save(self, record) -> None:  # noqa: ANN001
            saved["record"] = record

    monkeypatch.setattr(codex_oauth, "_credential_store", lambda: FakeStore())

    codex_oauth._save(
        {"connected_at": "2026-01-01T00:00:00Z", "auth_method": "cli_oauth"},
        {"tokens": {"access_token": "tok", "refresh_token": "ref"}},
    )

    assert saved["record"].connected_at == "2026-01-01T00:00:00Z"
    assert saved["record"].secret["codex_auth"]["tokens"]["refresh_token"] == "ref"
