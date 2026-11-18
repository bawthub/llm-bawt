"""TASK-2472: ChatGPT model discovery declares the codex CLI version the live
bridges publish (TASK-2470 mansion keys), falling back to config."""
from __future__ import annotations

import json

import pytest

from llm_bawt.service import model_discovery as md


class FakeRedis:
    def __init__(self, data, *, fail=False):
        self.data = data
        self.fail = fail

    def scan_iter(self, match=None, count=None):
        if self.fail:
            raise ConnectionError("redis down")
        prefix = match.rstrip("*")
        return iter([k for k in self.data if k.startswith(prefix)])

    def get(self, key):
        return self.data.get(key)


def _rec(cli):
    return json.dumps({"versions": {"schema": 1, "harness": {"openai-codex-cli-bin": cli,
                                                             "claude-agent-sdk": "0.2.1"}}})


@pytest.fixture
def redis(monkeypatch):
    holder = {"client": FakeRedis({})}
    monkeypatch.setattr("llm_bawt.campus_events._redis", lambda _config=None: holder["client"])
    return holder


def test_highest_published_version_wins_and_nulls_and_junk_are_skipped(redis):
    redis["client"] = FakeRedis({
        "agent:mansion:versions:claude-code": _rec("0.170.0"),
        "agent:mansion:versions:codex": _rec("0.99.0"),       # numeric, not lexical, compare
        "agent:mansion:versions:other": _rec(None),           # not installed
        "agent:mansion:versions:bad": "not json",
        "agent:house:h1:versions:codex": _rec("9.9.9"),       # house previews are ignored
    })
    assert md.published_codex_cli_version() == "0.170.0"


@pytest.mark.parametrize("client", [None, FakeRedis({}), FakeRedis({}, fail=True),
                                    FakeRedis({"agent:mansion:versions:x": _rec(None)})])
def test_nothing_published_returns_none(redis, client):
    redis["client"] = client
    assert md.published_codex_cli_version() is None


class _Cfg:
    CODEX_CLIENT_VERSION = "0.163.0"


@pytest.mark.parametrize("published,expected", [("0.170.0", "0.170.0"), (None, "0.163.0")])
def test_discovery_declares_published_version_else_config(monkeypatch, published, expected):
    from types import SimpleNamespace

    monkeypatch.delenv("CODEX_BRIDGE_MODELS_URL", raising=False)
    monkeypatch.setattr(md, "published_codex_cli_version", lambda _config=None: published)
    monkeypatch.setattr("llm_bawt.service.usage.codex_oauth.get_access_token",
                        lambda: SimpleNamespace(state="ok", token="t", account_id="a"))
    seen = {}

    def fake_request(**kwargs):
        seen["version"] = kwargs["client_version"]
        raise md.ModelDiscoveryError("stop here")

    monkeypatch.setattr(md, "chatgpt_models_request", fake_request)
    with pytest.raises(md.ModelDiscoveryError):
        md.ChatGPTSubscriptionDiscoveryProvider(_Cfg()).fetch()
    assert seen["version"] == expected
