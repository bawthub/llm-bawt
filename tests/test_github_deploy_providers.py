"""TASK-997: release-deploy GitHub credentials are DB-stored ApiKeyAdapters."""
from __future__ import annotations

import httpx
import pytest

from llm_bawt.service.providers import github_deploy
from llm_bawt.service.providers.base import ConnectionRecord
from llm_bawt.service.providers.github_deploy import GhcrPullAdapter, GitHubReleaseAdapter
from llm_bawt.service.providers.registry import _ADAPTER_CLASSES


class FakeStore:
    def __init__(self):
        self.record = None

    def load(self, provider):
        return self.record if self.record and self.record.provider == provider else None

    def save(self, record):
        self.record = record


def _adapter(cls):
    adapter = object.__new__(cls)
    adapter.store = FakeStore()
    return adapter


def _github(monkeypatch, *, status=200, scopes="", login="zenoran"):
    seen = []

    def fake_get(url, headers, timeout):
        seen.append(headers["Authorization"])
        return httpx.Response(status, json={"login": login}, headers={"X-OAuth-Scopes": scopes},
                              request=httpx.Request("GET", url))
    monkeypatch.setattr(github_deploy.httpx, "get", fake_get)
    return seen


def test_both_adapters_are_registered_api_key_providers():
    assert _ADAPTER_CLASSES["github-release"] is GitHubReleaseAdapter
    assert _ADAPTER_CLASSES["ghcr-pull"] is GhcrPullAdapter


def test_release_token_connects_with_login_as_account(monkeypatch):
    seen = _github(monkeypatch)
    adapter = _adapter(GitHubReleaseAdapter)
    result = adapter.set_api_key(" github_pat_x ")
    assert result.ok and result.account == "zenoran" and seen == ["Bearer github_pat_x"]
    assert adapter.api_key() == "github_pat_x"


@pytest.mark.parametrize("scopes, ok", [("read:packages", True), ("repo, write:packages", True),
                                        ("repo", False), ("", False)])
def test_ghcr_pull_requires_packages_scope(monkeypatch, scopes, ok):
    _github(monkeypatch, scopes=scopes)
    adapter = _adapter(GhcrPullAdapter)
    result = adapter.set_api_key("ghp_x")
    assert result.ok is ok
    assert (adapter.store.record is not None) is ok
    if not ok:
        assert "read:packages" in result.detail


def test_rejected_token_is_not_saved(monkeypatch):
    _github(monkeypatch, status=401)
    adapter = _adapter(GitHubReleaseAdapter)
    assert adapter.set_api_key("bad").ok is False and adapter.store.record is None


def test_pull_auth_uses_login_and_token_or_none():
    adapter = _adapter(GhcrPullAdapter)
    assert adapter.pull_auth() is None
    adapter.store.record = ConnectionRecord(provider="ghcr-pull", status="connected", auth_method="api_key",
                                            account="zenoran", secret={"api_key": "ghp_x"})
    assert adapter.pull_auth() == {"username": "zenoran", "password": "ghp_x"}


def test_descriptor_exposes_setup_help_without_secret(monkeypatch):
    adapter = _adapter(GhcrPullAdapter)
    adapter.store.record = ConnectionRecord(provider="ghcr-pull", status="connected", auth_method="api_key",
                                            account="zenoran", secret={"api_key": "ghp_x"})
    descriptor = adapter.descriptor()
    assert descriptor["auth_methods"] == ["api_key"] and descriptor["category"] == "deploy"
    assert "read:packages" in descriptor["setup_url"] and "ghp_x" not in repr(descriptor)
