"""A transport failure minting a GitHub token is temporary, not "not connected" (TASK-2534)."""
from __future__ import annotations

import pytest

from llm_bawt.service.providers import github as gh
from llm_bawt.skills import source_config
from llm_bawt.skills.source_store import GitSpec, SourceError


class _Adapter(gh.GitHubAdapter):
    def __init__(self, exc):  # noqa: D107 - skip the real adapter setup
        self._exc = exc

    def git_credential_token(self, *, repository=None):
        raise self._exc


def _mint(monkeypatch, exc):
    from llm_bawt.service.providers import registry
    monkeypatch.setattr(registry, "get_adapter", lambda config, name: _Adapter(exc))
    spec = GitSpec.__new__(GitSpec)
    object.__setattr__(spec, "url", "https://github.com/example/skills.git")
    return source_config._github_token(None)(spec)


def test_a_missing_connection_reads_as_not_connected(monkeypatch):
    assert _mint(monkeypatch, gh.GitHubNotConnectedError("x")) is None


def test_a_network_failure_is_a_temporary_error(monkeypatch):
    with pytest.raises(SourceError) as exc:
        _mint(monkeypatch, ConnectionError("secret-ish upstream body"))
    assert exc.value.status == 503 and exc.value.code == "credential_mint_failed"
    assert "not connected" not in str(exc.value) and "upstream body" not in str(exc.value)
