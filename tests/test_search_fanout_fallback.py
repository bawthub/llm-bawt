"""TASK-1619: a default web_search falls back when every keyed web provider fails.

Brave 429s used to sink the whole default fan-out (DuckDuckGo is benched while
Brave is keyed). A named provider must still never be swapped, and the result
must report each provider's failure truthfully.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_bawt.search import multi
from llm_bawt.search.base import SearchError, SearchProvider, SearchResult

BRAVE_429 = "Brave Search is busy: too many searches are queued for its one-per-second limit."


class _Client:
    def __init__(self, provider: SearchProvider, *, fail: str | None = None, status: int | None = None):
        self.provider = provider
        self.fail = fail
        self.status = status
        self.calls = 0

    def search(self, query, max_results):
        self.calls += 1
        if self.fail:
            raise SearchError(self.provider, self.fail, status_code=self.status)
        return [SearchResult(title=f"{self.provider.value} hit", url=f"https://{self.provider.value}.example/{query}",
                             snippet="s", source=self.provider)]


@pytest.fixture
def clients(monkeypatch):
    made: dict[SearchProvider, _Client] = {
        SearchProvider.BRAVE: _Client(SearchProvider.BRAVE, fail=BRAVE_429, status=429),
        SearchProvider.DUCKDUCKGO: _Client(SearchProvider.DUCKDUCKGO),
    }
    monkeypatch.setattr(multi, "available_providers", lambda config: [SearchProvider.BRAVE])
    monkeypatch.setattr(multi, "is_search_available", lambda provider=None: True)
    monkeypatch.setattr(multi, "get_search_client",
                        lambda config, provider, max_results, fallback: made.get(provider))
    monkeypatch.setattr(multi, "get_search_unavailable_reason", lambda config, p=None: f"{p} unavailable")
    return made


def _search(**kw):
    return asyncio.run(multi.search_all_detailed(SimpleNamespace(), "q", **kw))


def test_default_fanout_falls_back_to_duckduckgo_on_brave_429(clients):
    out = _search()
    assert out.providers == ["duckduckgo"]
    assert [r.source for r in out.results] == [SearchProvider.DUCKDUCKGO]
    assert out.errors == {"brave": BRAVE_429}
    assert out.fallbacks == {"duckduckgo": ["brave"]}


def test_named_provider_is_never_swapped(clients):
    out = _search(providers=["brave"])
    assert out.providers == [] and out.results == []
    assert out.errors == {"brave": BRAVE_429}
    assert out.fallbacks == {}
    assert clients[SearchProvider.DUCKDUCKGO].calls == 0


def test_no_fallback_when_brave_answers(clients):
    clients[SearchProvider.BRAVE].fail = None
    out = _search()
    assert out.providers == ["brave"] and out.errors == {} and out.fallbacks == {}
    assert clients[SearchProvider.DUCKDUCKGO].calls == 0


def test_no_fallback_while_another_keyed_web_provider_answers(clients, monkeypatch):
    clients[SearchProvider.TAVILY] = _Client(SearchProvider.TAVILY)
    monkeypatch.setattr(multi, "available_providers", lambda config: [SearchProvider.BRAVE, SearchProvider.TAVILY])
    out = _search()
    assert out.providers == ["tavily"]
    assert out.errors == {"brave": BRAVE_429} and out.fallbacks == {}
    assert clients[SearchProvider.DUCKDUCKGO].calls == 0


def test_failed_fallback_reports_both_errors(clients):
    clients[SearchProvider.DUCKDUCKGO].fail = "duckduckgo search failed (RatelimitException)."
    out = _search()
    assert out.providers == [] and out.results == [] and out.fallbacks == {}
    assert set(out.errors) == {"brave", "duckduckgo"}


def _real_ddg(text):
    """Production DuckDuckGoClient with only ddgs's network call stubbed."""
    from llm_bawt.search.ddgs_client import DuckDuckGoClient

    client = DuckDuckGoClient()
    client._client = SimpleNamespace(text=text)
    return client


def _raise(exc):
    def text(*a, **kw):
        raise exc
    return text


def test_real_duckduckgo_rate_limit_is_a_failure_not_an_answer(clients):
    from ddgs.exceptions import RatelimitException

    clients[SearchProvider.DUCKDUCKGO] = _real_ddg(_raise(RatelimitException("202 Ratelimit")))
    out = _search()
    assert out.providers == [] and out.results == [] and out.fallbacks == {}
    assert out.errors["brave"] == BRAVE_429
    assert "rate limiting" in out.errors["duckduckgo"]


def test_real_duckduckgo_other_failure_is_reported(clients):
    clients[SearchProvider.DUCKDUCKGO] = _real_ddg(_raise(RuntimeError("connection reset")))
    out = _search(providers=["duckduckgo"])
    assert out.providers == []
    assert out.errors == {"duckduckgo": "DuckDuckGo search failed (RuntimeError)."}


def test_real_duckduckgo_no_results_stays_an_empty_answer(clients):
    from ddgs.exceptions import DDGSException

    clients[SearchProvider.DUCKDUCKGO] = _real_ddg(_raise(DDGSException("No results found.")))
    out = _search()
    assert out.providers == ["duckduckgo"] and out.results == []
    assert out.fallbacks == {"duckduckgo": ["brave"]}
    assert "duckduckgo" not in out.errors


def test_real_duckduckgo_results_are_returned(clients):
    clients[SearchProvider.DUCKDUCKGO] = _real_ddg(
        lambda *a, **kw: [{"title": "t", "href": "https://example.com/a", "body": "b"}])
    out = _search()
    assert out.providers == ["duckduckgo"] and [r.url for r in out.results] == ["https://example.com/a"]


def test_fallback_runs_when_the_keyed_client_cannot_be_built(clients):
    del clients[SearchProvider.BRAVE]  # key disconnected after provider selection
    out = _search()
    assert out.providers == ["duckduckgo"]
    assert set(out.errors) == {"brave"}
    assert out.fallbacks == {"duckduckgo": ["brave"]}


def test_named_provider_that_cannot_be_built_is_not_swapped(clients):
    del clients[SearchProvider.BRAVE]
    out = _search(providers=["brave"])
    assert out.providers == [] and out.fallbacks == {}
    assert clients[SearchProvider.DUCKDUCKGO].calls == 0


def test_mcp_web_search_reports_the_fallback(clients):
    from llm_bawt.mcp_server.search_tools import web_search

    res = asyncio.run(web_search("q"))
    assert res["providers"] == ["duckduckgo"] and res["count"] == 1
    assert res["errors"] == {"brave": BRAVE_429}
    assert res["fallbacks"] == {"duckduckgo": ["brave"]}
    assert "brave failed" in res["note"] and "duckduckgo" in res["note"]
    assert "error" not in res
