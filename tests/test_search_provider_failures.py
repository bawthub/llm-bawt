"""TASK-2248: provider failures are not successful empty answers (no network)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from llm_bawt.search import ddgs_client, multi, tavily_client
from llm_bawt.search.base import SearchError, SearchProvider, SearchResult
from llm_bawt.search.reddit_client import RedditSearchClient
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall

PRIVATE = "FAKE-SECRET-MUST-NOT-APPEAR"


@pytest.fixture
def tavily(monkeypatch):
    monkeypatch.setattr(tavily_client, "is_tavily_available", lambda: True)
    client = tavily_client.TavilyClient(api_key=PRIVATE)
    client._client = Mock()
    client._client.search.return_value = {"results": []}
    return client


@pytest.fixture
def ddg(monkeypatch):
    monkeypatch.setattr(ddgs_client, "is_ddgs_available", lambda: True)
    client = ddgs_client.DuckDuckGoClient()
    client._client = Mock()
    client._client.news.return_value = []
    return client


@pytest.fixture
def reddit():
    client = RedditSearchClient(client_id="test-client", client_secret=PRIVATE)
    client._client = Mock()
    client._client.post.return_value.json.return_value = {"access_token": "test-token", "expires_in": 3600}
    client._client.get.return_value.json.return_value = {"data": {"children": []}}
    return client


@pytest.mark.parametrize("method", ["search", "search_news"])
def test_tavily_exception_is_sanitized_failure(tavily, method, caplog):
    tavily._client.search.side_effect = RuntimeError(PRIVATE)
    with pytest.raises(SearchError) as caught:
        getattr(tavily, method)("query")
    assert caught.value.provider is SearchProvider.TAVILY
    assert "failed" in str(caught.value)
    assert PRIVATE not in str(caught.value) + caplog.text
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("method", ["search", "search_news"])
@pytest.mark.parametrize("missing", ["library", "key"])
def test_tavily_unavailable_is_not_empty(tavily, method, missing, monkeypatch):
    if missing == "library":
        monkeypatch.setattr(tavily_client, "is_tavily_available", lambda: False)
    else:
        tavily._api_key = ""
    with pytest.raises(SearchError, match="unavailable"):
        getattr(tavily, method)("query")
    tavily._client.search.assert_not_called()


@pytest.mark.parametrize("method", ["search", "search_news"])
def test_tavily_genuine_empty_stays_empty(tavily, method):
    assert getattr(tavily, method)("query") == []


@pytest.mark.parametrize("stage", ["post", "get"])
@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_reddit_http_failure_is_sanitized_and_retains_status(reddit, stage, status, caplog):
    request = httpx.Request("GET", f"https://example.invalid/?token={PRIVATE}")
    response = httpx.Response(status, request=request)
    getattr(reddit._client, stage).return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        PRIVATE, request=request, response=response)
    with pytest.raises(SearchError) as caught:
        reddit.search("query")
    assert caught.value.provider is SearchProvider.REDDIT
    assert caught.value.status_code == status
    assert f"HTTP {status}" in str(caught.value)
    assert PRIVATE not in str(caught.value) + caplog.text
    if stage == "post":
        reddit._client.get.assert_not_called()


@pytest.mark.parametrize("stage", ["post", "get"])
@pytest.mark.parametrize("error", [httpx.ConnectError(PRIVATE), ValueError(PRIVATE)])
def test_reddit_request_and_decode_failures_are_not_empty(reddit, stage, error, caplog):
    getattr(reddit._client, stage).side_effect = error
    with pytest.raises(SearchError, match="failed") as caught:
        reddit.search_reddit("query")
    assert PRIVATE not in str(caught.value) + caplog.text


def test_reddit_missing_credentials_or_token_is_not_empty(reddit):
    reddit._client_secret = ""
    with pytest.raises(SearchError, match="credentials"):
        reddit.search("query")
    reddit._client.post.assert_not_called()
    reddit._client_secret = PRIVATE
    reddit._client.post.return_value.json.return_value = {}
    with pytest.raises(SearchError, match="no access token"):
        reddit.search("query")
    reddit._client.get.assert_not_called()


def test_reddit_blank_query_and_genuine_empty_remain_empty(reddit):
    assert reddit.search("   ") == []
    reddit._client.post.assert_not_called()
    assert reddit.search("query") == []
    assert reddit.search("query again") == []
    reddit._client.post.assert_called_once()
    assert reddit._client.get.call_count == 2


@pytest.mark.parametrize("error", [RuntimeError(PRIVATE), httpx.ReadTimeout(PRIVATE)])
def test_ddg_news_failure_is_sanitized(ddg, error, caplog):
    ddg._client.news.side_effect = error
    with pytest.raises(SearchError, match="news search failed") as caught:
        ddg.search_news("query")
    assert PRIVATE not in str(caught.value) + caplog.text


def test_ddg_news_missing_library_is_not_empty(ddg, monkeypatch):
    monkeypatch.setattr(ddgs_client, "is_ddgs_available", lambda: False)
    with pytest.raises(SearchError, match="unavailable"):
        ddg.search_news("query")
    ddg._client.news.assert_not_called()


def test_ddg_news_rate_limit_retains_status(ddg, caplog):
    from ddgs.exceptions import RatelimitException

    ddg._client.news.side_effect = RatelimitException(PRIVATE)
    with pytest.raises(SearchError) as caught:
        ddg.search_news("query")
    assert caught.value.status_code == 429
    assert "rate limiting" in str(caught.value)
    assert PRIVATE not in str(caught.value) + caplog.text


def test_ddg_news_genuine_empty_and_no_results_exception_stay_empty(ddg):
    from ddgs.exceptions import DDGSException

    assert ddg.search_news("query") == []
    ddg._client.news.side_effect = DDGSException("No results found.")
    assert ddg.search_news("query") == []
    # Only the exact library sentinel is an empty answer.
    ddg._client.news.side_effect = RuntimeError("No results found.")
    with pytest.raises(SearchError):
        ddg.search_news("query")


@pytest.mark.parametrize("failed", [False, True])
def test_real_tavily_failure_allows_default_fallback_but_empty_answer_does_not(
    tavily, monkeypatch, failed,
):
    fallback = Mock()
    fallback.search.return_value = [
        SearchResult("Fallback", "https://example.invalid/", "snippet", source=SearchProvider.DUCKDUCKGO)]
    clients = {SearchProvider.TAVILY: tavily, SearchProvider.DUCKDUCKGO: fallback}
    if failed:
        tavily._client.search.side_effect = RuntimeError(PRIVATE)
    monkeypatch.setattr(multi, "available_providers", lambda config: [SearchProvider.TAVILY])
    monkeypatch.setattr(multi, "is_search_available", lambda provider=None: True)
    monkeypatch.setattr(multi, "get_search_client", lambda config, provider, **kwargs: clients[provider])
    outcome = asyncio.run(multi.search_all_detailed(SimpleNamespace(), "query"))
    if failed:
        assert outcome.providers == ["duckduckgo"]
        assert set(outcome.errors) == {"tavily"}
        assert outcome.fallbacks == {"duckduckgo": ["tavily"]}
        assert len(outcome.results) == 1
        fallback.search.assert_called_once()
    else:
        assert outcome.providers == ["tavily"] and outcome.results == []
        assert not outcome.errors and not outcome.fallbacks
        fallback.search.assert_not_called()


def test_real_reddit_failure_is_not_an_answer_in_mcp(reddit, monkeypatch):
    from llm_bawt.mcp_server.search_tools import web_search

    reddit._client.post.side_effect = RuntimeError(PRIVATE)
    monkeypatch.setattr(multi, "get_search_client", lambda *args, **kwargs: reddit)
    response = asyncio.run(web_search("query", provider="reddit"))
    assert response["providers"] == [] and response["results"] == []
    assert set(response["errors"]) == {"reddit"} and "error" in response
    assert PRIVATE not in str(response)


@pytest.mark.parametrize("provider,kind", [("tavily", "web"), ("tavily", "news"),
                                          ("reddit", "reddit"), ("ddg", "news")])
def test_real_provider_failures_reach_direct_tool_as_errors(request, provider, kind):
    client = request.getfixturevalue(provider)
    if provider == "tavily":
        client._client.search.side_effect = RuntimeError(PRIVATE)
    elif provider == "reddit":
        client._client.post.side_effect = RuntimeError(PRIVATE)
    else:
        client._client.news.side_effect = RuntimeError(PRIVATE)
    executor = ToolExecutor(user_id="test-user", search_client=client)
    response = executor.execute(ToolCall(
        name="search", arguments={"type": kind, "query": "query"}, raw_text=""))
    assert 'status="error"' in response
    assert "failed" in response and "No results found" not in response
    assert PRIVATE not in response
