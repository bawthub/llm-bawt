"""Tests for Brave Search client."""

from unittest.mock import Mock, patch

import httpx
import pytest

from llm_bawt.search.base import SearchError, SearchProvider
from llm_bawt.search import brave_client
from llm_bawt.search.brave_client import BRAVE_API_BASE, BraveSearchClient, is_brave_available


@pytest.fixture(autouse=True)
def _no_pacing_waits(monkeypatch):
    """Unit tests don't wait out Brave's 1/s window; sleeps are recorded instead."""
    slept: list[float] = []
    brave_client._PACER.reset(0.0)
    monkeypatch.setattr(brave_client, "_sleep", slept.append)
    yield slept
    brave_client._PACER.reset()


class TestBraveSearchClient:
    """Tests for BraveSearchClient."""

    def test_provider_constant(self):
        """Verify provider is set correctly."""
        assert BraveSearchClient.PROVIDER == SearchProvider.BRAVE
        assert BraveSearchClient.REQUIRES_API_KEY is True

    def test_init_with_api_key(self):
        """Test client initialization."""
        client = BraveSearchClient(api_key="test-key", max_results=10)
        assert client._api_key == "test-key"
        assert client.max_results == 10
        assert client.is_available() is True

    def test_init_without_api_key(self):
        """Test client without API key."""
        client = BraveSearchClient(api_key="", max_results=5)
        assert client.is_available() is False

    @pytest.mark.parametrize("call", [
        lambda c: c.search("q"), lambda c: c.search_news("q"),
        lambda c: c.search_reddit("q"), lambda c: c.search_with_summary("q"),
    ])
    def test_no_api_key_is_an_error_not_an_empty_answer(self, call):
        with pytest.raises(SearchError, match="has no API key"):
            call(BraveSearchClient(api_key=""))

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_success(self, mock_client_class):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "Test description",
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search("test query")

        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].source == SearchProvider.BRAVE

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_news(self, mock_client_class):
        """Test news search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "News Article",
                    "url": "https://news.example.com",
                    "description": "Breaking news",
                    "age": "2 hours ago",
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search_news("test query", time_range="d")

        assert len(results) == 1
        assert "2 hours ago" in results[0].snippet

    @patch("llm_bawt.search.brave_client.httpx.Client")
    def test_search_reddit(self, mock_client_class):
        """Test reddit search uses web endpoint with site filter."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Reddit Thread",
                        "url": "https://www.reddit.com/r/test/comments/abc123/thread/",
                        "description": "Interesting discussion",
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search_reddit("best mechanical keyboard", time_range="w")

        assert len(results) == 1
        assert results[0].title == "Reddit Thread"
        assert results[0].url.startswith("https://www.reddit.com/")

        mock_client.get.assert_called_once()
        called_path = mock_client.get.call_args.args[0]
        called_params = mock_client.get.call_args.kwargs["params"]
        assert called_path == "/web/search"
        assert called_params["q"].startswith("site:reddit.com")
        assert called_params["freshness"] == "pw"


def _brave_answering(status: int, body: object = None, *, raw: bytes | None = None) -> BraveSearchClient:
    """A Brave client whose transport answers every request with ``status``."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert "test-key" not in str(request.url)
        if raw is not None:
            return httpx.Response(status, content=raw)
        return httpx.Response(status, json=body if body is not None else {})

    client = BraveSearchClient(api_key="test-key")
    client._client = httpx.Client(base_url=BRAVE_API_BASE, transport=httpx.MockTransport(handler))
    return client


class TestBraveFailuresAreErrors:
    """TASK-1562: an upstream failure raises; only a real empty answer is []."""

    @pytest.mark.parametrize("status,code", [
        (401, None), (403, None), (422, "SUBSCRIPTION_TOKEN_INVALID"),
    ])
    def test_a_rejected_key_says_reconnect(self, status, code):
        body = {"type": "ErrorResponse", "error": {"code": code, "status": status}} if code else {}
        client = _brave_answering(status, body)
        with pytest.raises(SearchError) as caught:
            client.search("q")
        message = str(caught.value)
        assert caught.value.provider is SearchProvider.BRAVE
        assert caught.value.status_code == status
        assert "rejected the API key" in message and "Settings → Providers" in message
        assert f"HTTP {status}" in message
        assert "test-key" not in message

    def test_rate_limit(self):
        with pytest.raises(SearchError, match=r"rate limit exceeded \(HTTP 429, RATE_LIMITED\)"):
            _brave_answering(429, {"error": {"code": "RATE_LIMITED"}}).search("q")

    def test_other_status_names_status_and_known_code(self):
        with pytest.raises(SearchError) as caught:
            _brave_answering(400, {"error": {"code": "OPTION_NOT_IN_PLAN"}}).search("q")
        assert str(caught.value) == "Brave Search failed (HTTP 400, OPTION_NOT_IN_PLAN)."

    @pytest.mark.parametrize("status", [401, 429, 500])
    def test_an_unknown_body_code_is_never_echoed(self, status, caplog):
        """Upstream-controlled text (here, the key itself) maps by status only."""
        with pytest.raises(SearchError) as caught:
            _brave_answering(status, {"error": {"code": "test-key"}}).search("q")
        assert "test-key" not in str(caught.value)
        assert f"(HTTP {status})" in str(caught.value)
        assert "test-key" not in caplog.text

    def test_a_non_json_error_body_still_maps_by_status(self):
        with pytest.raises(SearchError, match=r"failed \(HTTP 502\)"):
            _brave_answering(502, raw=b"<html>bad gateway</html>").search("q")

    def test_unreachable(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        client = BraveSearchClient(api_key="test-key")
        client._client = httpx.Client(base_url=BRAVE_API_BASE, transport=httpx.MockTransport(handler))
        with pytest.raises(SearchError, match="unreachable \\(ConnectError\\)"):
            client.search("q")

    def test_unreadable_success_body(self):
        with pytest.raises(SearchError, match="unreadable response"):
            _brave_answering(200, raw=b"not json").search("q")

    @pytest.mark.parametrize("call", [
        lambda c: c.search_news("q"),
        lambda c: c.search_reddit("q"),
        lambda c: c.search_with_summary("q"),
    ])
    def test_every_endpoint_raises(self, call):
        with pytest.raises(SearchError):
            call(_brave_answering(401))

    def test_an_empty_answer_is_an_empty_list(self):
        assert _brave_answering(200, {"web": {"results": []}}).search("q") == []


class TestBraveAvailability:
    """Tests for availability checks."""

    def test_is_brave_available(self):
        """httpx should always be available."""
        assert is_brave_available() is True


FREE = {"x-ratelimit-policy": "1;w=1, 2000;w=2592000", "x-ratelimit-limit": "1, 2000"}


def _brave_sequence(*answers) -> tuple[BraveSearchClient, list[httpx.Request]]:
    """A client whose transport replays ``(status, headers)`` answers in order."""
    seen: list[httpx.Request] = []
    queue = list(answers)

    def handler(request):
        seen.append(request)
        status, headers = queue.pop(0)
        body = {"web": {"results": [{"title": "t", "url": "https://e.x"}]}} if status == 200 else {
            "error": {"code": "RATE_LIMITED"}}
        return httpx.Response(status, json=body, headers=headers)

    client = BraveSearchClient(api_key="test-key")
    client._client = httpx.Client(base_url=BRAVE_API_BASE, transport=httpx.MockTransport(handler))
    return client, seen


class TestBravePacing:
    """TASK-2139: Free plan allows 1 request/second; pace instead of failing."""

    def test_concurrent_calls_are_spaced_by_the_interval(self, monkeypatch):
        import threading
        import time as real_time
        monkeypatch.setattr(brave_client, "_sleep", real_time.sleep)
        pacer = brave_client._Pacer(0.2)
        starts: list[float] = []
        threads = [threading.Thread(target=lambda: (pacer.wait(), starts.append(real_time.monotonic())))
                   for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        starts.sort()
        assert all(b - a >= 0.18 for a, b in zip(starts, starts[1:]))

    def test_the_interval_is_learned_from_the_policy_header(self):
        client, _ = _brave_sequence((200, FREE))
        client.search("q")
        assert brave_client._PACER.interval == pytest.approx(1.05)

    def test_a_per_second_429_is_retried_once_after_the_reset(self, _no_pacing_waits):
        limited = {**FREE, "x-ratelimit-remaining": "0, 1931", "x-ratelimit-reset": "1, 1987654"}
        ok = {**FREE, "x-ratelimit-remaining": "0, 1930", "x-ratelimit-reset": "1, 1987653"}
        client, seen = _brave_sequence((429, limited), (200, ok))
        assert [r.title for r in client.search("q")] == ["t"]
        assert len(seen) == 2
        assert max(_no_pacing_waits) >= 1.0

    def test_a_second_per_second_429_fails(self):
        limited = {**FREE, "x-ratelimit-remaining": "0, 1931", "x-ratelimit-reset": "1, 1987654"}
        client, seen = _brave_sequence((429, limited), (429, limited))
        with pytest.raises(SearchError, match="rate limit exceeded"):
            client.search("q")
        assert len(seen) == 2

    def test_monthly_exhaustion_fails_at_once_and_says_when(self):
        spent = {**FREE, "x-ratelimit-remaining": "0, 0", "x-ratelimit-reset": "1, 1987654"}
        client, seen = _brave_sequence((429, spent))
        with pytest.raises(SearchError) as caught:
            client.search("q")
        assert len(seen) == 1
        assert caught.value.status_code == 429
        assert str(caught.value) == "Brave Search's monthly quota is used up (HTTP 429). It resets in about 23 days."
