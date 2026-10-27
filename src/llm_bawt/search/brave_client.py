"""Brave Search client implementation."""

import logging
from typing import Any

import httpx

from .base import SearchClient, SearchError, SearchProvider, SearchResult

logger = logging.getLogger(__name__)

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"

_FRESHNESS = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}


class BraveSearchClient(SearchClient):
    """Brave Search API client.

    An upstream failure (rejected key, rate limit, outage) raises
    :class:`SearchError`; an empty list means Brave answered and found nothing.
    """

    PROVIDER = SearchProvider.BRAVE
    REQUIRES_API_KEY = True

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout: int = 10,
        include_summary: bool = False,
        safesearch: str = "moderate",
    ):
        """Initialize Brave Search client.

        Args:
            api_key: Brave Search API key
            max_results: Default number of results to return
            timeout: Request timeout in seconds
            include_summary: Include AI-generated summary (requires Pro plan)
            safesearch: Filter level - "off", "moderate", or "strict"
        """
        super().__init__(max_results=max_results)
        self._api_key = api_key
        self._timeout = timeout
        self._include_summary = include_summary
        self._safesearch = safesearch
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=BRAVE_API_BASE,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                timeout=self._timeout,
            )
        return self._client

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET ``path`` and return the JSON body, or raise :class:`SearchError`."""
        try:
            response = self._get_client().get(path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise _status_error(e.response) from None
        except httpx.RequestError as e:
            logger.error("Brave search request failed: %s", type(e).__name__)
            raise SearchError(self.PROVIDER, f"Brave Search is unreachable ({type(e).__name__}).") from None
        except ValueError:
            logger.error("Brave search returned a non-JSON body")
            raise SearchError(self.PROVIDER, "Brave Search returned an unreadable response.") from None

    def _params(self, query: str, max_results: int, time_range: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"q": query, "count": max_results, "safesearch": self._safesearch}
        if time_range in _FRESHNESS:
            params["freshness"] = _FRESHNESS[time_range]
        return params

    def _web_results(self, data: dict[str, Any], *, extra_snippets: bool = True) -> list[SearchResult]:
        results = []
        for raw in data.get("web", {}).get("results", []):
            snippet = raw.get("description", "")
            extra = raw.get("extra_snippets", []) if extra_snippets else []
            if extra:
                snippet = f"{snippet}\n{' '.join(extra[:2])}"
            results.append(SearchResult(
                title=raw.get("title", ""),
                url=raw.get("url", ""),
                snippet=snippet,
                score=self._normalize_score(raw),
                source=self.PROVIDER,
                raw=raw,
            ))
        return results

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a Brave web search."""
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results
        params = self._params(query, max_results)
        if region:
            params["country"] = region
        if self._include_summary:
            params["summary"] = "1"

        return self._web_results(self._get_json("/web/search", params))[:max_results]

    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search Brave news."""
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results
        data = self._get_json("/news/search", self._params(query, max_results, time_range))

        results = []
        for raw in data.get("results", []):
            snippet = raw.get("description", "")
            age = raw.get("age", "")
            if age:
                snippet = f"[{age}] {snippet}"
            results.append(SearchResult(
                title=raw.get("title", ""),
                url=raw.get("url", ""),
                snippet=snippet,
                score=None,
                source=self.PROVIDER,
                raw=raw,
            ))
        return results[:max_results]

    def search_reddit(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search Reddit content via Brave web search."""
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results
        params = self._params(f"site:reddit.com {query}".strip(), max_results, time_range)
        return self._web_results(self._get_json("/web/search", params))[:max_results]

    def search_with_summary(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], str | None]:
        """Search with AI-generated summary (requires Pro plan)."""
        if not self._api_key:
            return [], None

        max_results = max_results or self.max_results
        params = {**self._params(query, max_results), "summary": "1"}
        data = self._get_json("/web/search", params)

        summary = (data.get("summarizer") or {}).get("summary")
        return self._web_results(data, extra_snippets=False)[:max_results], summary

    def _normalize_score(self, raw: dict) -> float | None:
        """Normalize Brave ranking to a 0.0-1.0 score."""
        return None

    def is_available(self) -> bool:
        """Check if Brave Search is properly configured."""
        return is_brave_available() and bool(self._api_key)

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if self._client:
            self._client.close()


def _status_error(response: httpx.Response) -> SearchError:
    """Map a Brave error response to a user-facing :class:`SearchError`.

    Brave's error body carries a machine code (``SUBSCRIPTION_TOKEN_INVALID``,
    ``RATE_LIMITED``, …); only that code is echoed, never the request headers.
    """
    status = response.status_code
    code = None
    try:
        error = response.json().get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            code = error["code"][:64]
    except Exception:  # noqa: BLE001 — a non-JSON error body still maps by status
        pass
    detail = f"HTTP {status}" + (f", {code}" if code else "")

    if status in (401, 403) or code == "SUBSCRIPTION_TOKEN_INVALID":
        logger.error("Brave rejected the API key (%s)", detail)
        message = f"Brave Search rejected the API key ({detail}). Reconnect it in Settings → Providers."
    elif status == 429:
        logger.warning("Brave rate limit exceeded (%s)", detail)
        message = f"Brave Search rate limit exceeded ({detail})."
    else:
        logger.error("Brave search HTTP error (%s)", detail)
        message = f"Brave Search failed ({detail})."
    return SearchError(SearchProvider.BRAVE, message, status_code=status)


def is_brave_available() -> bool:
    """Check if httpx is installed (core dependency)."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False
