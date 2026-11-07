"""Brave Search client implementation."""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .base import SearchClient, SearchError, SearchProvider, SearchResult

logger = logging.getLogger(__name__)

BRAVE_API_BASE = "https://api.search.brave.com/res/v1"

_FRESHNESS = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}

# Brave error-body codes safe to show; anything else is reported by status only.
_KNOWN_CODES = frozenset({
    "SUBSCRIPTION_TOKEN_INVALID", "RATE_LIMITED", "OPTION_NOT_IN_PLAN",
    "VALIDATION", "RESOURCE_NOT_FOUND", "UNAUTHORIZED", "FORBIDDEN",
})


DEFAULT_INTERVAL = 1.05  # Free plan: 1 request per 1-second window, plus margin
_MARGIN = 0.05
_MAX_RETRY_DELAY = 2.0
_sleep = time.sleep  # indirection so tests can record waits


@dataclass(frozen=True)
class _RateLimits:
    """Brave's ``x-ratelimit-*`` headers: comma lists, shortest window first.

    e.g. policy ``1;w=1, 2000;w=2592000``, remaining ``0, 1931``,
    reset ``1, 1987654`` (seconds until each window resets).
    """

    policy: tuple[tuple[int, int], ...] = ()
    remaining: tuple[int, ...] = ()
    reset: tuple[int, ...] = ()

    @classmethod
    def from_headers(cls, headers: Any) -> "_RateLimits":
        if not isinstance(headers, (httpx.Headers, dict)):
            return cls()

        def ints(name: str) -> tuple[int, ...]:
            try:
                return tuple(int(v.strip()) for v in str(headers.get(name, "")).split(",") if v.strip())
            except ValueError:
                return ()

        policy = []
        for part in str(headers.get("x-ratelimit-policy", "")).split(","):
            m = re.fullmatch(r"\s*(\d+)\s*;\s*w\s*=\s*(\d+)\s*", part)
            if m:
                policy.append((int(m.group(1)), int(m.group(2))))
        return cls(tuple(policy), ints("x-ratelimit-remaining"), ints("x-ratelimit-reset"))

    def interval(self) -> float | None:
        """Spacing for the shortest window, or None when not advertised."""
        if not self.policy:
            return None
        limit, window = self.policy[0]
        return window / limit + _MARGIN if limit > 0 and window <= 60 else None

    def monthly_exhausted(self) -> bool:
        return len(self.remaining) > 1 and any(r <= 0 for r in self.remaining[1:])

    def retryable(self) -> bool:
        return not self.monthly_exhausted()

    def second_reset(self) -> float:
        base = self.reset[0] if self.reset else DEFAULT_INTERVAL
        return min(max(float(base), 0.0) + _MARGIN, _MAX_RETRY_DELAY)


class _Pacer:
    """Process-wide spacing of Brave calls: one slot per interval, in order."""

    def __init__(self, interval: float = DEFAULT_INTERVAL):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            _sleep(start - now)

    def defer(self, delay: float) -> None:
        """Push the next slot out to at least ``delay`` seconds from now."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + delay)

    def learn(self, limits: _RateLimits) -> None:
        interval = limits.interval()
        if interval is not None:
            self.interval = interval

    def reset(self, interval: float = DEFAULT_INTERVAL) -> None:
        with self._lock:
            self.interval = interval
            self._next = 0.0


_PACER = _Pacer()


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
        if not self._api_key:
            logger.error("Brave API key not configured")
            raise SearchError(self.PROVIDER, "Brave Search has no API key. Connect it in Settings → Providers.")
        try:
            for attempt in range(2):
                _PACER.wait()
                response = self._get_client().get(path, params=params)
                limits = _RateLimits.from_headers(getattr(response, "headers", None))
                _PACER.learn(limits)
                if response.status_code == 429 and attempt == 0 and limits.retryable():
                    delay = limits.second_reset()
                    logger.info("Brave per-second limit hit; retrying once in %.2fs", delay)
                    _PACER.defer(delay)
                    continue
                if response.status_code == 429 and limits.monthly_exhausted():
                    raise _quota_error(limits)
                response.raise_for_status()
                return response.json()
            raise AssertionError("unreachable")  # pragma: no cover
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
        max_results = max_results or self.max_results
        params = self._params(f"site:reddit.com {query}".strip(), max_results, time_range)
        return self._web_results(self._get_json("/web/search", params))[:max_results]

    def search_with_summary(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], str | None]:
        """Search with AI-generated summary (requires Pro plan)."""
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

    Brave's error body carries a machine code. Only a code from
    :data:`_KNOWN_CODES` is echoed: any other body text is upstream-controlled
    and could carry anything, including the key, so it maps by status alone.
    """
    status = response.status_code
    code = None
    try:
        error = response.json().get("error")
        if isinstance(error, dict) and error.get("code") in _KNOWN_CODES:
            code = error["code"]
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


def _quota_error(limits: _RateLimits) -> SearchError:
    """The plan's monthly quota is used up: say so and when it resets."""
    when = ""
    if len(limits.reset) > 1:
        days = limits.reset[1] / 86400
        when = " It resets today." if days < 1 else f" It resets in about {round(days)} day{'s' if round(days) != 1 else ''}."
    logger.warning("Brave monthly quota exhausted")
    return SearchError(
        SearchProvider.BRAVE,
        f"Brave Search's monthly quota is used up (HTTP 429).{when}",
        status_code=429,
    )


def is_brave_available() -> bool:
    """Check if httpx is installed (core dependency)."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False
