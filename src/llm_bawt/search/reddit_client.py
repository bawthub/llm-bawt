"""Reddit search client implementation using Reddit OAuth API."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from .base import SearchClient, SearchError, SearchProvider, SearchResult

logger = logging.getLogger(__name__)

REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"


def is_reddit_available() -> bool:
    """Check if httpx is installed (core dependency)."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False


class RedditSearchClient(SearchClient):
    """Search client backed by Reddit's official OAuth API."""

    PROVIDER = SearchProvider.REDDIT
    REQUIRES_API_KEY = True

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "llm-bawt/0.1",
        max_results: int = 5,
        timeout: int = 10,
    ):
        super().__init__(max_results=max_results)
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent.strip() or "llm-bawt/0.1"
        self._timeout = timeout
        self._client: httpx.Client | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _token_is_valid(self) -> bool:
        """Return True if cached token exists and is not near expiry."""
        if not self._access_token or not self._token_expires_at:
            return False
        return datetime.now(UTC) < self._token_expires_at

    def _ensure_token(self) -> bool:
        """Fetch OAuth token if needed."""
        if self._token_is_valid():
            return True
        if not self._client_id or not self._client_secret:
            logger.error("Reddit client credentials not configured")
            raise SearchError(self.PROVIDER, "Reddit search is unavailable: client credentials aren't configured.")

        try:
            client = self._get_client()
            response = client.post(
                REDDIT_AUTH_URL,
                auth=(self._client_id, self._client_secret),
                data={"grant_type": "client_credentials"},
                headers={
                    "User-Agent": self._user_agent,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            response.raise_for_status()
            data = response.json()

            token = data.get("access_token", "")
            expires_in = int(data.get("expires_in", 3600))
            if not token:
                logger.error("Reddit token response missing access_token")
                raise SearchError(self.PROVIDER, "Reddit authentication failed: no access token was returned.")

            # Refresh 60s before expiry.
            self._access_token = token
            self._token_expires_at = datetime.now(UTC) + timedelta(seconds=max(60, expires_in - 60))
            return True

        except SearchError:
            raise
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error("Reddit authentication failed (HTTP %s)", status)
            raise SearchError(self.PROVIDER, f"Reddit authentication failed (HTTP {status}).", status) from None
        except Exception as e:
            logger.error("Reddit authentication failed (%s)", type(e).__name__)
            raise SearchError(self.PROVIDER, f"Reddit authentication failed ({type(e).__name__}).") from None

    def _map_time_range(self, time_range: str | None) -> timedelta | None:
        """Map tool time range to relative cutoff window."""
        if not time_range:
            return None
        mapping = {
            "d": timedelta(days=1),
            "w": timedelta(weeks=1),
            "m": timedelta(days=30),
            "y": timedelta(days=365),
        }
        return mapping.get(str(time_range).strip().lower())

    def _build_result(self, raw: dict[str, Any]) -> SearchResult:
        """Convert Reddit listing child.data into SearchResult."""
        subreddit = raw.get("subreddit_name_prefixed") or raw.get("subreddit") or "r/unknown"
        score = raw.get("score")
        comments = raw.get("num_comments")
        created_utc = raw.get("created_utc")

        meta_bits: list[str] = [str(subreddit)]
        if score is not None:
            meta_bits.append(f"score {score}")
        if comments is not None:
            meta_bits.append(f"{comments} comments")
        if created_utc:
            try:
                ts = datetime.fromtimestamp(float(created_utc), tz=UTC).strftime("%Y-%m-%d")
                meta_bits.append(ts)
            except Exception:
                pass

        title = str(raw.get("title") or "").strip()
        body = str(raw.get("selftext") or "").strip()
        snippet = " | ".join(meta_bits)
        if body:
            body_one_line = " ".join(body.split())
            snippet = f"{snippet}\n{body_one_line[:400]}"

        permalink = str(raw.get("permalink") or "")
        url = f"https://www.reddit.com{permalink}" if permalink else str(raw.get("url") or "")

        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            score=None,
            source=self.PROVIDER,
            raw=raw,
        )

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a Reddit search."""
        del region  # Reddit search does not support region.
        return self.search_reddit(query=query, max_results=max_results)

    def search_reddit(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search Reddit posts/threads via OAuth API."""
        if not query.strip():
            return []
        self._ensure_token()

        max_results = max_results or self.max_results

        try:
            client = self._get_client()
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "User-Agent": self._user_agent,
            }

            # Ask for more than needed when time filtering so we can post-filter by created_utc.
            fetch_limit = max(max_results, 25)
            if time_range:
                fetch_limit = max(fetch_limit, 50)
            fetch_limit = min(fetch_limit, 100)

            response = client.get(
                f"{REDDIT_API_BASE}/search",
                params={
                    "q": query,
                    "sort": "new",
                    "limit": fetch_limit,
                    "raw_json": 1,
                    "restrict_sr": "false",
                    "type": "link",
                },
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            children = data.get("data", {}).get("children", [])

            cutoff: datetime | None = None
            window = self._map_time_range(time_range)
            if window is not None:
                cutoff = datetime.now(UTC) - window

            results: list[SearchResult] = []
            for child in children:
                raw = child.get("data", {})
                if cutoff is not None:
                    try:
                        created = datetime.fromtimestamp(float(raw.get("created_utc", 0)), tz=UTC)
                        if created < cutoff:
                            continue
                    except Exception:
                        continue
                results.append(self._build_result(raw))
                if len(results) >= max_results:
                    break

            return results

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.error("Reddit search failed (HTTP %s)", status)
            raise SearchError(self.PROVIDER, f"Reddit search failed (HTTP {status}).", status) from None
        except Exception as e:
            logger.error("Reddit search failed (%s)", type(e).__name__)
            raise SearchError(self.PROVIDER, f"Reddit search failed ({type(e).__name__}).") from None

    def is_available(self) -> bool:
        """Check if Reddit search is configured and available."""
        return is_reddit_available() and bool(self._client_id and self._client_secret)

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if self._client:
            self._client.close()
