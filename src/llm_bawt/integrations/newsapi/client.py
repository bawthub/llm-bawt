"""NewsAPI client for article search and top headlines.

Uses newsapi.org REST API:
- /v2/everything - Search articles by keyword across 150k+ sources
- /v2/top-headlines - Top headlines by country, category, or source
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"


class NewsAPIClient:
    """NewsAPI (newsapi.org) client."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 10,
        language: str = "en",
    ):
        # Callers resolve the key (llm_bawt.search.credentials.resolve_key);
        # no env read here, so a disconnected key stays off.
        self._api_key = api_key or ""
        self._timeout = timeout
        self._language = language
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=NEWSAPI_BASE,
                headers={
                    "Accept": "application/json",
                    "X-Api-Key": self._api_key,
                },
                timeout=self._timeout,
            )
        return self._client

    def _format_articles(self, articles: list[dict[str, Any]]) -> str:
        """Format NewsAPI articles for LLM consumption."""
        if not articles:
            return "No articles found."

        lines = [f"Found {len(articles)} articles:\n"]

        for i, article in enumerate(articles, 1):
            title = article.get("title") or "Untitled"
            url = article.get("url") or ""
            description = article.get("description") or ""
            published = article.get("publishedAt", "")
            source_name = article.get("source", {}).get("name", "")

            lines.append(f"{i}. **{title}**")
            meta_parts = []
            if source_name:
                meta_parts.append(source_name)
            if published:
                meta_parts.append(published[:10])
            if meta_parts:
                lines.append(f"   [{' | '.join(meta_parts)}]")
            if description:
                lines.append(f"   {description}")
            if url:
                lines.append(f"   {url}")
            lines.append("")

        return "\n".join(lines)

    def search(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "publishedAt",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        """Search articles via /v2/everything.

        Args:
            query: Search keywords
            max_results: Number of results (max 100)
            sort_by: 'relevancy', 'popularity', or 'publishedAt'
            from_date: Oldest article date (ISO 8601, e.g. '2026-01-01')
            to_date: Newest article date (ISO 8601)

        Returns:
            Formatted string of articles
        """
        if not self._api_key:
            return "Error: NewsAPI key not configured. Connect it in Settings → Providers."

        try:
            client = self._get_client()
            params: dict[str, Any] = {
                "q": query,
                "pageSize": min(max_results, 100),
                "sortBy": sort_by,
                "language": self._language,
            }
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date

            page = 1
            max_pages = 5
            articles: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            total_results = 0

            while len(articles) < max_results and page <= max_pages:
                page_params = {**params, "page": page}
                response = client.get("/everything", params=page_params)
                response.raise_for_status()
                data = response.json()
                if page == 1:
                    total_results = int(data.get("totalResults", 0) or 0)

                page_articles = data.get("articles", []) or []
                if not page_articles:
                    break

                for article in page_articles:
                    key = str(article.get("url") or article.get("title") or "")
                    if key and key in seen_keys:
                        continue
                    if key:
                        seen_keys.add(key)
                    articles.append(article)
                    if len(articles) >= max_results:
                        break

                if len(articles) >= max_results:
                    break

                if page * params["pageSize"] >= total_results:
                    break

                page += 1

            return self._format_articles(articles[:max_results])

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, "search")
        except httpx.RequestError as e:
            logger.error(f"NewsAPI search request failed: {e}")
            return f"Error: NewsAPI request failed: {e}"
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return f"Error: {e}"

    def headlines(
        self,
        query: str | None = None,
        max_results: int = 5,
        country: str = "us",
        category: str | None = None,
    ) -> str:
        """Get top headlines via /v2/top-headlines.

        Args:
            query: Optional keyword filter
            max_results: Number of results (max 100)
            country: 2-letter country code (e.g. 'us', 'gb', 'de')
            category: business, entertainment, general, health, science, sports, technology

        Returns:
            Formatted string of headlines
        """
        if not self._api_key:
            return "Error: NewsAPI key not configured. Connect it in Settings → Providers."

        try:
            client = self._get_client()
            params: dict[str, Any] = {
                "pageSize": min(max_results, 100),
            }
            if query:
                params["q"] = query
            if country:
                params["country"] = country.lower()
            if category:
                params["category"] = category.lower()

            page = 1
            max_pages = 5
            articles: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            total_results = 0

            while len(articles) < max_results and page <= max_pages:
                page_params = {**params, "page": page}
                response = client.get("/top-headlines", params=page_params)
                response.raise_for_status()
                data = response.json()
                if page == 1:
                    total_results = int(data.get("totalResults", 0) or 0)

                page_articles = data.get("articles", []) or []
                if not page_articles:
                    break

                for article in page_articles:
                    key = str(article.get("url") or article.get("title") or "")
                    if key and key in seen_keys:
                        continue
                    if key:
                        seen_keys.add(key)
                    articles.append(article)
                    if len(articles) >= max_results:
                        break

                if len(articles) >= max_results:
                    break

                if page * params["pageSize"] >= total_results:
                    break

                page += 1

            return self._format_articles(articles[:max_results])

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, "headlines")
        except httpx.RequestError as e:
            logger.error(f"NewsAPI headlines request failed: {e}")
            return f"Error: NewsAPI request failed: {e}"
        except Exception as e:
            logger.error(f"NewsAPI headlines failed: {e}")
            return f"Error: {e}"

    def _handle_http_error(self, e: httpx.HTTPStatusError, context: str) -> str:
        """Handle HTTP errors from NewsAPI."""
        if e.response.status_code == 401:
            logger.error("NewsAPI key is invalid")
            return "Error: NewsAPI key is invalid."
        elif e.response.status_code == 429:
            logger.warning("NewsAPI rate limit exceeded")
            return "Error: NewsAPI rate limit exceeded. Try again later."
        else:
            logger.error(f"NewsAPI {context} HTTP error: {e}")
            return f"Error: NewsAPI HTTP {e.response.status_code}"

    def is_available(self) -> bool:
        """Check if NewsAPI is configured."""
        return bool(self._api_key)

    def __del__(self) -> None:
        if self._client:
            self._client.close()
