"""DuckDuckGo search client.

Free search using the ddgs library (DuckDuckGo scraping).
No API key required, but subject to rate limiting.

Note: This is an unofficial library - for heavy production use,
consider Tavily or another official API.
"""

import logging
from typing import TYPE_CHECKING

from .base import SearchClient, SearchError, SearchResult, SearchProvider

if TYPE_CHECKING:
    from ddgs import DDGS

logger = logging.getLogger(__name__)

# ddgs raises its base exception with this exact message when every engine ran
# without error and found nothing; any other exception is a real failure.
_NO_RESULTS = "No results found."


def _is_no_results(exc: Exception) -> bool:
    return type(exc).__name__ == "DDGSException" and str(exc) == _NO_RESULTS

# Suppress noisy ddgs internal logging (backend errors for Wikipedia, etc.)
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)


def is_ddgs_available() -> bool:
    """Check if ddgs library is installed."""
    try:
        from ddgs import DDGS  # noqa: F401
        return True
    except ImportError:
        return False


class DuckDuckGoClient(SearchClient):
    """DuckDuckGo search via ddgs library.
    
    Free, no API key required. Good for development and light use.
    May be rate limited under heavy load.
    
    Note: ddgs uses multiple backends internally. Some may fail due to
    DNS issues or rate limiting. Errors from individual backends are
    logged but don't fail the overall search.
    """
    
    PROVIDER = SearchProvider.DUCKDUCKGO
    REQUIRES_API_KEY = False
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        super().__init__(max_results=max_results)
        self._client: "DDGS | None" = None
        self._timeout = timeout
    
    def _get_client(self) -> "DDGS":
        """Lazy-initialize the DDGS client with timeout."""
        if self._client is None:
            from ddgs import DDGS
            # Set timeout to avoid hanging on slow/broken backends
            self._client = DDGS(timeout=self._timeout)
        return self._client
    
    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a DuckDuckGo web search.
        
        Args:
            query: Search query
            max_results: Max results (default: self.max_results)
            region: Region code like 'us-en', 'uk-en', etc.
            
        Returns:
            List of SearchResult objects; empty only when the search found nothing.

        Raises:
            SearchError: The search couldn't run (library missing, rate limit,
                timeout, network), so a failure never reads as "no results".
        """
        if not is_ddgs_available():
            logger.error("ddgs library not installed. Install with: pip install ddgs")
            raise SearchError(self.PROVIDER, "DuckDuckGo search is unavailable: the ddgs library isn't installed.")

        max_results = max_results or self.max_results
        
        try:
            client = self._get_client()
            
            # ddgs.text() returns list of dicts with: title, href/url, body
            raw_results = client.text(
                query,
                max_results=max_results,
                region=region or "wt-wt",  # Worldwide by default
            )
            
            results = []
            for raw in raw_results:
                # Handle both 'href' and 'url' keys (ddgs has used both)
                url = raw.get("href") or raw.get("url", "")
                
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=url,
                    snippet=raw.get("body", ""),
                    score=None,  # DuckDuckGo doesn't provide relevance scores
                    source=self.PROVIDER,
                    raw=raw,
                ))
            
            logger.debug(f"DuckDuckGo search '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            if _is_no_results(e):
                return []
            # TASK-1619: report the failure; [] would read as an empty answer.
            if "ratelimit" in type(e).__name__.lower() or "ratelimit" in str(e).lower():
                logger.warning(f"DuckDuckGo rate limit hit: {e}")
                raise SearchError(self.PROVIDER, "DuckDuckGo is rate limiting searches. Try again shortly.",
                                  status_code=429) from None
            logger.error(f"DuckDuckGo search failed: {e}")
            raise SearchError(self.PROVIDER, f"DuckDuckGo search failed ({type(e).__name__}).") from None
    
    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search DuckDuckGo news.
        
        Args:
            query: Search query
            max_results: Max results
            time_range: 'd' (day), 'w' (week), 'm' (month), 'y' (year)
            
        Returns:
            List of SearchResult objects
        """
        if not is_ddgs_available():
            logger.error("ddgs library not installed")
            return []
        
        max_results = max_results or self.max_results
        
        try:
            client = self._get_client()
            
            # ddgs.news() has timelimit parameter
            raw_results = client.news(
                query,
                max_results=max_results,
                timelimit=time_range,  # d, w, m, y
            )
            
            results = []
            for raw in raw_results:
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=raw.get("body", ""),
                    score=None,
                    source=self.PROVIDER,
                    raw=raw,
                ))
            
            logger.debug(f"DuckDuckGo news search '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            if "ratelimit" in str(e).lower():
                logger.warning(f"DuckDuckGo rate limit hit: {e}")
            else:
                logger.error(f"DuckDuckGo news search failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if DuckDuckGo search is available."""
        return is_ddgs_available()
