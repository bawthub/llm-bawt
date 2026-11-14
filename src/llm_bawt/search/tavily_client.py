"""Tavily search client.

Production-ready search API designed specifically for LLM agents.
Provides relevance scores and optionally LLM-generated answers.

Requires API key: https://tavily.com/
Free tier: 1,000 searches/month
"""

import logging
from typing import TYPE_CHECKING

from .base import SearchClient, SearchError, SearchResult, SearchProvider

if TYPE_CHECKING:
    from tavily import TavilyClient as TavilyAPI

logger = logging.getLogger(__name__)


def is_tavily_available() -> bool:
    """Check if tavily-python library is installed."""
    try:
        from tavily import TavilyClient  # noqa: F401
        return True
    except ImportError:
        return False


class TavilyClient(SearchClient):
    """Tavily search client - optimized for LLM agents.
    
    Features:
    - Relevance scores for each result
    - Optional AI-generated answer summary
    - Fast response times (~180ms p50)
    - Clean, structured responses
    
    Configure the API key through a stored Tavily connection.
    LLM_BAWT_TAVILY_API_KEY is a deprecated environment fallback.
    """
    
    PROVIDER = SearchProvider.TAVILY
    REQUIRES_API_KEY = True
    
    def __init__(
        self, 
        api_key: str,
        max_results: int = 5,
        include_answer: bool = False,
        search_depth: str = "basic",
    ):
        """Initialize Tavily client.
        
        Args:
            api_key: Tavily API key
            max_results: Default max results per search
            include_answer: Whether to request AI-generated answer summary
            search_depth: 'basic' (1 credit) or 'advanced' (2 credits)
        """
        super().__init__(max_results=max_results)
        self._api_key = api_key
        self._include_answer = include_answer
        self._search_depth = search_depth
        self._client: "TavilyAPI | None" = None
    
    def _get_client(self) -> "TavilyAPI":
        """Lazy-initialize the Tavily client."""
        if self._client is None:
            from tavily import TavilyClient as TavilyAPI
            self._client = TavilyAPI(api_key=self._api_key)
        return self._client
    
    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a Tavily web search.
        
        Args:
            query: Search query
            max_results: Max results (default: self.max_results)
            region: Not used by Tavily, included for interface compatibility
            
        Returns:
            List of SearchResult objects with relevance scores
        """
        if not is_tavily_available():
            logger.error("tavily-python not installed. Install with: pip install tavily-python")
            raise SearchError(self.PROVIDER, "Tavily search is unavailable: tavily-python isn't installed.")

        if not self._api_key:
            logger.error("Tavily API key not configured")
            raise SearchError(self.PROVIDER, "Tavily search is unavailable: no API key is configured.")
        
        max_results = max_results or self.max_results
        
        try:
            client = self._get_client()
            
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=self._search_depth,
                include_answer=self._include_answer,
            )
            
            results = []
            
            # Extract results from response
            raw_results = response.get("results", [])
            for raw in raw_results:
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=raw.get("content", ""),
                    score=raw.get("score"),  # Tavily provides relevance scores
                    source=self.PROVIDER,
                    raw=raw,
                ))
            
            logger.debug(f"Tavily search '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error("Tavily search failed (%s)", type(e).__name__)
            raise SearchError(self.PROVIDER, f"Tavily search failed ({type(e).__name__}).") from None
    
    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search for news using Tavily.
        
        Tavily doesn't have a dedicated news endpoint, but we can
        use topic filtering or append 'news' to the query.
        
        Args:
            query: Search query
            max_results: Max results
            time_range: Time filter (Tavily supports 'd', 'w', 'm', 'y')
            
        Returns:
            List of SearchResult objects
        """
        ranges = {"d": "day", "w": "week", "m": "month", "y": "year"}
        if time_range is not None and time_range not in ranges:
            raise ValueError("time_range must be one of d, w, m, y")
        time_filter = {"time_range": ranges[time_range]} if time_range is not None else {}

        if not is_tavily_available():
            logger.error("tavily-python not installed")
            raise SearchError(self.PROVIDER, "Tavily news search is unavailable: tavily-python isn't installed.")

        if not self._api_key:
            logger.error("Tavily API key not configured")
            raise SearchError(self.PROVIDER, "Tavily news search is unavailable: no API key is configured.")
        
        max_results = max_results or self.max_results
        
        try:
            client = self._get_client()
            
            # Use topic='news' if available, otherwise append to query
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=self._search_depth,
                topic="news",  # Tavily supports topic filtering
                **time_filter,
            )
            
            results = []
            raw_results = response.get("results", [])
            for raw in raw_results:
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=raw.get("content", ""),
                    score=raw.get("score"),
                    source=self.PROVIDER,
                    raw=raw,
                ))
            
            logger.debug(f"Tavily news search '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error("Tavily news search failed (%s)", type(e).__name__)
            raise SearchError(self.PROVIDER, f"Tavily news search failed ({type(e).__name__}).") from None
    
    def is_available(self) -> bool:
        """Check if Tavily is properly configured."""
        return is_tavily_available() and bool(self._api_key)
