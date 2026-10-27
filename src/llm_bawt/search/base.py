"""Base search client interface.

Defines the abstract SearchClient that all providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SearchProvider(str, Enum):
    """Available search providers."""
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    BRAVE = "brave"
    REDDIT = "reddit"


class SearchError(Exception):
    """A provider answered with a failure instead of results.

    Raised rather than returning ``[]`` so a caller can tell "the provider
    failed" from "the provider found nothing". The message is user-facing and
    never contains the API key.
    """

    def __init__(self, provider: SearchProvider, message: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


@dataclass
class SearchResult:
    """A single search result.
    
    Attributes:
        title: Page title
        url: Full URL to the page
        snippet: Text snippet/description from the result
        score: Optional relevance score (0.0-1.0) if provider supports it
        source: Which search provider returned this result
        raw: Optional raw data from the provider for debugging
    """
    title: str
    url: str
    snippet: str
    score: float | None = None
    source: SearchProvider | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "source": self.source.value if self.source else None,
        }


class SearchClient(ABC):
    """Abstract base class for search clients.
    
    Implementations must provide:
    - search(): Perform a web search
    - search_news(): Search for news articles (optional, can fall back to search)
    """
    
    # Provider identifier
    PROVIDER: SearchProvider
    
    # Whether this provider requires an API key
    REQUIRES_API_KEY: bool = False
    
    def __init__(self, max_results: int = 5):
        """Initialize search client.
        
        Args:
            max_results: Default maximum results to return per search
        """
        self.max_results = max_results
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """Perform a web search.
        
        Args:
            query: Search query string
            max_results: Override default max results
            region: Optional region code (e.g., 'us-en', 'uk-en')
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search for news articles.
        
        Default implementation falls back to regular search.
        Providers can override for specialized news search.
        
        Args:
            query: Search query string
            max_results: Override default max results
            time_range: Time filter - 'd' (day), 'w' (week), 'm' (month)
            
        Returns:
            List of SearchResult objects
        """
        # Default: just do a regular search with "news" appended
        return self.search(f"{query} news", max_results=max_results)

    def search_reddit(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search Reddit posts/threads.

        Default implementation uses a site-filtered web query.
        Providers can override for better freshness/filtering support.

        Args:
            query: Search query string
            max_results: Override default max results
            time_range: Optional freshness filter - 'd', 'w', 'm', 'y'

        Returns:
            List of SearchResult objects
        """
        del time_range  # Not used by the default implementation
        return self.search(f"site:reddit.com {query}".strip(), max_results=max_results)
    
    def format_results_for_llm(self, results: list[SearchResult]) -> str:
        """Format search results for injection into LLM context.
        
        Produces a clean, readable format optimized for LLM consumption.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string suitable for tool result injection
        """
        if not results:
            return "No results found for your search query."
        
        lines = [f"Found {len(results)} results:\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   {result.snippet}")
            lines.append(f"   Source: {result.url}")
            if result.score is not None:
                lines.append(f"   Relevance: {result.score:.2f}")
            lines.append("")  # Blank line between results
        
        return "\n".join(lines)
    
    def is_available(self) -> bool:
        """Check if this search client is properly configured and available.
        
        Default implementation returns True for clients that don't require setup.
        Override for clients that need API keys or other configuration.
        """
        return True
