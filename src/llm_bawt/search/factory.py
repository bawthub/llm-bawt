"""Factory for creating search clients.

Handles provider selection, configuration, and fallback logic.
"""

import logging
from typing import TYPE_CHECKING

from .base import SearchClient, SearchProvider
from .credentials import search_key, unusable_reason

if TYPE_CHECKING:
    from ..utils.config import Config

logger = logging.getLogger(__name__)


def is_search_available(provider: SearchProvider | str | None = None) -> bool:
    """Check if search is available for the given provider.
    
    Args:
        provider: Specific provider to check, or None to check any
        
    Returns:
        True if search is available
    """
    if provider is None:
        # Check if any provider is available
        from .ddgs_client import is_ddgs_available
        from .tavily_client import is_tavily_available
        from .brave_client import is_brave_available
        from .reddit_client import is_reddit_available
        return is_ddgs_available() or is_tavily_available() or is_brave_available() or is_reddit_available()
    
    if isinstance(provider, str):
        try:
            provider = SearchProvider(provider.lower())
        except ValueError:
            return False
    
    if provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import is_ddgs_available
        return is_ddgs_available()
    elif provider == SearchProvider.TAVILY:
        from .tavily_client import is_tavily_available
        return is_tavily_available()
    elif provider == SearchProvider.BRAVE:
        from .brave_client import is_brave_available
        return is_brave_available()
    elif provider == SearchProvider.REDDIT:
        from .reddit_client import is_reddit_available
        return is_reddit_available()
    
    return False


def get_search_client(
    config: "Config",
    provider: SearchProvider | str | None = None,
    max_results: int | None = None,
) -> SearchClient | None:
    """Create a search client based on configuration.
    
    Provider selection priority:
    1. Explicit provider argument
    2. Config SEARCH_PROVIDER setting
    3. Tavily (if API key configured)
    4. Brave (if API key configured)
    5. Reddit API (if OAuth credentials configured)
    6. DuckDuckGo (free fallback)
    
    Args:
        config: Application config
        provider: Override provider selection
        max_results: Override default max results
        
    Returns:
        Configured SearchClient or None if search unavailable
    """
    # Determine provider
    if provider is not None:
        if isinstance(provider, str):
            try:
                provider = SearchProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown search provider: {provider}")
                provider = None
    
    if provider is None:
        # Check config for preferred provider
        config_provider = getattr(config, "SEARCH_PROVIDER", None)
        if config_provider:
            try:
                provider = SearchProvider(config_provider.lower())
            except ValueError:
                logger.warning(f"Unknown search provider in config: {config_provider}")
    
    if provider is None:
        # Auto-select based on availability
        reddit_client_id = getattr(config, "REDDIT_CLIENT_ID", None)
        reddit_client_secret = getattr(config, "REDDIT_CLIENT_SECRET", None)
        if search_key(config, SearchProvider.TAVILY) and is_search_available(SearchProvider.TAVILY):
            provider = SearchProvider.TAVILY
            logger.debug("Auto-selected Tavily (API key configured)")
        elif search_key(config, SearchProvider.BRAVE) and is_search_available(SearchProvider.BRAVE):
            provider = SearchProvider.BRAVE
            logger.debug("Auto-selected Brave (API key configured)")
        elif reddit_client_id and reddit_client_secret and is_search_available(SearchProvider.REDDIT):
            provider = SearchProvider.REDDIT
            logger.debug("Auto-selected Reddit API (credentials configured)")
        elif is_search_available(SearchProvider.DUCKDUCKGO):
            provider = SearchProvider.DUCKDUCKGO
            logger.debug("Auto-selected DuckDuckGo (free fallback)")
        else:
            logger.warning("No search provider available")
            return None
    
    # Get max results from config if not specified
    if max_results is None:
        max_results = getattr(config, "SEARCH_MAX_RESULTS", 5)
    
    # Ensure max_results is an int at this point
    max_results = int(max_results)
    
    # Create client (with fallback to DuckDuckGo if primary fails)
    if provider == SearchProvider.TAVILY:
        from .tavily_client import TavilyClient, is_tavily_available
        
        if not is_tavily_available():
            logger.warning("Tavily requested but tavily-python not installed, falling back to DuckDuckGo")
            # Fall back to DuckDuckGo
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None
        
        api_key = search_key(config, SearchProvider.TAVILY).value
        if not api_key:
            logger.warning("Tavily requested but no usable Tavily key, falling back to DuckDuckGo")
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None
        
        include_answer = getattr(config, "SEARCH_INCLUDE_ANSWER", False)
        search_depth = getattr(config, "SEARCH_DEPTH", "basic")
        
        return TavilyClient(
            api_key=api_key,
            max_results=max_results,
            include_answer=include_answer,
            search_depth=search_depth,
        )

    elif provider == SearchProvider.BRAVE:
        from .brave_client import BraveSearchClient, is_brave_available

        if not is_brave_available():
            logger.warning("Brave requested but httpx not installed, falling back to DuckDuckGo")
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None

        api_key = search_key(config, SearchProvider.BRAVE).value
        if not api_key:
            logger.warning("Brave requested but no usable Brave key, falling back to DuckDuckGo")
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None

        include_summary = getattr(config, "SEARCH_INCLUDE_ANSWER", False)
        safesearch = getattr(config, "BRAVE_SAFESEARCH", "moderate")
        timeout = getattr(config, "SEARCH_TIMEOUT", 10)

        return BraveSearchClient(
            api_key=api_key,
            max_results=max_results,
            timeout=timeout,
            include_summary=include_summary,
            safesearch=safesearch,
        )

    elif provider == SearchProvider.REDDIT:
        from .reddit_client import RedditSearchClient, is_reddit_available

        if not is_reddit_available():
            logger.warning("Reddit requested but httpx not installed, falling back to DuckDuckGo")
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None

        client_id = getattr(config, "REDDIT_CLIENT_ID", None)
        client_secret = getattr(config, "REDDIT_CLIENT_SECRET", None)
        user_agent = getattr(config, "REDDIT_USER_AGENT", "llm-bawt/0.1")
        if not client_id or not client_secret:
            logger.warning("Reddit requested but credentials are not configured, falling back to DuckDuckGo")
            from .ddgs_client import DuckDuckGoClient, is_ddgs_available
            if is_ddgs_available():
                timeout = getattr(config, "SEARCH_TIMEOUT", 10)
                return DuckDuckGoClient(max_results=max_results, timeout=timeout)
            logger.error("DuckDuckGo fallback also unavailable")
            return None

        timeout = getattr(config, "SEARCH_TIMEOUT", 10)
        return RedditSearchClient(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            max_results=max_results,
            timeout=timeout,
        )
    
    elif provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import DuckDuckGoClient, is_ddgs_available
        
        if not is_ddgs_available():
            logger.error("DuckDuckGo requested but ddgs not installed")
            return None
        
        # Get timeout from config (default 10 seconds)
        timeout = getattr(config, "SEARCH_TIMEOUT", 10)
        
        return DuckDuckGoClient(max_results=max_results, timeout=timeout)
    
    logger.error(f"Unknown search provider: {provider}")
    return None


def get_search_unavailable_reason(
    config: "Config",
    provider: SearchProvider | str | None = None,
) -> str:
    """Return a human-readable reason why search isn't available.

    Args:
        config: Application config
        provider: Optional provider override

    Returns:
        A message explaining how to enable search
    """
    # Normalize provider selection logic to match get_search_client
    resolved_provider: SearchProvider | None = None

    if provider is not None:
        if isinstance(provider, str):
            try:
                resolved_provider = SearchProvider(provider.lower())
            except ValueError:
                resolved_provider = None
        else:
            resolved_provider = provider
    else:
        config_provider = getattr(config, "SEARCH_PROVIDER", None)
        if config_provider:
            try:
                resolved_provider = SearchProvider(config_provider.lower())
            except ValueError:
                resolved_provider = None

    if resolved_provider is None:
        tavily_key = search_key(config, SearchProvider.TAVILY)
        if tavily_key:
            from .tavily_client import is_tavily_available
            if not is_tavily_available():
                return (
                    "Tavily API key is set but tavily-python isn't installed. "
                    "Install with: pipx runpip llm-bawt install tavily-python"
                )
            return "Tavily is configured but unavailable. Check your API key and network."

        brave_key = search_key(config, SearchProvider.BRAVE)
        if brave_key:
            from .brave_client import is_brave_available
            if not is_brave_available():
                return (
                    "Brave Search requires httpx. This should be installed by default. "
                    "Try: pip install httpx"
                )
            return "Brave Search is configured but unavailable. Check your API key and network."

        reddit_client_id = getattr(config, "REDDIT_CLIENT_ID", None)
        reddit_client_secret = getattr(config, "REDDIT_CLIENT_SECRET", None)
        if reddit_client_id or reddit_client_secret:
            from .reddit_client import is_reddit_available
            if not is_reddit_available():
                return (
                    "Reddit search requires httpx. This should be installed by default. "
                    "Try: pip install httpx"
                )
            if not reddit_client_id or not reddit_client_secret:
                return (
                    "Reddit search requires both REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
                )
            return "Reddit search is configured but unavailable. Check credentials and network."

        from .ddgs_client import is_ddgs_available
        if not is_ddgs_available():
            return (
                "DuckDuckGo search requires the ddgs package. "
                "Install with: ./install.sh --with-search "
                "or pipx runpip llm-bawt install ddgs"
            )
        return "No search provider available. Configure Tavily, Brave, or Reddit credentials, or install ddgs."

    if resolved_provider == SearchProvider.TAVILY:
        from .tavily_client import is_tavily_available
        api_key = search_key(config, SearchProvider.TAVILY)
        if not api_key:
            return unusable_reason("Tavily", api_key) or (
                "Tavily needs an API key. Connect it in Settings → Providers."
            )
        if not is_tavily_available():
            return (
                "Tavily search requires tavily-python. "
                "Install with: pipx runpip llm-bawt install tavily-python"
            )
        return "Tavily is configured but unavailable. Check your API key and network."

    if resolved_provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import is_ddgs_available
        if not is_ddgs_available():
            return (
                "DuckDuckGo search requires the ddgs package. "
                "Install with: ./install.sh --with-search "
                "or pipx runpip llm-bawt install ddgs"
            )
        return "DuckDuckGo search is configured but unavailable."

    if resolved_provider == SearchProvider.BRAVE:
        api_key = search_key(config, SearchProvider.BRAVE)
        if not api_key:
            return unusable_reason("Brave Search", api_key) or (
                "Brave Search needs an API key. Connect it in Settings → Providers "
                "(get one at https://api.search.brave.com/)."
            )
        from .brave_client import is_brave_available
        if not is_brave_available():
            return (
                "Brave Search requires httpx. This should be installed by default. "
                "Try: pip install httpx"
            )
        return "Brave Search is configured but unavailable. Check your API key and network."

    if resolved_provider == SearchProvider.REDDIT:
        client_id = getattr(config, "REDDIT_CLIENT_ID", None)
        client_secret = getattr(config, "REDDIT_CLIENT_SECRET", None)
        if not client_id or not client_secret:
            return (
                "Reddit search requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your config."
            )
        from .reddit_client import is_reddit_available
        if not is_reddit_available():
            return (
                "Reddit search requires httpx. This should be installed by default. "
                "Try: pip install httpx"
            )
        return "Reddit search is configured but unavailable. Check credentials and network."

    return "Web search not available."
