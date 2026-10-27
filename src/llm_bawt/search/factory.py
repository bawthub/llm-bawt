"""Factory for creating search clients.

Handles provider selection, configuration, and fallback logic.
"""

import logging
from typing import TYPE_CHECKING

from .base import SearchClient, SearchProvider
from .credentials import (
    INTEGRATIONS_SETTINGS,
    RedditCredentials,
    SearchKey,
    reddit_credentials,
    search_key,
    unusable_reason,
)

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


def _ddg_fallback(config: "Config", max_results: int, fallback: bool, why: str) -> SearchClient | None:
    """The chosen provider can't be used: DuckDuckGo stands in, or None.

    With ``fallback=False`` (a caller that asked for one provider by name) the
    answer is None, so the caller reports the provider's problem instead of
    serving another provider's results under its name (TASK-1562).
    """
    if not fallback:
        logger.warning("%s; not falling back", why)
        return None
    logger.warning("%s, falling back to DuckDuckGo", why)
    from .ddgs_client import DuckDuckGoClient, is_ddgs_available
    if is_ddgs_available():
        timeout = getattr(config, "SEARCH_TIMEOUT", 10)
        return DuckDuckGoClient(max_results=max_results, timeout=timeout)
    logger.error("DuckDuckGo fallback also unavailable")
    return None


#: Keyed providers in auto-select order: (label, library it needs, what it needs configured).
_KEYED = {
    SearchProvider.TAVILY: ("Tavily", "tavily-python", "Tavily key"),
    SearchProvider.BRAVE: ("Brave", "httpx", "Brave key"),
    SearchProvider.REDDIT: ("Reddit", "httpx", "Reddit credentials"),
}

Credentials = SearchKey | RedditCredentials


def _credentials(config: "Config", provider: SearchProvider) -> Credentials:
    if provider == SearchProvider.REDDIT:
        return reddit_credentials(config)
    return search_key(config, provider)


def _as_provider(value: SearchProvider | str | None, unknown: str) -> SearchProvider | None:
    if value is None or isinstance(value, SearchProvider):
        return value
    try:
        return SearchProvider(value.lower())
    except ValueError:
        logger.warning(unknown, value)
        return None


def _select(config: "Config", provider: SearchProvider | str | None,
            ) -> tuple[SearchProvider, Credentials | None] | None:
    """The provider to serve and its credentials, before any fallback.

    The explicit provider, then ``SEARCH_PROVIDER``, then the first keyed
    provider with usable credentials, then DuckDuckGo; None when none is available.
    """
    chosen = _as_provider(provider, "Unknown search provider: %s")
    if chosen is None:
        configured = getattr(config, "SEARCH_PROVIDER", None)
        if configured:
            chosen = _as_provider(configured, "Unknown search provider in config: %s")
    if chosen is None:
        for keyed, (label, _library, _needs) in _KEYED.items():
            creds = _credentials(config, keyed)
            if creds and is_search_available(keyed):
                logger.debug("Auto-selected %s (credentials configured)", label)
                return keyed, creds
        if is_search_available(SearchProvider.DUCKDUCKGO):
            logger.debug("Auto-selected DuckDuckGo (free fallback)")
            return SearchProvider.DUCKDUCKGO, None
        return None
    return chosen, (_credentials(config, chosen) if chosen in _KEYED else None)


def _unusable(provider: SearchProvider, creds: Credentials | None) -> str | None:
    """Why a selected keyed provider can't serve (library missing, no usable credentials)."""
    if provider not in _KEYED:
        return None
    label, library, needs = _KEYED[provider]
    if not is_search_available(provider):
        return f"{label} requested but {library} not installed"
    if not creds:
        return f"{label} requested but no usable {needs}"
    return None


def _served(config: "Config") -> tuple[SearchProvider | None, Credentials | None]:
    """What ``get_search_client(config)`` would serve now, without building it."""
    selected = _select(config, None)
    if selected is None:
        return None, None
    provider, creds = selected
    if _unusable(provider, creds) is None and (
            provider in _KEYED or is_search_available(SearchProvider.DUCKDUCKGO)):
        return provider, creds
    if is_search_available(SearchProvider.DUCKDUCKGO):
        return SearchProvider.DUCKDUCKGO, None
    return None, None


def client_is_current(config: "Config", client: object) -> bool:
    """Whether a cached default client is what ``get_search_client(config)`` would serve now.

    False when the provider that would serve changed (``SEARCH_PROVIDER``, a
    first or restored connection, a disconnect) or its credentials rotated, so
    a long-lived holder rebuilds it. That includes a DuckDuckGo stand-in once
    the provider it replaced is usable again.
    """
    provider, creds = _served(config)
    if getattr(client, "PROVIDER", None) != provider:
        return False
    if isinstance(creds, RedditCredentials):
        return (creds.client_id, creds.client_secret) == (
            getattr(client, "_client_id", None), getattr(client, "_client_secret", None))
    if isinstance(creds, SearchKey):
        return creds.value == getattr(client, "_api_key", None)
    return True


def get_search_client(
    config: "Config",
    provider: SearchProvider | str | None = None,
    max_results: int | None = None,
    *,
    fallback: bool = True,
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
        fallback: When the selected provider can't be used (no usable key,
            library missing), return DuckDuckGo instead. False returns None,
            so the caller can name the problem (``get_search_unavailable_reason``)
            rather than label DuckDuckGo results as that provider's.

    Returns:
        Configured SearchClient or None if search unavailable
    """
    selected = _select(config, provider)
    if selected is None:
        logger.warning("No search provider available")
        return None
    provider, creds = selected

    if max_results is None:
        max_results = getattr(config, "SEARCH_MAX_RESULTS", 5)
    max_results = int(max_results)

    why = _unusable(provider, creds)
    if why:
        return _ddg_fallback(config, max_results, fallback, why)
    timeout = getattr(config, "SEARCH_TIMEOUT", 10)

    if provider == SearchProvider.TAVILY:
        from .tavily_client import TavilyClient

        return TavilyClient(
            api_key=creds.value,
            max_results=max_results,
            include_answer=getattr(config, "SEARCH_INCLUDE_ANSWER", False),
            search_depth=getattr(config, "SEARCH_DEPTH", "basic"),
        )

    if provider == SearchProvider.BRAVE:
        from .brave_client import BraveSearchClient

        return BraveSearchClient(
            api_key=creds.value,
            max_results=max_results,
            timeout=timeout,
            include_summary=getattr(config, "SEARCH_INCLUDE_ANSWER", False),
            safesearch=getattr(config, "BRAVE_SAFESEARCH", "moderate"),
        )

    if provider == SearchProvider.REDDIT:
        from .reddit_client import RedditSearchClient

        return RedditSearchClient(
            client_id=creds.client_id,
            client_secret=creds.client_secret,
            user_agent=getattr(config, "REDDIT_USER_AGENT", "llm-bawt/0.1"),
            max_results=max_results,
            timeout=timeout,
        )

    if provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import DuckDuckGoClient, is_ddgs_available

        if not is_ddgs_available():
            logger.error("DuckDuckGo requested but ddgs not installed")
            return None
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

        if reddit_credentials(config):
            from .reddit_client import is_reddit_available
            if not is_reddit_available():
                return (
                    "Reddit search requires httpx. This should be installed by default. "
                    "Try: pip install httpx"
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
        creds = reddit_credentials(config)
        if not creds:
            return unusable_reason("Reddit", creds, INTEGRATIONS_SETTINGS) or (
                f"Reddit search needs a client ID and secret. Connect Reddit in {INTEGRATIONS_SETTINGS}."
            )
        from .reddit_client import is_reddit_available
        if not is_reddit_available():
            return (
                "Reddit search requires httpx. This should be installed by default. "
                "Try: pip install httpx"
            )
        return "Reddit search is configured but unavailable. Check credentials and network."

    return "Web search not available."
