"""Search-tool API keys: Brave, Tavily and NewsAPI (TASK-1476, TASK-826).

Each key is stored encrypted as ``provider_connection:<id>`` and read by
:mod:`llm_bawt.search.credentials`. Their legacy env vars and ``Config``
fields apply only while no record exists, and a disconnect leaves a
``disconnected`` record so the env key can't come back. These are tool
credentials, not model routes: they own no access path.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from .api_key import ApiKeyAdapter

_TIMEOUT_S = 15.0


class SearchKeyAdapter(ApiKeyAdapter):
    """A search tool's API key, shown under the search category."""

    keeps_disconnected_record = True
    #: Legacy env names, read only while no record exists.
    env_vars: ClassVar[tuple[str, ...]] = ()
    #: Legacy ``Config`` field (``LLM_BAWT_<name>`` or the config-dir .env).
    config_attr: ClassVar[str] = ""
    setup_url: ClassVar[str] = ""
    description: ClassVar[str] = ""
    #: Statuses that mean the upstream rejected the key.
    rejected_statuses: ClassVar[frozenset[int]] = frozenset({400, 401, 403})

    def descriptor(self) -> dict:
        return {
            **super().descriptor(),
            "category": "search",
            "setup_url": self.setup_url,
            "description": self.description,
        }

    @staticmethod
    def _probe_request(key: str) -> tuple[str, dict[str, str]]:
        """The key check's GET: also run by the cutover probe (TASK-1476)."""
        raise NotImplementedError

    def _probe(self, key: str) -> tuple[str | None, str | None]:
        """Validate a key with one authenticated GET; never echoes the key."""
        url, headers = self._probe_request(key)
        try:
            resp = httpx.get(url, headers=headers, timeout=_TIMEOUT_S)
        except httpx.HTTPError as exc:
            return None, f"{self.label} unreachable: {type(exc).__name__}"
        if resp.status_code in self.rejected_statuses:
            return None, f"{self.label} rejected the API key"
        if resp.is_error:
            return None, f"{self.label} key check returned HTTP {resp.status_code}"
        return self.label, None


class BraveSearchKeyAdapter(SearchKeyAdapter):
    id = "brave"
    label = "Brave Search"
    env_vars = ("LLM_BAWT_BRAVE_API_KEY", "BRAVE_API_KEY")
    config_attr = "BRAVE_API_KEY"
    setup_url = "https://api.search.brave.com/app/keys"
    description = "Optional web search. Without a key, web search uses DuckDuckGo."
    # Brave answers an invalid subscription token with 422.
    rejected_statuses = frozenset({400, 401, 403, 422})

    @staticmethod
    def _probe_request(key: str) -> tuple[str, dict[str, str]]:
        # One single-result query: Brave has no free key-check endpoint.
        return ("https://api.search.brave.com/res/v1/web/search?q=brave&count=1",
                {"X-Subscription-Token": key, "Accept": "application/json"})


class TavilySearchKeyAdapter(SearchKeyAdapter):
    id = "tavily"
    label = "Tavily"
    env_vars = ("LLM_BAWT_TAVILY_API_KEY", "TAVILY_API_KEY")
    config_attr = "TAVILY_API_KEY"
    setup_url = "https://app.tavily.com/"
    description = "Optional web search. Without a key, web search uses DuckDuckGo."

    @staticmethod
    def _probe_request(key: str) -> tuple[str, dict[str, str]]:
        # The usage endpoint checks the key without spending search credits.
        return "https://api.tavily.com/usage", {"Authorization": f"Bearer {key}"}


class NewsApiKeyAdapter(SearchKeyAdapter):
    id = "newsapi"
    label = "NewsAPI"
    env_vars = ("LLM_BAWT_NEWSAPI_API_KEY", "NEWSAPI_API_KEY")
    config_attr = "NEWSAPI_API_KEY"
    setup_url = "https://newsapi.org/register"
    description = "Optional news search and headlines tool (newsapi.org)."

    @staticmethod
    def _probe_request(key: str) -> tuple[str, dict[str, str]]:
        return "https://newsapi.org/v2/top-headlines/sources", {"X-Api-Key": key}


SEARCH_KEY_ADAPTERS: tuple[type[SearchKeyAdapter], ...] = (
    BraveSearchKeyAdapter,
    TavilySearchKeyAdapter,
    NewsApiKeyAdapter,
)

__all__ = [
    "SEARCH_KEY_ADAPTERS",
    "BraveSearchKeyAdapter",
    "NewsApiKeyAdapter",
    "SearchKeyAdapter",
    "TavilySearchKeyAdapter",
]
