"""Tests for Reddit integration in search factory."""

from dataclasses import dataclass

from llm_bawt.search.base import SearchProvider
from llm_bawt.search.factory import get_search_client, get_search_unavailable_reason


@dataclass
class _Config:
    SEARCH_PROVIDER: str | None = None
    SEARCH_MAX_RESULTS: int = 5
    SEARCH_TIMEOUT: int = 10
    SEARCH_INCLUDE_ANSWER: bool = False
    SEARCH_DEPTH: str = "basic"
    BRAVE_SAFESEARCH: str = "moderate"
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "llm-bawt/0.1"


def test_factory_returns_reddit_client_when_explicit_and_configured():
    config = _Config(
        REDDIT_CLIENT_ID="cid",
        REDDIT_CLIENT_SECRET="csecret",
        REDDIT_USER_AGENT="llm-bawt-test/1.0",
    )
    client = get_search_client(config, provider=SearchProvider.REDDIT)

    assert client is not None
    assert client.PROVIDER == SearchProvider.REDDIT


def test_factory_auto_selects_reddit_when_other_keys_missing():
    config = _Config(
        REDDIT_CLIENT_ID="cid",
        REDDIT_CLIENT_SECRET="csecret",
    )
    client = get_search_client(config)

    assert client is not None
    assert client.PROVIDER == SearchProvider.REDDIT


def test_unavailable_reason_says_where_to_connect_reddit():
    # TASK-1476: the credentials live in the Integrations record; the env
    # settings are a deprecated fallback, so the message points at the form.
    config = _Config()
    message = get_search_unavailable_reason(config, provider=SearchProvider.REDDIT)
    assert "Settings → Integrations" in message
