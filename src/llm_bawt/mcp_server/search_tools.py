"""Web search MCP tool (registered via import side-effect from server.py).

Exposes ``web_search`` to every agent bot on the bawthub MCP server. This is the
local replacement for the Claude CLI's Anthropic *server-side* ``WebSearch``
tool, which only works against api.anthropic.com and hangs on the proxy path
(grok / openai bots). With no provider specified it fans the query out to every
configured provider in parallel (Brave + Reddit + Tavily as available) and
returns the merged, source-tagged results.

Page retrieval is already covered for agent bots by the crawl4ai MCP server, so
there is no separate ``web_fetch`` tool here.
"""

from __future__ import annotations

import logging

from .server import mcp

logger = logging.getLogger(__name__)


def _x_error(query: str, exc: Exception) -> dict:
    from llm_bawt.integrations.x_api import XApiError

    if isinstance(exc, XApiError):
        return {"provider": "x", "query": query, "count": 0, "results": [],
                "error_code": exc.code, "error": str(exc)}
    logger.warning("X request failed unexpectedly")
    return {"provider": "x", "query": query, "count": 0, "results": [],
            "error_code": "internal_error", "error": "X request failed unexpectedly; check service health."}


@mcp.tool(name="x_search")
async def x_search(
    query: str,
    max_results: int = 10,
    start_time: str | None = None,
    end_time: str | None = None,
    next_token: str | None = None,
    sort_order: str = "recency",
    include_authors: bool = False,
) -> dict:
    """Search public X posts from the last seven days (paid, one page per call).

    sort_order: recency (newest first) or relevancy (X-ranked; best for summaries).
    Operators: from:, -is:retweet, -is:reply, lang:en, has:links, is:verified,
    min_likes:N, min_reposts:N (API names; min_faves/min_retweets are rejected).
    Results carry like/repost/reply/quote counts. include_authors adds usernames
    and follower counts (extra billed user reads). Never falls back to web search.
    """
    import asyncio

    from llm_bawt.integrations import x_api
    from llm_bawt.utils.config import config

    try:
        return await asyncio.to_thread(
            x_api.recent_search, config, query, max_results=max_results,
            start_time=start_time, end_time=end_time, next_token=next_token,
            sort_order=sort_order, include_authors=include_authors,
        )
    except Exception as exc:  # noqa: BLE001 - mapped to a structured, token-free error
        return _x_error(query, exc)


@mcp.tool(name="x_counts")
async def x_counts(
    query: str,
    granularity: str = "hour",
    start_time: str | None = None,
    end_time: str | None = None,
    next_token: str | None = None,
) -> dict:
    """Count X posts matching a query per minute|hour|day over the last seven days.

    Returns no posts: use it to size a topic and find peaks, then x_search those
    windows with sort_order=relevancy. Same operators and time rules as x_search.
    """
    import asyncio

    from llm_bawt.integrations import x_api
    from llm_bawt.utils.config import config

    try:
        return await asyncio.to_thread(
            x_api.recent_counts, config, query, granularity=granularity,
            start_time=start_time, end_time=end_time, next_token=next_token,
        )
    except Exception as exc:  # noqa: BLE001
        return _x_error(query, exc)


@mcp.tool(name="web_search")
async def web_search(
    query: str,
    max_results: int = 5,
    provider: str | None = None,
) -> dict:
    """Search the live web via local providers (Brave / Reddit / Tavily).

    With no ``provider``, fans the query out to every configured provider in
    parallel and merges the results, each tagged with its source. Use this for
    current events, docs, or anything outside the model's training data. For
    fetching a specific page's contents, use the crawl4ai tools instead.

    Args:
        query: The search query.
        max_results: Max results to request per provider (default 5).
        provider: Optional single provider to restrict to — one of
            ``brave``, ``reddit``, ``tavily``, ``duckduckgo``. Omit to fan out
            to all configured providers.

    Returns:
        Dict with ``query``, ``providers`` (the providers that answered),
        ``count``, and ``results`` (list of ``{title, url, snippet, score,
        source}``). A named provider is never stood in for by another.
        ``errors`` maps each provider that couldn't be used or failed
        upstream (rejected key, rate limit, outage) to the reason; when
        nothing came back, ``error`` says why.
    """
    from llm_bawt.search.multi import search_all_detailed
    from llm_bawt.utils.config import config

    q = (query or "").strip()
    if not q:
        return {"query": query, "providers": [], "count": 0, "results": [],
                "error": "Missing required parameter: query"}

    providers = [provider] if provider else None
    try:
        outcome = await search_all_detailed(
            config, q, max_results=max_results, providers=providers
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search failed for %r: %s", q, e)
        return {"query": q, "providers": [], "count": 0, "results": [],
                "error": f"Search failed: {e}"}

    response: dict = {
        "query": q,
        "providers": outcome.providers,
        "count": len(outcome.results),
        "results": [r.to_dict() for r in outcome.results],
    }
    if outcome.errors:
        response["errors"] = outcome.errors
        if not outcome.results:
            response["error"] = " ".join(outcome.errors.values())
    elif not outcome.providers:
        from llm_bawt.search import get_search_unavailable_reason
        response["error"] = get_search_unavailable_reason(config)

    logger.info("web_search %r → %d results from %s; failed: %s",
                q, len(outcome.results), outcome.providers, sorted(outcome.errors))
    return response
