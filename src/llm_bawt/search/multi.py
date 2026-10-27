"""Parallel multi-provider web search fan-out.

The single-provider ``factory.get_search_client`` picks exactly one backend by
priority. This module fans a query out to *every configured* provider at once
(Brave + Reddit + Tavily as available), runs their blocking ``search()`` calls
in parallel threads, and merges the results — tagged by source and deduped by
URL — so a caller with no provider preference gets coverage from all of them in
one shot.

DuckDuckGo is treated as a keyless *fallback*: it only joins the fan-out when no
keyed web provider (Brave/Tavily) is available, so we don't return redundant
web results alongside Brave.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import SearchError, SearchProvider, SearchResult
from .credentials import reddit_credentials, search_key
from .factory import get_search_client, get_search_unavailable_reason, is_search_available

if TYPE_CHECKING:
    from ..utils.config import Config

logger = logging.getLogger(__name__)

# Keyed web providers — if any of these is configured, DuckDuckGo stays on the
# bench (it'd just duplicate general web coverage).
_KEYED_WEB = (SearchProvider.BRAVE, SearchProvider.TAVILY)


def available_providers(config: "Config") -> list[SearchProvider]:
    """Resolve the set of providers to fan out to, in a stable order.

    Only genuinely-available providers are included (creds + library present).
    DuckDuckGo is appended only when no keyed web provider is available, so it
    acts as a fallback rather than a redundant second web source.
    """
    provs: list[SearchProvider] = []

    if search_key(config, SearchProvider.BRAVE) and is_search_available(SearchProvider.BRAVE):
        provs.append(SearchProvider.BRAVE)
    if search_key(config, SearchProvider.TAVILY) and is_search_available(SearchProvider.TAVILY):
        provs.append(SearchProvider.TAVILY)

    if reddit_credentials(config) and is_search_available(SearchProvider.REDDIT):
        provs.append(SearchProvider.REDDIT)

    has_keyed_web = any(p in _KEYED_WEB for p in provs)
    if not has_keyed_web and is_search_available(SearchProvider.DUCKDUCKGO):
        provs.append(SearchProvider.DUCKDUCKGO)

    return provs


def _dedupe_key(url: str) -> str:
    """Normalize a URL for cross-provider dedup (drop trailing slash / fragment)."""
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    return u.rstrip("/").lower()


@dataclass
class SearchOutcome:
    """What a fan-out returned, and what each provider did (TASK-1562).

    ``providers`` lists only the providers that answered; a provider that
    couldn't be used or failed upstream is in ``errors`` with a user-facing
    reason instead, so a broken provider never looks like "no results".
    """

    results: list[SearchResult] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


async def search_all(
    config: "Config",
    query: str,
    *,
    max_results: int = 5,
    providers: list[SearchProvider | str] | None = None,
) -> tuple[list[SearchResult], list[str]]:
    """``(results, providers_answered)`` from :func:`search_all_detailed`."""
    outcome = await search_all_detailed(config, query, max_results=max_results, providers=providers)
    return outcome.results, outcome.providers


async def search_all_detailed(
    config: "Config",
    query: str,
    *,
    max_results: int = 5,
    providers: list[SearchProvider | str] | None = None,
) -> SearchOutcome:
    """Fan ``query`` out to every configured provider in parallel and merge.

    Args:
        config: Application config (source of API keys / provider selection).
        query: The search query.
        max_results: Max results requested **per provider**.
        providers: Optional explicit provider list; defaults to
            ``available_providers(config)``. A named provider is never
            replaced by another: one that can't be used is reported in
            ``errors`` rather than answered by DuckDuckGo under its name.

    Returns:
        A :class:`SearchOutcome`. Results are interleaved round-robin across
        providers (so every provider is represented), deduped by URL, each
        tagged with its ``source``.
    """
    outcome = SearchOutcome()
    if providers is None:
        provider_list = available_providers(config)
    else:
        provider_list = []
        for p in providers:
            try:
                provider_list.append(p if isinstance(p, SearchProvider) else SearchProvider(str(p).lower()))
            except ValueError:
                logger.warning("Ignoring unknown search provider: %r", p)
                outcome.errors[str(p)] = (
                    f"Unknown search provider {str(p)[:32]!r}. "
                    f"Use one of: {', '.join(s.value for s in SearchProvider)}."
                )

    clients = []
    for prov in provider_list:
        client = get_search_client(config, provider=prov, max_results=max_results, fallback=False)
        if client is None:
            outcome.errors[prov.value] = get_search_unavailable_reason(config, prov)
        else:
            clients.append((prov, client))

    if not clients:
        return outcome

    async def _run(prov: SearchProvider, client) -> list[SearchResult] | None:
        try:
            # search() is blocking (httpx sync / ddgs) — offload to a thread so
            # all providers run concurrently.
            results = await asyncio.to_thread(client.search, query, max_results)
        except SearchError as e:
            outcome.errors[prov.value] = str(e)
            return None
        except Exception as e:  # noqa: BLE001 — one provider failing must not sink the rest
            logger.warning("Search provider %s failed: %s", prov.value, e)
            outcome.errors[prov.value] = f"{prov.value} search failed ({type(e).__name__})."
            return None
        # Backfill source in case a client left it unset.
        for r in results:
            if r.source is None:
                r.source = prov
        return results

    answers = await asyncio.gather(*[_run(prov, client) for prov, client in clients])
    outcome.providers = [prov.value for (prov, _), got in zip(clients, answers) if got is not None]
    grouped = [got for got in answers if got]

    # Round-robin interleave (rank 0 of every provider, then rank 1, …) so every
    # provider is represented even after dedup; drop cross-provider URL
    # duplicates, first occurrence wins.
    merged: list[SearchResult] = []
    seen: set[str] = set()
    max_len = max((len(g) for g in grouped), default=0)
    for i in range(max_len):
        for g in grouped:
            if i >= len(g):
                continue
            r = g[i]
            key = _dedupe_key(r.url)
            if key and key not in seen:
                seen.add(key)
                merged.append(r)

    outcome.results = merged
    return outcome
