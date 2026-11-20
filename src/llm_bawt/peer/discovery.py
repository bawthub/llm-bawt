"""Bots that paired installs share with us (TASK-2517).

Read live from each active peer's ``peer_bots_list``; nothing is cached, so a
peer that unshares or revokes disappears at the next read. Each bot is
addressed ``slug@<install_id>`` and carries ``location.runtime == "peer"``.
"""
from __future__ import annotations

import asyncio
import re

from . import client, identity, store

LIST_TIMEOUT = 4.0
_SLUG = re.compile(r"^[a-z0-9_][a-z0-9_-]{0,63}$")


def _rows(link: store.PeerLink, got: dict) -> list[dict]:
    out = []
    for b in got.get("bots", []):
        slug = str(b.get("slug") or "").lower() if isinstance(b, dict) else ""
        if not _SLUG.match(slug):  # peer data: keep only well-formed addresses
            continue
        out.append({"id": f"{slug}@{link.install_id}", "slug": slug,
                    "name": str(b.get("name") or slug)[:80],
                    "description": str(b.get("description") or "")[:200],
                    "location": {"runtime": "peer", "peer": link.label,
                                 "install_id": link.install_id}})
    return out


async def peer_bots(own: identity.Identity, link: store.PeerLink) -> dict:
    """``{available, reason?, bots}`` for one peer."""
    try:
        got = await asyncio.wait_for(client.call(own, link, "peer_bots_list"), LIST_TIMEOUT)
    except client.PeerUnavailable as exc:
        return {"available": False, "reason": exc.reason, "bots": []}
    except TimeoutError:
        return {"available": False, "reason": "peer did not answer in time", "bots": []}
    return {"available": True, "bots": _rows(link, got)}


async def all_peer_bots(engine) -> tuple[list[dict], list[dict]]:
    """Every active peer's shared bots, read in parallel.

    Returns ``(bots, unavailable)`` where ``unavailable`` names each peer that
    could not be read, so a caller never mistakes an offline peer for one that
    shares nothing.
    """
    with engine.begin() as conn:
        own = identity.load_or_create(conn)
        links = [lk for lk in store.list_links(conn) if lk.state == "active"]
    results = await asyncio.gather(*(peer_bots(own, lk) for lk in links))
    bots, unavailable = [], []
    for link, got in zip(links, results, strict=True):
        bots.extend(got["bots"])
        if not got["available"]:
            unavailable.append({"peer": link.label, "install_id": link.install_id,
                                "reason": got["reason"]})
    return bots, unavailable
