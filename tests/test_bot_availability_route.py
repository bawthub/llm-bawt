"""GET /v1/bots/available wraps the bots_list_available read (TASK-1712)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import llm_bawt.mcp_server.server as server  # noqa: F401  registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools
from llm_bawt.bot_access import BotAccessError
from llm_bawt.service.routes.bot_availability import bots_available


def _run(**kwargs):
    return asyncio.run(bots_available(**{"capability": None, "tag": None, **kwargs}))


def test_passes_filters_and_returns_full_rows(monkeypatch):
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return [{"slug": "fixture-coordinator", "capabilities": ["coordinate"], "state": {"turn": {"kind": "idle"}}}]

    monkeypatch.setattr(tools, "list_available_bots", fake)
    assert _run(capability="coordinate") == {"bots": [
        {"slug": "fixture-coordinator", "capabilities": ["coordinate"], "state": {"turn": {"kind": "idle"}}}]}
    assert seen == {"tag": None, "capability": "coordinate", "include_state": True, "detail": "full"}


@pytest.mark.parametrize(("error", "status"), [
    (ValueError("unknown capability 'x'"), 400),
    (BotAccessError("policy_unavailable"), 503),
])
def test_errors_never_return_a_partial_list(monkeypatch, error, status):
    async def fake(**_kwargs):
        raise error

    monkeypatch.setattr(tools, "list_available_bots", fake)
    with pytest.raises(HTTPException) as raised:
        _run(capability="x")
    assert raised.value.status_code == status
