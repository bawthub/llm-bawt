"""self_system_prompt edit note matches the session pin (TASK-1318).

A pinned agent session keeps its prompt until the next session boundary, so
the edit result must not promise "next turn" for those sessions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import httpx

from llm_bawt.mcp_server import catalog_contracts
from llm_bawt.mcp_server import self_tools as st


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def patch(self, url: str, json: dict, timeout: float) -> httpx.Response:
        request = httpx.Request("PATCH", url)
        body = {"system_prompt": json["system_prompt"], "name": "Probe", "bot_type": "agent"}
        return httpx.Response(200, json=body, request=request)


def test_edit_note_names_session_boundary(monkeypatch) -> None:
    async def fetch(bot_id: str) -> dict:
        return {"system_prompt": "old"}

    monkeypatch.setattr(st, "_fetch_bot_profile", fetch)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    res = asyncio.run(st.self_system_prompt(bot_id="probe", action="edit", new_prompt="new"))

    assert res["updated"] is True
    note = res["note"]
    assert "session boundary" in note
    for boundary in ("/new", "reset", "model switch", "/compact"):
        assert boundary in note
    assert not note.startswith("Takes effect on your next turn")


def test_catalog_summary_does_not_promise_next_turn() -> None:
    summary = catalog_contracts.TOOL_SUMMARIES["self_system_prompt"]
    assert "apply next turn" not in summary
    assert "boundary" in summary
