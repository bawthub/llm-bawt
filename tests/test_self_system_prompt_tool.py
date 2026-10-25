"""self_system_prompt edit note matches the session pin (TASK-1318).

Backends that pin the prompt per session (``pins_system_prompt``) keep it
until the next session boundary, so the edit result must not promise "next
turn" for them, and must still say "next turn" for unpinned bots.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import httpx
import pytest

from llm_bawt.agent_backends import list_backends
from llm_bawt.mcp_server import catalog_contracts
from llm_bawt.mcp_server import self_tools as st

_BOUNDARIES = ("session boundary", "/new", "reset", "model switch", "/compact")


def _edit(monkeypatch, profile: dict) -> dict:
    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def patch(self, url: str, json: dict, timeout: float) -> httpx.Response:
            body = {**profile, "system_prompt": json["system_prompt"]}
            return httpx.Response(200, json=body, request=httpx.Request("PATCH", url))

    async def fetch(bot_id: str) -> dict:
        return {**profile, "system_prompt": "old"}

    monkeypatch.setattr(st, "_fetch_bot_profile", fetch)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    res = asyncio.run(st.self_system_prompt(bot_id="probe", action="edit", new_prompt="new"))
    assert res["updated"] is True
    return res


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_pinned_backend_note_names_session_boundary(monkeypatch, backend: str) -> None:
    assert list_backends()[backend].pins_system_prompt is True
    note = _edit(monkeypatch, {"bot_type": "agent", "agent_backend": backend})["note"]
    for boundary in _BOUNDARIES:
        assert boundary in note
    assert "next turn" not in note


@pytest.mark.parametrize(
    "profile",
    [
        {"bot_type": "agent", "agent_backend": "openclaw"},
        {"bot_type": "chat"},
        {"bot_type": "agent", "agent_backend": "no-such-backend"},
    ],
)
def test_unpinned_bot_note_says_next_turn(monkeypatch, profile: dict) -> None:
    note = _edit(monkeypatch, profile)["note"]
    assert "next turn" in note
    assert "session boundary" not in note


def test_catalog_summary_distinguishes_pinned_sessions() -> None:
    summary = catalog_contracts.TOOL_SUMMARIES["self_system_prompt"]
    assert "next turn" in summary
    assert "pinned sessions at next boundary" in summary
