"""TASK-1320: agent-facing message tools drop ``avatar_render``; messages_find takes Nh/Nd.

The MCP message tools return storage rows with an ``author`` display envelope.
Agents get it without the base64 ``avatar_render``; the app's verified service
client (REST ``/v1/history``, ``/v1/memory/message`` → UI) keeps it.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access.context import bind_bot_access
from llm_bawt.mcp_server import agent_view, find_tools
from llm_bawt.mcp_server import server as tools

RENDER = "data:image/svg+xml;base64,AAAA"


def _author() -> dict:
    return {"entity_type": "bot", "entity_id": "fixture_bot", "display_name": "Fixture",
            "color": "#123456", "avatar": "https://example.test/a.png", "avatar_render": RENDER}


def _row(message_id: str = "m1") -> dict:
    return {"id": message_id, "role": "assistant", "content": "hello", "author": _author()}


def _access(kind: str):
    return SimpleNamespace(caller=SimpleNamespace(kind=kind))


@pytest.fixture
def storage(monkeypatch):
    async def search_all_messages(**kwargs):
        return [_row()]

    async def get_messages(**kwargs):
        return [_row("m1"), _row("m2")]

    async def get_message_by_id(**kwargs):
        return {"message": _row("m2"), "before": [_row("m1")], "after": [_row("m3")]}

    fake = SimpleNamespace(search_all_messages=search_all_messages, get_messages=get_messages,
                           get_message_by_id=get_message_by_id)
    monkeypatch.setattr(tools, "_get_storage", lambda: fake)
    return fake


def _call_all(kind: str | None) -> list:
    async def run():
        return [
            await tools.search_all_messages(query="hello"),
            await tools.get_messages(bot_id="fixture_bot"),
            await tools.get_message_by_id(bot_id="fixture_bot", message_id="m2", before=1, after=1),
        ]
    if kind is None:
        return asyncio.run(run())
    with bind_bot_access(_access(kind)):
        return asyncio.run(run())


def _authors(value) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, list):
        for item in value:
            found += _authors(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "author" and isinstance(item, dict):
                found.append(item)
            else:
                found += _authors(item)
    return found


@pytest.mark.parametrize("kind", ["turn", "approved", None])
def test_agent_callers_get_no_avatar_render_but_keep_display_fields(storage, kind):
    results = _call_all(kind)
    authors = [a for result in results for a in _authors(result)]
    assert len(authors) == 1 + 2 + 3
    for author in authors:
        assert "avatar_render" not in author
        assert (author["display_name"], author["color"], author["avatar"]) == (
            "Fixture", "#123456", "https://example.test/a.png")
    assert RENDER not in json.dumps(results)


def test_service_caller_keeps_avatar_render_for_rest_and_ui(storage):
    results = _call_all("service")
    authors = [a for result in results for a in _authors(result)]
    assert len(authors) == 6
    assert all(author["avatar_render"] == RENDER for author in authors)


def test_for_agent_never_mutates_storage_rows():
    row = _row()
    stripped = agent_view.for_agent([row])
    assert row["author"]["avatar_render"] == RENDER
    assert "avatar_render" not in stripped[0]["author"]
    assert agent_view.for_agent(None) is None


NOW = 1_800_000_000.0


@pytest.mark.parametrize(("value", "expected"), [
    ("24h", NOW - 24 * 3600),
    ("7d", NOW - 7 * 86400),
    (" 1H ", NOW - 3600),
    ("365d", NOW - 365 * 86400),
    ("8760h", NOW - 8760 * 3600),
    ("2026-06-01", 1_780_272_000.0),
    (1782857172, 1782857172.0),
    ("1782857172", 1782857172.0),
    (None, None),
])
def test_find_time_accepts_relative_iso_and_unix(value, expected):
    assert find_tools.parse_find_time("since", value, now=NOW) == expected


@pytest.mark.parametrize("value", ["0h", "0d", "366d", "8761h", "3w", "-2d", "1.5d", "yesterday",
                                   "2026-13-01", "10000d"])
def test_find_time_rejects_out_of_range_and_unknown(value):
    with pytest.raises(find_tools.InvalidTime) as caught:
        find_tools.parse_find_time("until", value, now=NOW)
    assert caught.value.field == "until"


def test_bad_time_answer_names_the_field_and_accepted_formats(monkeypatch):
    monkeypatch.setattr(find_tools, "trusted_caller",
                        lambda name, arguments: SimpleNamespace(kind="service", bot_id="b", user_id=None))
    answer = json.loads(asyncio.run(find_tools.messages_find(query="x", since="yesterday")))
    assert answer["status"] == "unavailable"
    assert answer["reason"] == "invalid_request"
    assert answer["field"] == "since"
    assert answer["accepted"] == list(find_tools.ACCEPTED_TIME_FORMATS)
    assert any("24h" in fmt for fmt in answer["accepted"])
