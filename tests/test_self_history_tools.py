"""Raw self-history tools retain signed-user ownership without live history."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.mcp_server import self_tools, server
from llm_bawt.mcp_server.approval_interceptor import ApprovedCallerContext
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from tests.test_bot_access_mcp import Resolver, identity, signer, turn  # noqa: F401
from tests.test_bot_access_service_self import service_call

pytestmark = pytest.mark.usefixtures("signer")


@pytest.fixture
def history_catalog():
    catalog = CatalogFastMCP("self-history-owners", bot_policy_resolver=Resolver())
    catalog.add_tool(self_tools.self_tail, name="self_tail")
    catalog.add_tool(self_tools.self_fwd, name="self_fwd")
    catalog.mark_tools_registered()
    return catalog


async def _invoke(catalog, name, args, *, user="user_a", kind="turn"):
    if kind == "approved":
        blocks = await catalog.call_approved_tool(
            name, args,
            expected_invocation_hash=canonical_invocation_hash(name, args),
            caller_context=ApprovedCallerContext(bot_id="public_fixture", user_id=user),
        )
        return json.loads(blocks[0].text)
    context = (service_call(name, args, bot="public_fixture", user=user)
               if kind == "service" else identity(turn(bot="public_fixture", user=user)))
    with context:
        return await catalog._tool_manager.call_tool(name, args)


def test_self_tool_catalog_matches_supported_tools():
    tools = asyncio.run(server.mcp.list_tools())
    assert {tool.name for tool in tools if tool.name.startswith("self_")} == {
        "self_tail", "self_fwd", "self_system_prompt", "self_model", "self_house",
    }


def test_tail_reads_raw_real_bubbles_without_writes(monkeypatch, history_catalog):
    storage = SimpleNamespace(get_messages=AsyncMock(return_value=[
        {"role": "summary", "content": "not a bubble"},
        {"role": "user", "content": "first", "timestamp": 1.0},
        {"role": "tool", "content": "not a bubble"},
        {"role": "assistant", "content": "second", "timestamp": 2.0},
        {"role": "user", "content": "third", "timestamp": 3.0},
    ]))
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    result = asyncio.run(_invoke(history_catalog, "self_tail", {"bot_id": "public_fixture", "count": 2}))
    storage.get_messages.assert_awaited_once_with(bot_id="public_fixture", raw=True, user_id="user_a")
    assert result["total_available"] == 3
    assert result["count_returned"] == 2
    assert [m["content"] for m in result["messages"]] == ["second", "third"]
    assert "first" not in result["transcript"]
    assert "[ASSISTANT @ 1970-01-01 00:00 UTC]\nsecond" in result["transcript"]
    assert result["truncated"] is False


def test_tail_minimum_count_and_empty_history(monkeypatch, history_catalog):
    storage = SimpleNamespace(get_messages=AsyncMock(return_value=[]))
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    result = asyncio.run(_invoke(history_catalog, "self_tail", {"bot_id": "public_fixture", "count": 0}))
    assert result["messages"] == []
    assert result["transcript"] == ""
    assert result["count_returned"] == result["total_available"] == 0
    assert result["truncated"] is False


def test_tail_formatter_keeps_existing_bound_and_reports_oldest_drop(monkeypatch):
    monkeypatch.setattr(self_tools, "_MAX_TAIL_CHARS", 30)
    transcript, count, truncated = self_tools._format_bubbles([
        {"role": "user", "content": "old" * 20},
        {"role": "assistant", "content": "new"},
    ])
    assert transcript == "[ASSISTANT]\nnew"
    assert count == 1
    assert truncated is True


@pytest.fixture
def interleaved_history(monkeypatch):
    rows = [
        {"user_id": "user_a", "role": "user", "content": "A first", "timestamp": 1},
        {"user_id": "user_b", "role": "user", "content": "B first", "timestamp": 2},
        {"user_id": "user_a", "role": "summary", "content": "A summary", "timestamp": 3},
        {"user_id": "user_a", "role": "assistant", "content": "A second", "timestamp": 4},
        {"user_id": "user_b", "role": "assistant", "content": "B second", "timestamp": 5},
        {"user_id": "user_a", "role": "user", "content": "A third", "timestamp": 6},
        {"user_id": "user_a", "role": "tool", "content": "A tool", "timestamp": 7},
        {"user_id": "user_b", "role": "user", "content": "B third", "timestamp": 8},
    ]

    async def get_messages(*, bot_id, raw=False, user_id=None):
        # Match storage's contract: an omitted owner reads bot-wide history.
        assert bot_id == "public_fixture"
        assert raw is True
        return [row for row in rows if user_id is None or row["user_id"] == user_id]

    read = AsyncMock(side_effect=get_messages)
    send = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(server, "_get_storage", lambda: SimpleNamespace(get_messages=read))
    monkeypatch.setattr(server, "send_message_to_bot", send)
    return read, send


def _args(name, count=2):
    if name == "self_tail":
        return {"bot_id": "public_fixture", "count": count}
    return {"sender_bot_id": "public_fixture", "target_bot_id": "receiver_fixture", "count": count}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["self_tail", "self_fwd"])
@pytest.mark.parametrize("kind", ["turn", "approved", "service"])
@pytest.mark.parametrize("user,prefix,other", [("user_a", "A", "B"), ("user_b", "B", "A")])
async def test_self_history_isolates_interleaved_users(history_catalog, interleaved_history, name, kind, user, prefix, other):
    read, send = interleaved_history
    result = await _invoke(history_catalog, name, _args(name), user=user, kind=kind)
    read.assert_awaited_once_with(bot_id="public_fixture", raw=True, user_id=user)
    assert result["total_available"] == 3
    assert result["truncated"] is False
    if name == "self_tail":
        assert [m["content"] for m in result["messages"]] == [f"{prefix} second", f"{prefix} third"]
        assert result["count_returned"] == 2
        transcript = result["transcript"]
        send.assert_not_awaited()
    else:
        assert result["success"] is True
        assert result["forwarded"] == 2
        send.assert_awaited_once()
        assert send.call_args.kwargs["sender_bot_id"] == "public_fixture"
        assert send.call_args.kwargs["target_bot_id"] == "receiver_fixture"
        transcript = send.call_args.kwargs["message"]
    assert f"{prefix} second" in transcript
    assert f"{prefix} third" in transcript
    assert f"{prefix} first" not in transcript
    assert f"{other} " not in transcript
    assert "summary" not in transcript
    assert "tool" not in transcript


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["self_tail", "self_fwd"])
@pytest.mark.parametrize("kind", ["unsigned", "direct", "userless_service"])
async def test_self_history_requires_user_before_read_or_forward(history_catalog, interleaved_history, name, kind):
    read, send = interleaved_history
    args = _args(name)
    with pytest.raises(Exception, match="caller_unavailable"):
        if kind == "direct":
            await getattr(self_tools, name)(**args)
        elif kind == "userless_service":
            await _invoke(history_catalog, name, args, user=None, kind="service")
        else:
            await history_catalog._tool_manager.call_tool(name, args)
    read.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_forward_does_not_fall_back_to_other_users_history(history_catalog, interleaved_history):
    read, send = interleaved_history
    result = await _invoke(history_catalog, "self_fwd", _args("self_fwd"), user="empty_user")
    read.assert_awaited_once_with(bot_id="public_fixture", raw=True, user_id="empty_user")
    assert result["success"] is False
    assert result["forwarded"] == result["total_available"] == 0
    send.assert_not_awaited()
