"""Inter-bot sends are attributed to the signed caller, never "unknown" (TASK-1241)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import llm_bawt.mcp_server.server as server  # registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools  # noqa: E402
from llm_bawt.mcp_server import approval_interceptor, task_association
from llm_bawt.mcp_server.inter_bot_sender import (
    SenderIdentityError,
    resolve_message_sender,
    resolve_sender_bot_id,
)
from llm_bawt import inter_bot_send as send
from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag, TagSnapshot
from llm_bawt.service.routes import inter_bot_deliveries as route
from llm_bawt.task_turn_context import TaskTurnContextError


def _signed(monkeypatch: pytest.MonkeyPatch, bot_id: str | None) -> None:
    def context():
        if bot_id is None:
            raise TaskTurnContextError("no capability")
        return SimpleNamespace(bot_id=bot_id, turn_id="turn-" + "c" * 32,
                               trigger_message_id="m")

    monkeypatch.setattr(task_association, "current_task_turn_context", context)


def _rejected_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capability header arrived but does not open."""
    _signed(monkeypatch, None)
    monkeypatch.setattr(task_association, "current_task_turn_capability", lambda: "not-a-token")


_CATALOG = sorted(str(tag) for tag in BehaviorTag)


class _Policy:
    """Stands in for ``BotPolicyResolver``; counts reads."""

    def __init__(self, tags: dict[str, list[str]] | None = None, *, error: bool = False):
        self.tags, self.error, self.reads = tags or {}, error, 0

    def __call__(self):
        return self

    async def resolve(self):
        self.reads += 1
        if self.error:
            raise BotAccessError("policy_unavailable")
        return SimpleNamespace(tags=TagSnapshot.from_rows([
            {"slug": slug, "include_in_global_search": True, "tags": sorted(tags), "catalog": _CATALOG}
            for slug, tags in sorted(self.tags.items())
        ]))


_OUTSIDE = {"outside": ["external", "inactive"], "al": [], "snark": []}


def _message_sender(claimed, policy) -> str:
    return asyncio.run(resolve_message_sender(claimed, policy))


# ---- resolver --------------------------------------------------------------

@pytest.mark.parametrize("claimed", [None, "", "  ", "unknown", "AL", " al "])
def test_signed_caller_fills_or_confirms_sender(monkeypatch, claimed) -> None:
    _signed(monkeypatch, "al")
    assert resolve_sender_bot_id(claimed) == "al"


def test_signed_caller_rejects_spoofed_sender(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    with pytest.raises(SenderIdentityError, match="does not match the signed caller"):
        resolve_sender_bot_id("gavel")


@pytest.mark.parametrize("claimed", [None, "", "unknown", " Unknown "])
def test_unsigned_caller_must_name_itself(monkeypatch, claimed) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="sender_bot_id is required"):
        resolve_sender_bot_id(claimed)


def test_unsigned_caller_with_explicit_sender_is_accepted(monkeypatch) -> None:
    _signed(monkeypatch, None)
    assert resolve_sender_bot_id(" Codex ") == "codex"


def test_approved_replay_uses_original_caller(monkeypatch) -> None:
    _signed(monkeypatch, None)
    approved = approval_interceptor.ApprovedCallerContext(bot_id="byte")
    token = approval_interceptor._approved_caller_context.set(approved)
    try:
        assert resolve_sender_bot_id(None) == "byte"
        with pytest.raises(SenderIdentityError):
            resolve_sender_bot_id("snark")
    finally:
        approval_interceptor._approved_caller_context.reset(token)


# ---- outside-agent default (TASK-1362) --------------------------------------

@pytest.mark.parametrize("claimed", [None, "", "unknown"])
def test_no_header_and_no_sender_defaults_to_the_outside_agent_bot(monkeypatch, claimed) -> None:
    _signed(monkeypatch, None)
    assert _message_sender(claimed, _Policy(_OUTSIDE)) == "outside"


def test_no_outside_agent_bot_is_a_setup_error(monkeypatch) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="no outside-agent bot is set up"):
        _message_sender(None, _Policy({"al": [], "snark": ["inactive"]}))


def test_an_external_bot_that_is_still_active_is_not_a_default(monkeypatch) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="no outside-agent bot is set up"):
        _message_sender(None, _Policy({"outside": ["external"], "al": []}))


def test_several_outside_agent_bots_must_be_named(monkeypatch) -> None:
    _signed(monkeypatch, None)
    policy = _Policy({"outside-b": ["external", "inactive"], "outside-a": ["external", "inactive"]})
    with pytest.raises(SenderIdentityError, match=r"\(outside-a, outside-b\); pass sender_bot_id"):
        _message_sender(None, policy)
    assert _message_sender("outside-b", policy) == "outside-b"


def test_unreadable_policy_fails_closed_to_naming_yourself(monkeypatch) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="sender_bot_id is required"):
        _message_sender(None, _Policy(error=True))


def test_rejected_capability_never_falls_back_to_the_outside_agent(monkeypatch) -> None:
    _rejected_capability(monkeypatch)
    policy = _Policy(_OUTSIDE)
    with pytest.raises(SenderIdentityError, match="sender_bot_id is required"):
        _message_sender(None, policy)
    assert policy.reads == 0


def test_rejected_capability_still_accepts_an_explicit_sender(monkeypatch) -> None:
    _rejected_capability(monkeypatch)
    policy = _Policy(_OUTSIDE)
    assert _message_sender(" Codex ", policy) == "codex"
    assert resolve_sender_bot_id("codex") == "codex"
    assert policy.reads == 0


@pytest.mark.parametrize("claimed", [None, "al"])
def test_a_signed_or_named_caller_never_reads_the_policy(monkeypatch, claimed) -> None:
    _signed(monkeypatch, "al" if claimed is None else None)
    policy = _Policy(_OUTSIDE)
    assert _message_sender(claimed, policy) == "al"
    assert policy.reads == 0


def test_signed_caller_still_rejects_a_spoof_of_the_outside_agent(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    with pytest.raises(SenderIdentityError, match="does not match the signed caller"):
        _message_sender("outside", _Policy(_OUTSIDE))


def test_sync_resolver_has_no_outside_agent_default(monkeypatch) -> None:
    _signed(monkeypatch, None)
    with pytest.raises(SenderIdentityError, match="sender_bot_id is required"):
        resolve_sender_bot_id(None)


# ---- tools -----------------------------------------------------------------

async def _no_refusal(_target_bot_id: str) -> None:
    """Every fixture target is active; tag refusals are covered in test_bot_tag_discovery."""
    return None


def _capture_enqueue(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def enqueue(**kwargs):
        calls.append(kwargs)
        return {"delivery_id": "delivery-1", "status": "QUEUED"}

    monkeypatch.setattr(server, "_enqueue_durable", enqueue)
    monkeypatch.setattr(tools, "_target_refusal", _no_refusal)
    return calls


def test_send_without_sender_is_stamped_with_signed_bot(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    calls = _capture_enqueue(monkeypatch)
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is True
    assert calls[0]["sender_bot_id"] == "al"


def test_send_with_spoofed_sender_has_no_side_effect(monkeypatch) -> None:
    _signed(monkeypatch, "al")
    calls = _capture_enqueue(monkeypatch)
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="snark", message="hi", sender_bot_id="gavel",
    ))
    assert result["dispatched"] is False and "signed caller" in result["error"]
    assert calls == []


def test_unattributed_send_without_context_is_refused(monkeypatch) -> None:
    _signed(monkeypatch, None)
    calls = _capture_enqueue(monkeypatch)
    monkeypatch.setattr(tools, "_policy_resolver", _Policy({"al": [], "snark": []}))
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is False and "no outside-agent bot is set up" in result["error"]
    assert calls == []


def test_unattributed_send_is_stamped_with_the_outside_agent_bot(monkeypatch) -> None:
    _signed(monkeypatch, None)
    calls = _capture_enqueue(monkeypatch)
    monkeypatch.setattr(tools, "_policy_resolver", _Policy(_OUTSIDE))
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is True
    assert calls[0]["sender_bot_id"] == "outside"


def test_send_with_a_rejected_capability_has_no_side_effect(monkeypatch) -> None:
    _rejected_capability(monkeypatch)
    calls = _capture_enqueue(monkeypatch)
    policy = _Policy(_OUTSIDE)
    monkeypatch.setattr(tools, "_policy_resolver", policy)
    result = asyncio.run(server.send_message_to_bot(target_bot_id="snark", message="hi"))
    assert result["dispatched"] is False and "sender_bot_id is required" in result["error"]
    assert calls == [] and policy.reads == 0


def test_compact_without_sender_uses_signed_bot(monkeypatch) -> None:
    _signed(monkeypatch, "snark")
    monkeypatch.setattr(tools, "agent_context_health", AsyncMock(return_value={
        "capabilities": {"compact": True},
    }))
    enqueue = AsyncMock(return_value={"delivery_id": "delivery-1"})
    monkeypatch.setattr(tools, "_enqueue_durable", enqueue)
    asyncio.run(tools.agent_context_compact("snark", "compact-1"))
    assert enqueue.call_args.kwargs["sender_bot_id"] == "snark"


def test_compact_without_any_identity_is_refused(monkeypatch) -> None:
    """The outside-agent default is bots_send_message only (TASK-1362)."""
    _signed(monkeypatch, None)
    monkeypatch.setattr(tools, "_policy_resolver", _Policy(_OUTSIDE))
    monkeypatch.setattr(tools, "agent_context_health", AsyncMock(return_value={
        "capabilities": {"compact": True},
    }))
    enqueue = AsyncMock()
    monkeypatch.setattr(tools, "_enqueue_durable", enqueue)
    result = asyncio.run(tools.agent_context_compact("snark", "compact-1"))
    assert result["success"] is False
    enqueue.assert_not_called()


# ---- route -----------------------------------------------------------------

@pytest.fixture
def delivery_route(monkeypatch):
    record = SimpleNamespace(id="fixture-delivery")
    store = Mock(engine=True)
    store.enqueue.return_value = (record, False)
    store.get.return_value = record
    store.payload.return_value = {}
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock(), wake=Mock())
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="nick"),
                              _inter_bot_dispatcher=dispatcher)
    bot = SimpleNamespace(agent_backend="claude-code", harness="claude-code")
    known = {"al", "snark", "outside"}
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(send, "BotManager", lambda config: SimpleNamespace(
        get_bot=lambda slug: bot if slug.strip().lower() in known else None,
    ))
    monkeypatch.setattr(send, "submission_result", lambda *a, **kw: (202, {"success": True}))
    return store


def _create(**kwargs):
    body = route.DeliveryCreate(target_bot_id="snark", message="hello",
                                session_policy="continue", **kwargs)
    return asyncio.run(route.create_delivery(body))


@pytest.mark.parametrize("fields", [
    {},
    {"sender_bot_id": "unknown"},
    {"sender_bot_id": "ghost"},
    {"sender_bot_id": "al", "author_entity_type": "bot", "author_entity_id": "unknown"},
])
def test_route_rejects_bot_author_that_is_not_a_real_bot(delivery_route, fields) -> None:
    with pytest.raises(HTTPException, match="real sender bot") as error:
        _create(**fields)
    assert error.value.status_code == 422
    delivery_route.enqueue.assert_not_called()


def test_route_accepts_real_bot_sender(delivery_route) -> None:
    _create(sender_bot_id="al")
    assert delivery_route.enqueue.call_args.kwargs["sender_bot_id"] == "al"


def test_route_accepts_the_outside_agent_bot_as_sender(delivery_route) -> None:
    """The route needs a real bot; an inactive one still is (TASK-1362)."""
    _create(sender_bot_id="outside")
    assert delivery_route.enqueue.call_args.kwargs["sender_bot_id"] == "outside"


def test_route_keeps_user_authored_workspace_delivery(delivery_route) -> None:
    """BawtHub's repos workspace posts author=nick with sender unknown."""
    _create(sender_bot_id="unknown", author_entity_type="user", author_entity_id="nick")
    kwargs = delivery_route.enqueue.call_args.kwargs
    assert kwargs["sender_bot_id"] == "unknown"
    assert kwargs["author"].entity_type == "user"
