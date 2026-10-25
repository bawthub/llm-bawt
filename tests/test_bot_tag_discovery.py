"""Discovery and send refusals come from bot tags (TASK-1329, BOT_TAGS.md §7).

``bots_list_available`` reads in-process: the policy snapshot first, then the
profile rows. ``bots_send_message`` refuses an ``inactive`` target before any
side effect. Fixture bots only.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import llm_bawt.mcp_server.server as server  # registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools  # noqa: E402
from llm_bawt.bot_access import BotAccessError, BotPolicyResolver

_CATALOG = ["coder", "inactive", "no_global_search", "private", "reviewer"]
_TAGS = {
    "fixture-public": ["reviewer"],
    "fixture-quiet": ["no_global_search"],
    "fixture-private": ["private"],
    "fixture-inactive": ["inactive"],
    "fixture-column": [],
}


def _rows() -> list[dict]:
    return [{"slug": slug, "include_in_global_search": slug != "fixture-column",
             "tags": tags, "catalog": _CATALOG} for slug, tags in sorted(_TAGS.items())]


def _profile(slug: str, **overrides: Any) -> SimpleNamespace:
    values = {"slug": slug, "name": slug.title(), "bot_type": None, "description": None,
              "default_model": "fixture-model", "agent_backend": "claude-code"}
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def policy(monkeypatch):
    state = {"down": False, "order": [], "profiles": [_profile(s) for s in sorted(_TAGS)]}

    class _Resolver:
        async def resolve(self, caller=None):
            state["order"].append("policy")
            if state["down"]:
                raise BotAccessError("policy_unavailable")
            return BotPolicyResolver._compile(_rows(), caller)

    def profile_rows():
        state["order"].append("profiles")
        return state["profiles"]

    def live_states(profiles):
        state["order"].append("state")
        state["state_reads"].append([row.slug for row in profiles])
        return {row.slug: {"turn": {"kind": "idle"}, "fixture": row.slug} for row in profiles}

    def role_capabilities():
        state["order"].append("capabilities")
        return state["capabilities"]

    state["state_reads"] = []
    # Stored order is the operator's; discovery reports the sorted union.
    state["capabilities"] = {"coder": ("implement",), "reviewer": ("review", "implement")}
    monkeypatch.setattr(tools, "_policy_resolver", _Resolver)
    monkeypatch.setattr(tools, "_role_capabilities", role_capabilities)
    monkeypatch.setattr(tools, "_profile_rows", profile_rows)
    monkeypatch.setattr(tools, "_live_states", live_states)
    return state


def test_discovery_lists_only_public_and_quiet_bots_with_their_tags(policy):
    listed = asyncio.run(tools.list_available_bots(include_state=False))

    assert [bot["slug"] for bot in listed] == ["fixture-public", "fixture-quiet"]
    assert listed[0] == {"slug": "fixture-public", "name": "Fixture-Public", "bot_type": "agent",
                         "description": "", "default_model": "fixture-model",
                         "agent_backend": "claude-code", "tags": ["reviewer"],
                         "capabilities": ["implement", "review"]}
    assert listed[1]["tags"] == ["no_global_search"]
    assert listed[1]["capabilities"] == []
    assert policy["order"] == ["policy", "capabilities", "profiles"]
    assert policy["state_reads"] == []


def test_state_is_read_once_for_the_visible_bots_only(policy):
    listed = asyncio.run(tools.list_available_bots())

    assert policy["order"] == ["policy", "capabilities", "profiles", "state"]
    assert policy["state_reads"] == [["fixture-public", "fixture-quiet"]]
    assert [bot["state"]["fixture"] for bot in listed] == ["fixture-public", "fixture-quiet"]


@pytest.mark.parametrize("tag", ["reviewer", " Reviewer "])
def test_the_tag_filter_keeps_only_matching_visible_bots(policy, tag):
    listed = asyncio.run(tools.list_available_bots(tag=tag))

    assert [bot["slug"] for bot in listed] == ["fixture-public"]
    assert policy["state_reads"] == [["fixture-public"]]


@pytest.mark.parametrize("tag", ["private", "inactive"])
def test_filtering_on_a_hiding_tag_lists_nothing_and_reads_no_state(policy, tag):
    assert asyncio.run(tools.list_available_bots(tag=tag)) == []
    assert policy["state_reads"] == []


def test_a_catalog_tag_no_visible_bot_carries_lists_nothing(policy):
    assert asyncio.run(tools.list_available_bots(tag="coder")) == []


def test_an_unknown_tag_raises_before_reading_profiles(policy):
    with pytest.raises(ValueError, match="unknown tag"):
        asyncio.run(tools.list_available_bots(tag="no-such-tag"))
    assert policy["order"] == ["policy"]


@pytest.mark.parametrize("capability", ["review", " review "])
def test_the_capability_filter_keeps_bots_whose_tags_grant_it(policy, capability):
    listed = asyncio.run(tools.list_available_bots(capability=capability, include_state=False))

    assert [bot["slug"] for bot in listed] == ["fixture-public"]


def test_a_capability_follows_the_tag_not_the_bot(policy):
    """A token moved to another tag (or a renamed tag) changes who it finds."""
    policy["capabilities"] = {"coder": ("review",), "reviewer": ()}

    assert asyncio.run(tools.list_available_bots(capability="review")) == []
    assert asyncio.run(tools.list_available_bots(include_state=False))[0]["capabilities"] == []


def test_an_unknown_capability_raises_before_reading_profiles(policy):
    with pytest.raises(ValueError, match="unknown capability"):
        asyncio.run(tools.list_available_bots(capability="no-such-capability"))
    assert policy["order"] == ["policy", "capabilities"]


def test_unreadable_capabilities_are_policy_unavailable(monkeypatch):
    class _Store:
        def __init__(self, _config):
            self.engine = None

    monkeypatch.setattr("llm_bawt.runtime_settings.BotProfileStore", _Store)

    with pytest.raises(BotAccessError, match="policy_unavailable"):
        tools._role_capabilities()


def test_a_profile_created_after_the_snapshot_is_hidden(policy):
    policy["profiles"].append(_profile("fixture-new"))

    assert "fixture-new" not in {b["slug"] for b in asyncio.run(tools.list_available_bots())}


def test_discovery_raises_instead_of_listing_unfiltered(policy):
    policy["down"] = True

    with pytest.raises(BotAccessError, match="policy_unavailable"):
        asyncio.run(tools.list_available_bots())
    assert policy["order"] == ["policy"]


def test_an_unreadable_profile_table_is_policy_unavailable(monkeypatch):
    class _Store:
        def __init__(self, _config):
            self.engine = None

    monkeypatch.setattr("llm_bawt.runtime_settings.BotProfileStore", _Store)

    with pytest.raises(BotAccessError, match="policy_unavailable"):
        tools._profile_rows()


def _sends(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def enqueue(**kwargs):
        calls.append(kwargs)
        return {"delivery_id": "delivery-1", "status": "QUEUED"}

    monkeypatch.setattr(server, "_enqueue_durable", enqueue)
    monkeypatch.setattr(server, "_find_delivery_by_key", lambda *a: asyncio.sleep(0))
    monkeypatch.setattr(server, "_check_bot_in_turn", lambda *a: asyncio.sleep(0))
    return calls


def _send(target: str, **kwargs) -> dict:
    return asyncio.run(server.send_message_to_bot(
        target_bot_id=target, message="hi", sender_bot_id="fixture-public", **kwargs))


@pytest.mark.parametrize("extra", [{}, {"delivery": "when_idle"},
                                   {"wait_for_reply": True, "idempotency_key": "k1"}])
def test_a_send_to_an_inactive_bot_is_refused_and_never_queued(policy, monkeypatch, extra):
    calls = _sends(monkeypatch)

    result = _send(" Fixture-Inactive ", **extra)

    assert result["dispatched"] is False and result["error"] == "target_inactive"
    assert calls == []


_ACCEPTED = {"delivery_id": "delivery-0", "status": "DELIVERED", "author": {"entity_type": "bot"}}


def _retries(monkeypatch, found: Any) -> dict:
    seen: dict = {"lookups": [], "awaited": [], "in_turn": 0}

    async def find(sender, target, key):
        seen["lookups"].append((sender, target, key))
        if isinstance(found, Exception):
            raise found
        return found

    async def await_delivery(receipt, _seconds):
        seen["awaited"].append(receipt)
        return {"success": True, "delivery_id": receipt["delivery_id"], "content": "done"}

    async def in_turn(_target):
        seen["in_turn"] += 1

    monkeypatch.setattr(server, "_find_delivery_by_key", find)
    monkeypatch.setattr(server, "_await_delivery", await_delivery)
    monkeypatch.setattr(server, "_check_bot_in_turn", in_turn)
    monkeypatch.setattr(server, "_bot_send_wait_ceiling_seconds", lambda: 300.0)
    return seen


@pytest.mark.parametrize("extra", [{}, {"delivery": "when_idle"}])
def test_an_async_same_key_retry_reattaches_after_the_target_goes_inactive(policy, monkeypatch, extra):
    calls = _sends(monkeypatch)
    seen = _retries(monkeypatch, dict(_ACCEPTED))

    result = _send(" Fixture-Inactive ", idempotency_key=" k1 ", **extra)

    assert result["delivery_id"] == "delivery-0" and result["duplicate"] is True
    assert result["dispatched"] is True and "error" not in result
    assert seen["lookups"] == [("fixture-public", "fixture-inactive", "k1")]
    assert calls == []


def test_a_waited_same_key_retry_reattaches_after_the_target_goes_inactive(policy, monkeypatch):
    calls = _sends(monkeypatch)
    seen = _retries(monkeypatch, dict(_ACCEPTED))

    result = _send("fixture-inactive", wait_for_reply=True, idempotency_key="k1")

    assert result["delivery_id"] == "delivery-0" and result["content"] == "done"
    assert [r["delivery_id"] for r in seen["awaited"]] == ["delivery-0"]
    assert seen["awaited"][0]["duplicate"] is True
    assert len(seen["lookups"]) == 1 and seen["in_turn"] == 0 and calls == []


@pytest.mark.parametrize("found", [None, RuntimeError("deliveries api down")])
@pytest.mark.parametrize("extra", [{}, {"wait_for_reply": True}])
def test_a_new_key_or_a_failed_lookup_is_still_refused(policy, monkeypatch, found, extra):
    calls = _sends(monkeypatch)
    seen = _retries(monkeypatch, found)

    result = _send("fixture-inactive", idempotency_key="k2", **extra)

    assert result["dispatched"] is False and result["error"] == "target_inactive"
    assert seen["awaited"] == [] and calls == []


def test_a_keyless_send_to_an_inactive_bot_never_looks_up_a_delivery(policy, monkeypatch):
    _sends(monkeypatch)
    seen = _retries(monkeypatch, dict(_ACCEPTED))

    assert _send("fixture-inactive")["error"] == "target_inactive"
    assert seen["lookups"] == []


@pytest.mark.parametrize("target", ["fixture-private", "fixture-quiet", "fixture-column"])
def test_private_and_quiet_bots_can_still_be_messaged(policy, monkeypatch, target):
    calls = _sends(monkeypatch)

    assert _send(target)["dispatched"] is True
    assert [c["target_bot_id"] for c in calls] == [target]


def test_an_unreadable_policy_refuses_the_send(policy, monkeypatch):
    calls = _sends(monkeypatch)
    policy["down"] = True

    result = _send("fixture-public")

    assert result["dispatched"] is False and result["error"] == "policy_unavailable"
    assert calls == []
