"""Discovery and send refusals come from bot tags (TASK-1329, BOT_TAGS.md §7).

``bots_list_available`` reads in-process: the policy snapshot first, then the
profile rows. ``bots_send_message`` refuses an ``inactive`` target before any
side effect. Fixture bots only.
"""

from __future__ import annotations

import asyncio
import time
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


_NO_ROUTE = {"model": None, "provider": None, "quota": "unknown", "quota_checked_at": None}


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

    def discovery_caller(arguments):
        state["caller_args"] = arguments
        if isinstance(state["caller"], Exception):
            raise state["caller"]
        return state["caller"]

    def route_facts(profiles, caller, turn_id):
        state["order"].append("routes")
        state["route_reads"].append(([row.slug for row in profiles], caller, turn_id))
        return ({row.slug: dict(state["routes"].get(row.slug, _NO_ROUTE)) for row in profiles},
                state["caller_provider"])

    state["state_reads"] = []
    state["route_reads"] = []
    state["caller"] = (None, None)
    state["routes"] = {}
    state["caller_provider"] = None
    # Stored order is the operator's; discovery reports the sorted union.
    state["capabilities"] = {"coder": ("implement",), "reviewer": ("review", "implement")}
    monkeypatch.setattr(tools, "_discovery_caller", discovery_caller)
    monkeypatch.setattr(tools, "_route_facts", route_facts)
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
                         "capabilities": ["implement", "review"], **_NO_ROUTE,
                         "same_provider_as_caller": None}
    assert listed[1]["tags"] == ["no_global_search"]
    assert listed[1]["capabilities"] == []
    assert policy["order"] == ["policy", "capabilities", "profiles", "routes"]
    assert policy["state_reads"] == []
    assert policy["caller_args"] == {"tag": None, "capability": None, "include_state": False}


def test_state_is_read_once_for_the_visible_bots_only(policy):
    listed = asyncio.run(tools.list_available_bots())

    assert policy["order"] == ["policy", "capabilities", "profiles", "routes", "state"]
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


# --- route facts (TASK-1249) ----------------------------------------------------


def _route(provider, quota="available"):
    return {"model": f"fixture-model@{provider}-path", "provider": provider,
            "quota": quota, "quota_checked_at": 1_800_000_000}


def test_a_signed_caller_is_left_out_and_compared_by_provider(policy):
    providers = {"fixture-public": "vendor-a", "fixture-quiet": "vendor-b"}
    policy["routes"] = {slug: _route(p) for slug, p in providers.items()}
    policy["caller"] = ("fixture-public", "turn-1")
    policy["caller_provider"] = "vendor-a"

    listed = asyncio.run(tools.list_available_bots(include_state=False))

    assert [b["slug"] for b in listed] == ["fixture-quiet"]
    assert listed[0]["same_provider_as_caller"] is False
    assert (listed[0]["provider"], listed[0]["quota"]) == ("vendor-b", "available")
    # The caller's own row still feeds the route read; the turn id rides along.
    assert policy["route_reads"] == [(["fixture-quiet", "fixture-public"], "fixture-public", "turn-1")]


def test_a_hidden_caller_still_names_its_route(policy):
    policy["caller"] = ("fixture-private", None)

    asyncio.run(tools.list_available_bots(include_state=False))

    assert policy["route_reads"][0][0] == ["fixture-public", "fixture-quiet", "fixture-private"]


def test_one_provider_for_everyone_marks_every_row_same_and_still_lists_them(policy):
    policy["routes"] = {s: _route("vendor-a") for s in _TAGS}
    policy["caller"] = ("fixture-column", None)
    policy["caller_provider"] = "vendor-a"

    listed = asyncio.run(tools.list_available_bots(capability="review", include_state=False))

    assert [(b["slug"], b["same_provider_as_caller"]) for b in listed] == [("fixture-public", True)]


@pytest.mark.parametrize("caller, caller_provider, row_provider", [
    ((None, None), "vendor-a", "vendor-a"),          # unsigned: never compared
    (("fixture-column", None), None, "vendor-a"),    # caller's provider unknown
    (("fixture-column", None), "vendor-a", None),    # row's provider unknown
])
def test_same_provider_is_null_unless_both_sides_are_known(policy, caller, caller_provider,
                                                            row_provider):
    policy["routes"] = {"fixture-public": _route(row_provider)}
    policy["caller"] = caller
    policy["caller_provider"] = caller_provider

    listed = asyncio.run(tools.list_available_bots(include_state=False))

    assert listed[0]["same_provider_as_caller"] is None


def test_an_unsigned_caller_is_not_excluded(policy):
    listed = asyncio.run(tools.list_available_bots(include_state=False))

    assert [b["slug"] for b in listed] == ["fixture-public", "fixture-quiet"]


def test_a_bad_credential_raises_before_reading_anything(policy):
    policy["caller"] = BotAccessError("caller_unavailable")

    with pytest.raises(BotAccessError, match="caller_unavailable"):
        asyncio.run(tools.list_available_bots())
    assert policy["order"] == []


class _Catalog:
    """Endpoint ids 1-3 bound to fixture paths; refs are model@path."""

    def __init__(self):
        def endpoint(eid, model, vendor, path):
            return SimpleNamespace(id=eid, ref=f"{model}@{path}",
                                   model=SimpleNamespace(key=model, vendor=vendor),
                                   access_path=SimpleNamespace(key=path))
        self.by_id = {1: endpoint(1, "fixture-a", "vendor-a", "path-sub"),
                      2: endpoint(2, "fixture-a", "vendor-a", "path-api"),
                      3: endpoint(3, "fixture-b", "vendor-b", "path-sub-b")}

    def resolve_endpoint(self, eid):
        return self.by_id[eid]

    def endpoint_by_ref(self, ref):
        return next((e for e in self.by_id.values() if e.ref == ref), None)


@pytest.fixture
def routes(monkeypatch):
    import llm_bawt.service.usage as usage
    from llm_bawt.service.usage.canonical import ProviderUsage, UsageLimit
    from llm_bawt.utils import config as config_module

    state = {"turn_models": {}}
    snaps = {"report-sub": ProviderUsage(
        provider="report-sub", display_name="Sub", status="ok", available=True,
        fetched_at=int(time.time()) - 60,
        limits=[UsageLimit(id="w", label="w", used_pct=40)])}
    adapters = {"path-sub": SimpleNamespace(provider="report-sub"),
                "path-sub-b": SimpleNamespace(provider="report-missing")}
    catalog = _Catalog()
    monkeypatch.setattr(config_module, "Config",
                        lambda: SimpleNamespace(ensure_model_catalog=lambda: catalog))
    monkeypatch.setattr(usage, "usage_adapter_for_access_path", adapters.get)
    monkeypatch.setattr(usage, "get_cached", lambda provider: snaps.get(provider))
    monkeypatch.setattr(tools, "_turn_model", lambda turn, bot: state["turn_models"].get((turn, bot)))
    return state


def test_route_facts_come_from_the_bound_endpoint_and_cached_usage(routes):
    rows = [_profile("fixture-sub", endpoint_id=1, default_model="fixture-b@path-sub-b"),
            _profile("fixture-api", endpoint_id=2),
            _profile("fixture-b", endpoint_id=3),
            _profile("fixture-unbound", endpoint_id=None),
            _profile("fixture-gone", endpoint_id=99)]

    facts, caller_provider = tools._route_facts(rows, None, None)

    # default_model never wins over the endpoint binding.
    assert facts["fixture-sub"] == {"model": "fixture-a@path-sub", "provider": "vendor-a",
                                    "quota": "available",
                                    "quota_checked_at": facts["fixture-sub"]["quota_checked_at"]}
    assert facts["fixture-sub"]["quota_checked_at"] is not None
    # Same vendor, a path no usage report owns: provider known, quota unknown.
    assert (facts["fixture-api"]["provider"], facts["fixture-api"]["quota"]) == ("vendor-a", "unknown")
    # An owning report with nothing cached yet is unknown, never available.
    assert facts["fixture-b"]["quota"] == "unknown"
    for slug in ("fixture-unbound", "fixture-gone"):
        assert facts[slug] == _NO_ROUTE
    assert caller_provider is None


def test_a_turn_callers_provider_is_the_model_its_turn_runs_on(routes):
    rows = [_profile("fixture-me", endpoint_id=1)]
    routes["turn_models"][("turn-1", "fixture-me")] = "fixture-b@path-sub-b"

    assert tools._route_facts(rows, "fixture-me", "turn-1")[1] == "vendor-b"
    # Without a turn the profile's endpoint names it.
    assert tools._route_facts(rows, "fixture-me", None)[1] == "vendor-a"
    # An unreadable or unknown turn model is unknown, not the profile's guess.
    assert tools._route_facts(rows, "fixture-me", "turn-2")[1] is None


def test_an_unreadable_catalog_reports_unknown_routes(monkeypatch):
    from llm_bawt.utils import config as config_module

    def broken():
        raise RuntimeError("catalog down")

    monkeypatch.setattr(config_module, "Config",
                        lambda: SimpleNamespace(ensure_model_catalog=broken))

    facts, caller_provider = tools._route_facts([_profile("fixture-x", endpoint_id=1)], None, None)

    assert facts == {"fixture-x": _NO_ROUTE} and caller_provider is None


def test_discovery_caller_maps_signed_and_unsigned_callers(monkeypatch):
    import llm_bawt.mcp_server.bot_access_identity as identity
    import llm_bawt.mcp_server.task_association as association
    from llm_bawt.bot_access import TrustedCaller

    seen = {}

    def caller_for(value):
        def trusted(name, arguments):
            seen["name"] = name
            if isinstance(value, Exception):
                raise value
            return value
        return trusted

    turn = SimpleNamespace(bot_id="fixture-me", user_id="fixture-user", turn_id="turn-1")
    monkeypatch.setattr(association, "current_task_turn_context", lambda: turn)

    monkeypatch.setattr(identity, "trusted_caller", caller_for(None))
    assert tools._discovery_caller({}) == (None, None)
    assert seen["name"] == "bots_list_available"

    monkeypatch.setattr(identity, "trusted_caller",
                        caller_for(TrustedCaller("fixture-me", None, "service")))
    assert tools._discovery_caller({}) == ("fixture-me", None)

    monkeypatch.setattr(identity, "trusted_caller",
                        caller_for(TrustedCaller("fixture-me", "fixture-user", "turn")))
    assert tools._discovery_caller({}) == ("fixture-me", "turn-1")

    monkeypatch.setattr(identity, "trusted_caller",
                        caller_for(TrustedCaller("fixture-other", "fixture-user", "turn")))
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        tools._discovery_caller({})

    monkeypatch.setattr(identity, "trusted_caller", caller_for(BotAccessError("caller_unavailable")))
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        tools._discovery_caller({})


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
