"""bots_profile_patch: caller, target and compare-and-set contract (TASK-1521).

Also the model switch and protected roles (TASK-1818).

The app route is replaced by a recorder that answers like
``PATCH /v1/bots/{slug}/profile``; the route itself is covered against
PostgreSQL in ``test_bot_profile_patch_pg.py``.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.mcp_server import bot_profile_tools as tools
from llm_bawt.mcp_server.catalog_contracts import TOOL_SUMMARIES
from llm_bawt.task_turn_context import TaskTurnContext

TARGET = "fixture-target"
HIDDEN = "fixture-hidden"
CALLER = "fixture-caller"


class _Response:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _App:
    """Answers GET with ``profile`` and PATCH with ``patch`` (or the merged profile)."""

    def __init__(self, profile: dict, patch: _Response | None = None):
        self.profile = profile
        self.patch = patch
        self.calls: list[tuple[str, str, dict | None]] = []

    async def __call__(self, method, slug, body=None):
        self.calls.append((method, slug, body))
        if method == "GET":
            return _Response(200, self.profile)
        if self.patch is not None:
            return self.patch
        merged = dict(self.profile)
        merged.update({k: v for k, v in body.items()
                       if k in tools.ALLOWED_FIELDS | {tools.HARNESS}})
        return _Response(200, merged)

    @property
    def writes(self):
        return [body for method, _slug, body in self.calls if method == "PATCH"]


def _profile(**fields):
    return {"slug": TARGET, "name": "Target", "description": "old", "color": None,
            "system_prompt": "fixture", "tags": ["inactive"], "agent_backend": "fixture-harness",
            "default_model": "fixture-model-a", "endpoint_id": 1, "harness": "fixture-harness",
            **fields}


#: The picker list for the target's runtime: what GET /v1/models/catalog/endpoints answers.
ENDPOINTS = [
    {"id": 1, "model_key": "fixture-model-a", "harness": "fixture-harness",
     "eligibility": {"eligible": True}},
    {"id": 2, "model_key": "fixture-model-b", "harness": "fixture-proxy",
     "eligibility": {"eligible": True}},
    {"id": 3, "model_key": "fixture-model-c", "harness": "fixture-harness",
     "eligibility": {"eligible": False, "reason": "fixture route not connected"}},
]


@pytest.fixture
def rig(monkeypatch):
    turn = TaskTurnContext(session_id="s", turn_id="t", trigger_message_id="m",
                           bot_id=CALLER, user_id="fixture-user", issued_at=0)
    access = SimpleNamespace(partitions={TARGET: TARGET, HIDDEN: HIDDEN, CALLER: CALLER},
                             private_bots=frozenset({HIDDEN}))
    app = _App(_profile())

    async def resolve():
        return access

    monkeypatch.setattr(tools, "current_task_turn_context", lambda: turn)
    monkeypatch.setattr(tools, "_resolve_access", resolve)
    monkeypatch.setattr(tools, "_app", app)
    monkeypatch.setattr(tools, "_protected_roles", lambda: frozenset({"fixture-manager"}))
    switchers = {CALLER}
    monkeypatch.setattr(tools, "_may_switch_models", lambda slug: slug in switchers)
    catalog_reads: list[str | None] = []

    async def picker(backend):
        catalog_reads.append(backend)
        return ENDPOINTS

    monkeypatch.setattr(tools, "_picker_endpoints", picker)
    return SimpleNamespace(app=app, access=access, switchers=switchers,
                           catalog_reads=catalog_reads)


def _call(**kwargs) -> dict:
    kwargs.setdefault("bot_id", TARGET)
    kwargs.setdefault("reason", "fixture reason")
    return json.loads(asyncio.run(tools.bots_profile_patch(**kwargs)))


def test_an_unsigned_call_is_refused_before_anything_is_read(monkeypatch):
    monkeypatch.setattr(tools, "_resolve_access", lambda: pytest.fail("policy read unsigned"))
    monkeypatch.setattr(tools, "_app", lambda *_a: pytest.fail("app reached unsigned"))

    assert _call(changes={"description": "x"})["error"] == "unauthorized"


def test_a_private_bot_and_an_unknown_bot_get_the_same_refusal(rig):
    hidden = _call(bot_id=HIDDEN, changes={"description": "x"})
    unknown = _call(bot_id="fixture-nobody", changes={"description": "x"})

    assert hidden == unknown == {"error": "access_denied", "message": "no such editable bot"}
    assert rig.app.calls == []


def test_a_bot_that_reads_back_private_is_refused(rig):
    rig.app.profile = _profile(tags=["private"])

    assert _call(changes={"description": "x"})["error"] == "access_denied"
    assert rig.app.writes == []


@pytest.mark.parametrize("tags", [["inactive", "private"], ["private"]])
def test_the_private_tag_cannot_be_added(rig, tags):
    assert _call(changes={"tags": tags})["error"] == "private_tag_locked"
    assert rig.app.writes == []


@pytest.mark.parametrize("field", ["agent_backend", "harness", "bot_type",
                                   "include_in_global_search", "agent_backend_config"])
def test_runtime_route_and_privacy_fields_are_refused(rig, field):
    result = _call(changes={field: "x"})

    assert result["error"] == "field_not_allowed"
    assert rig.app.calls == []


@pytest.mark.parametrize("kwargs", [
    {"changes": {}}, {"changes": {"description": "x"}, "reason": " "},
    {"changes": {"tags": "inactive"}},
    {"changes": {"description": "x"}, "expected": {"color": None}},
])
def test_malformed_calls_are_refused(rig, kwargs):
    assert _call(**kwargs)["error"] == "invalid_request"
    assert rig.app.calls == []


def test_every_write_is_pinned_to_what_was_read_and_stamped_from_the_turn(rig):
    result = _call(changes={"description": "new", "color": "#123456"}, reason=" why ")

    assert rig.app.writes == [{
        "description": "new", "color": "#123456", "updated_by": f"bot:{CALLER}",
        "change_note": "why", "expected_tags": ["inactive"],
        "expected_fields": {"description": "old", "color": None}}]
    assert result == {
        "bot_id": TARGET, "updated_by": f"bot:{CALLER}", "reason": "why",
        "changed": {"description": {"before": "old", "after": "new"},
                    "color": {"before": None, "after": "#123456"}},
        "rollback": {"changes": {"description": "old", "color": None},
                     "expected": {"description": "new", "color": "#123456"}}}


def test_a_tag_write_pins_the_tag_set_and_sends_no_field_pin(rig):
    result = _call(changes={"tags": ["no_global_search", "inactive"]})

    assert rig.app.writes[0]["tags"] == ["no_global_search", "inactive"]
    assert rig.app.writes[0]["expected_tags"] == ["inactive"]
    assert "expected_fields" not in rig.app.writes[0]
    assert result["changed"] == {"tags": {"before": ["inactive"],
                                          "after": ["inactive", "no_global_search"]}}


def test_the_callers_expected_value_is_checked_and_sent(rig):
    stale = _call(changes={"description": "new"}, expected={"description": "older"})
    assert stale["error"] == "conflict" and stale["current"] == {"description": "old"}
    assert rig.app.writes == []

    _call(changes={"description": "new"}, expected={"description": "old"})
    assert rig.app.writes[0]["expected_fields"] == {"description": "old"}


def test_unchanged_fields_are_dropped_and_an_all_unchanged_call_writes_nothing(rig):
    assert _call(changes={"description": "old", "tags": ["inactive"]})["error"] == "no_change"
    assert rig.app.writes == []

    result = _call(changes={"description": "old", "color": "#123456"})
    assert set(result["changed"]) == {"color"}
    assert "description" not in rig.app.writes[0]


def test_a_conflict_from_the_app_passes_through(rig):
    rig.app.patch = _Response(409, {"detail": {
        "code": "fields_changed", "message": "changed", "current": {"description": "theirs"}}})

    result = _call(changes={"description": "new"})

    assert result == {"error": "conflict", "message": "changed", "code": "fields_changed",
                      "current": {"description": "theirs"}}


def test_a_bot_turned_private_between_read_and_write_is_access_denied(rig):
    rig.app.patch = _Response(409, {"detail": {
        "code": "tags_changed", "message": "changed", "current_tags": ["inactive", "private"]}})

    assert _call(changes={"description": "new"})["error"] == "access_denied"


def test_a_policy_outage_fails_closed(rig, monkeypatch):
    async def down():
        raise BotAccessError("policy_unavailable")

    monkeypatch.setattr(tools, "_resolve_access", down)

    assert _call(changes={"description": "x"})["error"] == "policy_unavailable"
    assert rig.app.calls == []


def test_the_catalog_summary_keeps_the_write_contract_inline():
    summary = TOOL_SUMMARIES["bots_profile_patch"]
    for words in ("compare-and-set", "409 writes nothing", "Private bots and own tags refused", "rollback"):
        assert words in summary


@pytest.mark.parametrize("target", [CALLER, f" {CALLER.upper()} "])
def test_a_caller_cannot_change_its_own_tags(rig, target):
    result = _call(bot_id=target, changes={"tags": ["coder"]})

    assert result["error"] == "self_tag_edit_refused"
    assert rig.app.calls == []


def test_a_caller_may_edit_its_own_other_fields(rig):
    result = _call(bot_id=CALLER, changes={"description": "new"})

    assert result["changed"] == {"description": {"before": "old", "after": "new"}}


@pytest.mark.parametrize("held, wanted", [
    (["inactive"], ["inactive", "fixture-manager"]),
    (["inactive", "fixture-manager"], ["inactive"]),
])
def test_a_grant_manager_role_cannot_be_added_or_removed(rig, held, wanted):
    """A bot that could hand out manager roles could ordain a peer that ordains it."""
    rig.app.profile = _profile(tags=held)

    result = _call(changes={"tags": wanted})

    assert result["error"] == "tag_locked"
    assert result["tags"] == ["fixture-manager"]
    assert rig.app.writes == []


def test_a_tag_write_that_keeps_a_held_manager_role_passes(rig):
    rig.app.profile = _profile(tags=["inactive", "fixture-manager"])

    result = _call(changes={"tags": ["fixture-manager", "no_global_search"]})

    assert result["changed"]["tags"]["after"] == ["fixture-manager", "no_global_search"]


def test_unreadable_protected_roles_fail_closed_for_a_tag_write(rig, monkeypatch):
    def down():
        raise RuntimeError("settings store down")

    monkeypatch.setattr(tools, "_protected_roles", down)

    assert _call(changes={"tags": ["no_global_search"]})["error"] == "unavailable"
    assert rig.app.writes == []
    assert _call(changes={"description": "new"})["changed"]


# --- Model switch (TASK-1818) ------------------------------------------------


def test_a_model_switcher_writes_exactly_what_a_picker_pick_writes(rig):
    result = _call(changes={"endpoint_id": 2}, reason="quota")

    assert rig.catalog_reads == ["fixture-harness"]
    assert rig.app.writes == [{
        "default_model": "fixture-model-b", "endpoint_id": 2, "harness": "fixture-proxy",
        "updated_by": f"bot:{CALLER}", "change_note": "quota", "expected_tags": ["inactive"],
        "expected_fields": {"default_model": "fixture-model-a", "endpoint_id": 1,
                            "harness": "fixture-harness"}}]
    assert result["changed"] == {
        "default_model": {"before": "fixture-model-a", "after": "fixture-model-b"},
        "endpoint_id": {"before": 1, "after": 2},
        "harness": {"before": "fixture-harness", "after": "fixture-proxy"}}
    # harness follows the endpoint, so the rollback names only the switch.
    assert result["rollback"] == {
        "changes": {"default_model": "fixture-model-a", "endpoint_id": 1},
        "expected": {"default_model": "fixture-model-b", "endpoint_id": 2}}


def test_the_rollback_of_a_model_switch_is_itself_a_valid_switch(rig):
    rollback = _call(changes={"endpoint_id": 2})["rollback"]
    rig.app.profile = _profile(default_model="fixture-model-b", endpoint_id=2,
                               harness="fixture-proxy")

    undone = _call(**rollback)

    assert undone["changed"]["endpoint_id"] == {"before": 2, "after": 1}
    assert rig.app.writes[-1]["harness"] == "fixture-harness"


def test_a_matching_default_model_is_accepted_with_its_endpoint(rig):
    result = _call(changes={"endpoint_id": 2, "default_model": "fixture-model-b"})

    assert result["changed"]["endpoint_id"]["after"] == 2


def test_a_caller_without_the_model_switch_capability_is_refused(rig):
    rig.switchers.clear()

    for changes in ({"endpoint_id": 2}, {"endpoint_id": 2, "description": "new"}):
        assert _call(changes=changes)["error"] == "capability_required"
    assert rig.app.calls == [] and rig.catalog_reads == []
    # Other fields stay open to every bot.
    assert _call(changes={"description": "new"})["changed"]


def test_an_unreadable_capability_fails_closed(rig, monkeypatch):
    def down(_slug):
        raise RuntimeError("settings store down")

    monkeypatch.setattr(tools, "_may_switch_models", down)

    assert _call(changes={"endpoint_id": 2})["error"] == "unavailable"
    assert rig.app.calls == []


def test_a_model_switcher_may_switch_its_own_model(rig):
    rig.app.profile = _profile(slug=CALLER)

    assert _call(bot_id=CALLER, changes={"endpoint_id": 2})["changed"]["endpoint_id"]["after"] == 2


@pytest.mark.parametrize("changes, error", [
    ({"endpoint_id": 99}, "unknown_endpoint"),
    ({"endpoint_id": 2, "default_model": "fixture-model-a"}, "model_endpoint_mismatch"),
    ({"endpoint_id": 3}, "endpoint_ineligible"),
    ({"endpoint_id": None, "default_model": "fixture-model-a"}, "model_endpoint_mismatch"),
])
def test_a_switch_the_catalog_refuses_writes_nothing(rig, changes, error):
    assert _call(changes=changes)["error"] == error
    assert rig.app.writes == []


@pytest.mark.parametrize("changes", [
    {"default_model": "fixture-model-b"}, {"endpoint_id": "2"}, {"endpoint_id": True},
    {"endpoint_id": 2, "default_model": 5},
])
def test_malformed_model_switches_are_refused_before_anything_is_read(rig, changes):
    assert _call(changes=changes)["error"] == "invalid_request"
    assert rig.app.calls == [] and rig.catalog_reads == []


def test_an_unreadable_catalog_fails_closed(rig, monkeypatch):
    async def down(_backend):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(tools, "_picker_endpoints", down)

    assert _call(changes={"endpoint_id": 2})["error"] == "unavailable"
    assert rig.app.writes == []


def test_a_null_endpoint_clears_the_model_like_the_pickers_clear(rig):
    result = _call(changes={"endpoint_id": None})

    assert rig.catalog_reads == []
    assert {k: rig.app.writes[0][k] for k in ("default_model", "endpoint_id")} == {
        "default_model": None, "endpoint_id": None}
    assert "harness" not in rig.app.writes[0]
    assert set(result["changed"]) == {"default_model", "endpoint_id"}


def test_a_model_switch_of_a_private_bot_is_refused(rig):
    assert _call(bot_id=HIDDEN, changes={"endpoint_id": 2})["error"] == "access_denied"
    rig.app.profile = _profile(tags=["private"])
    assert _call(changes={"endpoint_id": 2})["error"] == "access_denied"
    assert rig.app.writes == []


def test_a_stale_model_switch_is_a_conflict(rig):
    stale = _call(changes={"endpoint_id": 2}, expected={"endpoint_id": 7})
    assert stale["error"] == "conflict" and stale["current"] == {"endpoint_id": 1}
    assert rig.app.writes == []

    rig.app.patch = _Response(409, {"detail": {
        "code": "fields_changed", "message": "changed", "current": {"endpoint_id": 3}}})
    raced = _call(changes={"endpoint_id": 2})
    assert raced["error"] == "conflict" and raced["code"] == "fields_changed"


@pytest.mark.parametrize("held, wanted", [
    (["inactive"], ["inactive", "fixture-coordinator"]),
    (["inactive", "fixture-coordinator"], ["inactive"]),
])
def test_a_model_switch_role_cannot_be_added_or_removed(rig, monkeypatch, held, wanted):
    monkeypatch.setattr(tools, "_protected_roles",
                        lambda: frozenset({"fixture-manager", "fixture-coordinator"}))
    rig.app.profile = _profile(tags=held)

    result = _call(changes={"tags": wanted})

    assert result["error"] == "tag_locked"
    assert result["tags"] == ["fixture-coordinator"]
    assert rig.app.writes == []


def test_protected_roles_union_both_configured_capabilities(monkeypatch):
    from llm_bawt.bot_tags import grants

    monkeypatch.setattr(tools, "_global_settings", lambda: {
        grants.MANAGER_CAPABILITY_SETTING: "fixture-manage",
        tools.MODEL_SWITCH_CAPABILITY_SETTING: " fixture-coordinate "})
    roles = {"fixture-manage": {"fixture-manager"}, "fixture-coordinate": {"fixture-coordinator"}}
    asked: list[str] = []

    def manager_roles(_conn, capability, **_kw):
        asked.append(capability)
        return frozenset(roles[capability])

    monkeypatch.setattr(grants, "manager_roles", manager_roles)
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda _cfg: _Engine())

    assert tools._protected_roles() == {"fixture-manager", "fixture-coordinator"}
    assert sorted(asked) == ["fixture-coordinate", "fixture-manage"]


def test_unset_capabilities_protect_nothing_and_let_no_bot_switch(monkeypatch):
    monkeypatch.setattr(tools, "_global_settings", lambda: {})
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine",
                        lambda _cfg: pytest.fail("no capability configured, nothing to read"))

    assert tools._protected_roles() == frozenset()
    assert tools._may_switch_models(CALLER) is False


class _Engine:
    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
