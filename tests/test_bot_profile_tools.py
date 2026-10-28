"""bots_profile_patch: caller, target and compare-and-set contract (TASK-1521).

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
        merged.update({k: v for k, v in body.items() if k in tools.ALLOWED_FIELDS})
        return _Response(200, merged)

    @property
    def writes(self):
        return [body for method, _slug, body in self.calls if method == "PATCH"]


def _profile(**fields):
    return {"slug": TARGET, "name": "Target", "description": "old", "color": None,
            "system_prompt": "fixture", "tags": ["inactive"], **fields}


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
    monkeypatch.setattr(tools, "_manager_roles", lambda: frozenset({"fixture-manager"}))
    return SimpleNamespace(app=app, access=access)


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


@pytest.mark.parametrize("field", ["default_model", "agent_backend", "endpoint_id", "harness",
                                   "bot_type", "include_in_global_search", "agent_backend_config"])
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


def test_unreadable_manager_roles_fail_closed_for_a_tag_write(rig, monkeypatch):
    def down():
        raise RuntimeError("settings store down")

    monkeypatch.setattr(tools, "_manager_roles", down)

    assert _call(changes={"tags": ["no_global_search"]})["error"] == "unavailable"
    assert rig.app.writes == []
    assert _call(changes={"description": "new"})["changed"]
