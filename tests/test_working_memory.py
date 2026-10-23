"""Unit proof for working-memory contracts, policy and the trusted tools (TASK-1281)."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from llm_bawt.mcp_server import working_memory_tools as tools
from llm_bawt.mcp_server.catalog_contracts import TOOL_SUMMARIES, tool_reference
from llm_bawt.working_memory.contracts import (
    WorkingMemoryError,
    create_fingerprint,
    normalize_changes,
    normalize_create,
    normalize_idempotency_key,
    normalize_item_id,
)
from llm_bawt.working_memory.policy import WorkingMemoryPolicy
from llm_bawt.working_memory.store import Actor, WorkingMemoryStore


def _code(call) -> str:
    with pytest.raises(WorkingMemoryError) as raised:
        call()
    return raised.value.code


@pytest.mark.parametrize("key", [
    "Ship slice 1", "", None, 42,
    "{9b2f1c3e-4d5a-4b6c-8d7e-0f1a2b3c4d5e}", "9b2f1c3e4d5a4b6c8d7e0f1a2b3c4d5e",
])
def test_idempotency_key_must_be_canonical_uuid(key):
    assert _code(lambda: normalize_idempotency_key(key)) == "invalid_idempotency_key"


def test_idempotency_key_is_lowercased():
    key = str(uuid.uuid4())
    assert normalize_idempotency_key(f" {key.upper()} ") == key


def test_fingerprint_covers_normalized_create_with_defaults():
    explicit = normalize_create({"objective": "Café ", "status": "planned", "kind": "personal", "refs": []})
    implicit = normalize_create({"objective": "Café"})
    assert explicit == implicit
    assert create_fingerprint(explicit) == create_fingerprint(implicit)
    assert create_fingerprint(implicit) != create_fingerprint(normalize_create({"objective": "Café", "priority": 1}))


@pytest.mark.parametrize("fields, code", [
    ({"objective": "x" * 161}, "cap_exceeded"),
    ({"objective": "ok", "next_action": "x" * 401}, "cap_exceeded"),
    ({"objective": "ok", "refs": [{"kind": "k", "id": str(n)} for n in range(21)]}, "cap_exceeded"),
    ({"objective": "ok", "refs": [{"kind": "k", "id": "i", "excerpt": "x" * 201}]}, "cap_exceeded"),
    ({"objective": "ok", "refs": [{"kind": "k", "id": "i", "body": "nope"}]}, "invalid_request"),
    ({"objective": "  "}, "invalid_request"),
    ({"objective": "ok", "status": "done"}, "invalid_request"),
    ({"objective": "ok", "priority": True}, "invalid_request"),
    ({"objective": "ok", "kind": "initiative_link"}, "invalid_request"),
    ({"objective": "ok", "kind": "initiative_link", "initiative_task_id": "TASK-1", "next_action": "n"}, "invalid_request"),
    ({"objective": "ok", "initiative_task_id": "TASK-1"}, "invalid_request"),
    ({"objective": "ok", "kind": "initiative_link", "initiative_task_id": "not a key!"}, "invalid_request"),
])
def test_create_validation(fields, code):
    assert _code(lambda: normalize_create(fields)) == code


def test_initiative_link_normalizes_task_key():
    payload = normalize_create({"objective": "ok", "kind": "initiative_link", "initiative_task_id": "task-12"})
    assert payload["initiative_task_id"] == "TASK-12"


def test_changes_clear_and_link_restrictions():
    assert normalize_changes("personal", {"next_action": None, "priority": None, "checkpoint": " "}) == {
        "next_action": None, "priority": None, "checkpoint": None,
    }
    assert _code(lambda: normalize_changes("personal", {"objective": None})) == "invalid_request"
    assert _code(lambda: normalize_changes("personal", {"kind": "initiative_link"})) == "invalid_request"
    assert _code(lambda: normalize_changes("initiative_link", {"blockers": ["b"]})) == "invalid_request"
    assert _code(lambda: normalize_changes("personal", {"blockers": ["b"] * 6})) == "cap_exceeded"


def test_bad_item_id_is_not_found():
    assert _code(lambda: normalize_item_id("../etc")) == "not_found"


@pytest.mark.parametrize("scope_type, value", [("global", True), ("global", False), ("bot", "true"), ("bot", 1)])
def test_setting_refused_without_reading_the_database(scope_type, value):
    policy = WorkingMemoryPolicy(engine=None)
    assert _code(lambda: policy.validate_setting_write(scope_type, "vex", "working_memory_enabled", value)) == "invalid_setting"
    policy.validate_setting_write("global", "*", "some_other_key", "anything")
    policy.validate_setting_write("bot", "vex", "working_memory_enabled", False)


def test_protected_bot_refused_before_any_read():
    policy = WorkingMemoryPolicy(engine=None)  # any read would raise AttributeError
    assert _code(lambda: policy.require_tools("Mira")) == "protected_bot"
    assert _code(lambda: policy.require_owner_access("mira")) == "protected_bot"
    assert _code(lambda: policy.validate_setting_write("bot", "mira", "working_memory_enabled", True)) == "invalid_setting"


class _DownEngine:
    """A PostgreSQL engine whose database is unreachable."""

    class dialect:  # noqa: N801 - mirrors Engine.dialect
        name = "postgresql"

    def _down(self, *_args, **_kwargs):
        raise OperationalError("SELECT 1", None, Exception("connection refused"))

    begin = connect = _down


def test_storage_failure_is_unavailable_not_empty():
    store = WorkingMemoryStore(_DownEngine())
    assert _code(lambda: store.list_items("vex", "nick")) == "unavailable"
    assert _code(lambda: store.create("vex", "nick", {"objective": "x"}, idempotency_key=str(uuid.uuid4()),
                                      actor=Actor("bot", "vex"))) == "unavailable"
    assert _code(lambda: WorkingMemoryStore(create_engine("sqlite://"))) == "unavailable"


def _tool(coro) -> dict:
    return json.loads(asyncio.run(coro))


def test_tools_refuse_without_capability(monkeypatch):
    monkeypatch.setattr(tools, "_service", lambda: pytest.fail("service reached without a capability"))
    assert _tool(tools.working_memory_list())["error"] == "unauthorized"
    assert _tool(tools.working_memory_create(objective="x", idempotency_key=str(uuid.uuid4())))["error"] == "unauthorized"
    assert _tool(tools.working_memory_update(item_id=str(uuid.uuid4()), expected_revision=1))["error"] == "unauthorized"


def test_update_tool_rejects_set_and_clear_of_the_same_field():
    result = _tool(tools.working_memory_update(
        item_id=str(uuid.uuid4()), expected_revision=1, next_action="n", clear=["next_action"],
    ))
    assert result["error"] == "invalid_request"


def test_tools_are_catalogued_under_memory_history():
    for name in ("working_memory_list", "working_memory_create", "working_memory_update"):
        assert name in TOOL_SUMMARIES
        assert tool_reference(name) == "bawthub-mcp/references/memory-history.md"
