from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pydantic import TypeAdapter

from llm_bawt.mcp_server import task_association, task_tools
from llm_bawt.mcp_server.task_step_types import STEP_TYPES, TaskStepInput


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_associate_current_tool_accepts_only_task_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_associate(task_ref: str) -> dict:
        calls.append(task_ref)
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    assert run(task_tools.associate_task_to_current_turn("TASK-701")) == {
        "ok": True,
        "taskRef": "TASK-701",
    }
    assert calls == ["TASK-701"]


def test_signed_association_links_only_after_success_with_returned_cuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from llm_bawt.utils import db
    from llm_bawt.working_memory.service import WorkingMemoryService

    monkeypatch.setattr(WorkingMemoryService, "__init__", lambda self, engine: None)
    ctx = SimpleNamespace(bot_id="test-bot", user_id="test-user", session_id="sess",
                          turn_id="turn-abc", trigger_message_id="message")
    monkeypatch.setattr(task_association, "current_task_turn_context", lambda: ctx)
    monkeypatch.setattr(db, "get_shared_engine", lambda config: object())
    calls = []

    def linked(self, context, canonical):
        calls.append((context, canonical))
        return {"item": {"id": "memory-item"}}

    monkeypatch.setattr(WorkingMemoryService, "link_associated", linked)

    class Response:
        def raise_for_status(self):
            calls.append("association_committed")

        def json(self):
            return {"association": {"taskId": "canonical-cuid"}}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def put(self, path, json):
            calls.append((path, json["botId"]))
            return Response()

    monkeypatch.setattr(task_association.httpx, "AsyncClient", Client)
    result = run(task_association.associate_current_task("TASK-900"))
    assert calls == [("/internal/tasks/TASK-900/associations", "test-bot"),
                     "association_committed", (ctx, "canonical-cuid")]
    assert result["workingMemoryLink"] == {
        "state": "linked", "taskId": "canonical-cuid", "itemId": "memory-item"}


def test_memory_failure_does_not_uncommit_signed_association(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_bawt.working_memory.contracts import WorkingMemoryError
    from llm_bawt.working_memory.service import WorkingMemoryService

    monkeypatch.setattr(WorkingMemoryService, "__init__", lambda self, engine: None)

    def unavailable(self, context, canonical):
        raise WorkingMemoryError("task_link_audit_pending", "migration is pending")

    monkeypatch.setattr(WorkingMemoryService, "link_associated", unavailable)
    monkeypatch.setattr(task_association, "current_task_turn_context", lambda: type("Context", (), {
        "bot_id": "test-bot", "user_id": "test-user", "session_id": "sess",
        "turn_id": "turn-abc", "trigger_message_id": "message",
    })())
    from llm_bawt.utils import db
    monkeypatch.setattr(db, "get_shared_engine", lambda config: object())

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def put(self, path, json):
            return type("Response", (), {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"association": {"taskId": "canonical-cuid"}},
            })()

    monkeypatch.setattr(task_association.httpx, "AsyncClient", Client)
    result = run(task_association.associate_current_task("TASK-900"))
    assert result["ok"] is True and result["association"]["taskId"] == "canonical-cuid"
    assert result["workingMemoryLink"]["error"] == "task_link_audit_pending"


def test_task_catalog_resolves_short_ref_to_verified_cuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_bawt.working_memory import task_links
    from llm_bawt.working_memory.contracts import WorkingMemoryError

    seen = []
    row = {"id": "canonical-cuid", "shortId": "TASK-42", "title": "A task",
           "status": "IN_PROGRESS", "agentBotId": "some-bot"}

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["auth"] is not None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, path):
            seen.append(path)
            return self

        def raise_for_status(self):
            pass

        def json(self):
            return row

    monkeypatch.setattr(task_links.httpx, "Client", Client)
    catalog = task_links.TaskCatalog()
    assert catalog.get("task-42").id == "canonical-cuid"
    assert seen == ["/api/tasks/tasks/TASK-42"]
    assert catalog.get("canonical-cuid").short_id == "TASK-42"
    row["id"] = "wrong-cuid"
    with pytest.raises(WorkingMemoryError, match="does not match"):
        catalog.get("canonical-cuid")


def test_create_can_immediately_associate_new_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    async def fake_post(path: str, json: dict, headers: dict | None = None) -> dict:
        calls.append((path, json))
        return {"id": "task-id", "shortId": "TASK-900", "title": json["title"]}

    async def fake_associate(task_ref: str) -> dict:
        calls.append(("associate", task_ref))
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    result = run(task_tools.create_task(
        "Follow-up",
        bot_id="loopy",
        associate_current_turn=True,
    ))

    assert result["task"]["shortId"] == "TASK-900"
    assert result["currentTurnAssociation"]["ok"] is True
    assert calls[-1] == ("associate", "TASK-900")


def test_create_without_flag_preserves_existing_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {"id": "task-id", "shortId": "TASK-900"}

    async def fake_post(path: str, json: dict, headers: dict | None = None) -> dict:
        return created

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    assert run(task_tools.create_task("Follow-up")) is created


def test_create_step_schema_exposes_canonical_enum() -> None:
    schema = TypeAdapter(TaskStepInput).json_schema()
    assert schema["properties"]["type"]["enum"] == list(STEP_TYPES)


def test_create_rejects_invalid_step_type_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_post(*args: Any, **kwargs: Any) -> dict:
        raise AssertionError("invalid step types must not reach BawtHub")

    monkeypatch.setattr(task_tools, "_api_post", unexpected_post)

    result = run(task_tools.create_task(
        "Follow-up",
        steps=[{"title": "Run tests", "type": "TEST"}],  # type: ignore[list-item]
    ))

    assert result == {
        "error": (
            "steps[0].type must be one of: PLAN, READ_FILE, EDIT_FILE, "
            "CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW"
        )
    }


def test_create_forwards_valid_typed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict] = []

    async def fake_post(path: str, json: dict, headers: dict | None = None) -> dict:
        sent.append(json)
        return {"id": "task-id", "shortId": "TASK-900"}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    result = run(task_tools.create_task(
        "Follow-up",
        steps=[{"title": "Run tests", "type": "RUN_COMMAND"}],
    ))

    assert result["shortId"] == "TASK-900"
    assert sent[0]["steps"] == [
        {"title": "Run tests", "type": "RUN_COMMAND"}
    ]


def test_update_can_associate_without_rewriting_task_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_associate(task_ref: str) -> dict:
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    assert run(task_tools.update_task(
        "TASK-701",
        associate_current_turn=True,
    )) == {"ok": True, "taskRef": "TASK-701"}


def test_update_can_claim_and_associate_current_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_patch(path: str, json: dict, headers: dict | None = None) -> dict:
        return {
            "id": "task-id",
            "shortId": "TASK-701",
            "title": "Traceability",
            "status": json["status"],
            "steps": [],
        }

    async def fake_associate(task_ref: str) -> dict:
        return {"ok": True, "taskRef": task_ref}

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    monkeypatch.setattr(task_tools, "associate_current_task", fake_associate)

    result = run(task_tools.update_task(
        "TASK-701",
        status="IN_PROGRESS",
        bot_id="loopy",
        associate_current_turn=True,
    ))

    assert result["task"]["status"] == "IN_PROGRESS"
    assert result["currentTurnAssociation"] == {
        "ok": True,
        "taskRef": "TASK-701",
    }
