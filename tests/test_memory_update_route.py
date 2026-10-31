"""HTTP and tool errors from a failed memory update (TASK-1854)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError
from llm_bawt.service.routes.memory import update_memory
from llm_bawt.service.schemas import MemoryUpdateRequest
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall


@pytest.mark.parametrize("failure, expected", [
    (MemoryUpdateDuplicateError("memory update conflicts with another duplicate"), "duplicate"),
    (MemoryUpdateConflictError("memory changed concurrently; retry the update"), "retry"),
])
def test_rest_memory_update_reports_conflict(failure, expected):
    def reject(**_kwargs):
        raise failure

    client = SimpleNamespace(update_memory=reject)
    with pytest.raises(HTTPException) as exc:
        update_memory("memory-id", MemoryUpdateRequest(content="Changed"),
                      bot_id="example", client=client)
    assert exc.value.status_code == 409
    assert expected in str(exc.value.detail)


def test_rest_memory_update_leaves_unrelated_backend_failure_as_500():
    def fail(**_kwargs):
        raise RuntimeError("database unavailable")

    with pytest.raises(HTTPException) as exc:
        update_memory("memory-id", MemoryUpdateRequest(content="Changed"),
                      bot_id="example", client=SimpleNamespace(update_memory=fail))
    assert exc.value.status_code == 500


@pytest.mark.parametrize("failure, expected", [
    (MemoryUpdateDuplicateError("memory update conflicts with another duplicate"), "duplicate"),
    (MemoryUpdateConflictError("memory changed concurrently; retry the update"), "retry"),
])
def test_tool_update_shows_specific_conflict(failure, expected):
    def reject(**_kwargs):
        raise failure

    executor = ToolExecutor.__new__(ToolExecutor)
    executor.memory_client = SimpleNamespace(update_memory=reject)
    result = executor._memory_update(ToolCall(name="memory", arguments={
        "memory_id": "memory-id", "content": "Changed",
    }, raw_text=""))
    assert 'status="error"' in result
    assert expected in result
