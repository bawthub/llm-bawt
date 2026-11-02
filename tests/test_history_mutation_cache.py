from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service import dependencies
from llm_bawt.service.routes import history


class FakeMemoryClient:
    def __init__(
        self,
        *,
        message_count: int = 0,
        memories_cleared: bool = True,
        forgotten: bool = True,
    ) -> None:
        self.message_count = message_count
        self.memories_cleared = memories_cleared
        self.forgotten = forgotten
        self.calls: list[object] = []

    def clear_messages(self) -> int:
        self.calls.append("clear_messages")
        return self.message_count

    def clear_memories(self) -> bool:
        self.calls.append("clear_memories")
        return self.memories_cleared

    def ignore_message_by_id(self, message_id: str) -> bool:
        self.calls.append(("ignore_message_by_id", message_id))
        return self.forgotten


@pytest.fixture(autouse=True)
def reset_service():
    dependencies.set_service(None)
    yield
    dependencies.set_service(None)


def fake_service(client: FakeMemoryClient):
    return SimpleNamespace(
        _default_bot="nova",
        _available_models=[],
        _llm_bawt_cache={
            ("model", "nova", "user"): object(),
            ("model", "other", "user"): object(),
        },
        get_memory_client=lambda bot_id: client,
    )


def test_clear_history_uses_cached_client_for_messages_and_memories() -> None:
    client = FakeMemoryClient(message_count=4, memories_cleared=True)
    service = fake_service(client)
    dependencies.set_service(service)

    response = history.clear_history(bot_id="nova")

    assert response.success is True
    assert client.calls == ["clear_messages", "clear_memories"]
    assert ("model", "nova", "user") not in service._llm_bawt_cache
    assert ("model", "other", "user") in service._llm_bawt_cache


def test_direct_clear_reports_incomplete_memory_clear() -> None:
    client = FakeMemoryClient(message_count=0, memories_cleared=False)

    assert history._clear_history_direct(client, "nova") is False
    assert client.calls == ["clear_messages", "clear_memories"]


def test_operator_clear_failure_does_not_fall_back_to_one_user_or_load_model() -> None:
    client = FakeMemoryClient(memories_cleared=False)
    service = fake_service(client)
    service._available_models = ["synthetic"]
    service._get_llm_bawt = lambda *_a: pytest.fail("must not load a model or report a user-only clear as bot-wide")
    dependencies.set_service(service)
    with pytest.raises(HTTPException) as exc_info:
        history.clear_history(bot_id="nova")
    assert exc_info.value.status_code == 500
    assert ("model", "nova", "user") in service._llm_bawt_cache


def test_delete_message_uses_cached_client_and_evicts_bot_cache() -> None:
    client = FakeMemoryClient(forgotten=True)
    service = fake_service(client)
    dependencies.set_service(service)

    response = history.delete_message("message-123", bot_id="nova")

    assert response.success is True
    assert response.deleted_count == 1
    assert client.calls == [("ignore_message_by_id", "message-123")]
    assert ("model", "nova", "user") not in service._llm_bawt_cache
    assert ("model", "other", "user") in service._llm_bawt_cache


def test_delete_message_preserves_not_found_behavior() -> None:
    client = FakeMemoryClient(forgotten=False)
    dependencies.set_service(fake_service(client))

    with pytest.raises(HTTPException) as exc_info:
        history.delete_message("missing-message", bot_id="nova")

    assert exc_info.value.status_code == 404
    assert client.calls == [("ignore_message_by_id", "missing-message")]
