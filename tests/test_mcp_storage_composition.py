"""Storage facade contracts after the responsibility split (TASK-1222).

These are extraction/dispatch tests, not proof of protected-bot enforcement.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.mcp_server.storage import Memory, MemoryStorage, Message
from llm_bawt.mcp_server.storage_admin import StorageAdminMixin
from llm_bawt.mcp_server.storage_search import StorageSearchMixin
from llm_bawt.mcp_server.storage_sessions import StorageSessionsMixin


def test_facade_exposes_each_mixin_method_without_shadowing():
    """Removing an inheritance edge must break the public facade contract."""
    seen = set()
    for mixin in (StorageAdminMixin, StorageSearchMixin, StorageSessionsMixin):
        for name, method in vars(mixin).items():
            if not callable(method):
                continue
            assert name not in seen
            seen.add(name)
            assert getattr(MemoryStorage, name) is method


def test_result_types_keep_their_public_import_and_serialization():
    assert Memory(id="memory", content="fact").to_dict()["tags"] == ["misc"]
    result = Message(id="message", role="user", content="hello", timestamp=1.0)
    assert result.to_dict() == {
        "id": "message", "role": "user", "content": "hello",
        "timestamp": 1.0, "session_id": None, "author": None,
    }


def test_session_forwarding_preserves_read_and_mutation_provisioning():
    storage = object.__new__(MemoryStorage)
    manager = Mock()
    manager.get_session.return_value = {"id": "session"}
    manager.close_session.return_value = True
    storage.get_short_term_manager = Mock(return_value=manager)

    assert asyncio.run(storage.get_session("session", bot_id="public_fixture")) == {"id": "session"}
    storage.get_short_term_manager.assert_called_once_with("public_fixture")
    manager.get_session.assert_called_once_with("session")
    storage.get_short_term_manager.reset_mock()
    assert asyncio.run(storage.close_session("session", bot_id="public_fixture")) is True
    storage.get_short_term_manager.assert_called_once_with("public_fixture", provision_schema=True)
    manager.close_session.assert_called_once_with("session")


def test_admin_hydration_keeps_neighbor_boundaries():
    storage = object.__new__(MemoryStorage)
    raw = {"before": [{"id": "before"}], "message": {"id": "hit"}, "after": [{"id": "after"}]}
    backend = SimpleNamespace(get_message_by_id=Mock(return_value=raw))
    storage._get_backend = Mock(return_value=backend)
    storage.hydrate_message_authors = Mock(
        side_effect=lambda rows, **kwargs: [{**row, "hydrated": True} for row in rows]
    )

    result = asyncio.run(storage.get_message_by_id("public_fixture", "hit", before=1, after=1))
    storage._get_backend.assert_called_once_with("public_fixture")
    backend.get_message_by_id.assert_called_once_with("hit", before=1, after=1)
    storage.hydrate_message_authors.assert_called_once_with(
        [{"id": "before"}, {"id": "hit"}, {"id": "after"}], bot_id="public_fixture"
    )
    assert result == {
        "before": [{"id": "before", "hydrated": True}],
        "message": {"id": "hit", "hydrated": True},
        "after": [{"id": "after", "hydrated": True}],
    }
