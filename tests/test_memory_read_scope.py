"""TASK-2279 part 3: memory list/stats and profile reads narrow to a pinned owner.

The user-bound client carries TASK-2275's ownership authority (embedded) or
service principal (MCP), so the routes only have to choose the right client.
Without an owner they behave exactly as before (operator client, all profiles).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import memory as routes
from llm_bawt.service.routes import profiles as profile_routes


@pytest.fixture
def clients(monkeypatch):
    by_owner = {
        None: Mock(name="operator"),
        "alice": Mock(name="alice"),
        "bob": Mock(name="bob"),
    }
    by_owner[None].list_memories.return_value = [{"id": "a1"}, {"id": "b1"}]
    by_owner["alice"].list_memories.return_value = [{"id": "a1"}]
    by_owner["bob"].list_memories.return_value = [{"id": "b1"}]
    for owner, client in by_owner.items():
        client.stats.return_value = {"messages": {"owner": owner}, "memories": {}}
    svc = Mock()
    svc.get_memory_client.side_effect = lambda bot_id, user_id=None: by_owner[user_id]
    monkeypatch.setattr(routes, "get_service", lambda: svc)
    return svc, by_owner


def test_list_with_owner_uses_only_that_users_client(clients):
    svc, by_owner = clients
    resp = routes.list_memories(limit=5, user_id="alice", bot_id="b")
    svc.get_memory_client.assert_called_once_with("b", "alice")
    assert [r.id for r in resp.results] == ["a1"]
    by_owner["bob"].list_memories.assert_not_called()
    by_owner[None].list_memories.assert_not_called()


def test_list_without_owner_is_unchanged(clients):
    svc, _ = clients
    resp = routes.list_memories(limit=5, user_id=None, bot_id="b")
    svc.get_memory_client.assert_called_once_with("b")
    assert [r.id for r in resp.results] == ["a1", "b1"]


def test_stats_with_and_without_owner(clients):
    svc, _ = clients
    assert routes.get_memory_stats(user_id="bob", bot_id="b").messages == {"owner": "bob"}
    assert routes.get_memory_stats(user_id=None, bot_id="b").messages == {"owner": None}
    assert svc.get_memory_client.call_args_list[0].args == ("b", "bob")
    assert svc.get_memory_client.call_args_list[1].args == ("b",)


@pytest.mark.parametrize("call", [
    lambda: routes.list_memories(limit=5, user_id="  ", bot_id="b"),
    lambda: routes.get_memory_stats(user_id="", bot_id="b"),
])
def test_blank_owner_is_refused_not_widened(clients, call):
    svc, _ = clients
    with pytest.raises(HTTPException) as err:
        call()
    assert err.value.status_code == 400
    svc.get_memory_client.assert_not_called()


class _Profiles:
    def __init__(self):
        self.detail_calls: list[str] = []

    def list_profiles(self, entity_type):
        return [SimpleNamespace(entity_id=e) for e in ("nick", "Alice", "bob")]

    def get_profile(self, _t, entity_id):
        self.detail_calls.append(entity_id)
        return SimpleNamespace(
            email=None, display_name=entity_id, description=None, color=None,
            avatar=None, avatar_render=None, is_service=False, summary=None,
            summary_updated_at=None, created_at=None)

    def get_all_attributes(self, _t, _e):
        return []


@pytest.fixture
def profiles(monkeypatch):
    manager = _Profiles()
    monkeypatch.setattr(profile_routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(profile_routes, "get_profile_manager", lambda _c: manager)
    return manager


def test_profiles_filtered_by_entity_id(profiles):
    resp = profile_routes.list_profiles(entity_type="user", entity_id="alice")
    assert resp.total_count == 1
    assert profiles.detail_calls == ["Alice"]


def test_profiles_untyped_filter_and_unfiltered(profiles):
    assert profile_routes.list_profiles(entity_id="bob").total_count == 2  # user + bot rows
    assert profiles.detail_calls == ["bob", "bob"]
    assert profile_routes.list_profiles(entity_type="user").total_count == 3


def test_profiles_blank_entity_id_refused(profiles):
    with pytest.raises(HTTPException) as err:
        profile_routes.list_profiles(entity_type="user", entity_id=" ")
    assert err.value.status_code == 400
    assert profiles.detail_calls == []
