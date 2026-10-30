"""TASK-1774 r1 Blockers: history must never cross users outside the DB filter.

B1: a DB-backed HistoryManager whose read fails must not fall back to a
bot-wide file, and file mode keeps one file per (bot, user).
B2: the /v1/history/context-seed preview builds the seed for the requesting
user, never DEFAULT_USER.
"""

import json
from types import SimpleNamespace

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service.routes import history_seed
from llm_bawt.utils.history import HistoryManager


class _Console:
    def print(self, *_a, **_k) -> None:
        pass


def _manager(tmp_path, user_id, db_backend=None) -> HistoryManager:
    config = SimpleNamespace(HISTORY_FILE=str(tmp_path / "history.json"))
    client = SimpleNamespace(console=_Console())
    return HistoryManager(
        client, config, db_backend=db_backend, bot_id="dash", user_id=user_id,
    )


class _FailingDb:
    def get_messages(self, **_kw):
        raise RuntimeError("db down")


def test_db_read_failure_does_not_fall_back_to_shared_file(tmp_path) -> None:
    # Another user's history sitting in every candidate file name.
    alice = _manager(tmp_path, "alice")
    alice.messages = [Message(role="user", content="alice secret", timestamp=1.0)]
    alice.save_history()
    bot_wide = tmp_path / "history_dash.json"
    bot_wide.write_text(json.dumps([m.to_dict() for m in alice.messages]))

    bob = _manager(tmp_path, "bob", db_backend=_FailingDb())
    bob.load_history()

    assert bob.messages == []


def test_file_mode_keeps_one_file_per_user(tmp_path) -> None:
    alice = _manager(tmp_path, "alice")
    alice.messages = [Message(role="user", content="alice secret", timestamp=1.0)]
    alice.save_history()

    bob = _manager(tmp_path, "bob")
    assert bob.history_file != alice.history_file
    bob.load_history()
    assert bob.messages == []

    alice_again = _manager(tmp_path, "alice")
    alice_again.load_history()
    assert [m.content for m in alice_again.messages] == ["alice secret"]


@pytest.mark.parametrize("first,second", [("a@b", "a_b"), ("a/b", "a_b"), ("a%40b", "a@b")])
def test_distinct_users_never_share_a_file(tmp_path, first, second) -> None:
    owner = _manager(tmp_path, first)
    owner.messages = [Message(role="user", content="first secret", timestamp=1.0)]
    owner.save_history()

    other = _manager(tmp_path, second)
    assert other.history_file != owner.history_file
    other.load_history()
    assert other.messages == []


@pytest.mark.parametrize("user", ["../../etc/x", "..", "a/../../b", "x\x00y", "/abs"])
def test_file_name_cannot_escape_the_history_dir(tmp_path, user) -> None:
    from pathlib import Path

    m = _manager(tmp_path, user)
    path = Path(m.history_file)
    assert path.parent == tmp_path
    assert "\x00" not in path.name and "/" not in path.name


@pytest.mark.parametrize("user", ["nick", "alice-2", "voicetest_loopy", "a.b"])
def test_plain_ids_keep_a_readable_name(tmp_path, user) -> None:
    assert _manager(tmp_path, user).history_file.endswith(f"history_dash__{user}.json")


class _Stop(Exception):
    pass


class _Service:
    _default_bot = "dash"
    config = SimpleNamespace(DEFAULT_USER="nick")

    def __init__(self) -> None:
        self.users: list[str] = []

    def _resolve_request_model(self, model, bot_id, local_mode=False):
        return "m", None

    def _get_llm_bawt(self, model_alias, bot_id, user_id):
        self.users.append(user_id)
        raise _Stop


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_context_seed_is_built_for_the_requesting_user(user) -> None:
    service = _Service()
    with pytest.raises(_Stop):
        history_seed.build_context_seed("dash", None, service, user_id=user)
    assert service.users == [user]


def test_context_seed_without_user_fails_closed() -> None:
    service = _Service()
    with pytest.raises(ValueError):
        history_seed.build_context_seed("dash", None, service)
    assert service.users == []


def test_context_seed_route_requires_user_and_passes_it(monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    seen = {}

    def fake_build(bot_id, model, service, **kw):
        seen.update(kw)
        return {"messages": []}

    monkeypatch.setattr(history_seed, "get_service", lambda: _Service())
    monkeypatch.setattr(history_seed, "build_context_seed", fake_build)
    app = FastAPI()
    app.include_router(history_seed.router)
    client = TestClient(app)

    assert client.get("/v1/history/context-seed?bot_id=dash").status_code == 422
    assert seen == {}
    resp = client.get("/v1/history/context-seed?bot_id=dash&user_id=bob")
    assert resp.status_code == 200
    assert seen["user_id"] == "bob"
