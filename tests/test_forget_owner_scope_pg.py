"""TASK-2279: count/time forget touches only the caller's own threads.

A bot's message partition holds every user's conversations. Forgetting "the
last N" or "the last M minutes" must select within the caller's sessions (and
one thread when given), move exactly that ID set, and delete only memories
sourced from those same IDs.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.runtime_settings import BotProfileStore
from tests.private_postgres import private_app_pg  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.integration


@pytest.fixture
def backend(private_app_pg):  # noqa: F811
    slug = f"fscope-{uuid.uuid4().hex[:12]}"
    BotProfileStore(private_app_pg).upsert({"slug": slug, "name": slug, "system_prompt": "fixture"})
    return PostgreSQLMemoryBackend(private_app_pg, bot_id=slug)


def _session(backend, user_id: str) -> str:
    sid = str(uuid.uuid4())
    with backend.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sessions (id, bot_id, user_id, started_at, status) "
            "VALUES (:id, :bot, :user, NOW(), 'archived')"
        ), {"id": sid, "bot": backend.bot_id, "user": user_id})
    return sid


def _msg(backend, session_id: str, ts: float) -> str:
    mid = str(uuid.uuid4())
    backend.add_message(mid, "user", f"m-{mid[:6]}", ts, session_id=session_id, extract_memory=False)
    return mid


def _memory(backend, source_ids: list[str]) -> str:
    mem_id = str(uuid.uuid4())
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {backend._memories_table_name}
                (id, content, importance, source_message_ids, created_at)
            VALUES (:id, :content, 0.5, CAST(:src AS jsonb), NOW())
        """), {"id": mem_id, "content": f"fact {mem_id[:6]}", "src": json.dumps(source_ids)})
    return mem_id


def _canonical(backend) -> set[str]:
    with backend.engine.connect() as conn:
        return {r.id for r in conn.execute(text(f"SELECT id FROM {backend._messages_table_name}"))}


def _memories(backend) -> set[str]:
    with backend.engine.connect() as conn:
        return {r.id for r in conn.execute(text(f"SELECT id FROM {backend._memories_table_name}"))}


@pytest.fixture
def two_users(backend):
    """Alice has two threads; Bob's messages are the newest in the partition."""
    now = time.time()
    a1, a2, b1 = _session(backend, "alice"), _session(backend, "alice"), _session(backend, "bob")
    alice_t1 = [_msg(backend, a1, now - 50 + i) for i in range(2)]
    alice_t2 = [_msg(backend, a2, now - 40 + i) for i in range(2)]
    bob = [_msg(backend, b1, now - 10 + i) for i in range(3)]
    mems = {
        "alice_t2": _memory(backend, [alice_t2[-1]]),
        "alice_t1": _memory(backend, [alice_t1[-1]]),
        "bob": _memory(backend, [bob[-1]]),
    }
    return {"a1": a1, "a2": a2, "b1": b1, "alice_t1": alice_t1,
            "alice_t2": alice_t2, "bob": bob, "mems": mems}


def test_count_forget_stays_in_caller_thread(backend, two_users):
    d = two_users
    result = backend.forget_owned_messages(user_id="alice", session_id=d["a2"], count=5)

    assert result["message_ids"] == sorted(d["alice_t2"])
    assert result["messages_ignored"] == 2
    assert result["memories_deleted"] == 1
    assert _canonical(backend) == set(d["alice_t1"]) | set(d["bob"])
    assert _memories(backend) == {d["mems"]["alice_t1"], d["mems"]["bob"]}


def test_time_forget_never_touches_another_user(backend, two_users):
    d = two_users
    result = backend.forget_owned_messages(user_id="alice", minutes=5)

    assert result["messages_ignored"] == 4
    assert _canonical(backend) == set(d["bob"])
    assert _memories(backend) == {d["mems"]["bob"]}


def test_ignore_and_preview_are_owner_scoped(backend, two_users):
    d = two_users
    preview = backend.preview_recent_messages(2, user_id="bob")
    assert [m["id"] for m in preview] == d["bob"][-2:]
    assert backend.ignore_recent_messages(10, user_id="alice", session_id=d["a1"]) == 2
    assert backend.ignore_messages_since_minutes(5, user_id="bob") == 3
    assert _canonical(backend) == set(d["alice_t2"])
    # ignore never deletes memories; only forget does.
    assert len(_memories(backend)) == 3


def test_session_of_another_user_selects_nothing(backend, two_users):
    d = two_users
    result = backend.forget_owned_messages(user_id="alice", session_id=d["b1"], count=10)
    assert result["messages_ignored"] == 0
    assert result["memories_deleted"] == 0
    assert set(d["bob"]) <= _canonical(backend)


def test_owner_is_required(backend, two_users):
    with pytest.raises(ValueError):
        backend.forget_owned_messages(user_id="", count=1)
    with pytest.raises(TypeError):
        backend.ignore_recent_messages(1)  # type: ignore[call-arg]
