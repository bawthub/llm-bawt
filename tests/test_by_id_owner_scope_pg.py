"""TASK-2279: by-ID forget, restore and memory delete honor a pinned owner.

Without ``user_id`` these stay the operator's bot-wide operations.
"""

from __future__ import annotations

import time

import pytest

from tests.private_postgres import private_app_pg  # noqa: F401  (pytest fixture)
from tests.test_forget_owner_scope_pg import (  # noqa: F401  (fixtures)
    _canonical,
    _memories,
    _memory,
    _msg,
    _session,
    backend,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def pair(backend):  # noqa: F811
    now = time.time()
    a, b = _session(backend, "alice"), _session(backend, "bob")
    alice = [_msg(backend, a, now - 20 + i) for i in range(2)]
    bob = [_msg(backend, b, now - 10 + i) for i in range(2)]
    return {"alice": alice, "bob": bob}


def test_forget_by_id_refuses_another_users_message(backend, pair):  # noqa: F811
    bob_msg = pair["bob"][0]
    assert backend.ignore_message_by_id(bob_msg, user_id="alice") is False
    assert backend.ignore_message_by_id(bob_msg[:8], user_id="alice") is False
    assert bob_msg in _canonical(backend)

    assert backend.ignore_message_by_id(pair["alice"][0][:8], user_id="alice") is True
    assert pair["alice"][0] not in _canonical(backend)
    # Operator path is unchanged.
    assert backend.ignore_message_by_id(bob_msg) is True


def test_restore_is_owner_scoped(backend, pair):  # noqa: F811
    for mid in (pair["alice"][0], pair["bob"][0]):
        assert backend.ignore_message_by_id(mid) is True

    assert backend.restore_ignored_messages(user_id="alice") == 1
    live = _canonical(backend)
    assert pair["alice"][0] in live and pair["bob"][0] not in live

    assert backend.restore_ignored_messages() == 1
    assert pair["bob"][0] in _canonical(backend)


def test_memory_delete_needs_every_source_owned(backend, pair):  # noqa: F811
    own = _memory(backend, [pair["alice"][0]])
    mixed = _memory(backend, [pair["alice"][1], pair["bob"][0]])
    bobs = _memory(backend, [pair["bob"][1]])
    sourceless = _memory(backend, [])

    for mem in (mixed, bobs, sourceless):
        assert backend.delete_memory(mem, user_id="alice") is False
    assert backend.delete_memory(own[:8], user_id="alice") is True
    assert _memories(backend) == {mixed, bobs, sourceless}

    # A forgotten source still counts as the owner's.
    assert backend.ignore_message_by_id(pair["bob"][1]) is True
    assert backend.delete_memory(bobs, user_id="bob") is True
    # Operator path is unchanged.
    assert backend.delete_memory(sourceless) is True
