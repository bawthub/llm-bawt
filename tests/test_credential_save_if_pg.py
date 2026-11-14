"""TASK-2267: CredentialStore.save_if is an atomic compare-and-save on a private PostgreSQL."""

from __future__ import annotations

import threading

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_CONNECTED, ConnectionRecord, CredentialStore
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

PROVIDER = "save-if-fixture"


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    config = require_test_db_config()
    engine = get_shared_engine(config)
    assert engine is not None
    yield CredentialStore(config)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM runtime_settings WHERE key = :k"), {"k": f"provider_connection:{PROVIDER}"})


def record(token: str) -> ConnectionRecord:
    return ConnectionRecord(provider=PROVIDER, status=STATUS_CONNECTED, secret={"t": token})


def token_is(expected):
    return lambda current: current is not None and current.secret.get("t") == expected


def test_save_if_saves_only_when_row_matches(store):
    store.save(record("r0"))

    assert store.save_if(record("r1"), token_is("other")) is False
    assert store.load(PROVIDER).secret["t"] == "r0"
    assert store.save_if(record("r1"), token_is("r0")) is True
    assert store.load(PROVIDER).secret["t"] == "r1"


def test_save_if_on_deleted_row_does_not_resurrect(store):
    store.save(record("r0"))
    store.delete(PROVIDER)

    assert store.save_if(record("r1"), token_is("r0")) is False
    assert store.load(PROVIDER) is None


def test_concurrent_reconnect_waits_for_locked_compare_and_wins(store):
    store.save(record("r0"))
    checked, release = threading.Event(), threading.Event()

    def matches(current):
        checked.set()
        assert release.wait(10)
        return token_is("r0")(current)

    saver = threading.Thread(target=lambda: store.save_if(record("r1"), matches))
    saver.start()
    assert checked.wait(10)  # row is now locked by save_if
    reconnect = threading.Thread(target=lambda: store.save(record("reconnected")))
    reconnect.start()
    reconnect.join(0.5)
    assert reconnect.is_alive()  # blocked behind the row lock
    release.set()
    saver.join(10)
    reconnect.join(10)

    assert store.load(PROVIDER).secret["t"] == "reconnected"
