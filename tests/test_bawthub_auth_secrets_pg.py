"""TASK-1472: create-if-absent holds under concurrent writers on real PostgreSQL."""

from __future__ import annotations

import threading

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from llm_bawt.bawthub_auth.secret_store import BawtHubAuthSecretStore
from llm_bawt.service.providers import crypto
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

WRITERS = 12


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = get_shared_engine(require_test_db_config())
    assert engine is not None
    store = BawtHubAuthSecretStore(engine)
    yield store
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM bawthub_auth_secrets"))


def _race(write, count: int = WRITERS) -> list[tuple[str, bool]]:
    barrier = threading.Barrier(count)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        value = f"value-{index}"
        try:
            barrier.wait()
            outcome = write(value)
        except BaseException as error:  # noqa: BLE001 - surfaced below
            with lock:
                errors.append(error)
            return
        with lock:
            results.append((value, outcome))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert not errors, errors
    assert len(results) == count
    return results


def test_concurrent_session_secret_create_has_one_winner(store):
    results = _race(lambda value: store.create_if_absent("session-secret", value))
    winners = [value for value, created in results if created]
    assert len(winners) == 1
    assert store.get("session-secret") == winners[0]
    with store.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT count(*) FROM bawthub_auth_secrets WHERE name='session-secret'")).scalar() == 1


def test_concurrent_oidc_upsert_creates_once_and_keeps_a_written_value(store):
    results = _race(lambda value: store.put("oidc/fixture-idp", value))
    assert sum(created for _, created in results) == 1
    assert store.get("oidc/fixture-idp") in {value for value, _ in results}


def test_existing_session_secret_is_never_replaced(store):
    assert store.create_if_absent("session-secret", "original")
    results = _race(lambda value: store.create_if_absent("session-secret", value))
    assert not any(created for _, created in results)
    assert store.get("session-secret") == "original"
