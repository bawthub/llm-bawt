"""Stub the stored ``provider_connection:<id>`` records the key readers see (TASK-1476).

``outcomes`` maps a provider id to a key string, ``None`` (no record) or a
:class:`ConnectionUnavailable` to raise. Ids not listed read as no record.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace

from llm_bawt.service.connection_secrets import ConnectionSecret


def stub_stored_keys(monkeypatch, outcomes: dict[str, object]) -> list[str]:
    """Patch the strict reader and the service config; returns the ids read."""
    reads: list[str] = []

    def fake_read(_config, kind, connection_id, _spec, **_kwargs):
        assert kind == "provider"
        reads.append(connection_id)
        outcome = outcomes.get(connection_id)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return ConnectionSecret(kind=kind, id=connection_id, meta={}, secret={"api_key": outcome})

    monkeypatch.setattr("llm_bawt.service.connection_secrets.read_connection", fake_read)
    monkeypatch.setattr(
        "llm_bawt.service.dependencies.get_service",
        lambda: SimpleNamespace(config=SimpleNamespace()),
    )
    return reads


@contextmanager
def real_credential_store(monkeypatch) -> Iterator[tuple[object, Callable[[], None]]]:
    """A real CredentialStore over SQLite with a swappable Fernet key.

    Yields ``(store, rotate_root)``; ``rotate_root()`` swaps the Fernet key so
    every stored secret becomes undecryptable.
    """
    from cryptography.fernet import Fernet
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel

    from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
    from llm_bawt.service.providers import base, crypto

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE UNIQUE INDEX idx_runtime_settings_scope_key ON runtime_settings (scope_type, scope_id, key)"
        ))
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    runtime.config = None
    monkeypatch.setattr(base, "RuntimeSettingsStore", lambda config: runtime)
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)

    def rotate_root():
        other = Fernet(Fernet.generate_key())
        monkeypatch.setattr(crypto, "_get_fernet", lambda: other)

    try:
        yield base.CredentialStore(SimpleNamespace()), rotate_root
    finally:
        engine.dispose()
