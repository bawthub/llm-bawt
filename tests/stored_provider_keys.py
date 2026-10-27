"""Stub the stored ``provider_connection:<id>`` records the key readers see (TASK-1476).

``outcomes`` maps a provider id to a key string, ``None`` (no record) or a
:class:`ConnectionUnavailable` to raise. Ids not listed read as no record.
"""

from __future__ import annotations

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
