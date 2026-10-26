"""Read a stored connection secret for a trusted consumer (TASK-1476).

One place answers "where does secret X live": the encrypted CredentialStore.
``provider`` connections are ``provider_connection:<id>`` rows,
``integration`` connections are ``integration_connection:<id>`` rows.

:func:`read_connection` returns only the fields named in a :class:`FieldSpec`,
never the whole record. A record that exists but can't be used (not
connected, undecryptable or missing a required field) raises
:class:`ConnectionUnavailable`, so a caller never falls back to a legacy env
value for it. Only an absent record returns ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.service.providers.base import STATUS_CONNECTED, CredentialStore

Kind = Literal["provider", "integration"]
KINDS: tuple[str, ...] = ("provider", "integration")


@dataclass(frozen=True)
class FieldSpec:
    """The fields a consumer may read from one connection.

    ``meta``/``secret`` are required; ``meta_any`` needs at least one present;
    ``meta_optional`` is returned when present.
    """

    secret: tuple[str, ...]
    meta: tuple[str, ...] = ()
    meta_any: tuple[str, ...] = ()
    meta_optional: tuple[str, ...] = field(default=())


# The connections BawtHub's server may read through
# GET /v1/internal/connections/{kind}/{id}. Field names are a frozen contract
# with BawtHub (TASK-1476 step "Contract: BawtHub reads secrets ...").
BAWTHUB_READABLE: dict[tuple[str, str], FieldSpec] = {
    ("provider", "xai"): FieldSpec(secret=("api_key",)),
    ("integration", "unraid"): FieldSpec(secret=("api_key",), meta=("url",)),
    ("integration", "object-store"): FieldSpec(
        secret=("access_key_id", "secret_access_key"), meta=("endpoint", "region", "bucket"),
    ),
    ("integration", "azure-speech"): FieldSpec(
        secret=("key",), meta_any=("region", "endpoint"), meta_optional=("region", "endpoint"),
    ),
    ("integration", "home-assistant"): FieldSpec(secret=("token",), meta=("url",)),
}


class ConnectionUnavailable(Exception):
    """A record exists (or the store is down) but can't be used.

    ``code`` is one of ``not_connected``, ``unreadable``, ``store_unavailable``.
    It never carries a value.
    """

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ConnectionSecret:
    kind: str
    id: str
    meta: dict[str, str]
    secret: dict[str, str]


def _store(config, kind: str) -> CredentialStore:
    if kind == "provider":
        return CredentialStore(config)
    if kind == "integration":
        return IntegrationStore(config)
    raise ValueError(f"unknown connection kind: {kind!r}")


def _present(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def read_connection(config, kind: str, connection_id: str, spec: FieldSpec,
                    *, store: CredentialStore | None = None) -> ConnectionSecret | None:
    """The fields ``spec`` names, or ``None`` when no record exists."""
    store = store if store is not None else _store(config, kind)
    if not store.available:
        raise ConnectionUnavailable("store_unavailable")
    record = store.load(connection_id)
    if record is None:
        return None
    if record.status != STATUS_CONNECTED:
        raise ConnectionUnavailable("not_connected")
    secret = record.secret or {}
    meta = record.meta or {}
    # An undecryptable secret loads as {} (CredentialStore.load logs it), so a
    # missing required field covers both "undecryptable" and "malformed".
    if not all(_present(secret.get(name)) for name in spec.secret):
        raise ConnectionUnavailable("unreadable")
    if not all(_present(meta.get(name)) for name in spec.meta):
        raise ConnectionUnavailable("unreadable")
    if spec.meta_any and not any(_present(meta.get(name)) for name in spec.meta_any):
        raise ConnectionUnavailable("unreadable")
    meta_out = {name: meta[name] for name in (*spec.meta, *spec.meta_optional) if _present(meta.get(name))}
    return ConnectionSecret(
        kind=kind,
        id=connection_id,
        meta=meta_out,
        secret={name: secret[name] for name in spec.secret},
    )


__all__ = [
    "BAWTHUB_READABLE",
    "ConnectionSecret",
    "ConnectionUnavailable",
    "FieldSpec",
    "KINDS",
    "read_connection",
]
