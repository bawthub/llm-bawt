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

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from sqlmodel import Session, select

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_CONNECTED, CredentialStore

logger = logging.getLogger(__name__)

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


def _read_row(store: CredentialStore, connection_id: str) -> str | None:
    """The raw JSON stored for one connection, or ``None`` when no row exists.

    Reads the exact row rather than going through ``CredentialStore.load``:
    ``load`` and ``get_scope_settings`` treat a row that isn't a JSON object
    (or isn't JSON at all) as absent, and absent is the one answer that lets
    a consumer fall back to env.
    """
    try:
        with Session(store._store.engine) as session:
            row = session.exec(
                select(RuntimeSetting).where(
                    RuntimeSetting.scope_type == "global",
                    RuntimeSetting.scope_id == "*",
                    RuntimeSetting.key == store._key(connection_id),
                )
            ).first()
    except Exception:  # noqa: BLE001 - any read failure means the store is down
        logger.warning("connection store read failed for %s", connection_id)
        raise ConnectionUnavailable("store_unavailable") from None
    return None if row is None else row.value_json


def _decode(value_json: str) -> tuple[dict, dict, dict]:
    """``(raw, meta, secret)`` from a stored row; ``unreadable`` when malformed."""
    try:
        raw = json.loads(value_json)
    except Exception:  # noqa: BLE001
        raise ConnectionUnavailable("unreadable") from None
    if not isinstance(raw, dict):
        raise ConnectionUnavailable("unreadable")
    meta = raw.get("meta") or {}
    secret: object = {}
    enc = raw.get("secret_enc")
    if enc:
        try:
            secret = json.loads(crypto.decrypt(enc))
        except Exception:  # noqa: BLE001 - never log the exception: it may echo input
            raise ConnectionUnavailable("unreadable") from None
    if not isinstance(meta, dict) or not isinstance(secret, dict):
        raise ConnectionUnavailable("unreadable")
    return raw, meta, secret


def read_connection(config, kind: str, connection_id: str, spec: FieldSpec,
                    *, store: CredentialStore | None = None) -> ConnectionSecret | None:
    """The fields ``spec`` names, or ``None`` when no row exists."""
    try:
        store = store if store is not None else _store(config, kind)
        available = store.available
    except Exception:  # noqa: BLE001
        raise ConnectionUnavailable("store_unavailable") from None
    if not available:
        raise ConnectionUnavailable("store_unavailable")
    value_json = _read_row(store, connection_id)
    if value_json is None:
        return None
    raw, meta, secret = _decode(value_json)
    if raw.get("status") != STATUS_CONNECTED:
        raise ConnectionUnavailable("not_connected")
    # A missing or blank required field is "unreadable", never "absent".
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
