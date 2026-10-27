"""Read a stored connection secret for a trusted consumer (TASK-1476).

One place answers "where does secret X live": the encrypted CredentialStore.
``provider`` connections are ``provider_connection:<id>`` rows,
``integration`` connections are ``integration_connection:<id>`` rows.

:func:`read_connection` returns only the fields named in a :class:`FieldSpec`,
never the whole record. A record that exists but can't be used (not
connected, undecryptable or missing a required field) raises
:class:`ConnectionUnavailable`, so a caller never falls back to a legacy env
value for it. Only an absent record returns ``None``.

:func:`resolve_integration` is the one record-first, env-fallback rule for
credential integrations whose readers still accept legacy settings.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sqlmodel import Session, select

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.credential_integrations import CREDENTIAL_INTEGRATIONS, CredentialIntegration
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


def _integration_spec(definition: CredentialIntegration) -> FieldSpec:
    """The read contract for a credential integration, from its own fields."""
    return FieldSpec(
        secret=definition.secret_names,
        meta=definition.required_meta,
        meta_any=definition.requires_one_of,
        meta_optional=definition.optional_meta,
    )


def integration_spec(integration_id: str) -> FieldSpec:
    """The read contract for one credential integration, for in-process readers."""
    return _integration_spec(CREDENTIAL_INTEGRATIONS[integration_id])


# The connections BawtHub's server may read through
# GET /v1/internal/connections/{kind}/{id}. Field names are a frozen contract
# with BawtHub (TASK-1476 step "Contract: BawtHub reads secrets ..."). The
# integration entries come from the same definitions the credentials form
# writes, so the writer and this reader can't drift apart.
BAWTHUB_READABLE: dict[tuple[str, str], FieldSpec] = {
    ("provider", "xai"): FieldSpec(secret=("api_key",)),
    **{("integration", d.id): _integration_spec(d) for d in CREDENTIAL_INTEGRATIONS.values()
       if d.bawthub_readable},
}


# The connections an agent bridge may read through
# GET /v1/internal/bridge/connections/{kind}/{id} (X-Bridge-Token broker gate).
# Only credentials a bridge process itself presents belong here.
BRIDGE_READABLE: dict[tuple[str, str], FieldSpec] = {
    ("integration", "openclaw"): integration_spec("openclaw"),
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


#: Integrations whose legacy-settings warning was already logged, one per process.
_legacy_warned: set[str] = set()


def has_connection_store(config) -> bool:
    """Whether ``config`` names a database, so a stored record can exist.

    Without database credentials (a CLI-only install) the legacy settings are
    the only source. A configured but unreachable database is a store outage
    and stays fail-closed. A duck-typed config without the field counts as no
    database.
    """
    from llm_bawt.utils.config import has_database_credentials  # noqa: PLC0415 — import cycle

    return hasattr(config, "POSTGRES_PASSWORD") and has_database_credentials(config)


def resolve_integration(config, integration_id: str, legacy: Mapping[str, str | None], *,
                        legacy_names: Sequence[str]) -> ConnectionSecret | None:
    """One credential integration's fields: the stored record, else its legacy settings.

    ``legacy`` maps the integration's field names to the deprecated settings'
    values. They apply only when no record exists and every required field
    has one, with one warning per process naming ``legacy_names``, never a
    value. A record that exists but can't be used raises
    :class:`ConnectionUnavailable`, so a disconnect never brings env back.
    Resolve per operation, so a saved or rotated credential applies without a
    restart.
    """
    spec = integration_spec(integration_id)
    if has_connection_store(config):
        found = read_connection(config, "integration", integration_id, spec)
        if found is not None:
            return found
    values = {name: value.strip() for name, value in legacy.items() if _present(value)}
    if not all(name in values for name in (*spec.secret, *spec.meta)):
        return None
    if spec.meta_any and not any(name in values for name in spec.meta_any):
        return None
    if integration_id not in _legacy_warned:
        _legacy_warned.add(integration_id)
        logger.warning(
            "%s credentials read from deprecated settings %s because no connection is stored; "
            "save them under Integrations instead",
            CREDENTIAL_INTEGRATIONS[integration_id].label, "/".join(legacy_names),
        )
    return ConnectionSecret(
        kind="integration",
        id=integration_id,
        meta={name: values[name] for name in (*spec.meta, *spec.meta_optional) if name in values},
        secret={name: values[name] for name in spec.secret},
    )


__all__ = [
    "BAWTHUB_READABLE",
    "BRIDGE_READABLE",
    "ConnectionSecret",
    "ConnectionUnavailable",
    "FieldSpec",
    "KINDS",
    "has_connection_store",
    "integration_spec",
    "read_connection",
    "resolve_integration",
]
