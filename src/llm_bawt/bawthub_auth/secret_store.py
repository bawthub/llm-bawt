"""Encrypted named secrets for BawtHub auth (TASK-1467 §10, TASK-1472).

BawtHub keeps its session-signing secret and OIDC client secrets here, read
through the service-token-gated ``/v1/internal/secrets/bawthub-auth/*`` API.

Values are encrypted with the provider-credential Fernet envelope
(``service.providers.crypto``). They live in their own table, not in
``runtime_settings``: the generic ``/v1/settings`` routes list and overwrite
that table without the service token, and an overwritten session secret would
let a caller forge BawtHub sessions. Reads accept only ciphertext, so a
plaintext row written behind the API's back is refused rather than trusted.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine

from llm_bawt.service.providers import crypto
from llm_bawt.utils.schema import SchemaBootstrapGuard

SESSION_SECRET = "session-secret"
_OIDC_NAME = re.compile(r"oidc/[a-z0-9][a-z0-9_-]{0,62}")
MAX_VALUE_CHARS = 8192
# ON CONFLICT DO NOTHING is the create-if-absent fence; both dialects have it.
_UPSERT_DIALECTS = {"postgresql": postgresql, "sqlite": sqlite}

_metadata = MetaData()
_secrets = Table(
    "bawthub_auth_secrets", _metadata,
    Column("name", String(80), primary_key=True),
    Column("value_enc", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class SecretUnreadable(Exception):
    """A stored value can't be decrypted (wrong key, or not ciphertext)."""


def valid_secret_name(name: str) -> bool:
    """Only ``session-secret`` and ``oidc/<providerId>`` exist."""
    return name == SESSION_SECRET or _OIDC_NAME.fullmatch(name) is not None


def is_create_only(name: str) -> bool:
    """``session-secret`` is written once (create-if-absent); OIDC ones upsert."""
    return name == SESSION_SECRET


class BawtHubAuthSecretStore:
    """Get, create-if-absent and upsert; there is deliberately no list."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        self._guard.run(engine, "bawthub-auth-secrets-v1", lambda conn: _metadata.create_all(conn))

    def get(self, name: str) -> str | None:
        _require_name(name)
        with self.engine.connect() as conn:
            stored = conn.execute(
                select(_secrets.c.value_enc).where(_secrets.c.name == name)
            ).scalar_one_or_none()
        if stored is None:
            return None
        if not crypto.is_encrypted(stored):
            raise SecretUnreadable(name)
        try:
            return crypto.decrypt(stored)
        except Exception as error:  # noqa: BLE001 - any decrypt failure is unreadable
            raise SecretUnreadable(name) from error

    def create_if_absent(self, name: str, value: str) -> bool:
        """Store ``value`` unless ``name`` exists. True when this call created it."""
        _require_name(name)
        with self.engine.begin() as conn:
            return self._insert(conn, name, value)

    def put(self, name: str, value: str) -> bool:
        """Create or replace ``name``. True when this call created it."""
        _require_name(name)
        with self.engine.begin() as conn:
            if self._insert(conn, name, value):
                return True
            conn.execute(
                update(_secrets).where(_secrets.c.name == name)
                .values(value_enc=_encrypt(value), updated_at=datetime.now(UTC))
            )
            return False

    def _insert(self, conn, name: str, value: str) -> bool:
        now = datetime.now(UTC)
        dialect = _UPSERT_DIALECTS[conn.dialect.name]
        statement = dialect.insert(_secrets).values(
            name=name, value_enc=_encrypt(value), created_at=now, updated_at=now,
        ).on_conflict_do_nothing(index_elements=[_secrets.c.name])
        return conn.execute(statement).rowcount == 1


def _require_name(name: str) -> None:
    if not valid_secret_name(name):
        raise ValueError("unknown secret name")


def _encrypt(value: str) -> str:
    if not value or len(value) > MAX_VALUE_CHARS:
        raise ValueError("secret value must be 1..%d characters" % MAX_VALUE_CHARS)
    return crypto.encrypt(value)
