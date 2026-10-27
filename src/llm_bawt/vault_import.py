"""One-shot import of the retired pgcrypto vault into the CredentialStore (TASK-1477).

Older tenants kept provider API keys in ``vault.secrets`` (pgcrypto, one
``BAWTHUB_MASTER_KEY``). The product now has one root (``LLM_BAWT_SECRET_KEY``)
and one encrypted store, so this command copies what the vault holds into
``provider_connection:<id>`` records and leaves the vault itself untouched:

* only the names in :data:`MAPPING` are ever decrypted or imported. Anything
  else is reported ``unmapped`` and stays in the vault. Adding a mapping is a
  code change with a test, never data;
* the write is :meth:`CredentialStore.create_if_absent`. A destination that
  has any row (connected, or an operator's disconnected one) is
  ``skipped_exists`` and never overwritten;
* the insert is read back through the same transaction and its key compared
  (by SHA-256, in process) with the vault plaintext before ``imported`` is
  recorded. A mismatch rolls the insert back;
* every write for one destination (ledger rows, the insert and its
  read-back) happens in one transaction under a per-destination advisory
  lock, after re-reading the ledger, the destination and the vault
  ciphertext there. A decision made from a stale snapshot of any of them
  writes nothing (a changed source reports ``failed:source_changed`` without
  touching the ledger), so a concurrent importer can't bring back a key, old
  or rotated, that the operator disconnected after another import;
* nothing is imported unless the active encryption key is the persisted root
  (env or key file): a record encrypted under a key that exists only in this
  process would be unreadable by the next one (``failed:ephemeral_root``);
* progress is durable per name in ``vault_import_ledger``. ``failed`` retries
  on the next run. A ``conflict`` (two names for one destination holding
  different keys) is re-evaluated while the destination is still absent, so
  removing one of them from the vault resolves it. ``imported``,
  ``skipped_exists`` and ``unmapped`` are final unless the vault ciphertext
  changed AND the destination is absent, so an operator who disconnects an
  imported provider never gets the old key back from a rerun.

The master key is read from a file (``--master-key-file`` or
``BAWTHUB_MASTER_KEY_FILE``) and reaches SQL only as a bound parameter. Output,
the ledger and logs carry names, destinations and outcome codes, never
values; database errors are reported by type only because their text can
include bound parameters.

Run inside the app image: ``python -m llm_bawt.vault_import``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, select, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine

from llm_bawt.service.providers.base import AUTH_API_KEY, STATUS_CONNECTED, ConnectionRecord, CredentialStore
from llm_bawt.utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

MASTER_KEY_FILE_ENV = "BAWTHUB_MASTER_KEY_FILE"
SECRET_FIELD = "api_key"
SOURCE_META = {"source": "vault-import"}

#: Vault name -> CredentialStore provider id. Closed by design (see module doc).
MAPPING: dict[str, str] = {
    "ANTHROPIC_API_KEY": "anthropic-api",
    "OPENAI_API_KEY": "openai-api",
    "XAI_API_KEY": "xai",
    "LLM_BAWT_XAI_API_KEY": "xai",
}

IMPORTED = "imported"
SKIPPED_EXISTS = "skipped_exists"
UNMAPPED = "unmapped"
CONFLICT = "conflict"
FAILED = "failed"
FINAL = frozenset({IMPORTED, SKIPPED_EXISTS, UNMAPPED})

_metadata = MetaData()
ledger = Table(
    "vault_import_ledger", _metadata,
    Column("name", String(128), primary_key=True),
    Column("vault_updated_at", DateTime(timezone=True), nullable=True),
    Column("source_sha256", String(64), nullable=False),
    Column("destination", String(128), nullable=True),
    Column("outcome", String(32), nullable=False),
    Column("detail_code", String(64), nullable=True),
    Column("at", DateTime(timezone=True), nullable=False),
)
_DIALECTS = {"postgresql": postgresql, "sqlite": sqlite}


class _VerifyMismatch(Exception):
    """The inserted record did not read back as the vault plaintext."""


@dataclass(frozen=True)
class VaultRow:
    name: str
    updated_at: datetime | None
    source_sha256: str  # of the ciphertext bytes, never the plaintext


@dataclass(frozen=True)
class Outcome:
    """One name's result. Public: names and codes only."""

    name: str
    destination: str | None
    outcome: str
    detail_code: str | None = None
    processed: bool = True

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "destination": self.destination, "outcome": self.outcome,
                "detail_code": self.detail_code, "processed": self.processed}


@dataclass
class _Group:
    destination: str
    rows: list[VaultRow] = field(default_factory=list)


def _sha256(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    return hashlib.sha256(data).hexdigest()


def _error_code(error: BaseException) -> str:
    # Never str(error): a DBAPI error's text includes its bound parameters.
    return f"error:{type(error).__name__}"[:64]


class VaultImporter:
    """Copies mapped vault names into the CredentialStore, never overwriting."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine, store: CredentialStore, master_key: str | None):
        self.engine = engine
        self.store = store
        self._master_key = master_key

    # --- vault reads ---------------------------------------------------------
    def vault_present(self) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT to_regclass('vault.secrets')")).scalar() is not None

    def _list_vault(self) -> list[VaultRow]:
        with self.engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            rows = conn.execute(text(
                "SELECT name, updated_at, ciphertext FROM vault.secrets ORDER BY name")).all()
        return [VaultRow(name, updated_at, _sha256(ciphertext)) for name, updated_at, ciphertext in rows]

    def _decrypt(self, row: VaultRow) -> str:
        """The stripped plaintext for ``row``, read-only. Raises on any failure."""
        if not self._master_key:
            raise LookupError("no_master_key")
        with self.engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            found = conn.execute(
                text("SELECT ciphertext, pgp_sym_decrypt(ciphertext, :key) FROM vault.secrets WHERE name = :name"),
                {"key": self._master_key, "name": row.name},
            ).first()
        if found is None:
            raise LookupError("vanished")
        ciphertext, plaintext = found
        if _sha256(ciphertext) != row.source_sha256:
            raise LookupError("changed_during_import")
        value = (plaintext or "").strip()
        if not value:
            raise LookupError("empty_value")
        return value

    # --- ledger --------------------------------------------------------------
    def _ledger(self) -> dict[str, Any]:
        self._guard.run(self.engine, "vault-import-ledger-v1", lambda conn: _metadata.create_all(conn))
        with self.engine.connect() as conn:
            return {row.name: row for row in conn.execute(select(ledger)).all()}

    @staticmethod
    def _record(conn, row: VaultRow, destination: str | None, outcome: str, detail: str | None) -> None:
        values = {"vault_updated_at": row.updated_at, "source_sha256": row.source_sha256,
                  "destination": destination, "outcome": outcome, "detail_code": detail,
                  "at": datetime.now(timezone.utc)}
        statement = _DIALECTS[conn.dialect.name].insert(ledger).values(name=row.name, **values)
        conn.execute(statement.on_conflict_do_update(index_elements=[ledger.c.name], set_=values))

    def _record_all(self, rows: Iterable[VaultRow], destination: str | None, outcome: str,
                    detail: str | None, conn=None) -> list[Outcome]:
        rows = list(rows)
        if conn is None:
            with self.engine.begin() as own:
                for row in rows:
                    self._record(own, row, destination, outcome, detail)
        else:
            for row in rows:
                self._record(conn, row, destination, outcome, detail)
        return [Outcome(row.name, destination, outcome, detail) for row in rows]

    # --- decisions -----------------------------------------------------------
    def _due(self, row: VaultRow, previous, destination_exists: bool) -> bool:
        if previous is None or previous.outcome == FAILED:
            return True
        if previous.outcome == CONFLICT:
            return not destination_exists
        return previous.source_sha256 != row.source_sha256 and not destination_exists

    def run(self) -> list[Outcome]:
        if not self.vault_present():
            return []
        rows = self._list_vault()
        seen = self._ledger()
        # Recording "imported" under a key the next process can't read would
        # strand the credential, so mapped names fail until the root persists.
        root_problem = self.store.durable_key_problem()
        results: list[Outcome] = []
        groups: dict[str, _Group] = {}
        for row in rows:
            destination = MAPPING.get(row.name)
            if destination is None:
                results.extend(self._unmapped(row, seen.get(row.name)))
            else:
                groups.setdefault(destination, _Group(destination)).rows.append(row)
        for group in groups.values():
            results.extend(self._import_group(group, seen, root_problem))
        return sorted(results, key=lambda outcome: outcome.name)

    def _unmapped(self, row: VaultRow, previous) -> list[Outcome]:
        if previous is not None and previous.outcome == UNMAPPED and previous.source_sha256 == row.source_sha256:
            return [Outcome(row.name, None, UNMAPPED, previous.detail_code, processed=False)]
        return self._record_all([row], None, UNMAPPED, None)

    def _split(self, group: _Group, seen: dict[str, Any], exists: bool) -> tuple[list[VaultRow], list[Outcome]]:
        """``(due rows, settled outcomes)`` for ``group`` against ledger ``seen``."""
        due = [row for row in group.rows if self._due(row, seen.get(row.name), exists)]
        settled = [Outcome(row.name, group.destination, seen[row.name].outcome, seen[row.name].detail_code,
                           processed=False) for row in group.rows if row not in due]
        return due, settled

    def _import_group(self, group: _Group, seen: dict[str, Any], root_problem: str | None) -> list[Outcome]:
        """Decide outside the fence (decryption is slow), then commit inside it.

        ``seen`` and the existence check here are only a pre-filter: another
        importer may have finished since they were read. :meth:`_commit`
        re-reads both under the destination lock and writes only what is
        still due.
        """
        with self.engine.connect() as conn:
            exists = self.store.exists_in(conn, group.destination)
        due, settled = self._split(group, seen, exists)
        if not due:
            return settled
        if exists:
            return self._commit(group, SKIPPED_EXISTS, None)
        if root_problem:
            return self._commit(group, FAILED, root_problem)
        values: dict[str, str] = {}
        for row in group.rows:
            try:
                values[row.name] = self._decrypt(row)
            except LookupError as error:
                return self._commit(group, FAILED, str(error.args[0]))
            except Exception as error:  # noqa: BLE001 - reported by type only
                return self._commit(group, FAILED, _error_code(error))
        if len(set(values.values())) > 1:
            return self._commit(group, CONFLICT, "names_disagree")
        return self._commit(group, IMPORTED, None, next(iter(values.values())))

    def _fence(self, conn, destination: str) -> None:
        """Serialize every importer's writes for ``destination`` until commit."""
        if conn.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
                         {"target": f"llm-bawt/vault-import/{destination}"})

    @staticmethod
    def _source_changed(conn, group: _Group) -> bool:
        """True when any of ``group``'s vault rows changed or vanished since listing."""
        names = [row.name for row in group.rows]
        live = {name: _sha256(ciphertext) for name, ciphertext in conn.execute(
            text("SELECT name, ciphertext FROM vault.secrets WHERE name = ANY(:names)"), {"names": names})}
        return any(live.get(row.name) != row.source_sha256 for row in group.rows)

    def _commit(self, group: _Group, outcome: str, detail: str | None, value: str | None = None,
                *, retry: bool = True) -> list[Outcome]:
        """Re-check eligibility under the destination fence, then write ``outcome``.

        One transaction holds the fence from the fresh ledger read through the
        insert, its read-back and the ledger write, so a decision made from a
        stale snapshot (another importer imported, then the operator
        disconnected) writes nothing.
        """
        destination = group.destination
        try:
            with self.engine.begin() as conn:
                self._fence(conn, destination)
                names = [row.name for row in group.rows]
                if self._source_changed(conn, group):
                    # Decided from ciphertext the vault no longer holds. Another
                    # importer may already own the newer source; write nothing
                    # and let the next run decide from the current one.
                    return [Outcome(name, destination, FAILED, "source_changed", processed=False)
                            for name in names]
                fresh = {row.name: row for row in conn.execute(select(ledger).where(ledger.c.name.in_(names)))}
                exists = self.store.exists_in(conn, destination)
                due, settled = self._split(group, fresh, exists)
                if not due:
                    return settled
                if exists:
                    detail = "created_concurrently" if outcome == IMPORTED else detail
                    return settled + self._record_all(due, destination, SKIPPED_EXISTS, detail, conn)
                if outcome != IMPORTED:
                    return settled + self._record_all(due, destination, outcome, detail, conn)
                record = ConnectionRecord(provider=destination, status=STATUS_CONNECTED, auth_method=AUTH_API_KEY,
                                          meta=dict(SOURCE_META), secret={SECRET_FIELD: value})
                if not self.store.create_if_absent(record, conn=conn):
                    # A writer outside the fence (a connect card) got there first.
                    return settled + self._record_all(due, destination, SKIPPED_EXISTS, "created_concurrently", conn)
                stored = self.store.load_in(conn, destination)
                if (stored is None or stored.status != STATUS_CONNECTED
                        or _sha256(str((stored.secret or {}).get(SECRET_FIELD) or "")) != _sha256(value or "")):
                    raise _VerifyMismatch
                return settled + self._record_all(due, destination, IMPORTED, None, conn)
        except _VerifyMismatch:
            failure = "verify_mismatch"
        except Exception as error:  # noqa: BLE001 - reported by type only
            if not retry:
                raise
            failure = _error_code(error)
        return self._commit(group, FAILED, failure, retry=False)


def read_master_key(path: str | None) -> str | None:
    """The key file's content, or None when no file is configured or present."""
    if not path:
        return None
    try:
        key = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return key or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_bawt.vault_import",
                                     description="Import mapped pgcrypto vault secrets into the credential store.")
    parser.add_argument("--master-key-file", default=os.environ.get(MASTER_KEY_FILE_ENV),
                        help=f"file holding the vault master key (default: ${MASTER_KEY_FILE_ENV})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    config = Config()
    engine = get_shared_engine(config)
    store = CredentialStore(config)
    if engine is None or not store.available:
        print(json.dumps({"error": "database_unavailable"}))
        return 2
    try:
        master_key = read_master_key(args.master_key_file)
    except OSError as error:
        print(json.dumps({"error": "master_key_unreadable", "detail_code": _error_code(error)}))
        return 2
    importer = VaultImporter(engine, store, master_key)
    if not importer.vault_present():
        print(json.dumps({"vault": "absent"}))
        return 0
    outcomes = importer.run()
    for outcome in outcomes:
        print(json.dumps(outcome.public()))
    return 1 if any(outcome.outcome == FAILED for outcome in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
