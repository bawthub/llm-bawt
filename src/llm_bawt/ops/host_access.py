"""Owner-managed host records for the host_key action (TASK-2430).

A record names one SSH destination (host, port, login user) and holds the
host keys the operator pinned: a ``scan`` job reads the offered keys, and
``confirm_host_keys`` pins those whose fingerprints the operator confirmed
(TASK-2520). Nothing is trusted on first use. Callers of the ops action pass only the
record id; ``pin`` resolves it at approval, ``check_pin`` re-resolves it at
dispatch, and ``settle`` folds a finished job back into the record.

Editing the destination clears its learned host keys: trust belongs to the
destination, not to the record id.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field as OutputField
from sqlalchemy import Column, DateTime, Text, inspect
from sqlmodel import Field, Session, SQLModel, select

from ..utils.schema import SchemaBootstrapGuard
from . import host_key

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")
_PREFIXES = ("host keys scanned: ", "host login verified: ", "bot key present on host: ", "bot key removed from host: ")
_FAILED_LOGIN = re.compile(
    r"login check failed: (?:timeout|host_key_mismatch|key_refused|dns_failed|connection_refused|unreachable|ssh_failed)"
)
_guard = SchemaBootstrapGuard()


def _utcnow():
    return datetime.now(timezone.utc)


def _iso(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class OpsHostAccess(SQLModel, table=True):
    __tablename__ = "ops_host_access"
    id: int | None = Field(default=None, primary_key=True)
    label: str = Field(index=True, unique=True, max_length=64)
    host: str = Field(max_length=253)
    port: int = 22
    login_user: str = Field(max_length=32)
    known_host_keys_json: str = Field(default="[]", sa_column=Column(Text, nullable=False))
    # unverified | scanned | trusted | verified | login_failed | key_present | key_removed | unknown
    status: str = Field(default="unverified", max_length=32)
    passwordless_sudo: bool | None = None
    last_op: str | None = Field(default=None, max_length=16)
    last_job_id: str | None = Field(default=None, max_length=64)
    last_result_json: str | None = Field(default=None, sa_column=Column(Text))
    last_checked_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_by: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))

    def to_api(self) -> dict:
        return {"id": self.id, "label": self.label, "host": self.host, "port": self.port,
                "login_user": self.login_user, "known_host_keys": json.loads(self.known_host_keys_json),
                "status": self.status, "passwordless_sudo": self.passwordless_sudo,
                "last_op": self.last_op, "last_job_id": self.last_job_id,
                "last_result": json.loads(self.last_result_json) if self.last_result_json else None,
                "last_checked_at": _iso(self.last_checked_at), "created_by": self.created_by,
                "created_at": _iso(self.created_at), "updated_at": _iso(self.updated_at)}


class HostAccessError(ValueError):
    pass


def bootstrap_host_access_schema(conn) -> None:
    """``ops_host_access`` DDL (TASK-2430). Runtime bootstrap or revision 0002."""
    SQLModel.metadata.create_all(bind=conn, tables=[OpsHostAccess.__table__])


def ensure_schema(engine) -> None:
    from ..utils.alembic_state import owns
    from .schema_revision import OPS_SCHEMA_REVISION

    if owns(engine, OPS_SCHEMA_REVISION):
        return  # TASK-631: revision 0002_ops_store owns this DDL
    if "ops_host_access" not in inspect(engine).get_table_names():
        _guard.run(engine, "ops-host-access-task2430-v1", bootstrap_host_access_schema)


def _validate_destination(fields: dict) -> dict:
    out = {}
    if "label" in fields:
        if not isinstance(fields["label"], str) or not _LABEL.fullmatch(fields["label"]):
            raise HostAccessError("label must be 1-64 letters, digits, spaces, dots, dashes or underscores")
        out["label"] = fields["label"]
    if "host" in fields:
        try:
            out["host"] = host_key.validate_host(fields["host"])
        except ValueError as exc:
            raise HostAccessError(str(exc)) from exc
    if "port" in fields:
        if type(fields["port"]) is not int or not 1 <= fields["port"] <= 65535:
            raise HostAccessError("port must be 1..65535")
        out["port"] = fields["port"]
    if "login_user" in fields:
        if not isinstance(fields["login_user"], str) or not host_key._USER.fullmatch(fields["login_user"]):
            raise HostAccessError("login_user must be a POSIX user name")
        out["login_user"] = fields["login_user"]
    return out


def list_hosts(engine) -> list[dict]:
    ensure_schema(engine)
    with Session(engine) as session:
        return [r.to_api() for r in session.exec(select(OpsHostAccess).order_by(OpsHostAccess.label)).all()]


INVENTORY_LIMIT = 25
CheckOperation = Literal["scan", "verify", "add", "remove"]


class HostInventoryEntry(BaseModel):
    label: str = OutputField(max_length=64)
    host: str = OutputField(max_length=253)
    port: int = OutputField(ge=1, le=65535)
    login_user: str = OutputField(max_length=32)
    last_checked_at: str | None = None
    last_check: CheckOperation | None = None
    login_status: Literal["registered_only", "known_failed", "verified", "unknown"]
    managed_config: Literal["recorded_installed_runtime_unknown", "installation_not_confirmed", "unknown"] = "unknown"


def read_host_inventory(engine) -> tuple[list[HostInventoryEntry], bool]:
    """Bounded bot projection of the installation registry, with no schema bootstrap.

    Worker receipts do not identify the caller's runtime or prove the current
    managed files are present there. Do not advertise an alias on that evidence.
    """
    host = OpsHostAccess
    query = select(
        host.label, host.host, host.port, host.login_user, host.status,
        host.last_checked_at, host.last_op, host.last_result_json,
    ).order_by(host.label).limit(INVENTORY_LIMIT + 1)
    with Session(engine) as session:
        rows = session.exec(query).all()

    entries = []
    for label, hostname, port, login, status, checked, op, result_json in rows[:INVENTORY_LIMIT]:
        last_check = op if op in host_key.OPS and checked is not None else None
        entry = HostInventoryEntry(
            label=label, host=hostname, port=port, login_user=login,
            last_checked_at=_iso(checked) if last_check else None,
            last_check=last_check, login_status="unknown",
        )
        if status in ("unverified", "scanned", "trusted"):
            entry.login_status = "registered_only"
        elif last_check == "verify" and status in ("login_failed", "verified"):
            try:
                result = json.loads(result_json or "null")
            except ValueError:
                result = None
            if isinstance(result, dict):
                if status == "login_failed":
                    # Other terminal failures can occur before SSH is attempted.
                    error = result.get("error")
                    if result.get("state") == "failed" and isinstance(error, str) and _FAILED_LOGIN.fullmatch(error):
                        entry.login_status = "known_failed"
                elif result.get("login_ok") is True:
                    entry.login_status = "verified"
                    config = result.get("bot_ssh_config")
                    if config == "installed":
                        entry.managed_config = "recorded_installed_runtime_unknown"
                    elif config == "not_writable":
                        entry.managed_config = "installation_not_confirmed"
        entries.append(entry)
    return entries, len(rows) > INVENTORY_LIMIT


def get_host(engine, host_id: int) -> OpsHostAccess | None:
    ensure_schema(engine)
    with Session(engine) as session:
        return session.get(OpsHostAccess, host_id)


def create_host(engine, fields: dict, *, actor: str | None = None) -> dict:
    ensure_schema(engine)
    if set(fields) != {"label", "host", "port", "login_user"}:
        raise HostAccessError("a host record takes exactly label, host, port and login_user")
    row = OpsHostAccess(**_validate_destination(fields), created_by=actor)
    with Session(engine) as session:
        if session.exec(select(OpsHostAccess).where(OpsHostAccess.label == row.label)).first():
            raise HostAccessError("label already in use")
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.to_api()


def update_host(engine, host_id: int, fields: dict) -> dict:
    ensure_schema(engine)
    if not fields or not set(fields) <= {"label", "host", "port", "login_user"}:
        raise HostAccessError("only label, host, port and login_user can be changed")
    values = _validate_destination(fields)
    with Session(engine) as session:
        row = session.get(OpsHostAccess, host_id)
        if row is None:
            raise KeyError(host_id)
        moved = any(values.get(k, getattr(row, k)) != getattr(row, k) for k in ("host", "port", "login_user"))
        for key, value in values.items():
            setattr(row, key, value)
        if moved:
            # A new destination is a new trust decision.
            row.known_host_keys_json, row.status, row.passwordless_sudo = "[]", "unverified", None
            # Drop the old destination's scan too: it must never be confirmable here.
            row.last_op = row.last_job_id = row.last_result_json = None
            row.last_checked_at = None
        row.updated_at = _utcnow()
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.to_api()


def delete_host(engine, host_id: int) -> bool:
    ensure_schema(engine)
    with Session(engine) as session:
        row = session.get(OpsHostAccess, host_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


def pin(engine, host_id: object, key_fingerprint: str, op: str) -> dict:
    """The approval-time target: everything the worker will use, from the record."""
    if type(host_id) is not int:
        raise HostAccessError("host_id must be an integer")
    row = get_host(engine, host_id)
    if row is None:
        raise HostAccessError(f"no host record {host_id}")
    keys = json.loads(row.known_host_keys_json)
    if op != "scan" and not keys:
        raise HostAccessError("pin this host's key first: scan it and confirm the fingerprint")
    return host_key.validate_target({"host_id": row.id, "host": row.host, "port": row.port,
                                     "login_user": row.login_user, "key_fingerprint": key_fingerprint,
                                     "known_host_keys": keys})


def check_pin(engine, target: dict) -> None:
    """At dispatch: the record must still be the destination that was approved."""
    host_key.validate_target(target)
    row = get_host(engine, target["host_id"])
    if row is None:
        raise HostAccessError("host record was deleted after approval")
    current = (row.host, row.port, row.login_user, json.loads(row.known_host_keys_json))
    if current != (target["host"], target["port"], target["login_user"], target["known_host_keys"]):
        raise HostAccessError("host record changed after approval; approve again")


def _result(text: str | None) -> dict | None:
    for prefix in _PREFIXES:
        if text and prefix in text:
            try:
                value = json.loads(text.split(prefix, 1)[1].strip().splitlines()[0])
                return value if isinstance(value, dict) else None
            except (ValueError, IndexError):
                return None
    return None


def settle(engine, job, snapshot: dict) -> None:
    """Fold a terminal host_key job back into its host record."""
    target, op = snapshot.get("host_key") or {}, (snapshot.get("spec") or {}).get("op")
    if not isinstance(target, dict) or "host_id" not in target:
        return
    ensure_schema(engine)
    result = _result(job.output_tail)
    with Session(engine) as session:
        row = session.get(OpsHostAccess, target["host_id"])
        if row is None or (row.host, row.port, row.login_user) != (target["host"], target["port"], target["login_user"]):
            return  # deleted or re-pointed since; this result is about another destination
        row.last_op, row.last_job_id, row.last_checked_at = op, job.id, _utcnow()
        if job.state == "succeeded" and result:
            row.last_result_json = json.dumps(result, sort_keys=True)
            if op == "scan":
                # Offered, not trusted: the operator confirms a fingerprint.
                row.status = "scanned" if row.status in ("unverified", "scanned", "login_failed") else row.status
            elif op == "verify":
                row.status = "verified"
                sudo = result.get("passwordless_sudo_probe")
                row.passwordless_sudo = sudo if isinstance(sudo, bool) else None
            else:
                row.status = "key_present" if op == "add" else "key_removed"
        else:
            row.last_result_json = json.dumps({"state": job.state, "error": (job.error_text or "")[:500]})
            row.status = "login_failed" if op == "verify" and job.state == "failed" else (
                row.status if op == "scan" else
                row.status if job.state == "failed" else "unknown")
        row.updated_at = _utcnow()
        session.add(row)
        session.commit()


def confirm_host_keys(engine, host_id: int, fingerprints: object) -> dict:
    """Pin the scanned host keys whose fingerprints the operator confirmed.

    Only keys from this record's latest successful scan can be pinned, and
    every confirmed fingerprint must name one of them. Replaces earlier pins.
    """
    ensure_schema(engine)
    # Exactly one: the operator checks one key's fingerprint on the host, and
    # only that key is trusted (never the other algorithms the scan offered).
    if (not isinstance(fingerprints, list) or len(fingerprints) != 1
            or any(not isinstance(f, str) or not host_key._FINGERPRINT.fullmatch(f) for f in fingerprints)):
        raise HostAccessError("confirm exactly one SHA256 host key fingerprint, the one you checked on the host")
    with Session(engine) as session:
        row = session.get(OpsHostAccess, host_id)
        if row is None:
            raise KeyError(host_id)
        last = json.loads(row.last_result_json) if row.last_result_json else {}
        scanned = last.get("scanned_host_keys") if row.last_op == "scan" else None
        if not isinstance(scanned, list) or not scanned:
            raise HostAccessError("scan this host first; there are no offered keys to confirm")
        by_fp = {}
        for entry in scanned:
            key = entry.get("key") if isinstance(entry, dict) else None
            try:
                host_key.validate_host_keys([key])
            except ValueError:
                continue
            by_fp[host_key.host_key_fingerprint(key)] = key
        missing = [f for f in fingerprints if f not in by_fp]
        if missing:
            raise HostAccessError("a confirmed fingerprint is not among the scanned host keys; scan again")
        row.known_host_keys_json = json.dumps(list(dict.fromkeys(by_fp[f] for f in fingerprints)))
        row.status, row.passwordless_sudo, row.updated_at = "trusted", None, _utcnow()
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.to_api()
