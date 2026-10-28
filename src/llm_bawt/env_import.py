"""Adopt allowlisted env credentials into encrypted records (TASK-1476).

Credentials that still reach the containers through ``.env`` move into the
same encrypted records the Settings cards write, so the env line can be
deleted without anyone reading or retyping a value:

* only the env names in :data:`SPECS` are ever read. Adding one is a code
  change with a test, never data. Bootstrap secrets are not adoptable;
* the source is this process's environment. Run it as a one-off container of
  the app image with the deployment's ``env_file``, so the values are exactly
  what the live ``.env`` holds, parsed by Compose itself;
* the write is :meth:`CredentialStore.create_if_absent` (provider records) or
  the same insert under ``integration_connection:<id>`` (integration records),
  so a row in any status, including an operator's disconnected one, is never
  overwritten;
* the insert is read back in the same transaction and every sourced field is
  compared (secrets by SHA-256, in process) before the transaction commits. A
  mismatch rolls the insert back;
* nothing is written unless the active encryption key is the persisted root.

Each destination reports ``removable``: true only when, after the run, its
record is connected and holds exactly the env values (a fresh import, or an
earlier record that already matches). Only those env names are safe to delete
from ``.env``; the report lists them in ``remove_names``. ``--check`` writes
nothing, so there it lists only names an existing record already matches,
never a ``would_import`` one. Output carries env
names, destinations and outcome codes, never values or hashes.

There is no ledger, unlike :mod:`llm_bawt.vault_import`. The env line is
deleted once its record is in, so a rerun has no source left to adopt. While
the line stays, a rerun only fills a destination that has no row at all.

Rollback of the surrounding cutover is restoring ``.env``. The imported records
hold the same values the env did and readers prefer the record, so leaving
them in place changes no credential a service sees.

Run inside the app image: ``python -m llm_bawt.env_import [--check] [--only DEST ...]``.
``--only`` limits a run to named destinations, so one with its own
precondition can be adopted in a later, separate run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore
from llm_bawt.integrations.credential_integrations import AUTH_METHOD, CREDENTIAL_INTEGRATIONS, CredentialIntegrations
from llm_bawt.service.providers.base import AUTH_API_KEY, STATUS_CONNECTED, ConnectionRecord, CredentialStore

logger = logging.getLogger(__name__)

PROVIDER = "provider"
INTEGRATION = "integration"
SOURCE_META = {"source": "env-import"}


@dataclass(frozen=True)
class Spec:
    """One destination record and the env names each of its fields comes from."""

    destination: str
    kind: str  # PROVIDER | INTEGRATION
    fields: tuple[tuple[str, tuple[str, ...]], ...]  # (record field, env aliases)

    @property
    def env_names(self) -> tuple[str, ...]:
        return tuple(name for _, aliases in self.fields for name in aliases)


def _api_key(destination: str, *names: str) -> Spec:
    return Spec(destination, PROVIDER, (("api_key", names),))


def _url_and_key(service: str) -> Spec:
    prefix = f"LLM_BAWT_{service.upper()}"
    return Spec(service, INTEGRATION, (("url", (f"{prefix}_URL",)), ("api_key", (f"{prefix}_API_KEY",))))


#: Closed by design (see module doc).
SPECS: tuple[Spec, ...] = (
    _api_key("zai", "ZAI_API_KEY"),
    _api_key("kimi-coding", "KIMI_CODING_API_KEY"),
    _api_key("brave", "LLM_BAWT_BRAVE_API_KEY"),
    _api_key("newsapi", "LLM_BAWT_NEWSAPI_API_KEY"),
    Spec("object-store", INTEGRATION, (
        ("endpoint", ("LLM_BAWT_S3_ENDPOINT",)),
        ("region", ("LLM_BAWT_S3_REGION",)),
        ("bucket", ("LLM_BAWT_S3_BUCKET",)),
        ("access_key_id", ("LLM_BAWT_S3_ACCESS_KEY",)),
        ("secret_access_key", ("LLM_BAWT_S3_SECRET_KEY",)),
    )),
    _url_and_key("sonarr"),
    _url_and_key("radarr"),
    _url_and_key("sabnzbd"),
)

# Outcomes. ``removable`` is decided separately (see :class:`Outcome`).
IMPORTED = "imported"
WOULD_IMPORT = "would_import"
PRESENT = "already_present"          # connected record already holds the env values
NO_SOURCE = "no_source"              # none of its env names is set
RECORD_DIFFERS = "record_differs"    # a connected record holds other values
DISCONNECTED = "disconnected"        # an operator's disconnect marker; never revived
MANAGED_ELSEWHERE = "managed_elsewhere"
CONFLICT = "conflict"                # two aliases for one field disagree
INCOMPLETE = "incomplete"            # the record would miss a required field
FAILED = "failed"
UNRESOLVED = frozenset({CONFLICT, INCOMPLETE, FAILED})


@dataclass(frozen=True)
class Outcome:
    destination: str
    env_names: tuple[str, ...]  # the names set in the environment
    outcome: str
    detail_code: str | None = None
    removable: bool = False

    def public(self) -> dict[str, Any]:
        return {"destination": self.destination, "env_names": list(self.env_names), "outcome": self.outcome,
                "detail_code": self.detail_code, "removable": self.removable}


class _VerifyMismatch(Exception):
    pass


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _error_code(error: BaseException) -> str:
    # Never str(error): DB errors can carry bound parameters.
    return f"error:{type(error).__name__}"[:64]


def _matches(record: ConnectionRecord | None, wanted_meta: Mapping[str, str],
             wanted_secret: Mapping[str, str]) -> bool:
    if record is None or record.status != STATUS_CONNECTED:
        return False
    meta, secret = record.meta or {}, record.secret or {}
    return (all(str(meta.get(k) or "") == v for k, v in wanted_meta.items())
            and all(_sha256(str(secret.get(k) or "")) == _sha256(v) for k, v in wanted_secret.items()))


class EnvImporter:
    def __init__(self, providers: CredentialStore, integrations: IntegrationStore,
                 environ: Mapping[str, str] | None = None):
        self.providers = providers
        self.integrations = integrations
        self.environ = os.environ if environ is None else environ

    def _store(self, spec: Spec) -> CredentialStore:
        return self.providers if spec.kind == PROVIDER else self.integrations

    def _source(self, spec: Spec) -> tuple[dict[str, str], tuple[str, ...], bool]:
        """``(field values, env names set, conflict)`` for ``spec``."""
        values: dict[str, str] = {}
        present: list[str] = []
        conflict = False
        for field, aliases in spec.fields:
            found = {name: self.environ[name].strip() for name in aliases if (self.environ.get(name) or "").strip()}
            present.extend(found)
            distinct = set(found.values())
            if len(distinct) > 1:
                conflict = True
            elif distinct:
                values[field] = distinct.pop()
        return values, tuple(present), conflict

    @staticmethod
    def _record(spec: Spec, values: dict[str, str]) -> tuple[ConnectionRecord, dict[str, str], dict[str, str]]:
        """The record to insert plus the meta/secret fields it must read back with."""
        if spec.kind == PROVIDER:
            secret = {"api_key": values["api_key"]}
            record = ConnectionRecord(provider=spec.destination, status=STATUS_CONNECTED, auth_method=AUTH_API_KEY,
                                      meta=dict(SOURCE_META), secret=dict(secret))
            return record, {}, secret
        definition = CREDENTIAL_INTEGRATIONS[spec.destination]
        # The Settings card's own validation and merge: same field contract.
        cleaned = CredentialIntegrations._validated(definition, values)
        record = CredentialIntegrations._merged(definition, spec.destination, None, cleaned)
        secret = {k: v for k, v in cleaned.items() if definition.field(k).secret}
        meta = {k: v for k, v in cleaned.items() if not definition.field(k).secret}
        return record, meta, secret

    def _existing(self, spec: Spec, present: tuple[str, ...], meta: dict[str, str],
                  secret: dict[str, str]) -> Outcome:
        record = self._store(spec).load(spec.destination)
        if record is None:
            # A row whose JSON can't be read: never overwritten, never adopted.
            return Outcome(spec.destination, present, MANAGED_ELSEWHERE, "unreadable_record")
        if spec.kind == INTEGRATION and record.auth_method != AUTH_METHOD:
            return Outcome(spec.destination, present, MANAGED_ELSEWHERE)
        if record.status != STATUS_CONNECTED:
            return Outcome(spec.destination, present, DISCONNECTED)
        if _matches(record, meta, secret):
            return Outcome(spec.destination, present, PRESENT, removable=True)
        return Outcome(spec.destination, present, RECORD_DIFFERS)

    def adopt(self, spec: Spec, *, check: bool = False) -> Outcome:
        values, present, conflict = self._source(spec)
        if not present:
            return Outcome(spec.destination, present, NO_SOURCE)
        if conflict:
            return Outcome(spec.destination, present, CONFLICT, "names_disagree")
        store = self._store(spec)
        try:
            record, meta, secret = self._record(spec, values)
        except IntegrationError as error:
            return Outcome(spec.destination, present, INCOMPLETE, str(error.args[0]) if error.args else None)
        except KeyError:
            return Outcome(spec.destination, present, INCOMPLETE, "missing_required_field")
        try:
            with store._store.engine.connect() as conn:
                exists = store.exists_in(conn, spec.destination)
            if exists:
                return self._existing(spec, present, meta, secret)
            problem = store.durable_key_problem()
            if problem:
                return Outcome(spec.destination, present, FAILED, problem)
            if check:
                # Nothing was written: the env line is still the only copy.
                return Outcome(spec.destination, present, WOULD_IMPORT)
            with store._store.engine.begin() as conn:
                if not store.create_if_absent(record, conn=conn):
                    raise LookupError("created_concurrently")
                if not _matches(store.load_in(conn, spec.destination), meta, secret):
                    raise _VerifyMismatch
        except LookupError:
            # A connect card got there first; judge that record like any other.
            return self._existing(spec, present, meta, secret)
        except _VerifyMismatch:
            return Outcome(spec.destination, present, FAILED, "verify_mismatch")
        except Exception as error:  # noqa: BLE001 - reported by type only
            return Outcome(spec.destination, present, FAILED, _error_code(error))
        return Outcome(spec.destination, present, IMPORTED, removable=True)

    def run(self, *, check: bool = False, only: frozenset[str] | None = None) -> list[Outcome]:
        return [self.adopt(spec, check=check) for spec in SPECS if only is None or spec.destination in only]


def remove_names(outcomes: list[Outcome]) -> list[str]:
    """Env names whose record now holds exactly their value: safe to delete from ``.env``."""
    return sorted(name for outcome in outcomes if outcome.removable for name in outcome.env_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_bawt.env_import",
                                     description="Adopt allowlisted env credentials into encrypted records.")
    parser.add_argument("--check", action="store_true", help="report what would happen; write nothing")
    parser.add_argument("--only", action="append", choices=[spec.destination for spec in SPECS],
                        help="limit the run to this destination (repeatable); default: all")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from llm_bawt.utils.config import Config

    config = Config()
    providers, integrations = CredentialStore(config), IntegrationStore(config)
    if not providers.available:
        print(json.dumps({"error": "database_unavailable"}))
        return 2
    only = frozenset(args.only) if args.only else None
    outcomes = EnvImporter(providers, integrations).run(check=args.check, only=only)
    for outcome in outcomes:
        print(json.dumps(outcome.public()))
    print(json.dumps({"check": args.check, "remove_names": remove_names(outcomes)}))
    return 1 if any(outcome.outcome in UNRESOLVED for outcome in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
