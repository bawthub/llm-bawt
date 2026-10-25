"""The BawtHub service token: where it comes from, how it is sent and checked.

Contract shared with BawtHub (TASK-1472):

* By default the token is **derived, not stored**: hex HMAC-SHA256 of the
  master Fernet key (``crypto.derive_key``) under :data:`DERIVATION_LABEL`. The
  master key already auto-generates and persists, so a fresh install needs no
  configuration, and nobody hand-sets a second secret.
* llm-bawt hands the token to BawtHub by writing it to a file in a shared
  volume (``token_export``); BawtHub reads that file through its
  ``BAWTHUB_SERVICE_TOKEN_FILE``. BawtHub never sees the master key.
* Operator override: ``BAWTHUB_SERVICE_TOKEN_FILE`` (trimmed contents, wins)
  or ``BAWTHUB_SERVICE_TOKEN``. A token file that is this process's own export
  path is not an override: reading back our own output would pin a stale token
  across a master-key rotation.
* llm-bawt sends ``Authorization: Bearer <token>`` on every BawtHub call. With
  no usable token it sends no header and logs that once.
* llm-bawt's internal secret API accepts only that bearer, compared in
  constant time, and fails closed when no token is available.

Rotating the derived token means rotating the master key (which also re-keys
every stored credential); there is no separate rotation path. Override files
are re-read on every call, and whenever the token in effect changes the export
is rewritten before that token is returned, so llm-bawt never sends or accepts
a token BawtHub's file doesn't hold yet. The value is never logged.
"""

from __future__ import annotations

import enum
import hmac
import logging
import os
import threading
from collections.abc import Generator, Mapping
from pathlib import Path

import httpx

from llm_bawt.service.providers import crypto

logger = logging.getLogger(__name__)

TOKEN_ENV = "BAWTHUB_SERVICE_TOKEN"
TOKEN_FILE_ENV = "BAWTHUB_SERVICE_TOKEN_FILE"
EXPORT_FILE_ENV = "BAWTHUB_SERVICE_TOKEN_EXPORT_FILE"
# Versioned so a future scheme can't collide with tokens issued under this one.
DERIVATION_LABEL = "bawthub-service-token/v1"
_BEARER = "bearer"


class TokenCheck(enum.Enum):
    """Outcome of checking a caller's ``Authorization`` header."""

    OK = "ok"
    UNCONFIGURED = "unconfigured"  # this process has no token: refuse everyone
    MISSING = "missing"
    INVALID = "invalid"


class ExportResult(enum.Enum):
    """Outcome of publishing the token to BawtHub's export file."""

    DISABLED = "disabled"  # no export path configured
    WRITTEN = "written"
    REMOVED = "removed"  # no usable token: any stale file is gone
    UNCHANGED = "unchanged"  # the export already holds the token in effect
    FAILED = "failed"


def derive_service_token() -> str:
    """The token derived from the master key; raises if the key is unusable."""
    return crypto.derive_key(DERIVATION_LABEL).hex()


def _same_file(first: str, second: str) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except (OSError, RuntimeError):
        return False


class ServiceTokenSource:
    """Resolves the token (override, else derived) and keeps BawtHub's copy in step.

    Reports a missing token once. When ``BAWTHUB_SERVICE_TOKEN_EXPORT_FILE`` is
    set, the token in effect is the one BawtHub's file holds: every change (an
    override file rotated in place, an override appearing or breaking) rewrites
    the export first, and only a written token is handed out. While a write
    fails (logged once, retried on every call) the last exported token stays in
    effect; until the first export succeeds there is none, so both sides fail
    closed together.
    """

    def __init__(self, environ: Mapping[str, str] | None = None):
        self._environ = os.environ if environ is None else environ
        self._reported: set[str] = set()
        self._derived: str | None = None
        self._lock = threading.Lock()
        # The token BawtHub's file holds, once this process has written it.
        self._exported = False
        self._published: str | None = None
        self._publish_lock = threading.Lock()
        self._publish_failing = False

    def read(self) -> str | None:
        """Return the token in effect, or ``None`` when none is usable."""
        token = self._resolve()
        if self._publish(token) is ExportResult.DISABLED:
            return token
        # What BawtHub's file holds, even if this call's write failed.
        return self._published if self._exported else None

    def sync(self, *, force: bool = False) -> ExportResult:
        """Publish the token in effect to the export file.

        ``force`` rewrites even an unchanged token (app startup: the volume may
        be new or hold a previous process's value).
        """
        return self._publish(self._resolve(), force=force)

    def _publish(self, token: str | None, *, force: bool = False) -> ExportResult:
        from .token_export import publish  # token_export imports this module

        raw = (self._environ.get(EXPORT_FILE_ENV) or "").strip()
        if not raw:
            return ExportResult.DISABLED
        if not force and self._exported and self._published == token:
            return ExportResult.UNCHANGED
        with self._publish_lock:
            if not force and self._exported and self._published == token:
                return ExportResult.UNCHANGED
            path = Path(raw)
            try:
                result = publish(path, token, self._environ)
            except (OSError, ValueError) as error:
                if not self._publish_failing:
                    self._publish_failing = True
                    logger.error("Could not export the BawtHub service token to %s (%s: %s); "
                                 "BawtHub keeps its previous copy until this succeeds",
                                 path, type(error).__name__, error)
                return ExportResult.FAILED
            self._published, self._exported = token, True
            self._publish_failing = False
        if result is ExportResult.REMOVED:
            logger.warning("No BawtHub service token to export; removed %s if present", path)
        else:
            logger.info("Exported the BawtHub service token to %s", path)
        return result

    def _resolve(self) -> str | None:
        path = (self._environ.get(TOKEN_FILE_ENV) or "").strip()
        export = (self._environ.get(EXPORT_FILE_ENV) or "").strip()
        if path and not (export and _same_file(path, export)):
            try:
                token = Path(path).read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as error:
                self._report("unreadable", "%s is set but its file can't be read (%s)",
                             TOKEN_FILE_ENV, type(error).__name__)
                return None
            if not token:
                self._report("empty-file", "%s points at an empty file", TOKEN_FILE_ENV)
                return None
            return self._found(token)
        token = (self._environ.get(TOKEN_ENV) or "").strip()
        if token:
            return self._found(token)
        return self._derive()

    def check(self, authorization: str | None) -> TokenCheck:
        """Check an ``Authorization`` header against the token in effect."""
        expected = self.read()
        if expected is None:
            return TokenCheck.UNCONFIGURED
        scheme, _, presented = (authorization or "").strip().partition(" ")
        presented = presented.strip()
        if scheme.lower() != _BEARER or not presented:
            return TokenCheck.MISSING
        if hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
            return TokenCheck.OK
        return TokenCheck.INVALID

    def _derive(self) -> str | None:
        # The master key is fixed for the process lifetime (crypto caches its
        # Fernet the same way), so derive once.
        if self._derived is None:
            try:
                self._derived = derive_service_token()
            except Exception as error:  # noqa: BLE001 — fail closed on any key problem
                self._report("underivable", "No BawtHub service token: no override is set "
                             "and the master key can't derive one (%s); BawtHub calls "
                             "carry no bearer", type(error).__name__)
                return None
        return self._found(self._derived)

    def _found(self, token: str) -> str:
        if self._reported:
            with self._lock:
                self._reported.clear()
            logger.info("BawtHub service token is available")
        return token

    def _report(self, reason: str, message: str, *args: object) -> None:
        with self._lock:
            if reason in self._reported:
                return
            self._reported.add(reason)
        logger.warning(message, *args)


class BawtHubServiceAuth(httpx.Auth):
    """httpx auth that adds the service bearer to each BawtHub request."""

    def __init__(self, source: ServiceTokenSource):
        self._source = source

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        token = self._source.read()
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


_SOURCE = ServiceTokenSource()
_AUTH = BawtHubServiceAuth(_SOURCE)


def service_token_source() -> ServiceTokenSource:
    """The process-wide token source (env read at call time)."""
    return _SOURCE


def service_auth() -> BawtHubServiceAuth:
    """httpx ``auth=`` for every llm-bawt client that calls BawtHub."""
    return _AUTH
