"""The BawtHub service token: where it comes from, how it is sent and checked.

Contract shared with BawtHub (agreed on TASK-1472):

* ``BAWTHUB_SERVICE_TOKEN_FILE`` names a secret file; its trimmed contents are
  the token and win over ``BAWTHUB_SERVICE_TOKEN``. Both containers (llm-bawt
  app, BawtHub frontend) are given the same value by the deployment.
* llm-bawt sends ``Authorization: Bearer <token>`` on every BawtHub call. With
  no token configured it sends no header and logs that once.
* llm-bawt's internal secret API accepts only that bearer, compared in
  constant time, and fails closed when no token is configured here.

The token is read at call time, so replacing the secret file takes effect on
the next request without a restart. The value is never logged.
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

logger = logging.getLogger(__name__)

TOKEN_ENV = "BAWTHUB_SERVICE_TOKEN"
TOKEN_FILE_ENV = "BAWTHUB_SERVICE_TOKEN_FILE"
_BEARER = "bearer"


class TokenCheck(enum.Enum):
    """Outcome of checking a caller's ``Authorization`` header."""

    OK = "ok"
    UNCONFIGURED = "unconfigured"  # this process has no token: refuse everyone
    MISSING = "missing"
    INVALID = "invalid"


class ServiceTokenSource:
    """Reads the token from the environment; reports a missing one once."""

    def __init__(self, environ: Mapping[str, str] | None = None):
        self._environ = os.environ if environ is None else environ
        self._reported: set[str] = set()
        self._lock = threading.Lock()

    def read(self) -> str | None:
        """Return the configured token, or ``None`` when none is usable."""
        path = (self._environ.get(TOKEN_FILE_ENV) or "").strip()
        if path:
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
        if not token:
            self._report("missing", "Neither %s nor %s is set; BawtHub calls carry no "
                         "service bearer", TOKEN_ENV, TOKEN_FILE_ENV)
            return None
        return self._found(token)

    def check(self, authorization: str | None) -> TokenCheck:
        """Check an ``Authorization`` header against the configured token."""
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

    def _found(self, token: str) -> str:
        if self._reported:
            with self._lock:
                self._reported.clear()
            logger.info("BawtHub service token is configured")
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
