"""The app's client for the ops control listener (TASK-1169 S3a).

``prepare`` returns the ops service's snapshot or raises ``OpsDispatchError``
with the ops service's own refusal code. Anything else (connection refused, a
timeout, a 5xx, a busy server after one retry, an old reconciler's 404/405, a
body that isn't the control protocol, a snapshot whose hash doesn't verify)
raises ``ops_unavailable``: the caller creates nothing and opens no approval.
There is no local fallback, since this process reads no Docker.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

import httpx

from .control_protocol import (CODE_PATTERN, EXECUTORS_PATH, PREPARE_PATH, PROTOCOL_FIELD, PROTOCOL_VERSION,
                               REFUSAL_STATUSES, control_token)
from .service import OpsDispatchError, OpsService
from .validation import canonical_json

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
#: The control server's Retry-After for ``ops_busy``.
BUSY_RETRY_SECONDS = 2.0
UNAVAILABLE = "ops_unavailable"


def _unavailable(detail: str) -> OpsDispatchError:
    return OpsDispatchError(UNAVAILABLE, f"ops_unavailable: the ops service could not prepare this: {detail}; "
                                         "nothing was created")


def _verified_snapshot(snapshot) -> dict:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("snapshot_hash"), str):
        raise _unavailable("the response carries no snapshot")
    body = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    if hashlib.sha256(canonical_json(body).encode()).hexdigest() != snapshot["snapshot_hash"]:
        raise _unavailable("the snapshot's content hash does not verify")
    return snapshot


class OpsControlClient:
    def __init__(self, base_url: str, *, token_source=control_token,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self._token_source = token_source
        self._timeout = timeout
        self._transport = transport
        self._sleep = sleep

    def prepare(self, operation_slug: str, args: dict | None) -> dict:
        body = self._call("POST", PREPARE_PATH, {"operation": operation_slug, "args": args})
        return _verified_snapshot(body.get("snapshot"))

    def executor_readiness(self) -> dict[str, str | None]:
        readiness = self._call("GET", EXECUTORS_PATH, None).get("executors")
        if not isinstance(readiness, dict) or not all(
                isinstance(kind, str) and (reason is None or isinstance(reason, str))
                for kind, reason in readiness.items()):
            raise _unavailable("the response carries no executor readiness")
        return readiness

    def _call(self, method: str, path: str, payload) -> dict:
        try:
            token = self._token_source()
        except Exception as exc:  # noqa: BLE001 - an unusable key means no call
            raise _unavailable(f"the ops control token is unavailable ({type(exc).__name__})") from exc
        headers = {"Authorization": f"Bearer {token}"}
        # One retry, only where nothing ran: a connection that never opened,
        # or a busy server that refused before doing any work.
        for attempt in (1, 2):
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                    response = client.request(method, self.base_url + path, headers=headers,
                                              **({"json": payload} if payload is not None else {}))
            except httpx.ConnectError as exc:
                if attempt == 1:
                    continue
                raise _unavailable(f"cannot connect ({type(exc).__name__})") from exc
            except httpx.HTTPError as exc:
                raise _unavailable(f"the request failed ({type(exc).__name__})") from exc
            body = self._protocol_body(response)
            if response.status_code == 503 and body is not None and body.get("code") == "ops_busy" and attempt == 1:
                self._sleep(BUSY_RETRY_SECONDS)
                continue
            return self._result(response, body)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _protocol_body(response) -> dict | None:
        try:
            body = response.json()
        except ValueError:
            return None
        if not isinstance(body, dict) or body.get(PROTOCOL_FIELD) != PROTOCOL_VERSION:
            return None
        return body

    @staticmethod
    def _result(response, body) -> dict:
        if body is None:
            raise _unavailable(f"HTTP {response.status_code} is not an ops control response")
        if response.status_code == 200:
            return body
        refusal = body.get("refusal")
        if (response.status_code in REFUSAL_STATUSES and isinstance(refusal, dict)
                and isinstance(refusal.get("code"), str) and CODE_PATTERN.fullmatch(refusal["code"])
                and isinstance(refusal.get("message"), str)):
            raise OpsDispatchError(refusal["code"], refusal["message"])
        code = body.get("code") if isinstance(body.get("code"), str) else "unknown"
        raise _unavailable(f"HTTP {response.status_code} {code}")


def build_admission_ops_service(config, store) -> OpsService:
    """The app's OpsService (TASK-1169 S3c): admits and reads, prepares through
    the ops service, and builds no Docker client or release coordinator."""
    return OpsService(store, admission_only=True, preparer=OpsControlClient(config.OPS_CONTROL_URL))


__all__ = ["OpsControlClient", "build_admission_ops_service"]
