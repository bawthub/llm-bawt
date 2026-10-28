"""The ops control contract between the app and the ops service (TASK-1169 S3).

The app admits and reads ops jobs; the ops service (the ops-reconciler
process) prepares, dispatches and reconciles them. The app's one request to
the ops service is "prepare this invocation", plus a readiness read for its
health report, over an internal listener separate from the GET-only health
surface on :8650.

Auth is a bearer derived from the master key under its own label, not the
BawtHub service token: only processes holding the master key (the app and the
ops service) can present it, and an operator's BawtHub token override, which
the ops service doesn't mount, never decides it.
"""
from __future__ import annotations

import hmac
import re

#: The product's own internal control port (compose-internal, never published).
CONTROL_PORT = 8651
PREPARE_PATH = "/v1/ops/prepare"
EXECUTORS_PATH = "/v1/ops/executors"
#: Versioned derivation label; a future scheme must not collide with this one.
TOKEN_LABEL = "ops-control-token/v1"
#: Every response the control server writes carries this field, so the client
#: can tell a coded refusal from any other server's 4xx (an old reconciler,
#: a misrouted request) and fail closed on the latter.
PROTOCOL_FIELD = "ops_control"
PROTOCOL_VERSION = 1
#: A refusal code is a short snake_case identifier.
CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: The statuses a prepare refusal (an ``OpsDispatchError``) answers with; any
#: other status is never a refusal the caller acts on.
REFUSAL_STATUS = {"operation_not_found": 404, "args_invalid": 400}
REFUSAL_DEFAULT_STATUS = 409
REFUSAL_STATUSES = frozenset({400, 404, 409})
MAX_BODY_BYTES = 64 * 1024


def control_token() -> str:
    """The control bearer, derived from the master key. Raises when the key is
    unusable (absent, or only in this process), so both sides fail closed."""
    from ..service.providers import crypto

    return crypto.derive_key(TOKEN_LABEL).hex()


def token_matches(authorization: str | None, token: str) -> bool:
    scheme, _, value = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(value.strip().encode(), token.encode())


def refusal_status(code: str) -> int:
    return REFUSAL_STATUS.get(code, REFUSAL_DEFAULT_STATUS)
