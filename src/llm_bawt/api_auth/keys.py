"""Per-caller API keys for the llm-bawt HTTP API (TASK-2380).

A key is derived, not stored: hex HMAC-SHA256 of the master-key subkey for
:data:`DERIVATION_LABEL` over the principal name. Callers send
``X-LLM-Bawt-Key: <principal>.<key>``; the principal selects which key to
derive and the comparison is constant time. Bumping the label version rotates
every key; a single principal is revoked through the denylist.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Iterable

HEADER = "x-llm-bawt-key"
DERIVATION_LABEL = "llm-bawt-api-key/v1"
_PRINCIPAL = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def valid_principal(name: str) -> bool:
    return bool(_PRINCIPAL.fullmatch(name or ""))


def _root() -> bytes:
    from llm_bawt.service.providers import crypto

    return crypto.derive_key(DERIVATION_LABEL)


def derive_key(principal: str, *, root: Callable[[], bytes] = _root) -> str:
    """The key for ``principal``; raises on an invalid name or unusable master key."""
    if not valid_principal(principal):
        raise ValueError(f"invalid principal name: {principal!r}")
    return hmac.new(root(), principal.encode(), hashlib.sha256).hexdigest()


def header_value(principal: str, *, root: Callable[[], bytes] = _root) -> str:
    return f"{principal}.{derive_key(principal, root=root)}"


class KeyCheck:
    """Outcome of checking one header value: ``principal`` set when valid."""

    __slots__ = ("principal", "reason")

    def __init__(self, principal: str | None, reason: str):
        self.principal = principal
        self.reason = reason

    @property
    def ok(self) -> bool:
        return self.principal is not None


def verify(value: str, *, denylist: Iterable[str] = (),
           root: Callable[[], bytes] = _root) -> KeyCheck:
    """Check a header value. Never logs or returns the key itself."""
    principal, sep, key = (value or "").strip().partition(".")
    if not sep or not key or not valid_principal(principal):
        return KeyCheck(None, "malformed")
    if principal in set(denylist):
        return KeyCheck(None, "revoked")
    try:
        expected = derive_key(principal, root=root)
    except Exception:  # unusable master key: nothing can be verified
        return KeyCheck(None, "unconfigured")
    if not hmac.compare_digest(expected.encode(), key.encode()):
        return KeyCheck(None, "invalid")
    return KeyCheck(principal, "ok")
