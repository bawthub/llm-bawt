"""Symmetric encryption for provider credentials at rest.

Provider credentials (OAuth tokens, API keys, GitHub App user tokens) are
persisted in the ``runtime_settings`` table. Nothing else in that table is
encrypted, so this module introduces a small Fernet-based envelope: the value we
store is ``enc:v1:<fernet-ciphertext>``. Anything without the ``enc:v1:`` prefix
is treated as plaintext (back-compat / never-encrypted values).

KEY SOURCE (in priority order):
  1. env ``LLM_BAWT_SECRET_KEY`` — a urlsafe-base64 32-byte Fernet key.
  2. a persisted key file ``~/.config/llm-bawt/secret.key`` (mode 0600). If
     absent it is generated once and written, so restarts stay stable.

IMPORTANT for tenants: the key MUST survive repaves or stored credentials become
unreadable. ``tenant.sh`` bakes ``LLM_BAWT_SECRET_KEY`` into the generated
``.env`` for exactly this reason — the file fallback only covers single-host dev.

DERIVED SECRETS: :func:`derive_key` turns the same master key into
domain-separated subkeys (HMAC-SHA256 keyed by the raw key, one label per use),
so a deployment needs no second hand-set secret. Rotating the master key
rotates every derived value with it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PREFIX = "enc:v1:"
_ENV_KEY = "LLM_BAWT_SECRET_KEY"

# Cached Fernet instance (lazy — cryptography import is deferred).
_fernet = None


def _key_file() -> Path:
    return Path.home() / ".config" / "llm-bawt" / "secret.key"


class EphemeralKeyError(RuntimeError):
    """The master key exists only in this process: it was generated and not saved."""


def _load_or_create_key(*, require_persisted: bool = False) -> bytes:
    """Resolve the Fernet key from env, else a persisted file, else generate.

    A generated key is published with an atomic create-if-absent, so two
    processes racing on a fresh install end up with the one key on disk. If it
    can't be saved, the key still works in-process for encryption this run;
    ``require_persisted`` callers (derived secrets, which must match across
    calls, processes and restarts) get :class:`EphemeralKeyError` instead.
    """
    from cryptography.fernet import Fernet

    env = os.getenv(_ENV_KEY)
    if env:
        return env.strip().encode()

    path = _key_file()
    if path.exists():
        return path.read_text().strip().encode()

    key = Fernet.generate_key()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _create_if_absent(path, key):
            return path.read_text().strip().encode()  # another process won the race
        logger.warning(
            "Generated a new provider-credential encryption key at %s. "
            "For tenants, set %s in the environment so it survives repaves.",
            path,
            _ENV_KEY,
        )
    except OSError as e:
        if require_persisted:
            raise EphemeralKeyError(f"could not persist the master key to {path}") from e
        logger.warning("Could not persist encryption key to %s: %s", path, e)
    return key


def _create_if_absent(path: Path, key: bytes) -> bool:
    """Atomically create ``path`` (0600) holding ``key``; False if it already exists.

    The key is written to a private temp file first and hard-linked into place,
    so no reader ever sees a partial or wider-mode key file.
    """
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        Path(tmp).unlink(missing_ok=True)


def _get_fernet():
    global _fernet
    if _fernet is None:
        from cryptography.fernet import Fernet

        _fernet = Fernet(_load_or_create_key())
    return _fernet


def derive_key(label: str) -> bytes:
    """A 32-byte subkey for ``label``: HMAC-SHA256(raw master key, label).

    The key is resolved like :func:`_get_fernet` and must be a valid Fernet
    key; an unusable key raises rather than deriving from garbage. So does a
    key that exists only in this process (generated, couldn't be saved): a
    subkey from it would differ on the next call, in another process and after
    a restart. Labels must be versioned and unique per use (e.g.
    ``"bawthub-service-token/v1"``).
    """
    if not label:
        raise ValueError("derive_key needs a label")
    from cryptography.fernet import Fernet

    key = _load_or_create_key(require_persisted=True)
    Fernet(key)  # validates: urlsafe-base64 of exactly 32 bytes
    raw = base64.urlsafe_b64decode(key)
    return hmac.new(raw, label.encode("utf-8"), hashlib.sha256).digest()


def encrypt(plaintext: str) -> str:
    """Return ``enc:v1:<ciphertext>`` for a plaintext string."""
    if plaintext is None:
        raise ValueError("cannot encrypt None")
    token = _get_fernet().encrypt(plaintext.encode()).decode()
    return f"{_PREFIX}{token}"


def decrypt(stored: str) -> str:
    """Reverse :func:`encrypt`. Values without the prefix are returned as-is."""
    if stored is None:
        raise ValueError("cannot decrypt None")
    if not stored.startswith(_PREFIX):
        return stored  # plaintext / legacy value
    token = stored[len(_PREFIX) :].encode()
    return _get_fernet().decrypt(token).decode()


def is_encrypted(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith(_PREFIX)
