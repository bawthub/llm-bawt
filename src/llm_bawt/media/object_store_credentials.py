"""Object-store (S3) credentials from the encrypted store first (TASK-1476).

The ``integration_connection:object-store`` record (the Integrations
credentials form, shared with BawtHub's storage layer) wins. Only when no
record exists are the legacy ``LLM_BAWT_S3_*`` env vars read, with a one-time
deprecation warning naming them, never a value.

A record that exists but can't be used (disconnected, undecryptable, store
down) raises :class:`ObjectStoreUnavailable` and never falls back to env: a
disconnect must not bring old keys back. Disconnect therefore keeps a
credential-free ``disconnected`` record (``keeps_disconnected_record``).

:class:`StoredS3Config` is the source an :class:`S3BlobBackend` asks before
each operation. It caches an answer for a short TTL so a hot blob path doesn't
read the store per object, and a saved or rotated key is used within that TTL
without a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from llm_bawt.media.object_store import BlobBackendUnavailable, S3Config, s3_config_from_env
from llm_bawt.service.connection_secrets import ConnectionUnavailable, integration_spec, read_connection

logger = logging.getLogger(__name__)

INTEGRATION_ID = "object-store"
#: Seconds a resolved answer is reused (BawtHub caches the same record ~60 s).
CACHE_TTL_SECONDS = 60.0
_ENV_NAMES = (
    "LLM_BAWT_S3_ENDPOINT",
    "LLM_BAWT_S3_REGION",
    "LLM_BAWT_S3_BUCKET",
    "LLM_BAWT_S3_ACCESS_KEY",
    "LLM_BAWT_S3_SECRET_KEY",
)
_env_fallback_warned = False


class ObjectStoreUnavailable(BlobBackendUnavailable):
    """No usable S3 credentials. Routes map it to 503 like any unreachable backend.

    ``code`` is ``not_connected``, ``unreadable``, ``store_unavailable`` or
    ``not_configured`` (no record and incomplete env).
    """

    def __init__(self, code: str):
        super().__init__(f"object store connection unavailable: {code}")
        self.code = code


def resolve_s3_config(config) -> S3Config | None:
    """The S3 config to use now, or ``None`` when none is set anywhere."""
    global _env_fallback_warned
    try:
        found = read_connection(config, "integration", INTEGRATION_ID, integration_spec(INTEGRATION_ID))
    except ConnectionUnavailable as error:
        raise ObjectStoreUnavailable(error.code) from None
    if found is not None:
        return S3Config(
            endpoint=found.meta["endpoint"].strip(),
            region=found.meta["region"].strip(),
            bucket=found.meta["bucket"].strip(),
            access_key=found.secret["access_key_id"].strip(),
            secret_key=found.secret["secret_access_key"].strip(),
        )
    cfg = s3_config_from_env()
    if cfg is not None and not _env_fallback_warned:
        _env_fallback_warned = True
        logger.warning(
            "Object store credentials read from deprecated settings %s; save them under Integrations instead",
            "/".join(_ENV_NAMES),
        )
    return cfg


class StoredS3Config:
    """Resolves the current :class:`S3Config` for each blob operation.

    Usable answers and definite refusals (``not_connected``, ``unreadable``,
    ``not_configured``) are reused for ``ttl`` seconds. ``store_unavailable``
    is never cached, so the next operation retries the store.
    """

    def __init__(self, config=None, *, ttl: float = CACHE_TTL_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self._config = config
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._expires = 0.0
        self._cached: S3Config | ObjectStoreUnavailable | None = None

    def _app_config(self):
        if self._config is None:
            from llm_bawt.utils.config import config

            self._config = config
        return self._config

    def current(self) -> S3Config:
        with self._lock:
            now = self._clock()
            if self._cached is None or now >= self._expires:
                try:
                    found = resolve_s3_config(self._app_config())
                    self._cached = found if found is not None else ObjectStoreUnavailable("not_configured")
                except ObjectStoreUnavailable as error:
                    if error.code == "store_unavailable":
                        self._cached = None
                        raise
                    self._cached = error
                self._expires = now + self._ttl
            cached = self._cached
        if isinstance(cached, ObjectStoreUnavailable):
            raise ObjectStoreUnavailable(cached.code)
        return cached


_default_source: StoredS3Config | None = None
_default_lock = threading.Lock()


def default_s3_config_source() -> StoredS3Config:
    """The process-wide source every env-selected S3 backend shares."""
    global _default_source
    with _default_lock:
        if _default_source is None:
            _default_source = StoredS3Config()
        return _default_source


__all__ = [
    "CACHE_TTL_SECONDS",
    "INTEGRATION_ID",
    "ObjectStoreUnavailable",
    "StoredS3Config",
    "default_s3_config_source",
    "resolve_s3_config",
]
