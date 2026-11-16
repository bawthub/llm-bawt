"""The ``api_auth_mode`` setting, resolved so that failures never open the API.

Contract (design rev 3, reviewed under TASK-2380):

* The global runtime setting ``api_auth_mode`` (``off|log|enforce``) is read
  through a 5 s cache. ``LLM_BAWT_API_AUTH_MODE`` is only the default for a
  store that was read successfully and has no row.
* A read error never changes the effective mode: a warm cache keeps the last
  value read; before the first successful read the mode is ``unresolved``,
  which the middleware treats as enforce-with-503.
* Only a successful read of ``off`` (the rollback write) lowers the mode.
* ``LLM_BAWT_API_AUTH_MODE_FORCE=off|log|enforce`` overrides everything, for an
  operator running without the database; it is logged at startup.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

SETTING_KEY = "api_auth_mode"
DENYLIST_KEY = "api_auth_denylist"
MODE_ENV = "LLM_BAWT_API_AUTH_MODE"
FORCE_ENV = "LLM_BAWT_API_AUTH_MODE_FORCE"
MODES = ("off", "log", "enforce")
UNRESOLVED = "unresolved"
TTL_SECONDS = 5.0


def _parse(value: Any) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in MODES else None


class ModeResolver:
    def __init__(self, settings: Callable[[], Mapping[str, Any]], *,
                 environ: Mapping[str, str] | None = None,
                 ttl: float = TTL_SECONDS, clock: Callable[[], float] = time.monotonic):
        env = os.environ if environ is None else environ
        self._settings = settings
        self._ttl = ttl
        self._clock = clock
        self._default = _parse(env.get(MODE_ENV)) or "off"
        self._force = _parse(env.get(FORCE_ENV))
        if (env.get(FORCE_ENV) or "").strip() and self._force is None:
            logger.error("%s=%r is not a mode; ignored", FORCE_ENV, env.get(FORCE_ENV))
        if self._force:
            logger.warning("API auth mode forced to %r by %s; the database setting is ignored",
                           self._force, FORCE_ENV)
        self._lock = threading.Lock()
        self._mode: str | None = None  # last successfully read mode
        self._denylist: frozenset[str] = frozenset()
        self._read_at: float | None = None
        self._failing = False

    def _refresh(self) -> None:
        now = self._clock()
        if self._read_at is not None and now - self._read_at < self._ttl:
            return
        with self._lock:
            if self._read_at is not None and now - self._read_at < self._ttl:
                return
            try:
                values = self._settings() or {}
            except Exception as error:  # keep last-known-good; never fall back to env
                if not self._failing:
                    logger.error("API auth settings unreadable (%s); keeping mode %s",
                                 type(error).__name__, self._mode or UNRESOLVED)
                self._failing = True
                self._read_at = now  # retry after one TTL, not every request
                return
            raw = values.get(SETTING_KEY)
            mode = self._default if raw is None else _parse(raw)
            if mode is None:  # a garbled row must not open anything
                logger.error("api_auth_mode=%r is not a mode; treating as enforce", raw)
                mode = "enforce"
            deny = values.get(DENYLIST_KEY) or []
            self._denylist = frozenset(str(p) for p in deny) if isinstance(deny, list) else frozenset()
            if self._failing:
                logger.info("API auth settings readable again; mode %s", mode)
            self._mode, self._read_at, self._failing = mode, now, False

    def mode(self) -> str:
        if self._force:
            return self._force
        self._refresh()
        return self._mode or UNRESOLVED

    def denylist(self) -> frozenset[str]:
        self._refresh()
        return self._denylist
