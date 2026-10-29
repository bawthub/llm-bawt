"""Loud maintenance failures (TASK-1704 slice A).

Botless background/utility calls (summaries, extraction, profile maintenance,
recall, utility completions) run on the global job model with nobody watching
the reply. When one fails today it falls back to heuristics or logs a line and
the failure is invisible.

``MaintenanceHealth`` keeps one entry per job: the latest outcome, the provider,
model and credential that served it, an error class and a failure count. It is
served at ``GET /v1/health/maintenance`` in the ``{"checks": [...]}`` shape the
BawtHub health aggregator renders as child rows, so a failing job becomes one
bell warning (repeats collapse into the count) that clears on the job's next
success.

State is process-local: a restart forgets failures, and the next scheduled run
of that job re-records them.

``classify_failure`` is the shared error classifier slice B's failover uses:
it reads the provider's error *code* first and the HTTP status second.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .logging import get_service_logger

log = get_service_logger(__name__)

# Error classes. Only ``spending_limit`` will ever justify credential failover.
SPENDING_LIMIT = "spending_limit"
RATE_LIMIT = "rate_limit"
AUTH = "auth"
TIMEOUT = "timeout"
UNAVAILABLE = "unavailable"
EMPTY_RESPONSE = "empty_response"
UNKNOWN = "unknown"

_SPENDING_CODE = re.compile(r"[\w-]*:spending-limit|insufficient_quota", re.I)
_STATUS_IN_TEXT = re.compile(r"(?:error code|status(?: code)?|HTTP)[:\s]+(\d{3})", re.I)


def _status_of(error: Any) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = _STATUS_IN_TEXT.search(str(error))
    return int(match.group(1)) if match else None


def classify_failure(error: Any) -> str:
    """Error class for an exception or an ``ERROR: …`` reply string.

    Code before status: a 403 is ``spending_limit`` only when the provider's
    code says so; a bare 403 is ``auth``. 429 is always ``rate_limit``.
    """
    text = str(error)
    status = _status_of(error)
    if status == 429:
        # Never spending_limit, whatever the message text says: failing over
        # on a rate-limit blip would burn paid credit.
        return RATE_LIMIT
    if status == 402 or (status in (None, 403) and _SPENDING_CODE.search(text)):
        return SPENDING_LIMIT
    if status in (401, 403):
        return AUTH
    name = type(error).__name__.lower()
    if "timeout" in name or "timed out" in text.lower():
        return TIMEOUT
    if (status is not None and status >= 500) or "connection" in name:
        return UNAVAILABLE
    return UNKNOWN


def safe_detail(error: Any) -> str:
    """Secret-free description: exception type and HTTP status only.

    Raw exception/reply text is never kept: provider errors can echo request
    content or credentials (even in a ``code`` field), and this lands in a bell reason and a log line.
    """
    kind = "error reply" if isinstance(error, str) else type(error).__name__
    parts = [kind]
    status = _status_of(error)
    if status is not None:
        parts.append(f"HTTP {status}")
    return ", ".join(parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobState:
    job: str
    ok: bool
    model: str | None = None
    provider: str | None = None
    credential: str | None = None
    error_class: str | None = None
    detail: str | None = None
    failures: int = 0
    since: str | None = None
    last_at: str | None = None


class MaintenanceHealth:
    """Thread-safe per-job outcome registry (background jobs run in a pool)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def record_success(self, job: str, *, model: str | None, provider: str | None,
                       credential: str | None) -> None:
        with self._lock:
            previous = self._jobs.get(job)
            if previous is not None and not previous.ok:
                log.info("Maintenance job %s recovered after %d failure(s)", job, previous.failures)
            self._jobs[job] = JobState(job=job, ok=True, model=model, provider=provider,
                                       credential=credential, last_at=_now())

    def record_failure(self, job: str, error_class: str, detail: str, *, model: str | None,
                       provider: str | None, credential: str | None) -> None:
        now = _now()
        with self._lock:
            previous = self._jobs.get(job)
            failing = previous is not None and not previous.ok
            self._jobs[job] = JobState(
                job=job, ok=False, model=model, provider=provider, credential=credential,
                error_class=error_class, detail=detail[:200],
                failures=(previous.failures if failing else 0) + 1,
                since=previous.since if failing else now, last_at=now,
            )
        log.error("Maintenance job %s failed (%s) on %s/%s: %s",
                  job, error_class, provider or "?", model or "?", detail[:200])

    def snapshot(self) -> list[JobState]:
        with self._lock:
            return [JobState(**vars(state)) for state in self._jobs.values()]

    def checks_payload(self) -> dict[str, Any]:
        """``{"checks": [{name, status, reason}]}`` for the BawtHub aggregator."""
        checks = []
        for state in sorted(self.snapshot(), key=lambda s: s.job):
            if state.ok:
                checks.append({"name": state.job, "status": "ok", "reason": None})
                continue
            served = "/".join(p for p in (state.provider, state.credential) if p) or "no client"
            times = f"{state.failures}x since {state.since}" if state.failures > 1 else f"at {state.since}"
            checks.append({
                "name": state.job, "status": "down",
                "reason": f"{state.error_class} on {served} ({state.model or 'no model'}), "
                          f"{times}: {state.detail}",
            })
        return {"checks": checks}


UNCONFIGURED = "unconfigured"


def record_unconfigured(job: str, setting: str) -> None:
    """A job that can't run because its model setting is unset fails visibly too."""
    _registry.record_failure(job, UNCONFIGURED, f"{setting} is not set",
                             model=None, provider=None, credential=None)


_registry = MaintenanceHealth()


def maintenance_health() -> MaintenanceHealth:
    return _registry


class MonitoredClient:
    """Delegating proxy that records each ``query``/``query_bounded`` outcome.

    Clients sometimes return an ``ERROR: …`` string instead of raising; that
    counts as a failure too. The reply/exception reaches the caller unchanged.
    """

    def __init__(self, client: Any, *, job: str, model: str | None, provider: str | None,
                 credential: str | None, health: MaintenanceHealth | None = None) -> None:
        self._client = client
        self._job = job
        self._labels = {"model": model, "provider": provider, "credential": credential}
        self._health = health or _registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def wrapped(self) -> Any:
        return self._client

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = getattr(self._client, method)(*args, **kwargs)
        except Exception as error:
            self._health.record_failure(self._job, classify_failure(error), safe_detail(error), **self._labels)
            raise
        if isinstance(result, str) and result.lstrip().startswith("ERROR:"):
            self._health.record_failure(self._job, classify_failure(result), safe_detail(result), **self._labels)
        elif isinstance(result, str) and not result.strip():
            self._health.record_failure(self._job, EMPTY_RESPONSE, "empty completion", **self._labels)
        else:
            self._health.record_success(self._job, **self._labels)
        return result

    def query(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("query", *args, **kwargs)

    def query_bounded(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("query_bounded", *args, **kwargs)
