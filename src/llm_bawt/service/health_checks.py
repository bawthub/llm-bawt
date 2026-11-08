"""Internal capability checks for the consolidated system-health view (TASK-1157).

Container liveness says a process exists; these say whether the capabilities
inside it work (e.g. the ops executor's receipt mount — a running app with a
missing mount silently parked every ops job in QUEUED for TASK-1145).

Response contract, consumed by BawtHub's ``/api/system/health`` aggregator,
which expands ``checks`` into child rows::

    {"status": "ok"|"degraded"|"down",
     "checks": [{"name": str, "status": str, "reason": str | None}]}

A check may add structured fields after those three (``deliveries.queue``
adds ``items`` and ``item_count``, TASK-1458); the aggregator ignores keys it
doesn't know, so ``reason`` must stand on its own.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..mcp_server.registry import tools_registration_complete
from ..ops.approved_admission import ADMISSION_WINDOW_SECONDS
from .http_errors import client_error_text

OK, DEGRADED, DOWN = "ok", "degraded", "down"
_RANK = {OK: 0, DEGRADED: 1, DOWN: 2}

# A queued job with no preflight error should be claimed within a few
# reconciler ticks (default 5s); an active job is re-reconciled every tick.
QUEUED_STALE_SECONDS = 120
ACTIVE_UNRECONCILED_SECONDS = 120
# An approved ops_run is admitted within a tick or two. The reconciler cannot
# report its own absence, so the app watches the oldest unadmitted approval.
ADMISSION_STALE_SECONDS = 60


def _check(name: str, status: str, reason: str | None = None) -> dict:
    return {"name": name, "status": status, "reason": reason}


def overall(checks: Iterable[dict]) -> str:
    return max((c["status"] for c in checks), key=_RANK.__getitem__, default=OK)


def check_database(engine) -> dict:
    if engine is None:
        return _check("database", DOWN, "database engine unavailable")
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return _check("database", OK)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return _check("database", DOWN, client_error_text(
            exc, "Health check: database unreachable", "Database unreachable"))


def check_redis(url: str) -> dict:
    if not url:
        return _check("redis", DOWN, "REDIS_URL is not configured")
    try:
        import redis

        client = redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)
        try:
            client.ping()
        finally:
            client.close()
        return _check("redis", OK)
    except Exception as exc:  # noqa: BLE001
        return _check("redis", DOWN, client_error_text(
            exc, "Health check: redis unreachable", "Redis unreachable"))


def check_ops_executors(ops) -> list[dict]:
    """Executor readiness as the process that runs them reports it: the ops
    service, for the app's admission-only OpsService (TASK-1169 S3c). When
    that process can't be asked, one ``ops.control`` row says so."""
    from ..ops.service import OpsDispatchError

    try:
        readiness = ops.executor_readiness()
    except OpsDispatchError as exc:
        return [_check("ops.control", DOWN, str(exc))]
    return [_check(f"ops.executor.{kind}", DOWN if reason else OK, reason)
            for kind, reason in sorted(readiness.items())]


def _age_seconds(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()


def check_ops_queue(store, now: datetime | None = None) -> dict:
    """Jobs the pipeline should have moved but has not.

    blocked  = QUEUED with a preflight error (operator must fix prerequisites)
    stale    = QUEUED, no error, unclaimed past QUEUED_STALE_SECONDS
    stranded = active but not reconciled recently (reconciler not running)
    """
    from ..ops.models import JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUEUED, JOB_RUNNING

    if store.engine is None:
        return _check("ops.queue", DOWN, "ops store database unavailable")
    now = now or datetime.now(timezone.utc)
    problems: list[str] = []
    status = OK
    for job in store.list_jobs(state=JOB_QUEUED, limit=50):
        if job.error_text:
            status = DOWN
            problems.append(f"{job.operation_slug} blocked: {job.error_text}")
        elif (_age_seconds(job.submitted_at, now) or 0) > QUEUED_STALE_SECONDS:
            status = max(status, DEGRADED, key=_RANK.__getitem__)
            problems.append(f"{job.operation_slug} queued but unclaimed")
    for state in (JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING):
        for job in store.list_jobs(state=state, limit=50):
            seen = job.last_reconcile_at or job.dispatched_at or job.submitted_at
            if (_age_seconds(seen, now) or 0) > ACTIVE_UNRECONCILED_SECONDS:
                status = max(status, DEGRADED, key=_RANK.__getitem__)
                problems.append(f"{job.operation_slug} {state} but not reconciled (is the reconciler running?)")
    return _check("ops.queue", status, "; ".join(problems) or None)


def check_ops_admission(approvals, now: datetime | None = None) -> dict:
    """Approved ops_run requests the reconciler has not admitted (TASK-1220)."""
    if approvals is None or approvals.engine is None:
        return _check("ops.admission", DOWN, "approval store database unavailable")
    now = now or datetime.now(timezone.utc)
    age = _age_seconds(approvals.oldest_unadmitted_ops_resolved_at(), now)
    if age is None or age <= ADMISSION_STALE_SECONDS:
        return _check("ops.admission", OK)
    status = DOWN if age > ADMISSION_WINDOW_SECONDS else DEGRADED
    return _check("ops.admission", status,
                  f"approved ops_run waiting {int(age)}s for admission (is the ops reconciler running?)")


def check_mcp_tool_registry() -> dict:
    if tools_registration_complete():
        return _check("mcp.tool_registry", OK)
    return _check("mcp.tool_registry", DEGRADED,
                  "MCP tools were not registered at startup; approved MCP calls are refused")


def check_system_events(service) -> dict:
    emitter = getattr(service, "_system_events", None)
    if emitter is None:
        return _check("system.events", DOWN, "event ledger unavailable")
    count, age = emitter.pending_health()
    failures = type(emitter).failed_appends
    if age > 60 or failures:
        return _check("system.events", DEGRADED,
                      f"{count} event(s) unpublished; oldest {int(age)}s; {failures} append failure(s) this process")
    return _check("system.events", OK, f"{count} event(s) awaiting fan-out")


#: Product prompt drift is exported by a daily sync; older drift means it stalled.
PROMPT_DRIFT_STALE_SECONDS = 48 * 3600


def check_prompt_sync(config: Any, now: datetime | None = None) -> dict:
    """Live product prompt edits not yet shipped as code defaults (TASK-2127)."""
    from ..prompt_registry import PromptResolver
    from ..prompt_sync import compute_drift

    resolver = PromptResolver(config)
    if resolver.store.engine is None:
        return _check("prompts.sync", DEGRADED, "prompt template DB unavailable")
    drift = compute_drift(resolver.store.list_rows(), resolver.definitions())
    if not drift:
        return _check("prompts.sync", OK, "product prompts match shipped defaults")
    now = now or datetime.now(timezone.utc)
    ages = [
        (now - datetime.fromisoformat(item.updated_at)).total_seconds()
        for item in drift if item.updated_at
    ]
    oldest = max(ages, default=0.0)
    keys = ", ".join(item.key for item in drift[:5])
    if len(drift) > 5:
        keys += f" and {len(drift) - 5} more"
    days = int(oldest // 86400)
    age = "today" if days == 0 else f"{days} day{'s' if days != 1 else ''}"
    noun = "prompt" if len(drift) == 1 else "prompts"
    reason = (f"{len(drift)} edited {noun} not yet synced to the shipped defaults; "
              f"oldest unsynced edit: {age}. Prompts: {keys}")
    return _check("prompts.sync", DEGRADED if oldest > PROMPT_DRIFT_STALE_SECONDS else OK, reason)


MEDIA_PROBE_TIMEOUT_SECONDS = 3.0


def _probe_media_root(root: Path) -> tuple[str, str | None]:
    try:
        try:
            os.listdir(root)  # not exists(): it can hide ESTALE as "missing"
            return OK, None
        except FileNotFoundError:
            # FsBlobBackend creates it on first write; the nearest existing
            # ancestor must still be readable.
            ancestor = root.parent
            while True:
                try:
                    os.listdir(ancestor)
                    return OK, f"{root} not created yet"
                except FileNotFoundError:
                    if ancestor.parent == ancestor:
                        raise
                    ancestor = ancestor.parent
    except OSError as exc:
        return DOWN, (f"media directory {root} is unreadable: "
                      f"{type(exc).__name__}: {exc.strerror or exc}")


class _MediaProbe:
    """At most one filesystem probe in flight per process.

    A hung mount blocks ``listdir`` uncancellably, so each health poll must
    not start another worker: while the previous one is stuck, polls wait on
    it (bounded) and report down.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._root: Path | None = None
        self._started = 0.0
        self._result: tuple[str, str | None] | None = None

    def _start_locked(self, root: Path) -> threading.Thread:
        def run() -> None:
            result = _probe_media_root(root)
            with self._lock:
                if self._worker is threading.current_thread():
                    self._result = result

        self._root, self._result = root, None
        self._started = time.monotonic()
        self._worker = threading.Thread(target=run, name="media-root-probe", daemon=True)
        self._worker.start()
        return self._worker

    def check(self, root: Path, timeout: float) -> dict:
        with self._lock:
            worker = self._worker
            if worker is None or not worker.is_alive():
                worker = self._start_locked(root)
        worker.join(timeout)
        with self._lock:
            if worker.is_alive():
                stuck = time.monotonic() - self._started
                return _check("media.storage", DOWN,
                              f"media directory {self._root} has not responded for {stuck:.0f}s")
            result, probed = self._result, self._root
        if result is None or probed != root:
            # A previous hung probe just finished, or it was for another root:
            # probe the requested root now (still single-flight).
            with self._lock:
                worker = self._worker if self._worker.is_alive() else self._start_locked(root)
            worker.join(timeout)
            with self._lock:
                if worker.is_alive() or self._result is None or self._root != root:
                    return _check("media.storage", DOWN,
                                  f"media directory {root} did not respond within {timeout:g}s")
                result = self._result
        status, reason = result
        return _check("media.storage", status, reason)


_media_probe = _MediaProbe()


def check_media_root(root: Path | None = None, *, timeout: float = MEDIA_PROBE_TIMEOUT_SECONDS) -> dict:
    """Whether the FS media blob root is readable (TASK-2149).

    A network mount can go stale under a running container ("Stale file
    handle") while the host path is fine; reads from it then fail until the
    mount is fixed. Skipped (ok) when blobs live only in S3. The probe runs
    on a single in-flight worker so a hung mount reports down without
    stalling the health response or piling up blocked threads.
    """
    from ..media.object_store import _env_backend_kind, _env_bool
    from ..media.store import resolve_media_root

    if root is None:
        if _env_backend_kind() != "fs" and not _env_bool("LLM_BAWT_S3_FALLBACK_FS"):
            return _check("media.storage", OK, "object storage; no local media directory")
        root = resolve_media_root()
    return _media_probe.check(Path(root), timeout)


def collect(config: Any, ops_store, ops_service, *, service=None) -> dict:
    from .runtime_health import RuntimeHealthChecks

    checks = [
        check_database(ops_store.engine),
        check_redis(getattr(config, "REDIS_URL", "") or ""),
    ]
    from .bridge_gate import bridge_gates

    probes = [
        ("ops.executors", lambda: check_ops_executors(ops_service)),
        ("ops.queue", lambda: [check_ops_queue(ops_store)]),
        ("bridge_gate", lambda: bridge_gates().health_checks()),
        ("prompts.sync", lambda: [check_prompt_sync(config)]),
        ("media.storage", lambda: [check_media_root()]),
    ]
    if service is not None:
        runtime = RuntimeHealthChecks(service)
        probes.extend([
            ("deliveries.queue", lambda: [runtime.deliveries()]),
            ("scheduler", runtime.scheduler),
            ("ops.admission", lambda: [check_ops_admission(
                getattr(service, "_tool_approval_policy_store", None))]),
            ("mcp.tool_registry", lambda: [check_mcp_tool_registry()]),
            ("system.events", lambda: [check_system_events(service)]),
        ])
    for name, probe in probes:
        try:
            checks.extend(probe())
        except Exception as exc:  # A broken capability must not hide the others.
            checks.append(_check(name, DOWN, f"{type(exc).__name__}: " + client_error_text(
                exc, f"Health check {name} failed", "Check failed")))
    return {"status": overall(checks), "checks": checks}
