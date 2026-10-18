"""Internal capability checks for the consolidated system-health view (TASK-1157).

Container liveness says a process exists; these say whether the capabilities
inside it work (e.g. the ops executor's receipt mount — a running app with a
missing mount silently parked every ops job in QUEUED for TASK-1145).

Response contract, consumed by BawtHub's ``/api/system/health`` aggregator,
which expands ``checks`` into child rows::

    {"status": "ok"|"degraded"|"down",
     "checks": [{"name": str, "status": str, "reason": str | None}]}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

OK, DEGRADED, DOWN = "ok", "degraded", "down"
_RANK = {OK: 0, DEGRADED: 1, DOWN: 2}

# A queued job with no preflight error should be claimed within a few
# reconciler ticks (default 5s); an active job is re-reconciled every tick.
QUEUED_STALE_SECONDS = 120
ACTIVE_UNRECONCILED_SECONDS = 120


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
        return _check("database", DOWN, f"{type(exc).__name__}: {exc}")


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
        return _check("redis", DOWN, f"{type(exc).__name__}: {exc}")


def check_ops_executors(ops) -> list[dict]:
    rows = []
    for kind, executor in sorted(ops.executors().items()):
        try:
            reason = executor.readiness()
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
        rows.append(_check(f"ops.executor.{kind}", DOWN if reason else OK, reason))
    return rows


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


def collect(config: Any, ops_store, ops_service) -> dict:
    checks = [
        check_database(ops_store.engine),
        check_redis(getattr(config, "REDIS_URL", "") or ""),
        *check_ops_executors(ops_service),
        check_ops_queue(ops_store),
    ]
    return {"status": overall(checks), "checks": checks}
