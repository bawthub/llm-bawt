"""TASK-1183: opt-in bell entries for an operation's job start and finish."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select, text
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.ops import JOB_CANCELLED, JOB_FAILED, JOB_LOST, JOB_SUCCEEDED, JOB_TIMED_OUT, OpsStore
from llm_bawt.ops import bell
from llm_bawt.ops.bell import FINISHED, STARTED, OpsBellOutbox
from llm_bawt.service import ops_bell_pump


def _store():
    store = object.__new__(OpsStore)
    store.engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                                 poolclass=StaticPool)
    store.events = None
    store._ensure_tables_exist()
    return store


def _op(store, **over):
    data = dict(slug="test.noop", title="No-op check", description="", enabled=True, target_host="",
                command_script='{"action":"restart","container_name":"test"}',
                args_schema_json='{"type":"object","additionalProperties":false,"properties":{}}',
                args_defaults_json="{}", timeout_seconds=60, start_delay_seconds=0,
                risk_level="low", category="restart")
    data.update(over)
    return store.create_operation(data)


def _job(store, op, *, key="k1", notify=True, legacy=False, user="test-user"):
    job = store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key=key,
                           caller_user_id=user, caller_bot_id="some-bot")
    operation = op.to_api()
    if legacy:
        operation.pop("notify_bell")
    else:
        operation["notify_bell"] = notify
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET invocation_snapshot_json=:s WHERE id=:id"),
                     {"s": json.dumps({"operation": operation}), "id": job.id})
    return job


def _rows(store):
    with store.engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(
            select(OpsBellOutbox.__table__).order_by(OpsBellOutbox.__table__.c.id))]


def test_flag_defaults_off_round_trips_and_is_validated():
    store = _store()
    op = _op(store)
    assert op.notify_bell is False and op.to_api()["notify_bell"] is False
    updated = store.update_operation(op.id, {"notify_bell": True})
    assert updated.notify_bell is True and updated.version == op.version + 1
    try:
        store.update_operation(op.id, {"notify_bell": "yes"})
    except ValueError as exc:
        assert "notify_bell" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-boolean notify_bell accepted")


def test_start_and_finish_stage_once_and_losing_writes_stage_nothing():
    store = _store()
    job = _job(store, _op(store))
    store.mark_dispatching(job.id, host_unit_name="u")
    assert _rows(store) == []
    store.mark_running(job.id)
    store.mark_running(job.id)  # lost CAS
    store.mark_terminal(job.id, state=JOB_SUCCEEDED, exit_code=0)
    store.mark_terminal(job.id, state=JOB_FAILED, exit_code=1)  # lost CAS
    assert [(r["event_key"], r["phase"]) for r in _rows(store)] == [
        (f"{job.id}:started", STARTED), (f"{job.id}:finished", FINISHED)]


def test_terminal_backfills_start_only_from_started_at():
    store = _store()
    op = _op(store)
    ran = _job(store, op, key="ran")
    store.mark_dispatching(ran.id, host_unit_name="u")
    store.mark_terminal(ran.id, state=JOB_SUCCEEDED, exit_code=0,
                        started_at=datetime.now(timezone.utc))
    never = _job(store, op, key="never")
    store.mark_terminal(never.id, state=JOB_CANCELLED)
    phases = {(r["job_id"], r["phase"]) for r in _rows(store)}
    assert phases == {(ran.id, STARTED), (ran.id, FINISHED), (never.id, FINISHED)}


def test_off_and_legacy_snapshots_stay_silent():
    store = _store()
    op = _op(store)
    for key, kwargs in (("off", {"notify": False}), ("legacy", {"legacy": True})):
        job = _job(store, op, key=key, **kwargs)
        store.mark_dispatching(job.id, host_unit_name="u")
        store.mark_running(job.id)
        store.mark_terminal(job.id, state=JOB_SUCCEEDED, exit_code=0)
    assert _rows(store) == []


def test_editing_the_operation_never_changes_an_admitted_job():
    store = _store()
    op = _op(store)
    job = _job(store, op, notify=False)
    store.update_operation(op.id, {"notify_bell": True})
    store.mark_dispatching(job.id, host_unit_name="u")
    store.mark_running(job.id)
    assert _rows(store) == []


def test_staging_failure_never_fails_the_job_write():
    store = _store()
    job = _job(store, _op(store))
    OpsBellOutbox.__table__.drop(store.engine)
    store.mark_dispatching(job.id, host_unit_name="u")
    assert store.mark_running(job.id).state == "running"
    assert store.get_job(job.id).state == "running"


def test_severity_and_payload_never_name_the_calling_bot():
    store = _store()
    op = _op(store)
    expected = {JOB_SUCCEEDED: "info", JOB_FAILED: "warning", JOB_TIMED_OUT: "warning",
                JOB_CANCELLED: "warning", JOB_LOST: "critical"}
    for i, (state, severity) in enumerate(expected.items()):
        job = _job(store, op, key=f"s{i}")
        store.mark_dispatching(job.id, host_unit_name="u")
        store.mark_running(job.id)
        store.mark_terminal(job.id, state=state, exit_code=0 if state == JOB_SUCCEEDED else None,
                            error_text=None if state == JOB_SUCCEEDED else "boom")
        row = store.get_job(job.id)
        finished = bell.payload_for(row, FINISHED)
        started = bell.payload_for(row, STARTED)
        assert finished["severity"] == severity and finished["state"] == state
        assert started["severity"] == "info" and started["state"] == "running"
        assert finished["userId"] == "test-user" and finished["operation"] == "test.noop"
        assert "No-op check" in finished["title"] and job.id in finished["body"]
        assert "some-bot" not in json.dumps([started, finished])
    assert bell.payload_for(store.get_job(job.id), FINISHED)["body"].count("boom") == 1


def test_missing_user_is_null_not_guessed():
    store = _store()
    job = _job(store, _op(store), user=None)
    assert bell.payload_for(store.get_job(job.id), STARTED)["userId"] is None


class _Receiver:
    def __init__(self, fail=0):
        self.fail = fail
        self.posts = []

    def handler(self, request):
        self.posts.append(json.loads(request.content))
        if self.fail:
            self.fail -= 1
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})


def _deliver(store, receiver):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(receiver.handler),
                                     base_url="http://bawthub") as http:
            return await ops_bell_pump.deliver(store.engine, http)
    return asyncio.run(go())


def _finish(store, job):
    store.mark_dispatching(job.id, host_unit_name="u")
    store.mark_running(job.id)
    store.mark_terminal(job.id, state=JOB_SUCCEEDED, exit_code=0)


def test_finish_waits_for_its_start_and_retries_after_an_outage():
    store = _store()
    job = _job(store, _op(store))
    _finish(store, job)
    receiver = _Receiver(fail=1)
    assert _deliver(store, receiver) == (1, 0)  # only the start; it fails
    assert [p["phase"] for p in receiver.posts] == [STARTED]
    assert _deliver(store, receiver) == (0, 0)  # start backing off; finish still waits
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_bell_outbox SET next_attempt_at=:t"),
                     {"t": datetime.now(timezone.utc) - timedelta(seconds=1)})
    assert _deliver(store, receiver) == (1, 1)
    assert _deliver(store, receiver) == (1, 1)
    assert _deliver(store, receiver) == (0, 0)
    assert [p["phase"] for p in receiver.posts] == [STARTED, STARTED, FINISHED]
    rows = _rows(store)
    assert all(r["delivered_at"] is not None for r in rows)
    assert rows[0]["attempts"] == 2 and rows[0]["last_error"] is None


def test_refused_entry_is_given_up_but_missing_route_retries():
    store = _store()
    job = _job(store, _op(store))
    store.mark_dispatching(job.id, host_unit_name="u")
    store.mark_running(job.id)

    class Status(_Receiver):
        def __init__(self, code):
            super().__init__()
            self.code = code

        def handler(self, request):
            self.posts.append(json.loads(request.content))
            return httpx.Response(self.code)

    assert _deliver(store, Status(404)) == (1, 0)  # older BawtHub mid-rollout
    row = _rows(store)[0]
    assert row["gave_up_at"] is None and row["last_error"] == "HTTP 404"
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_bell_outbox SET next_attempt_at=:t"),
                     {"t": datetime.now(timezone.utc) - timedelta(seconds=1)})
    assert _deliver(store, Status(422)) == (1, 0)
    row = _rows(store)[0]
    assert row["gave_up_at"] is not None and row["delivered_at"] is None
    assert _deliver(store, Status(200)) == (0, 0)


def test_claimed_rows_are_leased_and_old_rows_given_up():
    store = _store()
    job = _job(store, _op(store))
    _finish(store, job)
    first = bell.claim_due(store.engine)
    assert [r["phase"] for r in first] == [STARTED]
    assert bell.claim_due(store.engine) == []  # leased, and the finish waits on it
    later = datetime.now(timezone.utc) + bell.GIVE_UP_AFTER + timedelta(days=1)
    assert bell.claim_due(store.engine, now=later) == []
    assert all(r["gave_up_at"] is not None for r in _rows(store))


def test_pump_marks_notifications_topic_only_after_delivery(monkeypatch):
    from llm_bawt.service import state_events

    store = _store()
    job = _job(store, _op(store))
    _finish(store, job)
    receiver = _Receiver()
    marked = []

    class Service:
        _commit_wake_hub = None

    monkeypatch.setattr(state_events, "mark", lambda service, *topics: marked.extend(topics))

    def http_factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(receiver.handler), base_url="http://b")

    async def go():
        task = asyncio.create_task(ops_bell_pump.run_ops_bell_pump(
            Service(), lambda: store, http_factory=http_factory))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if len(receiver.posts) == 2:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    assert [p["phase"] for p in receiver.posts] == [STARTED, FINISHED]
    assert marked and set(marked) == {state_events.NOTIFICATIONS}
    assert state_events.NOTIFICATIONS in state_events.TOPICS
