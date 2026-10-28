"""TASK-1169 S3t: competing dispatchers and crash windows, on real PostgreSQL.

Conditional claims are a starting point, not proof of replica safety. Each test
runs two ``OpsService`` instances on separate engines (two processes sharing
one ledger) over ONE fake Docker daemon, driven through the real
``DockerExecutor``: its deterministic worker identity, receipt directory and
reconcile rules are the code under test, not a stand-in. Ordering comes from
Events and observed PostgreSQL lock waits; a crash is an exception raised out
of the dispatcher at the named point.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.ops import (JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, JOB_LOST, JOB_QUEUED, JOB_RUNNING,
                          OpsService)
from llm_bawt.ops.executor import DockerExecutor, worker_request
from llm_bawt.ops.store import OpsStore, slot_holders
from llm_bawt.ops.worker import atomic_json
from llm_bawt.ops.validation import canonical_json
from tests.pg_lock_wait import wait_blocked_by
from tests.test_ops_executor import IMAGE, FakeContainer, setup as docker_setup
from tests.test_ops_quiesce_pg import _admission
from tests.test_ops_service import _op_data

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

HANG_GUARD_S = 30.0
TARGET = "test"  # _op_data's container_name.


class Crash(RuntimeError):
    """The dispatcher process dies here: not an ExecutorError, so nothing handles it."""


class CountingDocker(DockerExecutor):
    """The real executor; counts submissions this instance made."""

    def __init__(self, client, root):
        super().__init__(client_factory=lambda: client, worker_image=IMAGE,
                         receipt_volume="receipts", receipt_root=str(root))
        self.submits = 0

    def dispatch(self, **kwargs):
        self.submits += 1
        return super().dispatch(**kwargs)


@pytest.fixture
def rig(private_app_pg, tmp_path, monkeypatch):
    _executor, client, _snapshot = docker_setup(tmp_path, monkeypatch)
    client.containers.rows[TARGET] = FakeContainer(status="running", labels={
        "com.docker.compose.project": "test", "com.docker.compose.service": "fake"})  # Not a GPU voice service.
    first = OpsStore(private_app_pg)
    # The other process: its own engine, so its own connections.
    second = OpsStore(private_app_pg, engine=create_engine(first.engine.url, pool_pre_ping=True))
    engines = [first.engine, second.engine]
    stores = [first, second]
    executors = [CountingDocker(client, tmp_path) for _ in stores]
    services = [OpsService(store, executor=executor) for store, executor in zip(stores, executors)]
    slugs = []

    def operation(**over):
        slug = f"test.compete-{uuid.uuid4().hex[:12]}"
        # The real DockerExecutor here runs the container launcher (TASK-1169 S4).
        stores[0].create_operation(_op_data(slug=slug, **{"execution_mode": "isolated", **over}))
        slugs.append(slug)
        return slug

    app = _admission(SimpleNamespace(store=stores[1], service=services[0]))

    def admit(slug):
        """The app admits (QUEUED row only); no dispatcher has looked yet."""
        return app.dispatch_job(operation_slug=slug, args={"target": "a"},
                                idempotency_key=uuid.uuid4().hex)["id"]

    def sql(statement, **params):
        with engines[0].begin() as conn:
            return conn.execute(text(statement), params)

    def past_grace(job_id):
        # get_job_status leaves a DISPATCHING job alone for 30s after its claim.
        sql("UPDATE ops_jobs SET dispatched_at = dispatched_at - interval '1 hour' WHERE id = :id", id=job_id)

    def creates():
        return len(client.containers.creates)

    yield SimpleNamespace(stores=stores, services=services, executors=executors, client=client, root=tmp_path,
                          app=app, operation=operation, admit=admit, sql=sql, past_grace=past_grace,
                          creates=creates, job=stores[0].get_job)
    try:
        for slug in slugs:
            sql("DELETE FROM ops_jobs WHERE operation_slug = :s", s=slug)
            sql("DELETE FROM ops_operation_revisions WHERE operation_slug = :s", s=slug)
            sql("DELETE FROM ops_operations WHERE slug = :s", s=slug)
    finally:
        engines[1].dispose()


# ── 1. Two dispatchers, one queued job ──────────────────────────────────────

def test_a_second_dispatcher_leaves_a_job_another_is_dispatching_untouched(rig):
    job_id = rig.admit(rig.operation())
    entered, release = threading.Event(), threading.Event()
    first, second = rig.executors
    original = first.preflight

    def held(snapshot):
        entered.set()
        assert release.wait(HANG_GUARD_S)
        original(snapshot)

    first.preflight = held
    with ThreadPoolExecutor(1) as pool:
        winner = pool.submit(rig.services[0]._dispatch_queued, rig.job(job_id))
        try:
            assert entered.wait(HANG_GUARD_S)
            # Mid-preflight on instance 0: instance 1's pass returns without effect.
            rig.services[1]._dispatch_queued(rig.stores[1].get_job(job_id))
            loser_view = rig.job(job_id)
            assert (loser_view.state, loser_view.error_text, loser_view.dispatched_at) == (JOB_QUEUED, None, None)
        finally:
            release.set()  # Before the pool's exit joins the held winner.
        winner.result(timeout=HANG_GUARD_S)
    assert (first.submits, second.submits, rig.creates()) == (1, 0, 1)
    assert rig.job(job_id).state == JOB_ACCEPTED
    # A later pass on either instance submits nothing more.
    for service in rig.services:
        service._dispatch_queued(rig.job(job_id))
    assert (first.submits, second.submits, rig.creates()) == (1, 0, 1)


def test_two_concurrent_claims_on_one_job_let_exactly_one_win(rig):
    """The claim CAS itself. Both claims queue behind a held operation-row lock,
    then run one after the other; the QUEUED condition is what refuses the
    second, so this fails for any interleaving if that condition is dropped."""
    slug = rig.operation()
    job_id = rig.admit(slug)
    holder = rig.stores[0].engine.connect()
    tx = holder.begin()
    pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
    holder.execute(text("SELECT 1 FROM ops_operations WHERE slug = :s FOR UPDATE"), {"s": slug})
    with ThreadPoolExecutor(2) as pool:
        claims = [pool.submit(store.claim_job, job_id) for store in rig.stores]
        try:
            wait_blocked_by(rig.stores[0].engine, pid, claims[0])
        finally:
            # Released before the pool's exit joins claimers blocked on it, so
            # a failed wait fails the test instead of hanging it.
            tx.commit()
            holder.close()
        results = sorted(claim.result(timeout=HANG_GUARD_S) for claim in claims)
    assert results == [False, True]
    job = rig.job(job_id)
    assert (job.state, job.error_text) == (JOB_DISPATCHING, None)


# ── 2. N jobs × 2 dispatchers, max_concurrent=k ─────────────────────────────

def test_competing_dispatchers_never_exceed_the_concurrency_cap(rig):
    k, n = 2, 6
    slug = rig.operation(max_concurrent=k)
    jobs = [rig.admit(slug) for _ in range(n)]
    observed = []
    lock = threading.Lock()
    for executor, store in zip(rig.executors, rig.stores):
        original = executor.dispatch

        def watched(_original=original, _store=store, **kwargs):
            # A submission happens only inside a claimed slot: count the slot
            # holders the ledger shows at this moment.
            with _store.engine.connect() as conn:
                holding = len(slot_holders(conn, slug))
            with lock:
                observed.append(holding)
            return _original(**kwargs)

        executor.dispatch = watched
    start = threading.Barrier(2, timeout=HANG_GUARD_S)

    def drive(service, order):
        start.wait()
        for job_id in order:
            service._dispatch_queued(rig.job(job_id))

    with ThreadPoolExecutor(2) as pool:
        runs = [pool.submit(drive, rig.services[0], jobs), pool.submit(drive, rig.services[1], jobs[::-1])]
        for run in runs:
            run.result(timeout=HANG_GUARD_S)
    states = [rig.job(job_id).state for job_id in jobs]
    assert observed and max(observed) <= k
    assert states.count(JOB_ACCEPTED) == k and states.count(JOB_QUEUED) == n - k
    assert len(observed) == k == rig.creates()
    # Each admitted job reached Docker at most once: one create per worker name.
    names = [config["name"] for _image, config in rig.client.containers.creates]
    assert len(names) == len(set(names))


# ── 3. Crash after the claim, before any submission ─────────────────────────

def test_a_claim_whose_dispatcher_died_before_submitting_is_lost_not_resubmitted(rig):
    """What the code does: the claim commits (DISPATCHING), the process dies
    before Docker is asked. The other instance leaves it alone through the
    30 s claim grace, then reconciles the deterministic worker identity, finds
    no worker, marks the job LOST ("side effects unknown, not replayed") and
    writes abandoned.json, so no later dispatch of that id can create one."""
    job_id = rig.admit(rig.operation())
    first, second = rig.executors

    def die(**_kwargs):
        raise Crash("killed between claim and submit")

    first.dispatch = die
    with pytest.raises(Crash):
        rig.services[0]._dispatch_queued(rig.job(job_id))
    assert rig.job(job_id).state == JOB_DISPATCHING
    rig.services[1].get_job_status(job_id)  # Inside the grace: untouched.
    assert rig.job(job_id).state == JOB_DISPATCHING
    rig.past_grace(job_id)
    rig.services[1].get_job_status(job_id)
    job = rig.job(job_id)
    assert (job.state, job.error_text) == (JOB_LOST, "worker missing; side effects unknown, not replayed")
    assert (rig.root / job_id / "abandoned.json").exists()
    for service in rig.services:
        service.get_job_status(job_id)
    assert (second.submits, rig.creates()) == (0, 0)


# ── 4. Crash after the submission, before the RUNNING write ─────────────────

def test_a_submitted_worker_whose_dispatcher_died_is_adopted_not_resubmitted(rig):
    job_id = rig.admit(rig.operation())
    first, second = rig.executors
    submit = first.dispatch

    def submit_then_die(**kwargs):
        submit(**kwargs)  # Docker created and started the worker...
        raise Crash("killed before the ledger heard back")  # ...and nobody wrote it down.

    first.dispatch = submit_then_die
    with pytest.raises(Crash):
        rig.services[0]._dispatch_queued(rig.job(job_id))
    assert (rig.job(job_id).state, rig.creates()) == (JOB_DISPATCHING, 1)
    rig.past_grace(job_id)
    rig.services[1].get_job_status(job_id)
    assert rig.job(job_id).state == JOB_ACCEPTED  # Found by its identity: llm-bawt-ops-<job id>.
    # The worker reports running; the next pass adopts that, still without a submission.
    snapshot = json.loads(rig.job(job_id).invocation_snapshot_json)
    digest = hashlib.sha256(canonical_json(worker_request(job_id, snapshot)).encode()).hexdigest()
    atomic_json(rig.root / job_id / "receipt.json", {"job_id": job_id, "request_hash": digest,
                                                     "state": "running", "started_at": "2026-10-28T12:00:00+00:00"})
    rig.services[1].get_job_status(job_id)
    assert rig.job(job_id).state == JOB_RUNNING
    assert (second.submits, rig.creates()) == (0, 1)
    assert rig.client.containers.rows[DockerExecutor.worker_name(job_id)].starts == 1


# ── 5. Catalog drift between admission and claim ────────────────────────────

def test_a_catalog_edit_after_admission_runs_the_admitted_snapshot(rig):
    """The code's rule (OpsService.prepare_invocation): the job executes the
    exact revision it was admitted with, whatever the catalog says by claim
    time. Only disable/delete is a claim-time interlock (next test)."""
    slug = rig.operation()
    job_id = rig.admit(slug)
    admitted = json.loads(rig.job(job_id).invocation_snapshot_json)
    rig.stores[0].update_operation(slug, {"command_script": '{"action":"restart","container_name":"other"}'})
    rig.services[1]._dispatch_queued(rig.job(job_id))
    assert rig.job(job_id).state == JOB_ACCEPTED
    request = json.loads((rig.root / job_id / "request.json").read_text())
    assert request["snapshot"] == admitted and request["snapshot"]["spec"]["container_name"] == TARGET


@pytest.mark.parametrize("change", ["disable", "delete"])
def test_an_operation_disabled_after_admission_is_refused_at_claim(rig, change):
    slug = rig.operation()
    job_id = rig.admit(slug)
    if change == "disable":
        rig.stores[0].update_operation(slug, {"enabled": False})
    else:
        rig.stores[0].soft_delete_operation(slug)
    rig.services[1]._dispatch_queued(rig.job(job_id))
    job = rig.job(job_id)
    assert (job.state, job.error_text) == (JOB_FAILED, "operation disabled before claim; no action submitted")
    assert job.dispatched_at is None and rig.creates() == 0


def test_a_disable_racing_the_claim_refuses_it(rig):
    """Disabled after the dispatcher's own check, before its claim: the claim
    re-reads the operation under its row lock and refuses."""
    slug = rig.operation()
    job_id = rig.admit(slug)
    executor = rig.executors[1]
    original = executor.preflight

    def disable_then_pass(snapshot):
        original(snapshot)
        rig.stores[0].update_operation(slug, {"enabled": False})

    executor.preflight = disable_then_pass
    rig.services[1]._dispatch_queued(rig.job(job_id))
    job = rig.job(job_id)
    assert (job.state, job.dispatched_at, rig.creates()) == (JOB_QUEUED, None, 0)
    executor.preflight = original
    rig.services[1]._dispatch_queued(rig.job(job_id))
    assert rig.job(job_id).state == JOB_FAILED and rig.creates() == 0


# ── 6. The admission-only app never reaches Docker ──────────────────────────

def test_the_admission_only_app_creates_the_row_and_only_the_reconciler_runs_it(rig, monkeypatch):
    import docker

    calls = []

    def unreachable(*args, **kwargs):
        calls.append(args)
        raise docker.errors.DockerException("no Docker socket in the app container")

    monkeypatch.setattr(docker, "from_env", unreachable)
    job_id = rig.admit(rig.operation())
    assert rig.job(job_id).state == JOB_QUEUED
    status = rig.app.get_job_status(job_id)  # A read; the app dispatches nothing.
    assert status["state"] == JOB_QUEUED and rig.creates() == 0
    rig.services[0].get_job_status(job_id)  # The reconciler's per-job pass (reconcile_active_jobs).
    assert rig.job(job_id).state == JOB_ACCEPTED
    assert (calls, rig.executors[0].submits, rig.executors[1].submits, rig.creates()) == ([], 1, 0, 1)
