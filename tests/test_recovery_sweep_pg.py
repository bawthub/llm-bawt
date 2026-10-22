"""TASK-1232 Phase 2: the recovery sweep against real PostgreSQL."""
from __future__ import annotations

import asyncio
import json
import threading
import uuid

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.recovery.orphan_turns import ProbeResult
from llm_bawt.service.recovery.sweep import RECOVERY_LOCK_KEY, RecoverySweep, SweepRecoveryHooks
from llm_bawt.service.turn_logs import TurnLogStore
from tests.test_inter_bot_delivery import _enqueue, delivery_store  # noqa: F401  (fixture)
from tests.test_purge_ops_pg import _dispatching, _ops_run

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(autouse=True)
def approvals(delivery_store):  # noqa: F811
    """Bootstraps every table the sweep reads, as the app does at startup."""
    from llm_bawt.approval_policies import ToolApprovalPolicyStore
    from llm_bawt.ops.store import OpsStore

    store, _, _ = delivery_store
    approvals = ToolApprovalPolicyStore(store.config)
    OpsStore(store.config)
    assert approvals.engine is not None
    yield approvals
    with approvals.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM ops_jobs"))
        conn.execute(sa_text("DELETE FROM tool_approval_requests"))


class Probe:
    def __init__(self, result: ProbeResult):
        self.result, self.calls = result, []

    async def probe(self, turn):
        self.calls.append(turn["turn_id"])
        return self.result


def _old_turn(store, bot, *, status="streaming", request_id="req-orphan", age_seconds=900, turn_id=None):
    turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                   response_text, agent_session_key, agent_request_id, ended_at)
            VALUES (:id, now() - make_interval(secs => :age), '/test', true, :bot, 'nick', :status, '', '',
                    :key, :req, NULL)
        """), {"id": turn_id, "age": age_seconds, "bot": bot, "status": status,
               "key": f"{bot}:nick" if request_id else None, "req": request_id})
    return turn_id


def _step(report, name):
    return next(step for step in report["steps"] if step["step"] == name)


def _ids(entries, key="turn_id"):
    return [entry[key] for entry in entries]


NOT_FOUND = ProbeResult("not_found", backend_matched=True, backend="claude-code", boot_id="boot-2",
                        stranded_result={"text": "the reply nobody read", "stream_id": "1792628773953-1"})


def test_a_not_found_orphan_settles_keeps_its_stranded_reply_and_a_rerun_is_a_no_op(delivery_store):  # noqa: F811
    store, _, target = delivery_store
    turn_id = _old_turn(store, target)
    sweep = RecoverySweep(store.engine, probe=Probe(NOT_FOUND))

    dry = asyncio.run(sweep.run(dry_run=True, bots=[target], steps=["orphan_turns"]))
    assert _ids(_step(dry, "orphan_turns")["planned"]) == [turn_id]
    assert TurnLogStore(store.config).get_turn(turn_id).ended_at is None

    report = asyncio.run(sweep.run(dry_run=False, bots=[target], steps=["orphan_turns"]))
    json.dumps(report)  # the admin API returns it as-is
    assert _ids(_step(report, "orphan_turns")["done"]) == [turn_id]
    row = TurnLogStore(store.config).get_turn(turn_id)
    assert (row.status, row.end_reason) == ("aborted", "aborted") and row.ended_at is not None  # A1
    recovery = json.loads(row.request_json)["recovery"]
    assert (recovery["cause"], recovery["actor"], recovery["probe"]["boot_id"]) == (
        "orphaned_by_restart", "recovery_sweep", "boot-2")
    assert recovery["sweep_run_id"] == report["sweep_run_id"]
    assert recovery["stranded_result"] == NOT_FOUND.stranded_result

    # N2: a re-entered leadership hook finds nothing left to do.
    asyncio.run(SweepRecoveryHooks(sweep).on_leadership_acquired())
    assert TurnLogStore(store.config).get_turn(turn_id).ended_at == row.ended_at


@pytest.mark.parametrize("result", [
    None,  # no bridge answers run.status yet
    ProbeResult("not_found", backend_matched=False, backend="codex"),
    ProbeResult("running", backend_matched=True, backend="claude-code"),
], ids=["unsupported", "wrong_backend", "running"])
def test_an_ambiguous_probe_leaves_the_turn_and_reports_why(delivery_store, result):  # noqa: F811
    store, _, target = delivery_store
    turn_id = _old_turn(store, target)
    report = asyncio.run(RecoverySweep(store.engine, probe=Probe(result) if result else None).run(
        dry_run=False, bots=[target], steps=["orphan_turns"]))
    step = _step(report, "orphan_turns")
    assert step["done"] == [] and _ids(step["ambiguous"]) == [turn_id]
    assert step["ambiguous"][0]["why"].startswith("run.status ")
    assert TurnLogStore(store.config).get_turn(turn_id).ended_at is None


def test_a_turn_without_a_bridge_request_id_is_never_probed(delivery_store):  # noqa: F811
    store, _, target = delivery_store
    turn_id = _old_turn(store, target, request_id=None)
    probe = Probe(NOT_FOUND)
    step = _step(asyncio.run(RecoverySweep(store.engine, probe=probe).run(
        dry_run=False, bots=[target], steps=["orphan_turns"])), "orphan_turns")
    assert _ids(step["ambiguous"]) == [turn_id] and probe.calls == []


def test_live_owners_and_young_turns_are_not_orphans(delivery_store, approvals):  # noqa: F811
    """S5 + r2-M4: a live TurnExecution, a delivery claim, a freshly leased
    approval continuation and a turn under the minimum age are all excluded."""
    from llm_bawt.service.turn_execution import TurnExecution, turn_executions

    store, sender, target = delivery_store
    # Claim first: a delivery claim reserves only an idle target.
    delivery, _ = _enqueue(store, sender, target, "claimed delivery")
    assert store.claim_next(target, claim_owner="live-owner", steer_capable=False)
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='streaming', created_at=now() - interval '15 minutes',"
            " agent_session_key='k', agent_request_id='r' WHERE id=:id"), {"id": delivery.turn_id})
    young = _old_turn(store, target, age_seconds=10)
    owned = _old_turn(store, target)
    execution = TurnExecution(owned, threading.Event(), is_agent=True, store=TurnLogStore(store.config))
    turn_executions.register(execution)
    request_id, _ = _dispatching(approvals)
    continuation_turn = _old_turn(store, target, turn_id="turn-" + approvals.get_request(request_id).continuation_id)
    probe = Probe(NOT_FOUND)
    sweep = RecoverySweep(store.engine, probe=probe)
    try:
        report = asyncio.run(sweep.run(dry_run=True, bots=[target], steps=["orphan_turns"]))
        assert _step(report, "orphan_turns")["found"] == [] and probe.calls == []
        # The continuation's lease lapses: it is no longer a live owner.
        with store.engine.begin() as conn:
            conn.execute(sa_text("UPDATE tool_approval_requests SET continuation_next_attempt_at ="
                                 " now() - interval '10 minutes' WHERE id=:id"), {"id": request_id})
        report = asyncio.run(sweep.run(dry_run=True, bots=[target], steps=["orphan_turns"]))
        assert _ids(_step(report, "orphan_turns")["found"]) == [continuation_turn]
    finally:
        turn_executions.remove(owned)
    assert young and owned


def _reserved_for(store, bot, owner):
    turn_id = f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                   request_json, response_text, ended_at)
            VALUES (:id, now(), '/v1/chat/completions', false, :bot, 'nick', 'reserved', '', :j, '', NULL)
        """), {"id": turn_id, "bot": bot, "j": json.dumps({"reservation_owner": owner})})
    return turn_id


def _lapse(store, request_id):
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE tool_approval_requests SET continuation_next_attempt_at ="
                             " now() - interval '10 minutes' WHERE id=:id"), {"id": request_id})


def test_a_stalled_continuations_reservation_is_released_only_after_its_claim_is_revoked(delivery_store, approvals):  # noqa: F811
    store, _, target = delivery_store
    request_id, token = _dispatching(approvals)
    turn_id = _reserved_for(store, target, token)
    _lapse(store, request_id)
    step = _step(asyncio.run(RecoverySweep(store.engine).run(
        dry_run=False, bots=[target], steps=["reservations"])), "reservations")
    assert _ids(step["done"]) == [turn_id] and step["done"][0]["action"] == "revoke_and_release"
    assert TurnLogStore(store.config).get_turn(turn_id) is None
    row = approvals.get_request(request_id)
    assert (row.continuation_state, row.continuation_claim_token) == ("pending", None)
    # The stalled claimant resumes: its fence fails, so it can't consume anything.
    with store.engine.begin() as conn:
        assert not approvals.continuation_claim_fence(request_id, token)(conn)


def test_a_renewal_between_detect_and_apply_is_a_reported_no_op(delivery_store, approvals):  # noqa: F811
    from llm_bawt.service.recovery.base import SweepScope
    from llm_bawt.service.recovery.reservations import OrphanedReservationsStep
    from datetime import datetime, timezone

    store, _, target = delivery_store
    request_id, token = _dispatching(approvals)
    turn_id = _reserved_for(store, target, token)
    _lapse(store, request_id)
    step = OrphanedReservationsStep()
    scope = SweepScope(engine=store.engine, now=datetime.now(timezone.utc), dry_run=False, bots=frozenset([target]))
    [candidate] = asyncio.run(step.detect(scope))
    plan = asyncio.run(step.plan(scope, candidate))
    assert plan.action == "revoke_and_release"
    assert approvals.renew_continuation_claim(request_id, token)
    assert asyncio.run(step.apply(scope, candidate, plan)) is None
    assert TurnLogStore(store.config).get_turn(turn_id).status == "reserved"
    assert approvals.get_request(request_id).continuation_claim_token == token


def test_loopy_a_reservation_consumed_between_detect_and_apply_keeps_its_live_claim(delivery_store, approvals):  # noqa: F811
    """Loopy B1: the claimant started the turn after detect, so its claim is live
    although its lease looked stale. The revoke used to run before the turn row
    was locked and commit when the release then failed."""
    from llm_bawt.service.recovery.base import SweepScope
    from llm_bawt.service.recovery.reservations import OrphanedReservationsStep
    from datetime import datetime, timezone

    store, _, target = delivery_store
    request_id, token = _dispatching(approvals)
    turn_id = _reserved_for(store, target, token)
    _lapse(store, request_id)
    step = OrphanedReservationsStep()
    scope = SweepScope(engine=store.engine, now=datetime.now(timezone.utc), dry_run=False, bots=frozenset([target]))
    [candidate] = asyncio.run(step.detect(scope))
    plan = asyncio.run(step.plan(scope, candidate))
    assert plan.action == "revoke_and_release"
    before = approvals.get_request(request_id)
    with store.engine.begin() as conn:  # the claimant consumes the reservation and starts its turn
        conn.execute(sa_text("UPDATE turn_logs SET status='streaming', agent_session_key='k',"
                             " agent_request_id='r' WHERE id=:id"), {"id": turn_id})
    assert asyncio.run(step.apply(scope, candidate, plan)) is None
    after = approvals.get_request(request_id)
    assert (after.continuation_state, after.continuation_claim_token, after.continuation_last_error) == (
        before.continuation_state, token, before.continuation_last_error)
    assert TurnLogStore(store.config).get_turn(turn_id).status == "streaming"
    # The claimant's heartbeat still holds its claim.
    assert approvals.renew_continuation_claim(request_id, token)


def test_a_cancelled_deliverys_reservation_is_released_and_a_live_claim_kept(delivery_store):  # noqa: F811
    store, sender, target = delivery_store
    live, _ = _enqueue(store, sender, target, "live claim")
    assert store.claim_next(target, claim_owner="owner", steer_capable=False)
    sweep = RecoverySweep(store.engine)
    step = _step(asyncio.run(sweep.run(dry_run=False, bots=[target], steps=["reservations"])), "reservations")
    assert step["done"] == [] and TurnLogStore(store.config).get_turn(live.turn_id).status == "reserved"
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='CANCELLED', claim_token=NULL WHERE id=:id"),
                     {"id": live.id})
    step = _step(asyncio.run(sweep.run(dry_run=False, bots=[target], steps=["reservations"])), "reservations")
    assert _ids(step["done"]) == [live.turn_id]
    assert TurnLogStore(store.config).get_turn(live.turn_id) is None


def test_a_requeued_deliverys_reservation_keeps_its_place(delivery_store):  # noqa: F811
    """Vex S2: recover_expired re-arms a QUEUED delivery's turn to reserved on
    purpose; releasing it would let an ordinary chat jump the queue."""
    store, sender, target = delivery_store
    requeued, _ = _enqueue(store, sender, target, "lease expired, requeued")
    assert store.claim_next(target, claim_owner="dead-app", steer_capable=False, lease_seconds=60)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second'"
                             " WHERE id=:id"), {"id": requeued.id})
    assert [row.status for row in store.recover_expired() if row.id == requeued.id] == ["QUEUED"]
    assert TurnLogStore(store.config).get_turn(requeued.turn_id).status == "reserved"
    step = _step(asyncio.run(RecoverySweep(store.engine).run(
        dry_run=False, bots=[target], steps=["reservations"])), "reservations")
    assert step["done"] == [] and step["ambiguous"] == []
    assert TurnLogStore(store.config).get_turn(requeued.turn_id).status == "reserved"


def test_failed_continuations_are_reported_as_one_bounded_summary(delivery_store, approvals):  # noqa: F811
    """Vex S1: failed rows are history; never list every one on every pass."""
    from llm_bawt.service.recovery.maintenance import FAILED_SAMPLE_LIMIT

    store, _, _ = delivery_store
    failed = []
    for _ in range(FAILED_SAMPLE_LIMIT + 2):
        request_id, token = _dispatching(approvals)
        approvals.mark_continuation_failed(request_id, error="boom", max_attempts=1, claim_token=token)
        assert approvals.get_request(request_id).continuation_state == "failed"
        failed.append(request_id)
    step = _step(asyncio.run(RecoverySweep(store.engine).run(dry_run=False, steps=["continuations"])),
                 "continuations")
    [summary] = step["ambiguous"]
    # Other files' rows may share the database; ours are the newest.
    assert summary["count"] >= len(failed) and len(summary["newest"]) == FAILED_SAMPLE_LIMIT
    assert set(summary["newest"]) <= set(failed) and step["done"] == []


def test_one_query_by_run_id_rebuilds_the_sweep_report(delivery_store, approvals):  # noqa: F811
    """Snark ruling (a): every row a sweep writes carries its actor and run id,
    and rows_for_run finds exactly those rows (a deleted reservation's stamp is
    on its producer)."""
    from llm_bawt.service.recovery.audit import rows_for_run
    from llm_bawt.service.tool_call_store import ToolCallStore

    store, sender, target = delivery_store
    # Claim first: a delivery claim reserves only an idle target.
    cancelled, _ = _enqueue(store, sender, target, "cancelled after its claim")
    assert store.claim_next(target, claim_owner="owner", steer_capable=False)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='CANCELLED', claim_token=NULL WHERE id=:id"),
                     {"id": cancelled.id})
    orphan = _old_turn(store, target)
    request_id, token = _dispatching(approvals)
    # Its own target: release_reservation needs an otherwise idle target.
    other = f"{target}-approval"
    stalled = _reserved_for(store, other, token)
    _lapse(store, request_id)
    tools = _old_turn(store, target, request_id=None)
    ToolCallStore(store.engine).save_start(turn_id=tools, bot_id=target, user_id="nick", call_id="c1", tool_name="Bash")
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='error', ended_at=now() WHERE id=:id"), {"id": tools})

    report = asyncio.run(RecoverySweep(store.engine, probe=Probe(NOT_FOUND)).run(
        dry_run=False, bots=[target, other], actor="nick"))
    assert report["actor"] == "nick"
    done = {step["step"]: step["done"] for step in report["steps"]}
    assert _ids(done["orphan_turns"]) == [orphan] and _ids(done["unresolved_tools"]) == [tools]
    assert sorted(_ids(done["reservations"])) == sorted([cancelled.turn_id, stalled])

    rows = rows_for_run(store.engine, report["sweep_run_id"])
    assert sorted(_ids(rows["turns"], "id")) == sorted([orphan, tools])
    assert _ids(rows["approvals"], "id") == [request_id]
    assert _ids(rows["deliveries"], "id") == [cancelled.id]
    assert rows["deliveries"][0]["recovery"]["released_reservation"] == cancelled.turn_id
    assert rows["jobs"] == []
    tool_stamp = next(r for r in rows["turns"] if r["id"] == tools)["recovery"]["tools_interrupted"]
    assert (tool_stamp["count"], tool_stamp["actor"]) == (1, "nick")


def test_an_unowned_ordinary_reservation_is_reported_never_released(delivery_store):  # noqa: F811
    store, _, target = delivery_store
    turn_id = f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                   request_json, response_text, ended_at)
            VALUES (:id, now(), '/v1/chat/completions', false, :bot, 'nick', 'reserved', '', '{}', '', NULL)
        """), {"id": turn_id, "bot": target})
    step = _step(asyncio.run(RecoverySweep(store.engine).run(
        dry_run=False, bots=[target], steps=["reservations"])), "reservations")
    assert _ids(step["ambiguous"]) == [turn_id] and step["done"] == []
    assert TurnLogStore(store.config).get_turn(turn_id).status == "reserved"


def test_a_purged_continuation_whose_claimant_went_quiet_is_suppressed(delivery_store, approvals):  # noqa: F811
    from llm_bawt.service.recovery.audit import RunStamp, rows_for_run
    from llm_bawt.service.recovery.purge_ops import purge_ops

    store, _, _ = delivery_store
    request_id, _ = _dispatching(approvals)
    purge_ops(store.engine, stamp=RunStamp.new("snark"))
    sweep = RecoverySweep(store.engine)
    step = _step(asyncio.run(sweep.run(dry_run=False, steps=["continuations"])), "continuations")
    assert step["found"] == []  # lease still fresh: the claimant's own writer suppresses it
    _lapse(store, request_id)
    report = asyncio.run(sweep.run(dry_run=False, steps=["continuations"]))
    assert _ids(_step(report, "continuations")["done"], "id") == [request_id]
    row = approvals.get_request(request_id)
    assert (row.continuation_state, row.continuation_claim_token) == ("suppressed", None)
    assert _ids(rows_for_run(store.engine, report["sweep_run_id"])["approvals"], "id") == [request_id]


def test_open_tool_calls_on_an_ended_turn_are_interrupted(delivery_store):  # noqa: F811
    from llm_bawt.service.tool_call_store import ToolCallStore

    store, _, target = delivery_store
    turn_id = _old_turn(store, target)
    calls = ToolCallStore(store.engine)
    calls.save_start(turn_id=turn_id, bot_id=target, user_id="nick", call_id="c1", tool_name="Bash")
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='error', ended_at=now() WHERE id=:id"), {"id": turn_id})
    step = _step(asyncio.run(RecoverySweep(store.engine).run(
        dry_run=False, bots=[target], steps=["unresolved_tools"])), "unresolved_tools")
    assert step["done"] == [{"turn_id": turn_id, "bot_id": target, "open_tools": 1, "action": "interrupt",
                             "interrupted": 1}]
    with store.engine.connect() as conn:
        ended, complete = conn.execute(sa_text(
            "SELECT ended_at, result_complete FROM tool_call_records WHERE turn_id=:id"), {"id": turn_id}).one()
    assert ended is not None and complete is False


def test_post_check_reports_each_bots_admission_state(delivery_store):  # noqa: F811
    store, _, target = delivery_store
    turn_id = _old_turn(store, target, age_seconds=10)
    step = _step(asyncio.run(RecoverySweep(store.engine).run(
        dry_run=True, bots=[target], steps=["post_check"])), "post_check")
    assert step["found"] == [{"bot_id": target, "state": "executing", "blocking_turn_id": turn_id}]


def test_purge_runs_only_when_asked(delivery_store, approvals):  # noqa: F811
    store, _, _ = delivery_store
    card = _ops_run(approvals, approved=False)
    sweep = RecoverySweep(store.engine)
    report = asyncio.run(sweep.run(dry_run=False))
    assert "purge_ops" not in {step["step"] for step in report["steps"]}
    assert approvals.get_request(card).status == "pending"
    step = _step(asyncio.run(sweep.run(dry_run=False, purge_ops=True, actor="nick", steps=["post_check"])), "purge_ops")
    assert _ids(step["done"], "id") == [card]
    assert approvals.get_request(card).status == "cancelled"


def test_a_mutating_pass_skips_while_another_holds_the_sweep(delivery_store):  # noqa: F811
    store, _, target = delivery_store
    turn_id = _old_turn(store, target)
    sweep = RecoverySweep(store.engine, probe=Probe(NOT_FOUND))
    with store.engine.connect() as holder, holder.begin():
        assert holder.execute(sa_text("SELECT pg_try_advisory_xact_lock(hashtext(:key))"),
                              {"key": RECOVERY_LOCK_KEY}).scalar()
        assert asyncio.run(sweep.run(dry_run=False, bots=[target])) == {"dry_run": False, "busy": True, "steps": []}
        # A dry run takes no lock.
        assert _ids(_step(asyncio.run(sweep.run(dry_run=True, bots=[target], steps=["orphan_turns"])),
                          "orphan_turns")["planned"]) == [turn_id]
    assert TurnLogStore(store.config).get_turn(turn_id).ended_at is None


def test_unknown_steps_are_refused(delivery_store):  # noqa: F811
    store, _, _ = delivery_store
    with pytest.raises(ValueError, match="unknown recovery steps"):
        asyncio.run(RecoverySweep(store.engine).run(dry_run=True, steps=["purge_everything"]))


def test_the_periodic_pass_skips_the_startup_only_tool_scan():
    calls = []

    class Sweep:
        async def run(self, **kwargs):
            calls.append(kwargs)
            return {"dry_run": False, "busy": False, "steps": []}  # RecoverySweep.run's shape

    hooks = SweepRecoveryHooks(Sweep())
    asyncio.run(hooks.on_leadership_acquired())
    asyncio.run(hooks.periodic())
    assert "steps" not in calls[0]  # leadership: every default step
    assert calls[1]["steps"] == list(RecoverySweep.PERIODIC_STEPS)
    assert "unresolved_tools" in RecoverySweep.DEFAULT_STEPS and "unresolved_tools" not in calls[1]["steps"]


def test_a_pass_skipped_as_busy_leaves_a_trace(caplog):
    class Sweep:
        async def run(self, **kwargs):
            return {"dry_run": False, "busy": True, "steps": []}

    with caplog.at_level("INFO", logger="llm_bawt.service.recovery.sweep"):
        asyncio.run(SweepRecoveryHooks(Sweep()).periodic())
    assert "recovery periodic pass skipped" in caplog.text
