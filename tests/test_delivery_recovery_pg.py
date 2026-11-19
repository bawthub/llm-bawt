"""MDv2 C1 honest steer status against the real store (TASK-2480).

DELIVERED only at STARTED; an accepted steer that may have run gets one
noted resend or UNKNOWN, never a fake success; Stop and exclusions show
their reason; evidence is recorded per attempt.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.delivery.recovery import settle_ended_parked
from llm_bawt.inter_bot_delivery import DELIVERED, QUEUED, STEERING
from tests.test_inter_bot_delivery import delivery_store  # noqa: F401  (fixture)
from tests.test_inter_bot_steer_generation_pg import _claim, _end_turn, _steer, rig  # noqa: F401
from tests.test_inter_bot_steer_lifecycle_pg import _apply, life  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _accepted(life):
    steering = _claim(life)
    assert steering.id == life.second.id and steering.status == STEERING
    row = _steer(life, steering)
    assert row.status == STEERING and row.claim_token is None
    return row


def _events(life, delivery_id):
    with life.store.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT attempt_no, kind FROM delivery_events WHERE delivery_id=:id ORDER BY id"
        ), {"id": delivery_id}).all()


def _sql(life, statement, **params):
    with life.store.engine.begin() as conn:
        conn.execute(sa_text(statement), params)


def test_an_accepted_steer_is_delivered_only_at_started(life):
    row = _accepted(life)
    assert [tuple(e) for e in _events(life, row.id)] == [(1, "CLAIMED"), (1, "ACCEPTED")]
    assert _apply(life, row, "started") == "started"
    delivered = life.store.get(row.id)
    assert delivered.status == DELIVERED and delivered.delivered_at is not None
    assert [tuple(e) for e in _events(life, row.id)] == [
        (1, "CLAIMED"), (1, "ACCEPTED"), (1, "STARTED")]
    assert [r.id for r in life.lifecycles] == [row.id]


def test_root_output_never_marks_an_accepted_addition_started(life):
    """Negative: the root run going on (and ending) proves nothing about the addition."""
    row = _accepted(life)
    head = life.store.get(life.first.id)
    assert life.store.mark_delivered(head.id, head.claim_token or "", response_model="m", response_chars=9)
    assert life.store.get(row.id).status == STEERING


def test_a_later_head_claims_while_the_target_run_is_open(life):
    _accepted(life)
    successor = _claim(life)
    assert successor.id == life.third.id and successor.status == STEERING


def test_run_end_without_started_resends_once_then_unknown(life):
    row = _accepted(life)
    _end_turn(life, life.first.turn_id)
    # The parked row is the head again: nothing later claims before recovery.
    assert _claim(life) is None
    settled = settle_ended_parked(life.store)
    original = life.store.get(row.id)
    assert original.status == "FAILED" and "resent once" in (original.last_error or "")
    child = next(r for r in settled if r.id != row.id)
    assert life.store.get(child.id).metadata["restart_retry_of"] == row.id
    assert ("1", "UNCERTAIN") in {(str(a), k) for a, k in _events(life, row.id)}
    # Replays settle nothing more.
    assert settle_ended_parked(life.store) == []
    # The child, if it too ends uncertain, is UNKNOWN: one resend per lineage.
    _sql(life, """UPDATE inter_bot_deliveries SET status='STEERING', kind='ADDITION',
                  claim_token=NULL, transport_accepted_at=now(), target_turn_id=:turn,
                  delivery_mode='steer' WHERE id=:id""", id=child.id, turn=life.first.turn_id)
    settle_ended_parked(life.store)
    assert life.store.get(child.id).last_error == "outcome_unknown: already_resent"


def test_stop_on_the_target_run_is_unknown_and_never_resent(life):
    row = _accepted(life)
    _sql(life, """UPDATE turn_logs SET status='aborted', ended_at=now(),
                  request_json=CAST(:req AS jsonb)::text WHERE id=:id""",
         id=life.first.turn_id, req=json.dumps({"abort_request": {"source": "chat_stop"}}))
    settled = settle_ended_parked(life.store)
    assert [r.id for r in settled] == [row.id]
    assert life.store.get(row.id).last_error == "outcome_unknown: stopped"
    # A later teardown pass never creates a child.
    assert settle_ended_parked(life.store) == []


def test_an_excluded_row_shows_its_reason(life):
    row = _accepted(life)
    _sql(life, """UPDATE inter_bot_deliveries SET metadata_json=metadata_json
                  || '{"prompt_schedule": {"id": "s"}}'::jsonb WHERE id=:id""", id=row.id)
    _end_turn(life, life.first.turn_id)
    settle_ended_parked(life.store)
    assert life.store.get(row.id).last_error == "outcome_unknown: excluded:prompt_schedule"


def test_a_started_addition_is_never_requeued_by_run_end(life):
    row = _accepted(life)
    assert _apply(life, row, "started") == "started"
    _end_turn(life, life.first.turn_id)
    assert settle_ended_parked(life.store) == []
    assert life.store.get(row.id).status == DELIVERED


def test_a_proven_rejection_of_a_parked_steer_requeues_at_the_next_attempt(life):
    row = _accepted(life)
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == "reopened"
    reopened = life.store.get(row.id)
    assert reopened.status == QUEUED
    with life.store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT attempt_no, ordinal FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": row.id}).one()[0] == 2


def _row(life, delivery_id):
    with life.store.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE id=:id"), {"id": delivery_id}).mappings().one()


def test_started_before_park_survives_lease_recovery_as_delivered(life):
    """Blocker 1: STARTED commits while the claim is still held, then the app dies."""
    steering = _claim(life)
    assert _apply(life, steering, "started") == "started"
    _sql(life, "UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 minute' WHERE id=:id",
         id=steering.id)
    life.store.recover_expired()
    row = life.store.get(steering.id)
    assert row.status == DELIVERED and row.claim_token is None


def test_a_rejection_after_started_is_a_visible_conflict(life):
    """Blocker 2: REJECTED after STARTED never keeps the success."""
    row = _accepted(life)
    assert _apply(life, row, "started") == "started"
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == "conflict"
    failed = _row(life, row.id)
    assert failed["status"] == "FAILED" and failed["last_error"] == "outcome_unknown: evidence_conflict"
    assert failed["outcome"] == "unknown"


def test_a_late_started_for_a_rejected_generation_is_a_conflict(life):
    """Blocker 2: the rejected k turns out to have run; the k+1 row must not run again."""
    row = _accepted(life)
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == "reopened"
    assert life.store.get(row.id).status == QUEUED
    _apply(life, row, "started")  # generation 0, already rejected
    failed = _row(life, row.id)
    assert failed["status"] == "FAILED" and failed["last_error"] == "outcome_unknown: evidence_conflict"


def test_a_late_started_after_uncertain_disposal_is_a_conflict(life):
    row = _accepted(life)
    _sql(life, """UPDATE inter_bot_deliveries SET metadata_json=metadata_json
                  || '{"prompt_schedule": {"id": "s"}}'::jsonb WHERE id=:id""", id=row.id)
    _end_turn(life, life.first.turn_id)
    settle_ended_parked(life.store)
    _apply(life, row, "started")
    failed = _row(life, row.id)
    assert failed["status"] == "FAILED" and failed["last_error"] == "outcome_unknown: evidence_conflict"


def test_not_run_at_max_attempts_fails_instead_of_requeueing(life):
    """Blocker 3: the NOT_RUN budget is bounded by max_attempts."""
    row = _accepted(life)
    _sql(life, "UPDATE inter_bot_deliveries SET max_attempts=1 WHERE id=:id", id=row.id)
    _apply(life, row, "cancelled", proof="sweep_receipt")
    failed = _row(life, row.id)
    assert failed["status"] == "FAILED" and failed["outcome"] == "not_run"
    assert ("1", "NOT_RUN") in {(str(a), k) for a, k in _events(life, row.id)}


def test_a_start_records_claimed_accepted_started_and_done(life):
    """Blocker 4: START evidence and DONE at completion."""
    head = life.store.get(life.first.id)
    assert life.store.note_start_started(head.id, head.claim_token or "")
    assert life.store.mark_delivered(head.id, head.claim_token or "", response_model="m", response_chars=3)
    assert [k for _a, k in _events(life, head.id)] == ["CLAIMED", "ACCEPTED", "STARTED", "DONE"]
    row = _row(life, head.id)
    assert row["kind"] == "START" and row["outcome"] == "done"


def test_a_started_start_torn_down_without_stop_is_resent_once(life):
    """Blocker 4: §4.4 covers an unfinished START teardown, not only host shutdown."""
    from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry

    head = life.store.get(life.first.id)
    assert life.store.note_start_started(head.id, head.claim_token or "")
    _sql(life, "UPDATE turn_logs SET status='aborted', ended_at=now() WHERE id=:id", id=head.turn_id)
    failed, child = fail_claim_with_restart_retry(life.store, head, head.claim_token or "", "run torn down")
    assert child is not None and failed.status == "FAILED"
    assert _row(life, head.id)["outcome"] == "resent"
    assert _row(life, child.id)["resent_from_id"] == head.id


def test_a_started_start_stopped_by_the_user_is_unknown(life):
    from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry

    head = life.store.get(life.first.id)
    assert life.store.note_start_started(head.id, head.claim_token or "")
    _sql(life, """UPDATE turn_logs SET status='aborted', ended_at=now(),
                  request_json=CAST(:req AS jsonb)::text WHERE id=:id""",
         id=head.turn_id, req=json.dumps({"abort_request": {"source": "chat_stop"}}))
    _failed, child = fail_claim_with_restart_retry(life.store, head, head.claim_token or "", "stopped")
    assert child is None
    row = _row(life, head.id)
    assert row["outcome"] == "unknown" and row["last_error"].startswith("outcome_unknown: ")


def test_new_rows_are_not_relabelled_legacy_on_a_later_boot(life):
    from llm_bawt.inter_bot_delivery_schema import bootstrap_delivery_schema

    head = life.store.get(life.first.id)
    life.store.mark_delivered(head.id, head.claim_token or "", response_model="m", response_chars=1)
    with life.store.engine.begin() as conn:
        bootstrap_delivery_schema(conn)
    assert _row(life, head.id)["outcome"] == "done"


def test_late_start_keeps_conflict_after_fallback_completed(life):
    """Codex r2: the rejection's own binding survives a fallback START rebinding."""
    old = _accepted(life)
    assert _apply(life, old, "cancelled", proof="sweep_receipt") == "reopened"
    _end_turn(life, life.first.turn_id)
    head = life.store.get(life.first.id)
    assert life.store.mark_delivered(head.id, head.claim_token or "", response_model=None, response_chars=0)
    fallback = _claim(life)
    assert fallback.id == old.id and fallback.status == "DISPATCHING"
    assert life.store.mark_transport_accepted(fallback.id, fallback.claim_token or "")
    assert life.store.note_start_started(fallback.id, fallback.claim_token or "")
    assert life.store.mark_delivered(fallback.id, fallback.claim_token or "", response_model=None, response_chars=1)
    _apply(life, old, "started")
    observed = _row(life, old.id)
    assert (observed["status"], observed["outcome"], observed["last_error"]) == (
        "FAILED", "unknown", "outcome_unknown: evidence_conflict")


@pytest.mark.parametrize("terminal, outcome", [("ok", "done"), ("aborted", "resent")])
def test_start_lease_recovery_stamps_terminal_evidence(life, terminal, outcome):
    """Codex r2: crash recovery applies the live path's terminal rules."""
    head = life.store.get(life.first.id)
    assert life.store.note_start_started(head.id, head.claim_token or "")
    _sql(life, "UPDATE turn_logs SET status=:status, ended_at=now() WHERE id=:id",
         id=head.turn_id, status=terminal)
    _sql(life, "UPDATE inter_bot_deliveries SET lease_expires_at=now()-interval '1 minute' WHERE id=:id",
         id=head.id)
    life.store.recover_expired()
    observed = _row(life, head.id)
    assert observed["outcome"] == outcome, (observed["status"], observed["outcome"], observed["last_error"])


def test_late_conflict_during_fallback_abort_is_not_resent(life):
    """Codex r3: conflict precedence holds on failure settlement; no auto resend."""
    from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry

    old = _accepted(life)
    assert _apply(life, old, "cancelled", proof="sweep_receipt") == "reopened"
    _end_turn(life, life.first.turn_id)
    head = life.store.get(life.first.id)
    assert life.store.mark_delivered(
        head.id, head.claim_token or "", response_model=None, response_chars=0
    )
    fallback = _claim(life)
    assert fallback.id == old.id and fallback.status == "DISPATCHING"
    assert life.store.mark_transport_accepted(fallback.id, fallback.claim_token or "")
    assert life.store.note_start_started(fallback.id, fallback.claim_token or "")
    _apply(life, old, "started")
    _sql(life, "UPDATE turn_logs SET status='aborted', ended_at=now() WHERE id=:id",
         id=fallback.turn_id)
    failed, child = fail_claim_with_restart_retry(
        life.store, fallback, fallback.claim_token or "", "fallback aborted"
    )
    observed = _row(life, old.id)
    assert child is None, (failed.status, observed["outcome"], child.id)
    assert observed["outcome"] == "unknown"
    assert observed["last_error"] == "outcome_unknown: evidence_conflict"
