"""Bot steer lifecycle against the real delivery store (TASK-1300).

A steered bot delivery is DELIVERED once the bridge accepted the command, but
"Interrupt now" can still sweep it before it runs. The bridge reports that
with a proof, and :func:`apply_bot_steer_lifecycle` re-queues the row at the
next generation. Everything else is fenced: stale or replayed events, other
turns, turn-mode rows, started commands and uncertain outcomes.
"""
from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy import event, text as sa_text

import llm_bawt.inter_bot_delivery as delivery_module
from llm_bawt.inter_bot_delivery import DELIVERED, DISPATCHING, QUEUED, STEERING
from llm_bawt.inter_bot_delivery_steer_fence import IGNORED, PENDING, REOPENED, steer_uncertain
from llm_bawt.service.inter_bot_dispatcher_steer import SteerGeneration
from llm_bawt.service.inter_bot_steer_lifecycle import STARTED, UNCERTAIN, apply_bot_steer_lifecycle
from tests.pg_lock_wait import wait_blocked_by
from tests.test_inter_bot_delivery import delivery_store  # noqa: F401  (fixture)
from tests.test_inter_bot_steer_generation_pg import _claim, _end_turn, _steer, rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def life(rig):  # noqa: F811
    rig.bridge.registers_after = 0  # the target's run is live: steers land
    rig.announced = []
    rig.dispatcher.announce_admitted = rig.announced.append
    rig.lifecycles = []
    rig.dispatcher.announce_lifecycle = rig.lifecycles.append
    return rig


def _event(record, state, *, proof=None, generation=0, origin="bot"):
    rpc = SteerGeneration.of(record).request_id.split("_g")[0]
    event = {
        "message_id": "cmd-uuid", "state": state, "origin": origin,
        "steer_rpc_id": rpc if generation == 0 else f"{rpc}_g{generation}",
        "run_request_id": "req-target",
    }
    if proof is not None:
        event["proof"] = proof
    return event


def _apply(life, record, state, *, turn_id=None, **kwargs):
    return apply_bot_steer_lifecycle(
        life.dispatcher, _event(record, state, **kwargs),
        target_turn_id=turn_id or life.first.turn_id,
    )


def _delivered(life):
    steering = _claim(life)
    assert steering.id == life.second.id and steering.status == STEERING
    row = _steer(life, steering)
    # TASK-2480: accepted is not delivered; it waits STEERING (claim released).
    assert row.status == STEERING and row.delivery_mode == "steer"
    assert row.claim_token is None
    return row


def _assert_reopened(row, *, attempts):
    assert row.status == QUEUED and row.delivery_mode == "steer_rejected"
    assert row.metadata["steer_generation"] == 1
    assert row.metadata["steer_rejection"]["generation"] == 0
    assert row.metadata.get("steer_rejected_pending") is None
    assert row.transport_accepted_at is None
    assert row.attempt_count == attempts


@pytest.mark.parametrize(("state", "proof"), [
    ("cancelled", "sweep_receipt"), ("discarded", "cli_terminal"), ("refused", "cli_terminal"),
])
def test_a_proven_rejection_reopens_a_delivered_steer_at_the_next_generation(life, state, proof):
    row = _delivered(life)
    assert _apply(life, row, state, proof=proof) == REOPENED
    reopened = life.store.get(row.id)
    _assert_reopened(reopened, attempts=row.attempt_count - 1)
    assert reopened.delivered_at is None
    assert reopened.metadata["steer_rejection"]["proof"] == proof
    assert [r.id for r in life.announced] == [row.id]  # QUEUED published, dispatcher woken


def test_the_reopened_steer_waits_for_its_target_turn_then_falls_back(life):
    row = _delivered(life)
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == REOPENED
    assert _claim(life) is None  # same open target turn; FIFO keeps the third behind it
    _end_turn(life, life.first.turn_id)
    head = life.store.get(life.first.id)  # the target turn's own delivery settles
    assert life.store.mark_delivered(head.id, head.claim_token or "", response_model=None, response_chars=0)
    fallback = _claim(life)
    assert fallback.id == row.id and fallback.status == DISPATCHING


def test_a_replayed_stale_or_foreign_event_changes_nothing(life):
    row = _delivered(life)
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == REOPENED
    after = life.store.get(row.id)
    assert _apply(life, row, "cancelled", proof="sweep_receipt") == IGNORED  # replay of g0
    assert _apply(life, row, "cancelled", proof="sweep_receipt", generation=5) == IGNORED
    assert _apply(life, row, "cancelled", proof="sweep_receipt", generation=1,
                  turn_id="turn-someone-else") == IGNORED
    assert life.store.get(row.id).metadata == after.metadata
    assert len(life.announced) == 1


def test_a_started_command_is_never_reopened(life):
    """TASK-2480: a later not-run proof is an evidence conflict, never a reopen."""
    row = _delivered(life)
    assert _apply(life, row, "started") == STARTED
    assert _apply(life, row, "cancelled", proof="cli_terminal") == "conflict"
    kept = life.store.get(row.id)
    assert kept.status == "FAILED" and kept.metadata["steer_started"] == {"generation": 0}
    assert kept.last_error == "outcome_unknown: evidence_conflict"
    assert life.announced == []


@pytest.mark.parametrize(("state", "proof"), [
    ("cancelled", None), ("queued", None), ("cancelled", "replayed"),
])
def test_an_unproven_or_non_terminal_state_changes_nothing(life, state, proof):
    row = _delivered(life)
    assert _apply(life, row, state, proof=proof) == IGNORED
    kept = life.store.get(row.id)
    assert kept.status == STEERING and kept.metadata == row.metadata


def test_an_uncertain_accepted_steer_is_resent_once_with_a_note(life):
    """TASK-2480: an accepted steer that may have run is never left DELIVERED.

    It fails visibly and gets exactly one system resend carrying the
    interruption note, linked to it; a repeat of the event changes nothing.
    """
    row = _delivered(life)
    assert _apply(life, row, "uncertain", proof="barrier_timeout") == UNCERTAIN
    kept = life.store.get(row.id)
    assert kept.status == "FAILED" and kept.delivered_at is None
    assert kept.metadata["steer_uncertain"] == {"generation": 0, "reason": "barrier_timeout"}
    [published] = life.lifecycles
    assert published.id == row.id and published.status == "FAILED"
    [child] = life.announced
    child = life.store.get(child.id)
    assert child.status == QUEUED and child.metadata["restart_retry_of"] == row.id
    assert child.author.entity_id == "system"
    with life.store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT resent_from_id FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": child.id}).scalar_one() == row.id
    assert _apply(life, row, "uncertain", proof="barrier_timeout") == IGNORED  # settled
    assert len(life.announced) == 1


def test_every_committed_change_advances_the_rows_revision(life):
    """Codex r2: the DELIVERED snapshot _steer publishes must be older, by
    revision, than the uncertain row published after it, whatever order the
    two events reach a consumer in."""
    queued = life.store.get(life.second.id)
    steering = _claim(life)
    assert steering.revision > queued.revision
    delivered = _steer(life, steering)
    assert delivered.revision > steering.revision
    assert _apply(life, delivered, "uncertain", proof="teardown") == UNCERTAIN
    [published] = life.lifecycles
    assert published.revision > delivered.revision
    assert published.revision == life.store.get(delivered.id).revision
    assert published.to_api()["revision"] == published.revision


def test_a_start_after_uncertain_disposal_is_a_visible_conflict(life):
    """TASK-2480: disposed of as uncertain (resent), then it turns out to have run."""
    row = _delivered(life)
    assert _apply(life, row, "uncertain", proof="teardown") == UNCERTAIN
    assert _apply(life, row, "started") == STARTED
    kept = life.store.get(row.id)
    assert kept.status == "FAILED" and kept.last_error == "outcome_unknown: evidence_conflict"


def test_an_uncertain_marker_stops_counting_once_a_claimed_command_starts(life):
    steering = _claim(life)
    assert _apply(life, steering, "uncertain", proof="teardown") == UNCERTAIN
    assert _apply(life, steering, "started") == STARTED
    assert steer_uncertain(life.store.get(steering.id)) is None


def test_user_origin_and_turn_mode_rows_are_untouched(life):
    row = _delivered(life)
    assert _apply(life, row, "cancelled", proof="sweep_receipt", origin="user") == IGNORED
    turn_row = life.store.get(life.first.id)
    assert turn_row.status == DISPATCHING and turn_row.delivery_mode == "turn"
    assert _apply(life, turn_row, "cancelled", proof="sweep_receipt") == IGNORED
    assert life.store.get(life.first.id).metadata == turn_row.metadata
    assert life.store.get(row.id).status == STEERING


# ── a rejection that lands while the claim is still STEERING ─────────────────


def _pending(life):
    steering = _claim(life)
    assert steering.status == STEERING
    assert _apply(life, steering, "cancelled", proof="sweep_receipt") == PENDING
    marked = life.store.get(steering.id)
    assert marked.status == STEERING
    assert marked.metadata["steer_rejected_pending"] == {"generation": 0, "proof": "sweep_receipt"}
    return steering


def test_the_claims_mark_delivered_settles_a_pending_rejection_as_a_requeue(life):
    steering = _pending(life)
    row = _steer(life, steering)  # the RPC returns ok; mark_delivered sees the marker
    _assert_reopened(row, attempts=steering.attempt_count - 1)


def test_a_rejection_racing_mark_delivered_still_reopens_the_steer(life, monkeypatch):
    """Codex r1 Blocker 1: a rejection committing between mark_delivered's
    marker check and its DELIVERED write must not leave the row DELIVERED.

    The rejection runs on a second connection the moment mark_delivered has
    read the claim. It has to wait for mark_delivered's row lock, then find
    DELIVERED and reopen it.
    """
    steering = _claim(life)
    assert steering.status == STEERING
    original = delivery_module.pending_rejection
    racer: dict = {}
    checked_out: dict[int, int] = {}  # thread -> backend pid of its latest pool checkout

    def note_checkout(dbapi_connection, _record, _proxy):
        checked_out[threading.get_ident()] = dbapi_connection.get_backend_pid()

    def check_while_a_rejection_races(row):
        if "thread" not in racer:
            holder_pid = checked_out[threading.get_ident()]  # mark_delivered's own connection
            racer["thread"] = threading.Thread(target=lambda: racer.setdefault(
                "outcome", _apply(life, steering, "cancelled", proof="sweep_receipt")))
            racer["thread"].start()
            # it waits on the claim's row lock
            wait_blocked_by(life.store.engine, holder_pid, racer["thread"])
        return original(row)

    monkeypatch.setattr(delivery_module, "pending_rejection", check_while_a_rejection_races)
    event.listen(life.store.engine, "checkout", note_checkout)
    try:
        settled = life.store.mark_delivered(
            steering.id, steering.claim_token or "", response_model=None, response_chars=0,
        )
    finally:
        event.remove(life.store.engine, "checkout", note_checkout)
    racer["thread"].join(timeout=10)
    assert not racer["thread"].is_alive()
    assert settled.status == DELIVERED
    assert racer["outcome"] == REOPENED
    _assert_reopened(life.store.get(steering.id), attempts=steering.attempt_count - 1)


def test_a_timed_out_requeue_settles_a_pending_rejection_at_the_next_generation(life):
    steering = _pending(life)
    row = life.store.requeue(steering.id, steering.claim_token or "", "steer timed out", delay_seconds=0)
    _assert_reopened(row, attempts=steering.attempt_count - 1)
    assert "not run" in (row.last_error or "")


def test_lease_recovery_settles_a_pending_rejection_at_the_next_generation(life):
    steering = _pending(life)
    with life.store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second' WHERE id=:id"
        ), {"id": steering.id})
    recovered = [r for r in life.store.recover_expired() if r.id == steering.id]
    assert recovered
    _assert_reopened(recovered[0], attempts=steering.attempt_count - 1)


def test_a_start_after_a_pending_rejection_is_a_conflict_when_the_claim_settles(life):
    """TASK-2480: both proofs for one attempt -> UNKNOWN, neither success nor reopen."""
    steering = _pending(life)
    assert _apply(life, steering, "started") == STARTED
    row = _steer(life, steering)
    assert row.status == "FAILED" and row.last_error == "outcome_unknown: evidence_conflict"
    assert row.metadata.get("steer_rejected_pending") is None
    assert row.metadata["steer_started"] == {"generation": 0}


def test_the_dispatcher_steers_with_the_id_the_parser_reads_back(life):
    steering = _claim(life)
    asyncio.run(life.dispatcher._steer(steering))
    rpc = life.bridge.request_ids[-1]
    assert SteerGeneration.parse_request_id(rpc) == (steering.id, 0)
