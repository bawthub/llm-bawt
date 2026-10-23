"""Graceful-shutdown claim handoff (TASK-1286 defect #1, slice B).

On shutdown the dispatcher expires only its own turn claims whose detached
worker provably exited. The next leader's first recover_expired then sends
each through the unchanged lease-expired branch, instead of the row waiting
out up to 300s of lease.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.turn_logs import TurnLogStore
from tests.test_inter_bot_delivery import _enqueue, delivery_store  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _lease_live(store, delivery_id) -> bool:
    with store.engine.connect() as conn:
        return bool(conn.execute(sa_text(
            "SELECT lease_expires_at > now() FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one())


def _running_claim(store, sender, target, owner="dispatcher-old"):
    """A turn claim whose bridge run was accepted and is still streaming."""
    record, _ = _enqueue(store, sender, target, "still running at shutdown")
    claim = store.claim_next(target, claim_owner=owner, steer_capable=False, lease_seconds=300)
    assert claim and claim.id == record.id and claim.delivery_mode == "turn"
    assert store.mark_transport_accepted(claim.id, claim.claim_token or "")
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": claim.turn_id})
    return claim


def test_expire_own_claims_touches_only_our_listed_turn_claims(delivery_store):
    store, sender, target = delivery_store
    claim = _running_claim(store, sender, target)

    assert store.expire_own_claims(claim_owner="dispatcher-peer", turn_ids=[claim.turn_id]) == []
    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=["turn-elsewhere"]) == []
    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=[]) == []
    assert _lease_live(store, claim.id)

    # I2: a steer row keeps its lease even when it matches owner and turn.
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET status='STEERING', delivery_mode='steer' WHERE id=:id"
        ), {"id": claim.id})
    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=[claim.turn_id]) == []
    assert _lease_live(store, claim.id)
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET status='DISPATCHING', delivery_mode='turn' WHERE id=:id"
        ), {"id": claim.id})

    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=[claim.turn_id]) == [claim.id]
    assert not _lease_live(store, claim.id)
    current = store.get(claim.id)
    # Only the lease moved: the row is still our DISPATCHING claim.
    assert (current.status, current.claim_token) == ("DISPATCHING", claim.claim_token)


def test_a_released_running_claim_is_requeued_for_the_next_dispatcher_to_reattach(delivery_store):
    store, sender, target = delivery_store
    claim = _running_claim(store, sender, target)

    # Before the handoff the new leader must leave a live claim alone.
    assert not [r for r in store.recover_expired(live_owner="dispatcher-new") if r.id == claim.id]
    assert store.get(claim.id).status == "DISPATCHING"

    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=[claim.turn_id]) == [claim.id]
    recovered = [r for r in store.recover_expired(live_owner="dispatcher-new") if r.id == claim.id]
    assert recovered and recovered[0].status == "QUEUED" and recovered[0].claim_token is None
    turn = TurnLogStore(store.config).get_turn(claim.turn_id)
    assert (turn.status, turn.ended_at) == ("reserved", None)  # re-armed, not settled

    again = store.claim_next(target, claim_owner="dispatcher-new", steer_capable=False)
    # Same row and turn: the deterministic bridge request id follows the
    # delivery id, so dispatch reattaches to the still-running run.
    assert again and (again.id, again.turn_id) == (claim.id, claim.turn_id)
    assert again.attempt_count == claim.attempt_count + 1


def test_a_turn_that_finished_before_shutdown_is_delivered_by_the_next_dispatcher(delivery_store):
    store, sender, target = delivery_store
    claim = _running_claim(store, sender, target)
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='ok', ended_at=now(), response_text='done' WHERE id=:id"
        ), {"id": claim.turn_id})

    assert store.expire_own_claims(claim_owner="dispatcher-old", turn_ids=[claim.turn_id]) == [claim.id]
    recovered = [r for r in store.recover_expired(live_owner="dispatcher-new") if r.id == claim.id]
    assert recovered and recovered[0].status == "DELIVERED"
