"""PostgreSQL tests for per-run steer capabilities (TASK-1243 P1).

A run's capabilities are pinned when ``bind_agent_bridge`` binds it, on its
turn row and on every unpinned gate member of that turn. The delivery claim
decides "steer or wait" from the active run's own pinned value; the bot's
current backend only stands in for a run that is not bound yet. The unit
halves (codec, route refusals, classifier) are in test_steer_capabilities.py.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_delivery import QUEUED, STEERING
from llm_bawt.service.turn_gates import GateScope
from tests.test_inter_bot_delivery import _enqueue, delivery_store  # noqa: F401  (fixture)
from tests.test_turn_gates_pg import _activate, _members, _spec, gated  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _streaming(engine, bot, *, backend=None, caps=None, request_id=None, user="nick"):
    turn_id = f"turn-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                   response_text, ended_at, agent_request_id, agent_backend,
                                   agent_session_key, agent_steer_caps)
            VALUES (:id, now(), '/test', true, :bot, :user, 'streaming', '', '', NULL, :req, :backend,
                    :session, :caps)
        """), {"id": turn_id, "bot": bot, "user": user, "req": request_id, "backend": backend,
               "session": "sess" if backend else None, "caps": caps})
    return turn_id


def _caps(engine, turn_id):
    with engine.connect() as conn:
        return conn.execute(sa_text("SELECT agent_steer_caps FROM turn_logs WHERE id=:id"),
                            {"id": turn_id}).scalar_one()


# --- pinned at bind -----------------------------------------------------------


def test_bind_pins_the_runs_caps_on_the_turn_and_its_unpinned_members(gated):
    store, bot = gated.store, gated.bot()
    turn_id = _streaming(store.engine, bot)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    assert _members(store, gate.id)[turn_id]["agent_steer_caps"] is None  # captured unbound

    store.bind_agent_bridge(turn_id=turn_id, backend="codex", session_key="s", request_id="req-1",
                            steer_caps="next")
    assert _caps(store.engine, turn_id) == "next"
    member = _members(store, gate.id)[turn_id]
    assert (member["agent_request_id"], member["agent_steer_caps"]) == ("req-1", "next")


def test_a_member_already_pinned_keeps_its_own_caps(gated):
    store, bot = gated.store, gated.bot()
    turn_id = _streaming(store.engine, bot)
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE gate_members SET agent_request_id='req-0', agent_steer_caps='next'"
                             " WHERE turn_id=:t"), {"t": turn_id})
    store.bind_agent_bridge(turn_id=turn_id, backend="codex", session_key="s", request_id="req-1",
                            steer_caps="")
    member = _members(store, gate.id)[turn_id]
    assert (member["agent_request_id"], member["agent_steer_caps"]) == ("req-0", "next")
    assert _caps(store.engine, turn_id) == ""  # '' is "none", never the legacy NULL


def test_a_capture_after_bind_copies_the_turns_caps(gated):
    store, bot = gated.store, gated.bot()
    turn_id = _streaming(store.engine, bot)
    store.bind_agent_bridge(turn_id=turn_id, backend="codex", session_key="s", request_id="req-1",
                            steer_caps="escalate,next")
    gate = _activate(store, _spec(gated.owner(), GateScope.for_bots(bot))).gate
    assert _members(store, gate.id)[turn_id]["agent_steer_caps"] == "escalate,next"


# --- the claim reads the active run's pinned caps -----------------------------


@pytest.mark.parametrize("backend, caps, bot_can_steer, steers", [
    # Legacy rows (no caps recorded) decode from their own backend, whatever
    # the bot's backend is now: a mid-run flip changes the next run only.
    ("claude-code", None, False, True),
    ("codex", None, True, False),
    # Pinned caps decide, not the backend's name.
    ("codex", "next", False, True),
    ("claude-code", "", True, False),
    # A run not bound yet has no pin: the bot's current answer stands in.
    (None, None, True, True),
    (None, None, False, False),
], ids=["legacy-claude", "legacy-codex", "pinned-steer", "pinned-none", "unbound-steer", "unbound-none"])
def test_the_claim_steers_on_the_active_runs_own_caps(delivery_store, backend, caps, bot_can_steer, steers):  # noqa: F811
    store, sender, target = delivery_store
    turn_id = _streaming(store.engine, target, backend=backend, caps=caps,
                         request_id="req-active" if backend else None)
    queued, _ = _enqueue(store, sender, target, "mid-turn reply")
    claimed = store.claim_next(target, claim_owner="owner", steer_capable=bot_can_steer)
    if steers:
        assert (claimed.id, claimed.status, claimed.target_turn_id) == (queued.id, STEERING, turn_id)
    else:
        assert claimed is None
        assert store.get(queued.id).status == QUEUED  # waits for the run to end
