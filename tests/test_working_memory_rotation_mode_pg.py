"""A claimed reset writes the admitted working-memory mode on its successor (10.1).

Real admission and claim on private PostgreSQL: the persisted intent, not the
delivery's session policy, decides the mode of an API reset.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, Origin,
)
from tests.test_new_session_policy_review_pg import claim_policy

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def successor_state(rig, session_id):
    with rig.engine.connect() as conn:
        metadata = conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:id"),
                                {"id": session_id}).scalar_one()
    return metadata["working_memory"]


def claim(rig, lifecycle, intent):
    record, reason = lifecycle._admit(intent)
    assert record is not None and reason is None
    claimed = lifecycle.store.claim_next(rig.bot, claim_owner="wm-mode", steer_capable=False)
    assert claimed is not None and claimed.id == record.id
    assert claimed.reset_status == "ROTATED" and claimed.old_session_id == rig.session_id
    return claimed


@pytest.mark.parametrize("origin", [Origin.CONTEXT_API, Origin.SESSION_API])
@pytest.mark.parametrize(("continuity", "flag", "mode"), [
    (Continuity.RETAIN, False, "suppressed"),
    (Continuity.CLEAN, None, "pinned"),
    (Continuity.CLEAN, True, "pinned"),
    (Continuity.CLEAN, False, "suppressed"),
])
def test_api_reset_successor_has_the_admitted_mode(rig, real_lifecycle, origin, continuity, flag, mode):
    intent = NewSessionIntent(
        bot_id=rig.bot, user_id=rig.user, author=AuthorReference.user(rig.user),
        continuity=continuity, admission=Admission.SKIP, origin=origin,
        idempotency_key=f"wm-mode:{rig.bot}", include_working_memory=flag,
    )
    claimed = claim(rig, real_lifecycle, intent)
    assert successor_state(rig, claimed.new_session_id) == {"mode": mode}


@pytest.mark.parametrize(("continuity", "mode"), [
    (Continuity.RETAIN, "pinned"), (Continuity.CLEAN, "suppressed"),
])
def test_bot_follow_up_successor_has_its_mode(rig, real_lifecycle, continuity, mode):
    intent = NewSessionIntent(
        bot_id=rig.bot, user_id=rig.user, author=AuthorReference.bot("test-sender"),
        continuity=continuity, admission=Admission.WAIT, origin=Origin.BOT_DELIVERY,
        idempotency_key=f"wm-mode:{rig.bot}", follow_up="Perform one task",
    )
    claimed = claim(rig, real_lifecycle, intent)
    assert successor_state(rig, claimed.new_session_id) == {"mode": mode}


@pytest.mark.parametrize(("policy", "mode"), [
    ("reset_retain_history", "pinned"), ("reset_without_history", "suppressed"),
])
def test_bare_policy_delivery_successor_has_its_mode(rig, real_lifecycle, policy, mode):
    claimed = claim_policy(rig, real_lifecycle, policy)
    assert successor_state(rig, claimed.new_session_id) == {"mode": mode}
