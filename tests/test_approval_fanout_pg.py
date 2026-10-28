"""Approver notices against an isolated PostgreSQL DB (TASK-1661).

Pending ops cards reach every eligible approver bot as one APPROVAL_REQUEST
delivery, are fenced at claim when eligibility goes away, and close with a
steer-only APPROVAL_RESOLVED follow-up (never to the resolver). Builds on the
TASK-1609 ``ops`` fixture: real tag, grant, approval and delivery stores.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.approval_models import REQ_DENIED
from llm_bawt.bot_identities import SYSTEM_BOT_ID
from llm_bawt.inter_bot_steer_only import STEER_ONLY_KEY
from llm_bawt.service import approval_fanout as fanout
from tests.test_ops_approvals_pg import BYSTANDER, HOLDER, REQUESTER, SECRET, USER, _private, ops  # noqa: F401
from tests.test_bot_grants_pg import grant_api  # noqa: F401
from tests.test_bot_tag_api_pg import api  # noqa: F401
from tests.test_bot_tags_pg import rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def fan(ops, monkeypatch):  # noqa: F811
    from llm_bawt import inter_bot_delivery
    from llm_bawt.utils.config import Config

    # The delivery store on the rig's database, beside the approval rows.
    monkeypatch.setattr(inter_bot_delivery, "get_shared_engine", lambda _config: ops.api.rig.engine)
    # Enqueue stamps the target's latest turn (maintenance barrier); the rig has no turn log.
    from llm_bawt.service.turn_log_schema import TurnLog
    TurnLog.__table__.create(ops.api.rig.engine, checkfirst=True)
    deliveries = inter_bot_delivery.InterBotDeliveryStore(Config())
    monkeypatch.setattr(fanout.ApprovalFanoutPump, "mode", lambda self: fanout.NOTIFY)
    ops.deliveries = deliveries
    ops.pump = fanout.ApprovalFanoutPump(deliveries)
    return ops


def _notices(ctx, *, target=None):
    with ctx.deliveries.engine.connect() as conn:
        rows = conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE sender_bot_id=:sys ORDER BY ordinal"
        ), {"sys": SYSTEM_BOT_ID}).mappings().all()
    return [dict(r) for r in rows if target is None or r["target_bot_id"] == target]


def _pair(ctx, request_id, approver):
    with ctx.deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            f"SELECT state FROM {fanout.TABLE} WHERE request_id=:r AND approver_bot_id=:b"
        ), {"r": request_id, "b": approver}).scalar()


def _claim(ctx, bot):
    return ctx.deliveries.claim_next(bot, claim_owner="fixture", steer_capable=True)


def _status(ctx, delivery_id):
    with ctx.deliveries.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT status, last_error FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).one()


def test_a_pending_card_reaches_each_eligible_approver_once_masked(fan):
    request_id = fan.card()
    fan.card(bot=HOLDER)                    # the holder's own request: no notice to itself

    first, second = fan.pump.run_pass(), fan.pump.run_pass()

    [notice] = _notices(fan)
    assert [n.id for n in first.notified] == [notice["id"]] and second.notified == []
    assert (notice["target_bot_id"], notice["message_kind"]) == (HOLDER, fanout.REQUEST_KIND)
    assert notice["idempotency_key"] == f"{fanout.REQUEST_KEY_PREFIX}{request_id}:{HOLDER}"
    assert notice["metadata_json"][fanout.REQUEST_ID_KEY] == request_id
    assert notice["payload_json"]["user"] == USER
    assert request_id in notice["message"] and "fixture.restart" in notice["message"]
    assert SECRET not in notice["message"] and "***" in notice["message"]
    assert _pair(fan, request_id, HOLDER) == fanout.REQUESTED


def test_bots_without_the_grant_and_private_requesters_get_nothing(fan):
    fan.card(bot=BYSTANDER)
    _private(fan, BYSTANDER)
    fan.card(user="fixture-other-user")     # another user's card: still the holder's to decide
    assert fan.pump.run_pass().errors == 0
    targets = {(n["target_bot_id"], n["payload_json"]["user"]) for n in _notices(fan)}
    assert targets == {(HOLDER, "fixture-other-user")}
    assert all(n["target_bot_id"] != BYSTANDER for n in _notices(fan))


def test_a_revoke_after_enqueue_is_fenced_at_claim(fan):
    request_id = fan.card()
    fan.pump.run_pass()
    [notice] = _notices(fan)
    assert fan.api.as_person(HOLDER, granted=False).status_code == 200

    assert _claim(fan, HOLDER) is None

    assert tuple(_status(fan, notice["id"])) == ("CANCELLED", fanout.INELIGIBLE)
    fan.pump.run_pass()
    assert _pair(fan, request_id, HOLDER) == fanout.REVOKED


def test_a_requester_made_private_after_enqueue_is_fenced_at_claim(fan):
    fan.card(bot=BYSTANDER)
    fan.pump.run_pass()
    [notice] = _notices(fan)
    _private(fan, BYSTANDER)

    assert _claim(fan, HOLDER) is None
    assert tuple(_status(fan, notice["id"])) == ("CANCELLED", fanout.INELIGIBLE)


def test_a_card_decided_before_its_notice_ran_withdraws_it(fan):
    request_id = fan.card()
    fan.pump.run_pass()
    [notice] = _notices(fan)
    fan.store.resolve_request(request_id, status=REQ_DENIED, resolved_by="fixture-person",
                              message="person reason", continuation_owner="server")

    fan.pump.run_pass()

    assert tuple(_status(fan, notice["id"])) == ("CANCELLED", fanout.WITHDRAWN)
    assert _pair(fan, request_id, HOLDER) == fanout.CANCELLED
    assert len(_notices(fan)) == 1          # no follow-up for a notice never seen


def test_a_seen_notice_gets_one_steer_only_follow_up_without_the_reason(fan):
    request_id = fan.card()
    fan.pump.run_pass()
    [notice] = _notices(fan)
    _deliver(fan, notice["id"])
    fan.store.resolve_request(request_id, status=REQ_DENIED, resolved_by="fixture-person",
                              message="person reason", continuation_owner="server")

    fan.pump.run_pass()
    fan.pump.run_pass()

    [_, follow_up] = _notices(fan)
    assert follow_up["message_kind"] == fanout.RESOLVED_KIND
    assert follow_up["payload_json"][STEER_ONLY_KEY] is True
    assert follow_up["idempotency_key"] == f"{fanout.RESOLVED_KEY_PREFIX}{request_id}:{HOLDER}"
    assert "person reason" not in follow_up["message"] and REQ_DENIED in follow_up["message"]
    assert _pair(fan, request_id, HOLDER) == fanout.FOLLOWED_UP


def test_the_resolver_gets_no_follow_up(fan):
    request_id = fan.card()
    fan.pump.run_pass()
    [notice] = _notices(fan)
    _deliver(fan, notice["id"])
    assert fan.resolve(HOLDER, request_id, reason="holder reason").status_code == 200

    fan.pump.run_pass()

    assert len(_notices(fan)) == 1
    assert _pair(fan, request_id, HOLDER) == fanout.SELF


def test_cards_pending_when_the_table_is_created_are_baselined(fan):
    request_id = fan.card()
    with fan.deliveries.engine.begin() as conn:
        conn.execute(sa_text(f"DROP TABLE {fanout.TABLE}"))
    fanout._schema_guard._completed.clear()
    fanout.ensure_fanout_schema(fan.deliveries.engine)

    fan.pump.run_pass()

    assert _notices(fan) == []
    assert _pair(fan, request_id, HOLDER) == fanout.BASELINE


def _deliver(ctx, delivery_id):
    """Mark a notice delivered, as a dispatcher run would."""
    with ctx.deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET status='DELIVERED', delivered_at=now() WHERE id=:id"
        ), {"id": delivery_id})


def test_history_marks_notice_rows_and_stubs_them_once_ineligible(fan, monkeypatch):
    from types import SimpleNamespace

    from llm_bawt.media import assets
    from llm_bawt.service.routes.history_approval_notice import hydrate_approver_notices_for_page

    request_id = fan.card()
    fan.pump.run_pass()
    [notice] = _notices(fan)
    message_id = str(notice["user_message_id"])
    with fan.deliveries.engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS sessions (id text, bot_id text, user_id text)")
        conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS messages (id text, bot_id text, session_id text, role text)")
        conn.execute(sa_text("INSERT INTO sessions VALUES ('s1', :b, :u)"), {"b": HOLDER, "u": USER})
        conn.execute(sa_text("INSERT INTO messages VALUES (:m, :b, 's1', 'user'), ('m-plain', :b, 's1', 'user')"),
                     {"m": message_id, "b": HOLDER})
    monkeypatch.setattr(assets, "_build_engine", lambda _config: fan.deliveries.engine)
    service = SimpleNamespace(config=None)
    page = [{"id": message_id, "role": "user"}, {"id": "m-plain", "role": "user"}]

    cards = hydrate_approver_notices_for_page(service, HOLDER, page, USER)
    assert set(cards) == {message_id}
    card = cards[message_id]
    assert (card["request_id"], card["kind"], card["available"]) == (request_id, fanout.REQUEST_KIND, True)
    assert card["detail"]["request_id"] == request_id and card["detail"]["reason"] is None
    assert SECRET not in str(card)
    assert hydrate_approver_notices_for_page(service, HOLDER, page, "fixture-other-user") == {}

    assert fan.api.as_person(HOLDER, granted=False).status_code == 200
    assert hydrate_approver_notices_for_page(service, HOLDER, page, USER)[message_id] == {
        "request_id": request_id, "kind": fanout.REQUEST_KIND, "available": False, "detail": None}
