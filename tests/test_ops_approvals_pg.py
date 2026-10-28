"""Delegated ops approval against an isolated PostgreSQL DB (TASK-1609).

A bot holding ``approve_ops`` lists and resolves another bot's pending
``ops_run`` card through ``/v1/ops-approvals``. The real routes run over HTTP
on the real tag and approval stores; the turn capability, the event
subscriber and the tool-card store are stand-ins. Fixture bots only.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.approval_models import (
    EXEC_PENDING,
    KIND_MCP,
    REQ_APPROVED,
    REQ_DENIED,
    REQ_PENDING,
    RESOLVED_VIA_BOT_GRANT,
    RESOLVED_VIA_USER,
)
from llm_bawt.approval_policies import ToolApprovalPolicyStore
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.service.routes import approval_policies as approval_routes
from llm_bawt.service.routes import ops_approvals as routes
from llm_bawt.service.routes import turn_logs as turn_log_routes
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER, TaskTurnContextError
from tests.bot_tag_fixtures import tag
from tests.test_approval_mcp_state import _mcp_req
from tests.test_bot_grants_pg import grant_api  # noqa: F401
from tests.test_bot_tag_api_pg import api  # noqa: F401
from tests.test_bot_tags_pg import rig, upsert  # noqa: F401

pytestmark = pytest.mark.integration

HOLDER = "fixture-target"      # granted approve_ops below
REQUESTER = "fixture-peer"     # asks for the ops run
BYSTANDER = "fixture-manager"  # holds no grant
USER = "fixture-user"
SECRET = "fixture-secret-value"
SNAPSHOT = {
    "snapshot_version": 1,
    "operation": {"slug": "fixture.restart", "title": "Restart fixture", "risk_level": "medium"},
    "schema": {"type": "object", "properties": {
        "service": {"type": "string"},
        "token": {"type": "string", "x-sensitive": True}}},
    "resolved_args": {"service": "fixture-svc", "token": SECRET},
}


@pytest.fixture
def ops(grant_api, monkeypatch):  # noqa: F811
    store = ToolApprovalPolicyStore(None, engine=grant_api.rig.engine)
    events, cards = [], []

    def open_turn(token):
        if not token.startswith("turn:"):
            raise TaskTurnContextError("bad turn capability")
        bot = token.removeprefix("turn:")
        return SimpleNamespace(bot_id=bot, user_id=USER, turn_id=f"turn-of-{bot}")

    async def xadd(stream, fields, **_kwargs):
        events.append((stream, json.loads(fields["payload"])))

    class Cards:
        def __init__(self, engine):
            pass

        def mark_approval_status(self, **kwargs):
            cards.append(("mark", kwargs))

        def resolve_approval_result(self, **kwargs):
            cards.append(("result", kwargs))

    monkeypatch.setattr(routes, "open_task_turn_context", open_turn)
    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(approval_routes, "_store", lambda: store)
    monkeypatch.setattr(approval_routes, "_subscriber",
                        lambda: SimpleNamespace(_redis=SimpleNamespace(xadd=xadd)))
    monkeypatch.setattr(approval_routes, "get_turn_log_store", lambda: SimpleNamespace(engine=object()))
    monkeypatch.setattr(approval_routes, "ToolCallStore", Cards)
    grant_api.client.app.include_router(routes.router)
    grant_api.client.app.include_router(approval_routes.router)
    assert grant_api.as_person(HOLDER).status_code == 200

    def card(*, bot=REQUESTER, user=USER, tool="ops_run", snapshot=SNAPSHOT, **over):
        request_id = f"req-{uuid.uuid4().hex}"
        fields = dict(
            request_id=request_id, tool_use_id=f"toolu_{uuid.uuid4().hex}",
            turn_id=f"turn-{uuid.uuid4().hex}", invocation_hash=uuid.uuid4().hex * 2,
            bot_id=bot, user_id=user, session_key=f"{bot}:{user}", tool_name=tool,
            prompt=f"Approve? {SECRET}", subject=f"args token={SECRET}",
            tool_arguments={"operation": "fixture.restart", "args": {"token": SECRET}},
            operations_snapshot=snapshot)
        store.record_mcp_request(**_mcp_req(**{**fields, **over}))
        return request_id

    def as_bot(bot, method, path, body=None, headers=None):
        hdrs = {TASK_TURN_CONTEXT_HEADER: f"turn:{bot}"} if headers is None else headers
        return grant_api.client.request(method, path, json=body, headers=hdrs)

    def resolve(bot, request_id, decision="approve", reason="fixture reason", **kw):
        return as_bot(bot, "POST", f"/v1/ops-approvals/{request_id}/resolve",
                      {"decision": decision, "reason": reason}, **kw)

    ctx = SimpleNamespace(api=grant_api, store=store, events=events, cards=cards, card=card,
                          resolve=resolve, listing=lambda bot, **kw: as_bot(bot, "GET", "/v1/ops-approvals", **kw))
    return ctx


def _code(response) -> str:
    return response.json()["detail"]["code"]


def _private(ops, slug):
    with ops.api.rig.engine.begin() as conn:
        tag(conn, slug, BehaviorTag.PRIVATE)


# --- resolve ------------------------------------------------------------------


def test_a_holder_approves_another_bots_ops_card_with_provenance(ops):
    request_id = ops.card()

    response = ops.resolve(HOLDER, request_id, reason="safe to restart")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "request_id": request_id, "status": REQ_APPROVED, "resolved_by": f"bot:{HOLDER}",
        "resolved_by_bot": HOLDER, "reason": "safe to restart",
        "awaiting_admission": True, "execution_state": EXEC_PENDING}
    row = ops.store.get_request(request_id)
    assert (row.status, row.resolved_by, row.resolved_via, row.resolved_turn_id) == (
        REQ_APPROVED, f"bot:{HOLDER}", RESOLVED_VIA_BOT_GRANT, f"turn-of-{HOLDER}")
    assert row.resolution_message == "safe to restart"
    # The approval waits for ops admission exactly as a person's does.
    assert [r.id for r in ops.store.find_admittable_ops_requests()] == [request_id]
    assert [kind for kind, _ in ops.cards] == ["mark"]
    [(stream, event)] = ops.events
    assert stream == f"events:{REQUESTER}:{USER}"
    assert {k: event[k] for k in ("request_id", "status", "resolved_by", "resolved_via",
                                  "resolved_by_bot", "resolution_message")} == {
        "request_id": request_id, "status": REQ_APPROVED, "resolved_by": f"bot:{HOLDER}",
        "resolved_via": RESOLVED_VIA_BOT_GRANT, "resolved_by_bot": HOLDER,
        "resolution_message": "safe to restart"}
    assert "result" not in event


def test_a_holder_denies_and_the_card_gets_the_denial(ops):
    request_id = ops.card()

    response = ops.resolve(HOLDER, request_id, decision="deny", reason="not now")

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["status"], body["resolved_by_bot"], body["reason"]) == (REQ_DENIED, HOLDER, "not now")
    assert body["awaiting_admission"] is False
    row = ops.store.get_request(request_id)
    assert (row.status, row.resolved_via) == (REQ_DENIED, RESOLVED_VIA_BOT_GRANT)
    assert ops.store.find_admittable_ops_requests() == []
    assert [kind for kind, _ in ops.cards] == ["result"]
    assert ops.events[0][1]["resolved_by_bot"] == HOLDER


@pytest.mark.parametrize("caller", [BYSTANDER, "fixture-nobody"])
def test_a_bot_without_the_grant_is_refused(ops, caller):
    request_id = ops.card()
    for response in (ops.resolve(caller, request_id), ops.listing(caller)):
        assert response.status_code == 403 and _code(response) == "grant_required"
    assert ops.store.get_request(request_id).status == REQ_PENDING


def test_a_private_holder_is_not_a_holder(ops):
    request_id = ops.card()
    _private(ops, HOLDER)
    response = ops.resolve(HOLDER, request_id)
    assert response.status_code == 403 and _code(response) == "grant_required"
    assert ops.store.get_request(request_id).status == REQ_PENDING


def test_a_revoke_takes_effect_on_the_next_call(ops):
    first, second = ops.card(), ops.card()
    assert ops.resolve(HOLDER, first).status_code == 200
    assert ops.api.as_person(HOLDER, granted=False).status_code == 200
    response = ops.resolve(HOLDER, second)
    assert response.status_code == 403 and _code(response) == "grant_required"


def test_a_holder_cannot_approve_its_own_request(ops):
    request_id = ops.card(bot=HOLDER)
    response = ops.resolve(HOLDER, request_id)
    assert response.status_code == 403 and _code(response) == "self_approval_refused"
    assert ops.store.get_request(request_id).status == REQ_PENDING and ops.events == []


@pytest.mark.parametrize("headers,code", [
    ({}, "turn_capability_required"),
    ({TASK_TURN_CONTEXT_HEADER: "forged"}, "turn_capability_invalid"),
])
def test_a_missing_or_forged_turn_capability_is_refused(ops, headers, code):
    request_id = ops.card()
    for response in (ops.resolve(HOLDER, request_id, headers=headers),
                     ops.listing(HOLDER, headers=headers)):
        assert response.status_code == 403 and _code(response) == code
    assert ops.store.get_request(request_id).status == REQ_PENDING


def test_a_card_that_is_not_an_ops_run_is_refused(ops):
    mcp_card = ops.card(tool="tasks_create", snapshot=None)
    harness = f"req-{uuid.uuid4().hex}"
    ops.store.record_request(
        request_id=harness, bot_id=REQUESTER, user_id=USER, turn_id="turn-h", backend="claude-code",
        tool_name="Bash", tool_arguments={"command": "true"}, subject="true", grant_key="g",
        policy_id=None, severity="high", prompt="Run it?")
    assert ops.store.get_request(harness).request_kind != KIND_MCP
    for request_id in (mcp_card, harness):
        response = ops.resolve(HOLDER, request_id)
        assert response.status_code == 409 and _code(response) == "not_ops_card"
        assert ops.store.get_request(request_id).status == REQ_PENDING


def test_cards_the_caller_cannot_see_are_not_found(ops):
    other_user = ops.card(user="fixture-other-user")
    hidden = ops.card(bot=BYSTANDER)
    _private(ops, BYSTANDER)
    for request_id in (other_user, hidden, "req-missing"):
        response = ops.resolve(HOLDER, request_id)
        assert response.status_code == 404 and _code(response) == "not_found"
    assert ops.store.get_request(other_user).status == REQ_PENDING
    assert ops.store.get_request(hidden).status == REQ_PENDING


@pytest.mark.parametrize("body", [
    {"decision": "approve", "reason": ""},
    {"decision": "cancel", "reason": "x"},
    {"decision": "approve", "reason": "x", "resolved_by": "nick"},
    {"decision": "approve", "reason": "x" * 1001},
])
def test_a_bad_body_changes_nothing(ops, body):
    request_id = ops.card()
    response = ops.api.client.post(f"/v1/ops-approvals/{request_id}/resolve", json=body,
                                   headers={TASK_TURN_CONTEXT_HEADER: f"turn:{HOLDER}"})
    assert 400 <= response.status_code < 500 and response.status_code != 404
    assert ops.store.get_request(request_id).status == REQ_PENDING


def test_an_already_resolved_card_reports_its_resolver(ops):
    request_id = ops.card()
    ops.store.resolve_request(request_id, status=REQ_DENIED, resolved_by="fixture-person",
                              message="person said no", continuation_owner="server")
    response = ops.resolve(HOLDER, request_id)
    assert response.status_code == 409 and _code(response) == "already_resolved"
    detail = response.json()["detail"]
    assert (detail["status"], detail["resolved_by"], detail["resolved_by_bot"], detail["reason"]) == (
        REQ_DENIED, "fixture-person", None, "person said no")


def test_the_loser_of_a_race_gets_the_winners_decision(ops, monkeypatch):
    """A person resolves after the route read the card as pending, before its CAS."""
    request_id = ops.card()
    stale = ops.store.get_request(request_id)
    real_get = ops.store.get_request
    calls = []

    def get_request(rid):
        calls.append(rid)
        if len(calls) == 1:
            ops.store.resolve_request(rid, status=REQ_DENIED, resolved_by="fixture-person",
                                      message="person first", continuation_owner="server")
            return stale
        return real_get(rid)

    monkeypatch.setattr(ops.store, "get_request", get_request)
    response = ops.resolve(HOLDER, request_id)

    assert response.status_code == 409 and _code(response) == "already_resolved"
    detail = response.json()["detail"]
    assert (detail["status"], detail["resolved_by"], detail["resolved_by_bot"], detail["reason"]) == (
        REQ_DENIED, "fixture-person", None, "person first")
    row = real_get(request_id)
    assert (row.resolved_by, row.resolved_via, row.resolved_turn_id) == (
        "fixture-person", RESOLVED_VIA_USER, None)


# --- the person route and the derived field -----------------------------------


@pytest.mark.parametrize("resolver", ["bot:fixture-target", "BOT:fixture-target", " bot:x"])
def test_the_person_route_refuses_a_bot_resolver_name(ops, resolver):
    request_id = ops.card()
    response = ops.api.client.post(f"/v1/chat/approvals/{request_id}/resolve",
                                   json={"decision": "approve", "resolved_by": resolver})
    assert response.status_code == 400 and _code(response) == "reserved_resolver"
    assert ops.store.get_request(request_id).status == REQ_PENDING


def test_resolved_by_bot_comes_only_from_a_bot_grant_resolution(ops):
    request_id = ops.card()
    ops.store.resolve_request(request_id, status=REQ_APPROVED, resolved_by="bot:fixture-target",
                              continuation_owner="server")
    row = ops.store.get_request(request_id)
    assert row.resolved_via == RESOLVED_VIA_USER and row.resolved_by_bot is None
    assert row.to_api()["resolved_by_bot"] is None and row.to_api()["resolved_via"] == RESOLVED_VIA_USER


def test_history_tool_cards_name_the_bot_that_resolved_them(ops, monkeypatch):
    by_bot = ops.card()
    assert ops.resolve(HOLDER, by_bot, reason="fixture: safe restart").status_code == 200
    by_person = ops.card()
    ops.store.resolve_request(by_person, status=REQ_APPROVED, resolved_by="bot:fixture-target",
                              continuation_owner="server")
    monkeypatch.setattr(turn_log_routes, "get_tool_approval_policy_store", lambda _config: ops.store)
    monkeypatch.setattr(turn_log_routes, "get_service", lambda: SimpleNamespace(config=None))
    calls = [{"approval_request_id": rid, "approval_status": "approved"} for rid in (by_bot, by_person)]
    calls.append({"approval_request_id": None, "approval_status": None})

    turn_log_routes._enrich_call_approvals(calls)

    assert calls[0] == {"approval_request_id": by_bot, "approval_status": "approved",
                        "approval_resolved_by_bot": HOLDER,
                        "approval_resolution_message": "fixture: safe restart"}
    assert "approval_resolved_by_bot" not in calls[1] and "approval_resolved_by_bot" not in calls[2]


# --- list -----------------------------------------------------------------------


def test_the_list_shows_resolvable_cards_masked_and_nothing_else(ops):
    shown = ops.card()
    ops.card(bot=HOLDER)                                     # own request
    ops.card(user="fixture-other-user")                      # another user's
    ops.card(tool="tasks_create", snapshot=None)             # not ops_run
    ops.card(snapshot={"operation": {"slug": "x"}})          # unreadable snapshot
    decided = ops.card()
    ops.store.resolve_request(decided, status=REQ_DENIED, resolved_by="fixture-person",
                              continuation_owner="server")
    hidden = ops.card(bot=BYSTANDER)
    _private(ops, BYSTANDER)

    response = ops.listing(HOLDER)

    assert response.status_code == 200, response.text
    [card] = response.json()["data"]
    assert card["request_id"] == shown and hidden != shown
    assert {k: card[k] for k in ("requested_by", "severity", "operation", "title", "risk_level", "args")} == {
        "requested_by": REQUESTER, "severity": "high", "operation": "fixture.restart",
        "title": "Restart fixture", "risk_level": "medium",
        "args": {"service": "fixture-svc", "token": "***"}}
    assert SECRET not in response.text
    assert set(card) == {"request_id", "requested_by", "created_at", "severity", "operation",
                         "title", "risk_level", "args"}


def test_older_cards_the_caller_cannot_see_do_not_fill_the_page(ops, monkeypatch):
    monkeypatch.setattr(routes, "LIST_LIMIT", 2)
    ops.card(bot=HOLDER)                                     # own requests, oldest
    ops.card(bot=HOLDER)
    ops.card(bot=BYSTANDER)                                  # private requester
    ops.card(bot="fixture-unknown-bot")                      # not a known bot
    shown = ops.card()
    _private(ops, BYSTANDER)

    response = ops.listing(HOLDER)

    assert response.status_code == 200, response.text
    assert [c["request_id"] for c in response.json()["data"]] == [shown]
