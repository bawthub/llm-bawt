"""Fan-out sends and ``steer_only`` without a database (TASK-1431 S1).

Covers the MCP mode resolver, the target selector and its expansion, the batch
route's refusals that must fire before any service or database access, the
per-target batch loop and the MCP ``targets`` routing. The PostgreSQL halves
(rate limit, keyed replay, claim settlement) are in
``test_inter_bot_steer_generation_pg.py``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import llm_bawt.mcp_server.server as server  # registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools  # noqa: E402
from llm_bawt import bot_tags, inter_bot_send as send
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.bot_tags.facade import TagSnapshot
from llm_bawt.inter_bot_send_targets import (
    MAX_FANOUT_TARGETS, SelectorError, TargetExpander, TargetSelector,
)
from llm_bawt.inter_bot_steer_only import STEER_ONLY, STEER_OR_IDLE, WHEN_IDLE
from llm_bawt.mcp_server import inter_bot_fanout, task_association
from llm_bawt.mcp_server.inter_bot_send_mode import (
    SendMode, SendModeError, require_one_recipient, resolve_send_mode,
)
from llm_bawt.service.routes import inter_bot_deliveries as route
from llm_bawt.task_turn_context import TaskTurnContextError

ROLE = "fixture_role"
SENDER = "fx-sender"


# ---- mode resolver ---------------------------------------------------------

def _mode(**overrides) -> SendMode:
    options = dict(delivery=None, queue_if_busy=False, fire_and_forget=None,
                   wait_for_reply=False, task_id=None, idempotency_key=None)
    return resolve_send_mode(**{**options, **overrides})


@pytest.mark.parametrize("spelling", ["steer_only", "steer-only", " STEER_ONLY "])
def test_steer_only_resolves_to_a_steer_that_never_waits(spelling):
    mode = _mode(delivery=spelling)
    assert (mode.delivery, mode.wait, mode.prefer_steer, mode.steer_only) == (
        STEER_ONLY, False, True, True)


@pytest.mark.parametrize("overrides", [
    {"queue_if_busy": True},
    {"wait_for_reply": True},
    {"fire_and_forget": False},
])
def test_steer_only_refuses_queueing_and_waiting(overrides):
    with pytest.raises(SendModeError):
        _mode(delivery="steer_only", **overrides)


def test_existing_modes_are_not_steer_only():
    assert not _mode().steer_only and _mode().delivery == STEER_OR_IDLE
    assert not _mode(delivery="when_idle").prefer_steer
    assert _mode(wait_for_reply=True).delivery == WHEN_IDLE


@pytest.mark.parametrize(("target", "targets", "ok"), [
    ("bot", None, True),
    ("", ["bot"], True),
    ("", {"all_agents": True}, True),
    ("bot", ["bot"], False),
    ("", None, False),
    ("  ", None, False),
])
def test_exactly_one_recipient_is_named(target, targets, ok):
    if ok:
        require_one_recipient(target, targets)
    else:
        with pytest.raises(SendModeError):
            require_one_recipient(target, targets)


# ---- selector --------------------------------------------------------------

def test_a_bare_list_is_an_explicit_bot_selector():
    selector = TargetSelector.parse([" Alpha ", "beta", "alpha"])
    assert selector.bots == ("alpha", "beta") and selector.explicit
    assert selector.to_json() == {"bots": ["alpha", "beta"]}


@pytest.mark.parametrize("raw", [
    {"all_agents": True}, {"backends": ["claude-code"]}, {"tags": [ROLE]}, {"bots": ["a"]},
])
def test_a_selector_round_trips(raw):
    assert TargetSelector.parse(raw).to_json() == raw


@pytest.mark.parametrize("raw", [
    {}, {"bots": ["a"], "tags": [ROLE]}, {"all_agents": "yes"}, {"everyone": True},
    {"bots": "a"}, {"bots": [""]}, "a", 3, [], {"all_agents": False},
])
def test_a_malformed_selector_is_refused(raw):
    with pytest.raises(SelectorError):
        TargetSelector.parse(raw)


def _bot(slug, *, backend="claude-code", bot_type="agent"):
    return SimpleNamespace(slug=slug, bot_type=bot_type, agent_backend=backend, harness=backend)


def _snapshot(*, private=(), inactive=(), roles=()):
    catalog = sorted({*BehaviorTag, ROLE})
    slugs = ["fx-a", "fx-b", "fx-c", "fx-chat", "fx-private", "fx-inactive", SENDER, "fx-plain"]
    return TagSnapshot.from_rows([{
        "slug": slug, "include_in_global_search": True, "catalog": catalog,
        "tags": [*([BehaviorTag.PRIVATE] if slug in private else []),
                 *([BehaviorTag.INACTIVE] if slug in inactive else []),
                 *([ROLE] if slug in roles else [])],
    } for slug in slugs])


def _expander():
    bots = [_bot("fx-a"), _bot("fx-b", backend="codex"), _bot("fx-c"),
            _bot("fx-chat", backend=None, bot_type="chat"), _bot("fx-private"),
            _bot("fx-inactive"), _bot(SENDER), _bot("fx-plain", backend="local")]
    snapshot = _snapshot(private=("fx-private",), inactive=("fx-inactive",),
                         roles=("fx-a", "fx-private", "fx-inactive", SENDER))
    return TargetExpander(bots, snapshot)


def _expand(raw, **kwargs):
    options = {"sender": SENDER, "bot_caller": True, **kwargs}
    return _expander().expand(TargetSelector.parse(raw), **options)


def test_all_agents_leaves_out_private_inactive_chat_non_durable_and_the_sender():
    assert _expand({"all_agents": True}).targets == ("fx-a", "fx-b", "fx-c")
    assert _expand({"all_agents": True}, include_self=True).targets == ("fx-a", "fx-b", "fx-c", SENDER)
    assert _expand({"all_agents": True}, exclude=["FX-B"]).targets == ("fx-a", "fx-c")


def test_backends_and_tags_select_inside_the_same_exclusions():
    assert _expand({"backends": ["codex"]}).targets == ("fx-b",)
    assert _expand({"tags": [ROLE]}).targets == ("fx-a",)


def test_an_unknown_tag_is_a_caller_error_not_an_empty_fan_out():
    with pytest.raises(SelectorError):
        _expander().validate_tags(TargetSelector.parse({"tags": ["no_such_role"]}))


def test_explicit_targets_get_per_target_refusals_and_private_reads_as_unknown():
    expansion = _expand(["fx-c", "fx-a", "ghost", "fx-private", "fx-inactive", "fx-plain", SENDER])
    assert expansion.targets == ("fx-a", "fx-c")
    refused = {r.target: (r.status_code, r.error) for r in expansion.refused}
    assert refused == {
        "ghost": (404, "unknown_target"),
        "fx-private": (404, "unknown_target"),
        "fx-inactive": (422, "target_inactive"),
        "fx-plain": (422, "target_not_durable"),
    }
    assert [r.detail for r in expansion.refused if r.target == "fx-private"] == [
        "Unknown target bot 'fx-private'"]


def test_a_user_caller_is_told_a_target_is_private():
    expansion = _expand(["fx-private"], bot_caller=False)
    assert [(r.status_code, r.error) for r in expansion.refused] == [(403, "target_private")]


# ---- request models --------------------------------------------------------

@pytest.mark.parametrize("fields", [
    {"steer_only": True, "prefer_steer": False},
    {"steer_only": True, "session_policy": "reset_without_history"},
    {"steer_only": True, "reset_session_before_delivery": True},
])
def test_a_steer_only_delivery_is_always_a_plain_steer(fields):
    with pytest.raises(ValidationError):
        send.DeliveryCreate(target_bot_id="fx-a", message="hi", **fields)


@pytest.mark.parametrize("field", ["lane", "target_run", "control", "target_bot_id", "prefer_steer"])
def test_the_batch_body_refuses_fields_it_does_not_define(field):
    with pytest.raises(ValidationError):
        send.BatchDeliveryCreate(targets=["fx-a"], message="hi", **{field: "x"})


@pytest.mark.parametrize(("mode", "prefer_steer", "steer_only"), [
    (STEER_OR_IDLE, True, False), (WHEN_IDLE, False, False), (STEER_ONLY, True, True),
])
def test_a_batch_mode_becomes_each_targets_delivery_flags(mode, prefer_steer, steer_only):
    body = send.BatchDeliveryCreate(targets=["fx-a"], message="hi", mode=mode,
                                    sender_bot_id=SENDER, idempotency_key="k", exclude=["fx-b"])
    one = body.for_target("fx-a", {"batch_id": "b"})
    assert (one.target_bot_id, one.prefer_steer, one.steer_only) == ("fx-a", prefer_steer, steer_only)
    assert (one.sender_bot_id, one.idempotency_key, one.metadata) == (SENDER, "k", {"batch_id": "b"})


# ---- batch route -----------------------------------------------------------

def _client():
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


@pytest.fixture
def no_service(monkeypatch):
    """Refusals that need no service or database must not reach either."""
    def unavailable():
        raise AssertionError("service touched")

    monkeypatch.setattr(route, "get_service", unavailable)
    monkeypatch.setattr(route, "_dispatcher", unavailable)


def _batch(**fields):
    body = {"sender_bot_id": SENDER, "message": "hi", "targets": ["fx-a", "fx-c"], **fields}
    return _client().post("/v1/inter-bot-deliveries/batch", json=body)


@pytest.mark.parametrize(("fields", "status", "error"), [
    ({}, 422, "fanout_key_required"),
    ({"idempotency_key": "k", "reply_mode": "waited"}, 422, "fanout_waited"),
    ({"idempotency_key": "k", "session_policy": "reset_retain_history"}, 422, "fanout_reset_forbidden"),
    ({"idempotency_key": "k", "reset_session_before_delivery": True}, 422, "fanout_reset_forbidden"),
])
def test_bot_fan_out_limits_refuse_before_any_lookup(no_service, fields, status, error):
    response = _batch(**fields)
    assert response.status_code == status, response.text
    assert response.json()["detail"]["error"] == error


def test_a_user_fan_out_may_wait_for_nothing_either(no_service):
    response = _batch(author_entity_type="user", author_entity_id="fixture-user", reply_mode="waited")
    assert response.json()["detail"]["error"] == "fanout_waited"


@pytest.mark.parametrize("key", ["batch_id", "batch", "control", "new_session_intent"])
def test_server_owned_metadata_is_refused_on_the_batch_route(no_service, key):
    response = _batch(idempotency_key="k", metadata={key: "forged"})
    assert response.status_code == 422 and "server-owned" in response.text


@pytest.mark.parametrize("field", ["lane", "target_run", "control"])
def test_the_batch_route_has_no_control_lane_fields(no_service, field):
    response = _batch(idempotency_key="k", **{field: {"turn_id": "t"}})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_a_malformed_selector_is_a_422(no_service):
    response = _batch(idempotency_key="k", targets={"bots": ["fx-a"], "all_agents": True})
    assert response.status_code == 422 and "exactly one" in response.text


@pytest.fixture
def user_batch(monkeypatch):
    """A user-authored, unkeyed batch: no rate-limit ledger, no database."""
    bots = [_bot("fx-a"), _bot("fx-c"), _bot("fx-private"), _bot(SENDER)]
    bots += [_bot(f"fx-n{i:02d}") for i in range(MAX_FANOUT_TARGETS)]
    rows = [{"slug": b.slug, "include_in_global_search": True,
             "tags": [BehaviorTag.PRIVATE] if b.slug == "fx-private" else []} for b in bots]

    class Tags:
        def __init__(self, _engine):
            pass

        def snapshot(self):
            return TagSnapshot.from_rows(rows)

    monkeypatch.setattr(bot_tags, "BotTags", Tags)
    monkeypatch.setattr(send, "BotManager", lambda _cfg: SimpleNamespace(list_bots=lambda: bots))
    monkeypatch.setattr(route, "get_service", lambda: SimpleNamespace(config=SimpleNamespace()))
    monkeypatch.setattr(route, "_dispatcher", lambda: SimpleNamespace(store=SimpleNamespace(engine=None)))
    admitted: list[Any] = []

    async def admit_one(_self, body, _request=None):
        if body.target_bot_id == "fx-c":
            raise send.DeliveryRefused(status_code=409, detail="gated")
        admitted.append(body)
        return 202, {"delivery_id": f"d-{body.target_bot_id}", "success": True}

    monkeypatch.setattr(send.DeliverySender, "_admit_one", admit_one)
    return admitted


def _user_batch(**fields):
    return _batch(author_entity_type="user", author_entity_id="fixture-user", **fields)


def test_a_batch_admits_per_target_and_reports_each_refusal(user_batch):
    response = _user_batch(targets=["fx-a", "fx-c", "ghost"], mode="steer_only")
    assert response.status_code == 202, response.text
    data = response.json()
    assert data["mode"] == STEER_ONLY and data["dispatched"] is True
    assert [d["delivery_id"] for d in data["deliveries"]] == ["d-fx-a"]
    assert {r["target"]: r["error"] for r in data["refused"]} == {
        "ghost": "unknown_target", "fx-c": "refused"}
    one = user_batch[0]
    assert one.steer_only and one.metadata["batch_id"] == data["batch_id"]
    assert one.metadata["batch"] == {"selector": {"bots": ["fx-a", "fx-c", "ghost"]},
                                     "exclude": [], "include_self": False,
                                     "targets": ["fx-a", "fx-c"]}


def test_a_batch_with_one_refused_target_answers_like_one_send(user_batch):
    response = _user_batch(targets=["ghost"])
    assert response.status_code == 404
    assert response.json()["detail"] == {"error": "unknown_target",
                                          "detail": "Unknown target bot 'ghost'"}
    assert user_batch == []


def test_a_batch_whose_targets_all_fail_differently_lists_each(user_batch):
    response = _user_batch(targets=["ghost", "fx-c"])
    assert response.status_code == 422
    data = response.json()
    assert data["dispatched"] is False and data["deliveries"] == []
    assert [r["target"] for r in data["refused"]] == ["ghost", "fx-c"]


def test_a_selector_that_matches_nothing_is_a_422(user_batch):
    response = _user_batch(targets={"backends": ["openclaw"]})
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "no_targets"


def test_more_than_the_cap_is_refused_before_any_row(user_batch):
    response = _user_batch(targets={"all_agents": True})
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "too_many_targets"
    assert user_batch == []


# ---- MCP targets routing ---------------------------------------------------

def _signed(monkeypatch, bot_id, turn_id="turn-" + "c" * 32):
    def context():
        if bot_id is None:
            raise TaskTurnContextError("no capability")
        return SimpleNamespace(bot_id=bot_id, turn_id=turn_id, trigger_message_id="m")

    monkeypatch.setattr(task_association, "current_task_turn_context", context)


@pytest.fixture
def mcp_sends(monkeypatch):
    _signed(monkeypatch, SENDER)
    calls: dict[str, list[dict[str, Any]]] = {"single": [], "batch": []}

    async def enqueue(**kwargs):
        calls["single"].append(kwargs)
        return {"delivery_id": "d-1", "status": "QUEUED"}

    async def batch(base_url, **kwargs):
        calls["batch"].append(kwargs)
        return {"batch_id": "b-1", "dispatched": True, "success": True,
                "deliveries": [], "refused": []}

    async def no_refusal(_target):
        return None

    monkeypatch.setattr(server, "_enqueue_durable", enqueue)
    monkeypatch.setattr(tools, "_target_refusal", no_refusal)
    monkeypatch.setattr(tools, "send_batch", batch)
    return calls


def _send(**kwargs):
    return asyncio.run(server.send_message_to_bot(message="hi", **kwargs))


def test_one_explicit_target_takes_the_single_path(mcp_sends):
    result = _send(targets=["FX-A"], delivery="steer_only")
    assert result["dispatched"] is True and mcp_sends["batch"] == []
    [call] = mcp_sends["single"]
    assert call["target_bot_id"] == "fx-a" and call["steer_only"] is True
    assert call["prefer_steer"] is True


def test_a_default_single_send_body_does_not_name_steer_only(mcp_sends):
    _send(target_bot_id="fx-a")
    assert "steer_only" not in mcp_sends["single"][0]


@pytest.mark.parametrize("targets", [["fx-a", "fx-c"], {"tags": [ROLE]}, [SENDER]])
def test_anything_else_is_one_batch(mcp_sends, targets):
    result = _send(targets=targets, idempotency_key="k", delivery="when_idle")
    assert result["batch_id"] == "b-1" and mcp_sends["single"] == []
    [call] = mcp_sends["batch"]
    assert call["targets"] == targets and call["exclude"] == []
    assert call["mode"].delivery == WHEN_IDLE
    fields = call["fields"]
    assert fields["sender_bot_id"] == SENDER and fields["idempotency_key"] == "k"
    assert "metadata" not in fields  # None fields are left to the route's defaults


def test_a_waited_inbound_sender_is_left_out_of_the_batch(mcp_sends, monkeypatch):
    async def inbound(_sender):
        return ("d-waiting", "fx-c")

    monkeypatch.setattr(tools, "_waited_inbound", inbound)
    result = _send(targets=["fx-a", "fx-c"], idempotency_key="k")
    assert mcp_sends["batch"][0]["exclude"] == ["fx-c"]
    assert result["skipped"] == [{"target": "fx-c", "in_reply_to_delivery_id": "d-waiting",
                                  "reason": "awaiting this turn's final answer"}]


@pytest.mark.parametrize("kwargs", [
    {"target_bot_id": "fx-a", "targets": ["fx-c"]},
    {},
    {"targets": {"bots": ["fx-a"], "all_agents": True}},
])
def test_a_bad_recipient_has_no_side_effect(mcp_sends, kwargs):
    result = _send(**kwargs)
    assert result["dispatched"] is False and result["error"]
    assert mcp_sends == {"single": [], "batch": []}


class _Client:
    def __init__(self, status_code, data, seen):
        self._response = SimpleNamespace(status_code=status_code, json=lambda: data)
        self._seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, timeout):
        self._seen.append((url, json))
        return self._response


def _post_batch(monkeypatch, status_code, data):
    seen: list[Any] = []
    monkeypatch.setattr(inter_bot_fanout.httpx, "AsyncClient", lambda: _Client(status_code, data, seen))
    result = asyncio.run(inter_bot_fanout.send_batch(
        "http://app", targets=["fx-a", "fx-c"], mode=_mode(delivery="steer_only"),
        exclude=["fx-b"], fields={"sender_bot_id": SENDER, "message": "hi"},
    ))
    return result, seen


def test_send_batch_posts_one_body_and_reports_the_receipt(monkeypatch):
    result, seen = _post_batch(monkeypatch, 202, {"batch_id": "b", "dispatched": True})
    assert seen == [("http://app/v1/inter-bot-deliveries/batch", {
        "sender_bot_id": SENDER, "message": "hi", "targets": ["fx-a", "fx-c"],
        "exclude": ["fx-b"], "mode": STEER_ONLY, "reply_mode": "async"})]
    assert result == {"batch_id": "b", "dispatched": True, "success": True, "status_code": 202}


def test_send_batch_reports_a_refusal_as_not_dispatched(monkeypatch):
    result, _ = _post_batch(monkeypatch, 429, {"detail": {
        "error": "fanout_rate_limited", "detail": "slow down"}})
    assert result == {"success": False, "dispatched": False, "mode": STEER_ONLY,
                      "error": "fanout_rate_limited", "detail": "slow down", "status_code": 429}
