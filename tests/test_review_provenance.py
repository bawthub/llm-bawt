"""TASK-1249: a review's models come from the reviewer's signed turn, never its arguments."""

import asyncio

import pytest
from cryptography.fernet import Fernet

from llm_bawt import task_turn_context as codec
from llm_bawt.mcp_server import review_provenance as provenance
from llm_bawt.mcp_server import task_association, task_review_tools
from llm_bawt.task_turn_context import mint_task_turn_context

REVIEWER = "fixture-reviewer"
AUTHOR = "fixture-author"
TURN = "turn-" + "a" * 32
SENDER_TURN = "turn-" + "b" * 32
TASK = {"id": "cfixturetask0001", "shortId": "TASK-7"}
AUTHOR_NOTE = "author_unresolved:no_commit_turn_record"


@pytest.fixture
def fernet(monkeypatch):
    instance = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: instance)
    return instance


def _capability(bot_id=REVIEWER):
    return mint_task_turn_context(
        session_id="12490000-0000-4000-8000-000000000011", turn_id=TURN,
        trigger_message_id="12490000-0000-4000-8000-000000000012",
        bot_id=bot_id, user_id="fixture-user",
    )


@pytest.fixture
def world(monkeypatch):
    """Turn logs, deliveries, catalog and BawtHub, all recorded."""
    state = {
        "models": {(TURN, REVIEWER): "model-r@path-r", (SENDER_TURN, AUTHOR): "model-a@path-a"},
        "vendors": {"model-r@path-r": "vendor-r", "model-a@path-a": "vendor-a"},
        "deliveries": [(AUTHOR, SENDER_TURN)],
        "task": dict(TASK),
        "posts": [], "routes": [], "paths": [], "gets": [], "delivery_reads": [],
    }

    def turn_model(turn_id, bot_id):
        if isinstance(state["models"], Exception):
            raise state["models"]
        return state["models"].get((turn_id, bot_id))

    def deliveries(turn_id, bot_id, refs):
        state["delivery_reads"].append((turn_id, bot_id, refs))
        if isinstance(state["deliveries"], Exception):
            raise state["deliveries"]
        return state["deliveries"]

    def provider_of(ref):
        if isinstance(state["vendors"], Exception):
            raise state["vendors"]
        return state["vendors"].get(ref) if ref else None

    async def fake_get(path, params=None):
        state["gets"].append(path)
        if isinstance(state["task"], Exception):
            raise state["task"]
        return state["task"]

    async def fake_post(path, json, headers=None):
        state["posts"].append(json)
        state["routes"].append("public")
        return {"id": "review-1"}

    async def fake_internal_post(path, json):
        # TASK-1615 slice 2: a signed review goes to the internal route.
        state["posts"].append(json)
        state["routes"].append("internal")
        state["paths"].append(path)
        return {"id": "review-1"}

    monkeypatch.setattr(provenance, "turn_model", turn_model)
    monkeypatch.setattr(provenance, "requesting_deliveries", deliveries)
    monkeypatch.setattr(provenance, "provider_of", provider_of)
    monkeypatch.setattr(task_review_tools, "_api_get", fake_get)
    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    monkeypatch.setattr(task_review_tools, "_internal_post", fake_internal_post)
    return state


def _review(capability=None, *, bot_id=REVIEWER, body="Reviewed on model-x@path-x."):
    binding = task_association.set_current_task_turn_capability(capability) if capability else None
    try:
        return asyncio.run(task_review_tools.add_task_review(
            "TASK-7", "IMPLEMENTATION", "APPROVE", body, bot_id, idempotency_key="TASK-7:r1",
        ))
    finally:
        if binding is not None:
            task_association.reset_current_task_turn_capability(binding)


def _provenance(payload):
    return {key: value for key, value in payload.items()
            if key not in {"stage", "verdict", "body", "targetRevision", "idempotencyKey",
                           "actorBotId"}}


def test_a_signed_review_records_turn_models_and_ignores_the_body(world, fernet):
    assert _review(_capability()) == {"id": "review-1"}
    [payload] = world["posts"]
    assert payload["body"] == "Reviewed on model-x@path-x."
    assert _provenance(payload) == {
        "provenance": "signed_turn",
        "reviewerModel": "model-r@path-r", "reviewerProvider": "vendor-r",
        "requesterBotId": AUTHOR, "requesterModel": "model-a@path-a",
        "authorModel": None, "authorProvider": None, "crossProvider": None,
        "provenanceNotes": [AUTHOR_NOTE],
    }
    assert payload["actorBotId"] == REVIEWER and world["routes"] == ["internal"]
    # Both task forms are matched; the delivery read is scoped to the signed turn.
    assert world["gets"] == ["/tasks/TASK-7"]
    assert world["delivery_reads"] == [(TURN, REVIEWER, [TASK["id"], TASK["shortId"]])]


def test_requester_model_never_stands_in_for_the_author(world, fernet):
    _review(_capability())
    [payload] = world["posts"]
    assert payload["requesterModel"] == "model-a@path-a"
    assert payload["authorModel"] is None
    assert payload["crossProvider"] is None  # differing providers prove nothing yet


def test_an_unsigned_review_says_so_and_reads_nothing(world):
    assert _review() == {"id": "review-1"}
    [payload] = world["posts"]
    assert _provenance(payload) == {"provenance": "unsigned"}
    assert world["routes"] == ["public"] and "actorBotId" not in payload
    assert world["gets"] == [] and world["delivery_reads"] == []


@pytest.mark.parametrize("credential", ["foreign", "forged"])
def test_a_foreign_or_forged_turn_is_refused_not_downgraded(world, fernet, credential):
    capability = _capability(bot_id="fixture-other") if credential == "foreign" else "not-a-token"
    assert _review(capability) == {"error": "caller_unavailable"}
    assert world["posts"] == []


@pytest.mark.parametrize(("deliveries", "note"), [
    ([], "requester_unresolved:no_delivery"),
    ([(AUTHOR, SENDER_TURN), ("fixture-third", "turn-t")], "requester_unresolved:multiple_senders"),
    ([(AUTHOR, None)], "requester_unresolved:no_sender_turn"),
    ([(AUTHOR, None), (AUTHOR, SENDER_TURN)], "requester_unresolved:multiple_sender_turns"),
    ([(AUTHOR, SENDER_TURN), (AUTHOR, "turn-other")], "requester_unresolved:multiple_sender_turns"),
    (RuntimeError("db down"), "requester_unresolved:deliveries_unavailable"),
])
def test_an_unprovable_requester_is_null_with_a_reason(world, fernet, deliveries, note):
    world["deliveries"] = deliveries
    _review(_capability())
    [payload] = world["posts"]
    assert (payload["requesterBotId"], payload["requesterModel"]) == (None, None)
    assert payload["provenanceNotes"] == sorted([note, AUTHOR_NOTE])
    assert payload["reviewerModel"] == "model-r@path-r"


def test_a_requester_without_a_turn_log_keeps_the_bot_but_not_a_model(world, fernet):
    world["models"].pop((SENDER_TURN, AUTHOR))
    _review(_capability())
    [payload] = world["posts"]
    assert (payload["requesterBotId"], payload["requesterModel"]) == (AUTHOR, None)
    assert "requester_unresolved:no_turn_log" in payload["provenanceNotes"]


@pytest.mark.parametrize("task", [RuntimeError("bawthub down"), {"error": "Task not found"}])
def test_an_unreadable_task_leaves_the_requester_unresolved(world, fernet, task):
    world["task"] = task
    _review(_capability())
    [payload] = world["posts"]
    assert payload["requesterBotId"] is None
    assert "requester_unresolved:task_unavailable" in payload["provenanceNotes"]
    assert world["delivery_reads"] == []


@pytest.mark.parametrize(("models", "vendors", "note", "model"), [
    ({}, {}, "reviewer_unresolved:no_turn_log", None),
    (RuntimeError("db down"), {}, "reviewer_unresolved:turn_log_unavailable", None),
    ({(TURN, REVIEWER): "model-z@path-z"}, {}, "reviewer_unresolved:unknown_endpoint", "model-z@path-z"),
    ({(TURN, REVIEWER): "model-r@path-r"}, RuntimeError("catalog"),
     "reviewer_unresolved:catalog_unavailable", "model-r@path-r"),
])
def test_an_unresolved_reviewer_still_posts_with_a_reason(world, fernet, models, vendors, note, model):
    world["models"], world["vendors"] = models, vendors
    world["deliveries"] = []
    assert _review(_capability()) == {"id": "review-1"}
    [payload] = world["posts"]
    assert (payload["reviewerModel"], payload["reviewerProvider"]) == (model, None)
    assert note in payload["provenanceNotes"]
    assert payload["provenanceNotes"] == sorted(payload["provenanceNotes"])


def test_a_retry_from_the_same_turn_sends_identical_provenance(world, fernet):
    world["deliveries"] = []
    _review(_capability())
    _review(_capability())
    assert world["posts"][0] == world["posts"][1]


# TASK-1615 slice 2: rechecks and finding resolutions need the signed turn.

def _signed(capability, call):
    binding = task_association.set_current_task_turn_capability(capability) if capability else None
    try:
        return asyncio.run(call)
    finally:
        if binding is not None:
            task_association.reset_current_task_turn_capability(binding)


def _recheck(**overrides):
    arguments = {
        "rechecks_review_id": "review-r1", "idempotency_key": "TASK-7:r2",
        "resolutions": [{"finding_id": "finding-1", "resolution": "FIXED", "revision": "abc1234"}],
    } | overrides
    return task_review_tools.add_task_review(
        "TASK-7", "IMPLEMENTATION", "APPROVE", "Blocker fixed.", REVIEWER, **arguments)


def test_a_signed_recheck_goes_to_the_recheck_route_with_wire_resolutions(world, fernet):
    assert _signed(_capability(), _recheck()) == {"id": "review-1"}
    [payload] = world["posts"]
    assert world["paths"] == ["/internal/tasks/TASK-7/reviews/review-r1/recheck"]
    assert payload["resolutions"] == [
        {"findingId": "finding-1", "resolution": "FIXED", "revision": "abc1234"}]
    assert payload["provenance"] == "signed_turn" and payload["actorBotId"] == REVIEWER
    assert payload["idempotencyKey"] == "TASK-7:r2"


def test_an_unsigned_recheck_is_refused_before_any_request(world):
    assert _signed(None, _recheck()) == {
        "error": "recheck_requires_signed_turn", "code": "recheck_requires_signed_turn"}
    assert world["posts"] == [] and world["gets"] == [] and world["delivery_reads"] == []


@pytest.mark.parametrize("credential", ["foreign", "forged"])
def test_a_recheck_with_a_bad_turn_is_refused(world, fernet, credential):
    capability = _capability(bot_id="fixture-other") if credential == "foreign" else "not-a-token"
    assert _signed(capability, _recheck()) == {"error": "caller_unavailable"}
    assert world["posts"] == []


def _resolve(bot_id=REVIEWER):
    return task_review_tools.resolve_review_finding(
        "TASK-7", "review-r1", "finding/2", "DEFERRED", bot_id, task="TASK-8", note="Later.")


def test_a_signed_resolution_posts_only_the_signed_marker(world, fernet):
    assert _signed(_capability(), _resolve()) == {"id": "review-1"}
    assert world["paths"] == ["/internal/tasks/TASK-7/reviews/review-r1/findings/finding%2F2/resolve"]
    assert world["posts"] == [{
        "actorBotId": REVIEWER, "resolution": "DEFERRED", "provenance": "signed_turn",
        "task": "TASK-8", "note": "Later.",
    }]
    # A resolution needs no model provenance, so it reads nothing else.
    assert world["gets"] == [] and world["delivery_reads"] == []


@pytest.mark.parametrize("capability", [None, "not-a-token", "foreign"])
def test_an_unsigned_or_bad_resolution_is_refused(world, fernet, capability):
    if capability == "foreign":
        capability = _capability(bot_id="fixture-other")
    expected = ({"error": "resolve_requires_signed_turn", "code": "resolve_requires_signed_turn"}
                if capability is None else {"error": "caller_unavailable"})
    assert _signed(capability, _resolve()) == expected
    assert world["posts"] == []


def test_the_signed_turn_check_names_the_tool_being_called(monkeypatch):
    from llm_bawt.mcp_server import bot_access_identity

    names = []

    def fake_trusted_caller(name, arguments):
        names.append(name)

    monkeypatch.setattr(bot_access_identity, "trusted_caller", fake_trusted_caller)
    assert provenance.signed_review_turn("bot", {}) is None
    assert provenance.signed_review_turn("bot", {}, tool="tasks_reviews_resolve_finding") is None
    assert names == ["tasks_reviews_add", "tasks_reviews_resolve_finding"]
