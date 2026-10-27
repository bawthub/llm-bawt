"""TASK-1585 slice 2: the context-waste report, its scope and its redaction."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.service.context_waste import (
    build_report, caller_scope, operator_scope, resolve_thresholds,
)
from llm_bawt.service.context_waste import report as waste
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_log_schema import TurnLog

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=timezone.utc)
ROWS = [
    {"slug": "alpha", "include_in_global_search": True, "tags": []},
    {"slug": "beta", "include_in_global_search": True, "tags": []},
    {"slug": "hidden", "include_in_global_search": True, "tags": ["no_global_search"]},
    {"slug": "secret", "include_in_global_search": True, "tags": ["private"]},
]
SECRET_VALUE = "/home/someone/private/notes.txt"


def access(caller_bot=None, user="owner"):
    caller = TrustedCaller(caller_bot, user, "turn") if caller_bot else None
    return BotPolicyResolver._compile(ROWS, caller)


@pytest.fixture
def engine():
    # One shared connection: the MCP tool runs the report in a worker thread,
    # and in-memory sqlite is otherwise one empty database per thread.
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    return engine


class Seeder:
    def __init__(self, engine):
        self.engine = engine
        self.turns = 0

    def turn(self, bot, *, user="owner", session_key="s1", age=timedelta(hours=1), usage=None):
        self.turns += 1
        turn_id = f"turn-{self.turns}"
        when = NOW - age
        with Session(self.engine) as session:
            session.add(TurnLog(id=turn_id, bot_id=bot, user_id=user, created_at=when,
                                ended_at=when, agent_session_key=session_key,
                                token_usage_json=usage))
            session.commit()
        return turn_id

    def call(self, turn_id, bot, tool="Read", *, user="owner", sha="a" * 64, tokens=2_000,
             accounting=2, media=0, complete=True, age=timedelta(hours=1), args=None):
        with Session(self.engine) as session:
            session.add(ToolCallRecord(
                turn_id=turn_id, bot_id=bot, user_id=user, tool_name=tool,
                created_at=NOW - age, result_sha256=sha, result_total_tokens=tokens,
                result_accounting=accounting, result_media_blocks=media,
                result_complete=complete,
                arguments_json=json.dumps(args if args is not None else {"file_path": SECRET_VALUE}),
            ))
            session.commit()


def scope(**kwargs):
    return operator_scope(access(), user_id="owner", days=7, now=NOW, **kwargs)


def report(engine, **kwargs):
    return build_report(engine, scope(**kwargs), resolve_thresholds(None))


def test_repeats_split_in_session_cross_session_and_fresh(engine):
    seed = Seeder(engine)
    first = seed.turn("alpha", session_key="s1", age=timedelta(hours=30))
    seed.call(first, "alpha", age=timedelta(hours=30))
    # Same session again: in-session repeat.
    again = seed.turn("alpha", session_key="s1", age=timedelta(hours=29))
    seed.call(again, "alpha", age=timedelta(hours=29))
    # Other session 2h later: cross-session repeat.
    other = seed.turn("alpha", session_key="s2", age=timedelta(hours=27))
    seed.call(other, "alpha", age=timedelta(hours=27))
    # Third session, 25h after s1 but 22h after s2: still a repeat.
    third = seed.turn("alpha", session_key="s3", age=timedelta(hours=5))
    seed.call(third, "alpha", age=timedelta(hours=5))

    repeats = report(engine)["repeats"]
    (in_session,) = repeats["in_session"]
    (cross,) = repeats["cross_session"]
    assert (in_session["calls"], in_session["repeats"], in_session["repeat_tokens"]) == (4, 1, 2_000)
    assert in_session["example_turns"] == [again]
    assert (cross["repeats"], cross["example_turns"]) == (2, [other, third])
    assert cross["sha"] == "a" * 12 and cross["alternative"].startswith("Read with offset")


def test_a_gap_over_24h_is_a_fresh_occurrence(engine):
    seed = Seeder(engine)
    seed.call(seed.turn("alpha", session_key="s1", age=timedelta(days=5)), "alpha", age=timedelta(days=5))
    seed.call(seed.turn("alpha", session_key="s2", age=timedelta(days=1)), "alpha", age=timedelta(days=1))
    assert report(engine)["repeats"]["cross_session"] == []


@pytest.mark.parametrize("override", [
    {"media": 1}, {"complete": False}, {"accounting": None}, {"tokens": 400},
])
def test_ineligible_rows_are_never_repeats(engine, override):
    seed = Seeder(engine)
    for _ in range(3):
        seed.call(seed.turn("alpha"), "alpha", **override)
    repeats = report(engine)["repeats"]
    assert repeats["in_session"] == repeats["cross_session"] == repeats["cross_tool"] == []


def test_missing_session_key_is_not_eligible(engine):
    seed = Seeder(engine)
    for _ in range(2):
        seed.call(seed.turn("alpha", session_key=None), "alpha")
    assert report(engine)["repeats"]["in_session"] == []


def test_cross_tool_identical_output_is_separate_and_never_in_per_tool_totals(engine):
    seed = Seeder(engine)
    seed.call(seed.turn("alpha"), "alpha", tool="Read")
    seed.call(seed.turn("alpha"), "alpha", tool="Grep", args={"pattern": "x"})
    repeats = report(engine)["repeats"]
    assert repeats["in_session"] == [] and repeats["cross_session"] == []
    (item,) = repeats["cross_tool"]
    assert (item["tools"], item["calls"], item["tokens"]) == (["Grep", "Read"], 2, 4_000)


def test_unverified_rows_are_listed_apart_and_unranked(engine):
    seed = Seeder(engine)
    turn = seed.turn("alpha")
    seed.call(turn, "alpha", tool="mcp__playwright__browser_take_screenshot",
              accounting=None, tokens=700_000)
    seed.call(turn, "alpha", tokens=9_000)
    result = report(engine)
    assert [(t["tool"], t["tokens"]) for t in result["totals"]["trusted"]] == [("Read", 9_000)]
    assert result["totals"]["unverified"][0]["tokens"] == 700_000
    # 700k unverified tokens never reach the oversized ranking.
    assert [entry["tool"] for entry in result["oversized"]] == []
    assert result["labels"]["unverified"].startswith("unverified accounting")


def test_oversized_uses_per_tool_thresholds_and_runtime_overrides(engine):
    seed = Seeder(engine)
    turn = seed.turn("alpha")
    seed.call(turn, "alpha", tool="mcp__bawthub__tasks_get", sha="b" * 64, tokens=5_000,
              args={"task_id": "TASK-1", "free_text_key": SECRET_VALUE})
    seed.call(turn, "alpha", tool="Read", sha="c" * 64, tokens=9_000)

    (entry,) = report(engine)["oversized"]
    assert (entry["tool"], entry["threshold"], entry["count"]) == ("mcp__bawthub__tasks_get", 4_000, 1)
    assert entry["alternative"] == "tasks_get_context or tasks_board"
    assert entry["top"][0]["arg_shape"] == {"keys": ["task_id"], "unknown_keys": 1}

    tuned = resolve_thresholds({"per_tool": {"Read": 8_500, "tasks_get": "bad"}, "default": -1})
    tools = {e["tool"] for e in build_report(engine, scope(), tuned)["oversized"]}
    assert tools == {"Read", "mcp__bawthub__tasks_get"}
    assert tuned["default"] == waste.DEFAULT_THRESHOLDS["default"]


def test_report_never_contains_argument_values_or_other_users(engine):
    seed = Seeder(engine)
    for _ in range(2):
        seed.call(seed.turn("alpha"), "alpha", tokens=20_000)
    # A stranger's turn holding an "owner" tool row, and vice versa: both out.
    seed.call(seed.turn("alpha", user="stranger"), "alpha", sha="d" * 64, tokens=50_000)
    seed.call(seed.turn("alpha"), "alpha", user="stranger", sha="e" * 64, tokens=60_000)

    result = report(engine)
    text = json.dumps(result)
    assert SECRET_VALUE not in text and "someone" not in text
    assert {t["max_tokens"] for t in result["totals"]["trusted"]} == {20_000}
    assert result["repeats"]["in_session"][0]["arg_shape"] == {
        "keys": ["file_path"], "unknown_keys": 0, "ranged": False}


@pytest.mark.parametrize("command,head", [
    ("git status --short", "git"),
    ('/bin/bash -lc "rg -n foo src"', "rg"),
    ("./scripts/secret-tool --token abc", "other"),
    ("unterminated 'quote", "other"),
])
def test_command_head_is_allowlisted_and_unwraps_codex_shell(command, head):
    shape = waste.arg_shape("shell", json.dumps({"command": command, "workdir": "/x"}))
    assert shape == {"keys": ["command", "workdir"], "unknown_keys": 0, "cmd_head": head}


def test_third_party_tool_keys_are_only_counted():
    assert waste.arg_shape("mcp__other__thing", json.dumps({"path": 1, "q": 2})) == {
        "keys": [], "unknown_keys": 2}
    # Codex records first-party tools by their bare name.
    assert waste.arg_shape("tasks_get", json.dumps({"task_id": 1}))["keys"] == ["task_id"]
    assert waste.arg_shape("Read", "{truncated")["unparsed"] is True


def _usage(**fields):
    base = dict(toolCallTokens=10, toolResultTokens=30, attachmentTokens=0,
                assistantMessageTokens=40, userMessageTokens=20)
    base.update(fields)
    return json.dumps({"context_breakdown": {"message_breakdown": base}})


def test_composition_counts_only_well_formed_breakdowns(engine):
    seed = Seeder(engine)
    seed.turn("alpha", usage=_usage())
    seed.turn("alpha", usage=_usage(toolResultTokens=70, userMessageTokens=0, assistantMessageTokens=0,
                                    toolCallTokens=30))
    for bad in (True, 1.5, -1, None):
        seed.turn("alpha", usage=_usage(attachmentTokens=bad))
    seed.turn("alpha", usage=_usage(toolCallTokens=0, toolResultTokens=0,
                                    assistantMessageTokens=0, userMessageTokens=0))
    seed.turn("alpha", usage=None)

    (day,) = report(engine)["composition"]
    assert (day["turns"], day["eligible_turns"], day["malformed_turns"]) == (8, 2, 5)
    assert (day["tool_result_tokens"], day["transcript_tokens"]) == (100, 200)
    assert day["share"] == 0.5 and day["coverage"] == 0.25


def test_operator_scope_excludes_private_and_denies_out_of_scope_filters():
    assert scope().bots == ("alpha", "beta", "hidden")
    assert scope(bot="Beta").bots == ("beta",)
    for bot in ("secret", "nobody"):
        with pytest.raises(BotAccessError, match="access_denied"):
            scope(bot=bot)
    with pytest.raises(BotAccessError, match="invalid_request"):
        operator_scope(access(), user_id="owner", days=0)


def test_caller_scope_is_aggregate_minus_private_including_self():
    # no_global_search bots are absent from another bot's aggregate scope.
    assert caller_scope(access("alpha"), now=NOW).bots == ("alpha", "beta")
    assert caller_scope(access("hidden"), now=NOW).bots == ("alpha", "beta", "hidden")
    # A private caller does not see even itself.
    assert caller_scope(access("secret"), now=NOW).bots == ("alpha", "beta")
    with pytest.raises(BotAccessError, match="access_denied"):
        caller_scope(access("secret"), bot="secret", now=NOW)
    with pytest.raises(BotAccessError, match="access_denied"):
        caller_scope(access("alpha"), bot="hidden", now=NOW)
    assert caller_scope(access("alpha", user="other"), now=NOW).user_id == "other"


def test_digest_is_compact_and_keeps_the_categories(engine):
    seed = Seeder(engine)
    for _ in range(3):
        seed.call(seed.turn("alpha", usage=_usage()), "alpha", tokens=20_000)
    result = waste.digest(report(engine))
    assert set(result["repeats"]) == {"in_session", "cross_session", "cross_tool"}
    assert result["composition_by_bot"] == [
        {"bot": "alpha", "turns": 3, "eligible_turns": 3, "malformed_turns": 0, "share": 0.3}]
    assert len(json.dumps(result)) < 8_000


def test_waste_route_precedes_the_turn_id_route():
    from llm_bawt.service.routes import all_routers
    from llm_bawt.service.routes.context_waste import router as waste_router
    from llm_bawt.service.routes.turn_logs import router as turn_logs_router
    assert all_routers.index(waste_router) < all_routers.index(turn_logs_router)


def test_mcp_tool_fails_closed(monkeypatch):
    from llm_bawt.mcp_server import context_waste_tools as tool

    monkeypatch.setattr(tool, "trusted_caller", lambda name, arguments: None)
    assert json.loads(asyncio.run(tool.context_waste_report())) == {
        "status": "unavailable", "reason": "caller_unavailable"}

    class Broken:
        def __init__(self, engine):
            pass

        async def resolve(self, caller):
            raise BotAccessError("policy_unavailable")

    monkeypatch.setattr(tool, "trusted_caller",
                        lambda name, arguments: TrustedCaller("alpha", "owner", "turn"))
    monkeypatch.setattr(tool, "_runtime", lambda: (object(), object()))
    monkeypatch.setattr(tool, "BotPolicyResolver", Broken)
    assert json.loads(asyncio.run(tool.context_waste_report())) == {
        "status": "unavailable", "reason": "policy_unavailable"}


def test_mcp_tool_returns_the_callers_digest(monkeypatch, engine):
    from llm_bawt.mcp_server import context_waste_tools as tool

    seed = Seeder(engine)
    for bot in ("alpha", "secret"):
        for _ in range(2):
            seed.call(seed.turn(bot), bot, tokens=20_000)

    class Fixed:
        def __init__(self, engine):
            pass

        async def resolve(self, caller):
            return access("alpha")

    monkeypatch.setattr(tool, "trusted_caller",
                        lambda name, arguments: TrustedCaller("alpha", "owner", "turn"))
    monkeypatch.setattr(tool, "_runtime", lambda: (engine, None))
    monkeypatch.setattr(tool, "_thresholds", lambda config: resolve_thresholds(None))
    monkeypatch.setattr(tool, "BotPolicyResolver", Fixed)
    result = json.loads(asyncio.run(tool.context_waste_report(days=30)))
    assert result["status"] == "ok" and result["bots"] == ["alpha", "beta"]
    assert {item["bot"] for item in result["repeats"]["in_session"]} == {"alpha"}


# ----- TASK-1590: declared per-tool keys and the HTTP route ---------------


def test_declared_keys_replace_the_fallback_per_tool():
    declared = {"tasks_get": frozenset({"task_id"})}
    args = json.dumps({"task_id": 1, "limit": 2})
    assert waste.arg_shape("mcp__bawthub__tasks_get", args, declared) == {
        "keys": ["task_id"], "unknown_keys": 1}
    # Absent from the registry: the static allowlist, which knows "limit".
    assert waste.arg_shape("mcp__bawthub__tasks_list", args, declared)["keys"] == ["limit", "task_id"]
    # A same-named third-party tool never borrows first-party keys.
    assert waste.arg_shape("mcp__other__tasks_get", args, declared)["keys"] == []
    # Schema evolution: a newly declared parameter shows without an allowlist edit.
    grown = {"tasks_get": frozenset({"task_id", "brand_new"})}
    assert waste.arg_shape("tasks_get", json.dumps({"brand_new": 1}), grown)["keys"] == ["brand_new"]


def test_declared_mcp_keys_follow_each_registered_schema():
    from mcp.server.fastmcp import FastMCP

    import llm_bawt.mcp_server.server  # noqa: F401  registers every tool
    from llm_bawt.mcp_server.registry import mcp
    from llm_bawt.service.context_waste.registry_keys import declared_mcp_keys

    keys = asyncio.run(declared_mcp_keys())
    tools = asyncio.run(FastMCP.list_tools(mcp))
    assert keys == {tool.name: frozenset(tool.inputSchema["properties"]) for tool in tools}
    assert keys["context_waste_report"] == {"bot", "days"}
    # The fallback holds no names that no registered tool declares any more.
    assert waste.FIRST_PARTY_MCP_KEYS <= frozenset().union(*keys.values())


def test_unreadable_registry_falls_back(monkeypatch):
    from mcp.server.fastmcp import FastMCP

    from llm_bawt.service.context_waste.registry_keys import declared_mcp_keys

    async def broken(self):
        raise RuntimeError("registry down")

    monkeypatch.setattr(FastMCP, "list_tools", broken)
    assert asyncio.run(declared_mcp_keys()) is None


@pytest.fixture
def waste_client(monkeypatch, engine):
    from functools import partial
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.service.routes import context_waste as route

    class Policy:
        def __init__(self, engine):
            pass

        def resolve_sync(self, caller):
            assert caller is None  # the operator view, never a bot caller
            return access()

    async def no_keys():
        return None

    monkeypatch.setattr(route, "get_turn_log_store", lambda: SimpleNamespace(engine=engine))
    monkeypatch.setattr(route, "BotPolicyResolver", Policy)
    monkeypatch.setattr(route, "load_thresholds", lambda: resolve_thresholds(None))
    monkeypatch.setattr(route, "declared_mcp_keys", no_keys)
    monkeypatch.setattr(route, "operator_scope", partial(operator_scope, now=NOW))
    app = FastAPI()
    app.include_router(route.router)
    return TestClient(app)


def test_waste_route_scopes_to_the_user_and_excludes_private_bots(engine, waste_client):
    seed = Seeder(engine)
    for bot in ("alpha", "secret"):
        for _ in range(2):
            seed.call(seed.turn(bot), bot, tokens=20_000)
    seed.call(seed.turn("alpha", user="stranger"), "alpha", user="stranger",
              sha="e" * 64, tokens=60_000)

    response = waste_client.get("/v1/turn-logs/waste", params={"user_id": "owner"})
    assert response.status_code == 200
    body = response.json()
    assert "secret" not in body["bots"] and {"alpha", "beta", "hidden"} <= set(body["bots"])
    assert {item["bot"] for item in body["repeats"]["in_session"]} == {"alpha"}
    assert {item["max_tokens"] for item in body["totals"]["trusted"]} == {20_000}
    assert "secret" not in response.text and SECRET_VALUE not in response.text

    stranger = waste_client.get("/v1/turn-logs/waste", params={"user_id": "stranger"}).json()
    assert {item["max_tokens"] for item in stranger["totals"]["trusted"]} == {60_000}


@pytest.mark.parametrize("params,status", [
    ({"user_id": "owner", "bot": "secret"}, 404),
    ({"user_id": "owner", "bot": "nobody"}, 404),
    ({"user_id": "owner", "days": 0}, 422),
    ({"user_id": "owner", "days": 91}, 422),
    ({}, 422),
])
def test_waste_route_denies_out_of_scope_requests(waste_client, params, status):
    response = waste_client.get("/v1/turn-logs/waste", params=params)
    assert response.status_code == status


def test_private_and_unknown_bots_are_denied_identically(waste_client):
    private = waste_client.get("/v1/turn-logs/waste", params={"user_id": "owner", "bot": "secret"})
    unknown = waste_client.get("/v1/turn-logs/waste", params={"user_id": "owner", "bot": "nobody"})
    assert private.json() == unknown.json() == {"detail": "access_denied"}
