"""Read-only call lookup against private PostgreSQL; synthetic identities only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text

from llm_bawt.retrieval.tool_calls import (
    ARG_READ_CHARS, DETAIL_BYTES, LIST_BYTES, CallRequest, ToolCallLookup,
)
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord
from tests.test_messages_find_pg import NOW, world  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def calls(world):  # noqa: F811 - imported shared pytest fixture
    with world.engine.begin() as conn:
        conn.execute(text("CREATE TABLE turn_logs (id text PRIMARY KEY, bot_id text, "
                          "user_id text, trigger_message_id text)"))
        conn.execute(text("CREATE TABLE ops_operations (slug text, args_schema_json text)"))
        ToolCallRecord.__table__.create(conn)
        ToolCallResultPayloadRecord.__table__.create(conn)
    world.lookup = ToolCallLookup(world.engine, clock=lambda: NOW)
    return world


def add(w, *, bot="pub", user="nick", session=None, trigger=None, author=("user", "nick"),
        args=None, result="result fixture", error=False, tool="Bash", age=60,
        payload=False, blob=None, digest=None, turn=None):
    if trigger is None:
        trigger = w.message(bot, session or f"s_{bot}", "synthetic trigger", author=author)
    turn = turn or f"turn_{w._n}"
    params = dict(turn=turn, bot=bot, user=user, trigger=trigger,
                  at=datetime.fromtimestamp(NOW - age, timezone.utc), tool=tool,
                  args=json.dumps(args if args is not None else {"command": "git status"}),
                  result=result, error=error, payload=payload, blob=blob,
                  chars=len(result), size=len(result.encode()),
                  sha=digest or hashlib.sha256(result.encode()).hexdigest())
    with w.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs VALUES (:turn,:bot,:user,:trigger) "
                          "ON CONFLICT DO NOTHING"), params)
        record = conn.execute(text(
            "INSERT INTO tool_call_records (turn_id,bot_id,user_id,created_at,tool_name,"
            "arguments_json,result_text,is_error,result_payload_available,result_blob_key,"
            "result_total_chars,result_total_bytes,result_sha256,result_complete,iteration) "
            "VALUES (:turn,:bot,:user,:at,:tool,:args,:result,:error,:payload,:blob,"
            ":chars,:size,:sha,true,1) RETURNING id"), params).scalar_one()
        if payload:
            conn.execute(text(
                "INSERT INTO tool_call_result_payloads VALUES "
                "(:id,:result,'text/plain',:chars,:size,:sha,true,:at)"),
                {**params, "id": record})
    return record


def raw(w, *, caller="pub", trigger_id=None, **kwargs):
    return w.lookup.lookup(w.access(caller), CallRequest(**kwargs), trigger_id=trigger_id)


def get(w, **kwargs):
    return json.loads(raw(w, **kwargs))


def ids(response):
    return [call["id"] for turn in response["turns"] for call in turn["calls"]]


def test_inventory_and_opt_in_detail(calls, monkeypatch):
    record = add(calls, args={"command": "git status", "token": "syntheticsecret"},
                 result='{"password":"syntheticsecret","ok":"actual result"}')
    from llm_bawt.service import tool_call_store
    monkeypatch.setattr(tool_call_store, "_get_tool_blob_backend",
                        lambda: pytest.fail("default lookup must not touch result backend"))
    listing = get(calls)
    assert ids(listing) == [record]
    call = listing["turns"][0]["calls"][0]
    assert call["args"]["value"]["command"] == "git status"
    assert "syntheticsecret" not in json.dumps(listing)
    assert "result" not in get(calls, id=record)
    detail = get(calls, id=record, include_result=True)
    assert "actual result" in detail["result"]["text"]
    assert "syntheticsecret" not in json.dumps(detail)
    assert detail["result"]["integrity"] == "verified"


@pytest.mark.parametrize("bot", ["priv", "tagged"])
def test_protected_id_and_filter_no_oracle(calls, bot):
    secret = add(calls, bot=bot, args={"x": "never disclosed"}, blob="synthetic")
    ordinary = add(calls)
    assert ids(get(calls)) == [ordinary]
    assert get(calls, id=secret, include_result=True) == get(calls, id=999999, include_result=True)
    assert get(calls, bot=bot) == get(calls, bot="nosuch") == get(calls, bot="empty")


@pytest.mark.parametrize("case", [
    "foreign_session", "foreign_turn", "foreign_call", "deleted_session", "wrong_session",
    "missing_trigger", "protected_author", "unknown_author", "partial_author", "current_trigger",
])
def test_trigger_and_owner_provenance(calls, case):
    session, user, author, trigger = "s_pub", "nick", ("user", "nick"), None
    if case == "foreign_session":
        session = calls.session("foreign", "pub", user="other")
    elif case == "deleted_session":
        session = calls.session("deleted", "pub", status="deleted")
    elif case == "wrong_session":
        session = "s_pub2"
    elif case == "foreign_turn":
        user = "other"
    elif case == "missing_trigger":
        trigger = "absent"
    elif case == "protected_author":
        author = ("bot", "priv")
    elif case == "unknown_author":
        author = ("bot", "unknown")
    elif case == "partial_author":
        author = ("bot", None)
    record = add(calls, session=session, user=user, author=author, trigger=trigger,
                 args={"x": "never disclosed"}, blob="synthetic")
    if case == "foreign_call":
        with calls.engine.begin() as conn:
            conn.execute(text("UPDATE tool_call_records SET user_id='other' WHERE id=:id"), {"id": record})
    if case == "current_trigger":
        with calls.engine.connect() as conn:
            trigger = conn.execute(text("SELECT trigger_message_id FROM turn_logs")).scalar_one()
    else:
        trigger = None
    assert get(calls, id=record, include_result=True, trigger_id=trigger) == {"status": "not_found"}
    assert ids(get(calls, trigger_id=trigger)) == []


def test_policy_reauthorization_on_detail_and_page(calls):
    first = add(calls, bot="pub2", age=10)
    second = add(calls, bot="pub2", age=20)
    listing = get(calls, limit=1)
    assert ids(listing) == [first]
    calls.flag("pub2", False)
    assert get(calls, id=second, include_result=True) == {"status": "not_found"}
    assert ids(get(calls, **listing["next"])) == []


def test_self_and_readable_other_calls(calls):
    mine = add(calls, bot="priv")
    other = add(calls, bot="pub2")
    assert set(ids(get(calls, caller="priv"))) == {mine, other}


def test_filters_order_null_errors_and_keyset(calls):
    first = add(calls, error=True, age=10)
    second = add(calls, error=None, age=10)
    third = add(calls, error=False, age=20, tool="Read")
    old = add(calls, age=90000)
    assert ids(get(calls)) == [second, first, third]
    assert ids(get(calls, errors_only=True)) == [first]
    assert ids(get(calls, tool_name="Read")) == [third]
    page = get(calls, limit=1)
    assert ids(page) == [second]
    assert ids(get(calls, limit=1, **page["next"])) == [first]
    assert ids(get(calls, turn_id="turn_4")) == [old]
    assert ids(get(calls, since=NOW - 15, until=NOW - 5)) == [second, first]


def test_budgets_and_complete_paging(calls):
    expected = [add(calls, args={"command": '\x01' * 5000 + "😀" * 5000},
                    result='\x02' * 20000 + "😀" * 20000, age=60) for _ in range(26)]
    response, seen = get(calls, limit=25), []
    while True:
        assert len(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()) <= LIST_BYTES
        seen.extend(ids(response))
        if response["next"] is None:
            break
        response = get(calls, limit=25, **response["next"])
    assert seen == list(reversed(expected))
    detail = raw(calls, id=expected[0], include_result=True, max_chars=8000)
    assert len(detail.encode()) <= DETAIL_BYTES
    assert json.loads(detail)["result"]["truncated"]


@pytest.mark.parametrize("payload", [False, True])
def test_full_and_truncated_hash(calls, payload):
    record = add(calls, result="é" * 20000, payload=payload)
    result = get(calls, id=record, include_result=True)["result"]
    assert result["truncated"] and result["integrity"] == "unverified"
    assert len(result["text"]) <= 4000
    record = add(calls, result="complete", payload=payload, digest="0" * 64)
    assert get(calls, id=record, include_result=True)["result"] == {
        "status": "unavailable", "reason": "integrity_mismatch"}


def test_object_prefix_and_whole_integrity(calls, monkeypatch):
    from llm_bawt.service import tool_call_store
    body = ("é" * 10000).encode()
    reads = []

    def read(key, byte_range):
        reads.append(byte_range)
        end = int(byte_range.split("-")[1]) + 1
        return SimpleNamespace(data=body[:end], total_size=len(body))

    monkeypatch.setattr(tool_call_store, "_get_tool_blob_backend",
                        lambda: SimpleNamespace(get_range=read))
    record = add(calls, result=body.decode(), blob="synthetic")
    assert get(calls, id=record, include_result=True)["result"]["integrity"] == "unverified"
    assert reads == ["bytes=0-16003"]
    body = b"small result"
    record = add(calls, result=body.decode(), blob="synthetic")
    assert get(calls, id=record, include_result=True)["result"]["integrity"] == "verified"
    body = b"tampered result"
    assert get(calls, id=record, include_result=True)["result"]["reason"] == "integrity_mismatch"
    monkeypatch.setattr(tool_call_store, "_get_tool_blob_backend", lambda: None)
    assert get(calls, id=record, include_result=True)["result"]["reason"] == "result_source_unavailable"


def test_ops_schema_sensitive_and_oversized_arguments(calls):
    with calls.engine.begin() as conn:
        conn.execute(text("INSERT INTO ops_operations VALUES ('synthetic', :schema)"), {
            "schema": json.dumps({"properties": {"arbitrary": {"x-sensitive": True}}})})
    record = add(calls, tool="mcp__bawthub__ops_run",
                 args={"operation": "synthetic", "args": {"arbitrary": "hide this", "keep": "yes"}})
    detail = get(calls, id=record)
    assert detail["call"]["args"]["value"]["args"] == {"arbitrary": "***", "keep": "yes"}
    record = add(calls, tool="ops_run",
                 args={"operation": "synthetic", "args": {"arbitrary": "x" * (ARG_READ_CHARS + 1)}})
    assert get(calls, id=record)["call"]["args"]["reason"] == "oversized_ops_args"


def test_no_writes_or_unauthorized_payload_reads(calls):
    record = add(calls, bot="priv", blob="synthetic")
    queries = []

    def observe(conn, cursor, statement, parameters, context, executemany):
        queries.append(statement)
        if "SELECT" in statement and "tool_call_records" in statement:
            assert conn.execute(text("SHOW transaction_read_only")).scalar() == "on"
            assert conn.execute(text("SHOW transaction_isolation")).scalar() == "repeatable read"

    event.listen(calls.engine, "before_cursor_execute", observe)
    try:
        assert get(calls, id=record, include_result=True) == {"status": "not_found"}
    finally:
        event.remove(calls.engine, "before_cursor_execute", observe)
    assert not any("arguments_json" in sql or "result_text" in sql
                   or "tool_call_result_payloads" in sql for sql in queries)
