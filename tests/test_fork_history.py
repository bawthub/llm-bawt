"""TASK-259: a forked thread seeds from the parent through the fork point."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.memory.fork_history import compose_fork_chain, compose_fork_history
from llm_bawt.models.message import Message
from llm_bawt.service.routes import history_seed
from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin


def msg(mid, role, ts, sid="parent", content=None):
    return Message(role=role, content=content or mid, timestamp=ts, db_id=mid, session_id=sid)


PARENT = [
    msg("p1", "user", 10), msg("p2", "assistant", 11),
    msg("p3", "user", 12), msg("p4", "assistant", 13),  # p4 is the fork point
    msg("p5", "user", 14, content="discarded branch"),
    msg("p6", "assistant", 15, content="discarded branch"),
]


def test_fork_pool_stops_at_the_cut_and_adds_the_forks_own_turns():
    own = [msg("f1", "user", 20, sid="fork"), msg("f2", "assistant", 21, sid="fork")]
    pool = compose_fork_history(PARENT, own, [], at_message_id="p4")
    assert [m.db_id for m in pool] == ["p1", "p2", "p3", "p4", "f1", "f2"]


def test_parent_summarized_after_the_cut_leaks_nothing():
    # Summary rows carry the end of the span they cover. One written after the
    # cut covers p1..p6; including it would leak the discarded branch.
    later = msg("s2", "summary", 15, content="summary mentioning discarded branch")
    earlier = msg("s1", "summary", 5, sid="older", content="older shared history")
    pool = compose_fork_history(PARENT, [], [earlier, later], at_message_id="p4")
    assert [m.db_id for m in pool] == ["s1", "p1", "p2", "p3", "p4"]
    assert not any("discarded" in m.content for m in pool)


def test_unknown_fork_message_refuses_rather_than_cutting_wrong():
    assert compose_fork_history(PARENT, [], [], at_message_id="elsewhere") is None


# TASK-2453: grand -> mid (forked at g2) -> child (forked at m1).
GRAND = [msg("g1", "user", 10, "grand"), msg("g2", "assistant", 11, "grand"),
         msg("g3", "user", 12, "grand", content="discarded branch")]
MID = [msg("m1", "user", 20, "mid"), msg("m2", "assistant", 21, "mid", content="discarded branch")]
CHILD = [msg("c1", "user", 30, "child")]
LINKS = {"mid": {"session_id": "grand", "at_message_id": "g2"}}


def _chain(threads, links, summaries=(), **kw):
    return compose_fork_chain(
        CHILD, summaries, session_id="child",
        fork_from={"session_id": "mid", "at_message_id": "m1"},
        read_rows=lambda sid: threads.get(sid, []), fork_of=links.get, **kw,
    )


def test_fork_of_a_fork_keeps_the_grandparents_history_through_each_cut():
    pool = _chain({"grand": GRAND, "mid": MID}, LINKS)
    assert [m.db_id for m in pool] == ["g1", "g2", "m1", "c1"]


def test_fork_chain_summaries_only_before_the_oldest_ancestor():
    # s_mid ends after grand began: it may describe grand's discarded branch.
    s_old = msg("s_old", "summary", 5, "older")
    s_mid = msg("s_mid", "summary", 12, "grand", content="discarded branch")
    pool = _chain({"grand": GRAND, "mid": MID}, LINKS, summaries=[s_old, s_mid])
    assert [m.db_id for m in pool] == ["s_old", "g1", "g2", "m1", "c1"]


def test_fork_chain_stops_at_a_cycle_and_at_the_depth_bound():
    looped = {"mid": LINKS["mid"], "grand": {"session_id": "mid", "at_message_id": "m1"}}
    pool = _chain({"grand": GRAND, "mid": MID}, looped)
    assert [m.db_id for m in pool] == ["g1", "g2", "m1", "c1"]

    pool = _chain({"grand": GRAND, "mid": MID}, LINKS, max_depth=0)
    assert [m.db_id for m in pool] == ["m1", "c1"]


def test_missing_ancestor_cut_keeps_the_direct_parents_history():
    broken = {"mid": {"session_id": "grand", "at_message_id": "gone"}}
    pool = _chain({"grand": GRAND, "mid": MID}, broken)
    assert [m.db_id for m in pool] == ["m1", "c1"]


LEAK = msg("s_leak", "summary", 13, "grand", content="discarded branch")
OLD = msg("s_old", "summary", 5, "older")


@pytest.mark.parametrize("threads,links,kw", [
    # Depth bound: grand is never read.
    ({"grand": GRAND, "mid": MID}, LINKS, {"max_depth": 0}),
    # Missing older cut: grand's fork message is gone.
    ({"grand": GRAND, "mid": MID}, {"mid": {"session_id": "grand", "at_message_id": "gone"}}, {}),
    # Cycle: grand claims to be forked from mid.
    ({"grand": GRAND, "mid": MID},
     {**LINKS, "grand": {"session_id": "mid", "at_message_id": "m1"}}, {}),
], ids=["depth", "missing-cut", "cycle"])
def test_truncated_fork_chain_admits_no_summaries(threads, links, kw):
    # s_leak (ts 13) predates mid's rows but summarizes grand past its cut.
    pool = _chain(threads, links, summaries=[OLD, LEAK], **kw)
    assert [m.db_id for m in pool if m.role == "summary"] == []
    assert not any("discarded" in m.content for m in pool)


def test_default_depth_bound_admits_no_summary_of_an_omitted_ancestor():
    # child -> s1 .. s10; s(i) is cut at m(i) (ts 200-10i); s10 has a discarded
    # row at 105 and a summary of it at 106, older than every collected row.
    threads = {f"s{i}": [msg(f"m{i}", "user", 200 - 10 * i, f"s{i}")] for i in range(1, 11)}
    threads["s10"].append(msg("d10", "assistant", 105, "s10", content="discarded branch"))
    links = {f"s{i}": {"session_id": f"s{i + 1}", "at_message_id": f"m{i + 1}"}
             for i in range(1, 10)}
    leak = msg("s_leak", "summary", 106, "s10", content="discarded branch")
    pool = compose_fork_chain(
        CHILD, [leak], session_id="child",
        fork_from={"session_id": "s1", "at_message_id": "m1"},
        read_rows=lambda sid: threads.get(sid, []), fork_of=links.get,
    )
    assert "m9" in [m.db_id for m in pool] and "m10" not in [m.db_id for m in pool]
    assert not any("discarded" in m.content for m in pool)


def test_mcp_client_loader_follows_the_chain():
    from llm_bawt.mcp_server.client_short_term import _MCPShortTermManager

    def rows(sid):
        return [{"id": m.db_id, "role": m.role, "content": m.content,
                 "timestamp": m.timestamp, "session_id": sid}
                for m in {"grand": GRAND, "mid": MID, "child": CHILD}.get(sid, [])]

    client = SimpleNamespace(
        get_messages=lambda session_id=None, summaries_only=False, user_id=None:
            [] if summaries_only else rows(session_id),
        get_session=lambda sid: {"id": sid, "session_metadata": {"forked_from": LINKS.get(sid)}},
    )
    manager = SimpleNamespace(_memory_client=client)
    pool = _MCPShortTermManager.load_fork_scoped(
        manager, session_id="child", parent_session_id="mid", at_message_id="m1", user_id="u")
    assert [m.db_id for m in pool] == ["g1", "g2", "m1", "c1"]


def _seed(binding, monkeypatch, scope="none"):
    built = Mock(return_value={"messages": [{"role": "user", "content": "x"}]})
    monkeypatch.setattr(history_seed, "build_context_seed", built)
    bot = SimpleNamespace(
        bot=SimpleNamespace(agent_backend="claude-code"),
        config_resolver=SimpleNamespace(resolve_config_setting=lambda key: SimpleNamespace(value=scope)),
    )
    out = history_seed.maybe_build_session_seed(bot, "fixture", "m", "hi", None, binding)
    return out, built


FORK = {"session_id": "parent", "at_message_id": "p4"}


def test_cold_fork_seeds_with_the_fork_bound_even_when_explicit_and_continuity_off(monkeypatch):
    out, built = _seed({"thread_session_id": "fork", "explicit_thread": True, "fork_from": FORK},
                       monkeypatch)
    assert out == [{"role": "user", "content": "x"}]
    assert built.call_args.kwargs["session_id"] == "fork"
    assert built.call_args.kwargs["fork_from"] == FORK


def test_warm_fork_resumes_its_own_provider_session(monkeypatch):
    out, built = _seed({"thread_session_id": "fork", "thread_resume_id": "sdk-1",
                        "fork_from": FORK}, monkeypatch, scope="inline+summaries")
    assert out is None
    built.assert_not_called()


def _binding(metadata):
    db = SimpleNamespace(get_session=lambda sid: {
        "id": sid, "bot_id": "b", "user_id": "u", "session_metadata": metadata})
    llm = SimpleNamespace(history_manager=SimpleNamespace(_db_backend=db),
                          bot=SimpleNamespace(slug="b"), user_id="u",
                          client=SimpleNamespace(_bot_config={}))
    return BridgeSessionResetMixin._resolve_thread_sdk_key(
        SimpleNamespace(), llm, {"thread_session_id": "fork"}, "claude-code")


def test_binding_carries_fork_only_without_a_provider_key():
    assert _binding({"forked_from": FORK})["fork_from"] == FORK
    keyed = _binding({"forked_from": FORK, "agent_session_keys": {"claude_code": "sdk-1"}})
    assert keyed["thread_resume_id"] == "sdk-1"
    assert "fork_from" not in keyed


@pytest.mark.asyncio
@pytest.mark.parametrize("named", ["parent", None])
@pytest.mark.parametrize("point,ok", [
    ({"id": "m1-full", "session_id": "parent", "role": "assistant"}, True),
    ({"session_id": "elsewhere", "role": "assistant"}, False),
    ({"session_id": "parent", "role": "summary"}, False),
    (None, False),
])
async def test_fork_create_checks_the_fork_point_and_keeps_the_parent_active(
    monkeypatch, point, ok, named,
):
    from fastapi import HTTPException

    from llm_bawt.service.routes import sessions

    storage = SimpleNamespace(
        get_session=AsyncMock(return_value={"id": "parent", "bot_id": "b", "user_id": "u"}),
        get_message_by_id=AsyncMock(return_value=point),
        rotate_session=AsyncMock(),
        create_inactive_session=AsyncMock(return_value="fork"),
    )
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)
    monkeypatch.setattr(sessions, "get_effective_bot_id", lambda b: "b")
    monkeypatch.setattr(sessions, "_resolve_user", lambda u: "u")
    body = sessions.CreateSessionRequest(
        activate=False, fork_from=sessions.ForkFrom(session_id=named, at_message_id="m1"))
    # Omitting the parent lets the message's own thread name it.
    expect_ok = ok or (named is None and point and point["session_id"] == "elsewhere")
    if expect_ok:
        assert (await sessions.create_session(body=body, bot_id="b", user="u")).session_id == "fork"
        meta = storage.create_inactive_session.call_args.kwargs["session_metadata"]
        assert meta["forked_from"] == {"session_id": point["session_id"],
                                       "at_message_id": point.get("id", "m1")}
        storage.rotate_session.assert_not_called()
    else:
        with pytest.raises(HTTPException) as exc:
            await sessions.create_session(body=body, bot_id="b", user="u")
        assert exc.value.status_code == 422
        storage.create_inactive_session.assert_not_called()
