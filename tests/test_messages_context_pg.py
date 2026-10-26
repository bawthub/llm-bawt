"""TASK-1323 ``messages_context`` on real PostgreSQL: anchors, neighbours, budget.

Private PostgreSQL, isolated schema, synthetic fixtures only (the
``messages_find`` world). Covers the §5.0 denial tests that apply to context
(4, 5, 6, 8, 9, 10, 12), the §5.2 order/budget/cursor contract and the §5.3
card-reference tests 2-5. Labels (§5.3 tests 1 and 6) are TASK-1487.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from llm_bawt.retrieval.context import CHARS_MAX, ContextRequest, MessageContext
from llm_bawt.retrieval.find import FindRequest
from llm_bawt.retrieval.handles import HandleStore
from tests.test_messages_find_pg import world  # noqa: F401  (the shared fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

NOT_FOUND = {"status": "not_found"}
EXPIRED = {"status": "unavailable", "reason": "handle_expired"}
INVALID = {"status": "unavailable", "reason": "invalid_request"}


@pytest.fixture
def ctx(world):  # noqa: F811
    world.cursors = HandleStore(per_owner=16)
    world.context = MessageContext(world.engine, world.handles, world.cursors)

    def call(caller="pub", user="nick", *, trigger_id=None, raw=False, **request):
        if "focus" in request and request["focus"] is not None:
            request["focus"] = tuple(request["focus"])
        out = world.context.context(world.access(caller, user), ContextRequest(**request),
                                    trigger_id=trigger_id)
        return out if raw else json.loads(out)
    world.ctx = call
    return world


def ids(resp: dict, side: str) -> list[str]:
    return [n["id"] for n in resp["neighbors"][side]]


def thread(w, session="s_pub", n=7, *, partition="pub", prefix="line"):
    """``n`` in-scope messages, oldest first, alternating user/bot."""
    out = []
    for i in range(n):
        if i % 2:
            out.append(w.message(partition, session, f"{prefix} {i}", role="assistant",
                                 author=("bot", partition), age=1000 - i * 10))
        else:
            out.append(w.message(partition, session, f"{prefix} {i}", age=1000 - i * 10))
    return out


# -- order, window, shape ---------------------------------------------------

def test_window_is_chronological_around_the_anchor(ctx):
    m = thread(ctx)
    resp = ctx.ctx(bot="pub", id=m[3], before=2, after=2)
    assert resp["status"] == "ok"
    anchor = resp["anchor"]
    assert (anchor["id"], anchor["bot"], anchor["session"], anchor["speaker"]) == (m[3], "pub", "s_pub", "bot")
    assert anchor["content"] == "line 3" and anchor["truncated"] is False and anchor["next_offset"] is None
    assert ids(resp, "before") == [m[1], m[2]] and ids(resp, "after") == [m[4], m[5]]
    assert [n["speaker"] for n in resp["neighbors"]["before"]] == ["bot", "user"]
    assert resp["more_before"] and resp["more_after"]
    full = ctx.ctx(bot="pub", id=m[3], before=10, after=10)
    assert ids(full, "before") == m[:3] and ids(full, "after") == m[4:]
    assert full["more_before"] is None and full["more_after"] is None


def test_default_window_is_three(ctx):
    m = thread(ctx, n=9)
    resp = ctx.ctx(bot="pub", id=m[4])
    assert ids(resp, "before") == m[1:4] and ids(resp, "after") == m[5:8]


def test_equal_timestamps_order_by_id(ctx):
    for mid in ("b", "a", "c"):
        ctx.message("pub", "s_pub", f"tie {mid}", age=500, mid=mid)
    with ctx.engine.begin() as conn:
        conn.execute(text("UPDATE messages SET timestamp = 1 WHERE id IN ('a','b','c')"))
    resp = ctx.ctx(bot="pub", id="b", before=5, after=5)
    assert ids(resp, "before") == ["a"] and ids(resp, "after") == ["c"]


def test_zero_sides_still_report_more(ctx):
    m = thread(ctx, n=3)
    resp = ctx.ctx(bot="pub", id=m[1], before=0, after=0)
    assert resp["neighbors"] == {"before": [], "after": []}
    assert resp["more_before"] and resp["more_after"]


@pytest.mark.parametrize("request_", [
    {"bot": "pub"}, {"id": "x"}, {}, {"bot": "pub", "id": ""},
    {"bot": "pub", "id": "x", "handle": "h", "card_ref": "c0.0"},
    {"bot": "pub", "id": "x", "before": 11}, {"bot": "pub", "id": "x", "after": -1},
    {"bot": "pub", "id": "x", "before": True}, {"bot": "pub", "id": "x", "max_chars": 999},
    {"bot": "pub", "id": "x", "max_chars": CHARS_MAX + 1}, {"bot": "pub", "id": "x", "offset": -1},
    {"bot": "pub", "id": "x", "focus": ["y" * 41]}, {"bot": "pub", "id": "x", "focus": ["a"] * 7},
    {"cursor": "k", "bot": "pub", "id": "x"},
])
def test_invalid_requests(ctx, request_):
    assert ctx.ctx(**request_) == INVALID


# -- §5.0 denial -------------------------------------------------------------

def test_4_out_of_scope_neighbours_are_skipped_and_never_counted(ctx):
    a = ctx.message("pub", "s_pub", "kept before", age=900)
    ctx.message("pub", "s_pub", "Message from bot 'priv': secret", author=(None, None), age=850)
    ctx.message("pub", "s_pub", "excluded delivery", author=("bot", "priv"), age=840)
    ctx.message("pub", "s_pub", "unknown delivery", author=("bot", "system"), age=830)
    ctx.message("pub", "s_pub", "blank bot reply", role="assistant", author=("bot", ""), age=820)
    anchor = ctx.message("pub", "s_pub", "anchor", age=800)
    ctx.session("s_pub_b", "pub")
    ctx.message("pub", "s_pub_b", "other session", age=790)
    ctx.message("pub", "s_pub", "assistant by another bot", role="assistant",
                author=("bot", "pub2"), age=780)
    b = ctx.message("pub", "s_pub", "kept after", role="assistant", author=("bot", "pub"), age=770)
    ctx.message("pub", "s_pub", "trailing legacy", author=(None, None), age=700)
    resp = ctx.ctx(bot="pub", id=anchor, before=1, after=1)
    assert ids(resp, "before") == [a] and ids(resp, "after") == [b]
    assert resp["more_before"] is None and resp["more_after"] is None
    text_ = json.dumps(resp)
    for word in ("secret", "excluded", "unknown", "blank", "other session", "another bot", "legacy"):
        assert word not in text_


def test_12_excluded_or_unknown_anchor_is_not_found(ctx):
    private = ctx.message("priv", "s_priv", "private reply", role="assistant", author=("bot", "priv"))
    tagged = ctx.message("tagged", "s_tagged", "tag-private reply", role="assistant",
                         author=("bot", "tagged"))
    public = ctx.message("pub", "s_pub", "public note")
    assert ctx.ctx(bot="priv", id=private) == NOT_FOUND
    assert ctx.ctx(bot="tagged", id=tagged) == NOT_FOUND
    assert ctx.ctx(bot="nosuch", id=public) == NOT_FOUND
    assert ctx.ctx(bot="pub", id="missing") == NOT_FOUND
    # Composite anchor: the right id under the wrong bot is not_found too.
    assert ctx.ctx(bot="pub2", id=public) == NOT_FOUND


def test_5_self_access_uses_caller_scope(ctx):
    own = ctx.message("priv", "s_priv", "own reply", role="assistant", author=("bot", "priv"), age=500)
    legacy = ctx.message("priv", "s_priv", "legacy reply", role="assistant", author=(None, None), age=400)
    after = ctx.message("priv", "s_priv", "own later", age=300)
    resp = ctx.ctx("priv", bot="priv", id=own)
    assert resp["status"] == "ok" and ids(resp, "after") == [after]
    assert ctx.ctx("priv", bot="priv", id=legacy) == NOT_FOUND


def test_6_user_pin_and_dead_sessions(ctx):
    ctx.session("s_eve", "pub", user="eve")
    eve = ctx.message("pub", "s_eve", "eve said", author=("user", "eve"))
    mine = ctx.message("pub", "s_eve", "nick in eve's session")
    assert ctx.ctx(bot="pub", id=eve) == NOT_FOUND
    assert ctx.ctx(bot="pub", id=mine) == NOT_FOUND
    assert ctx.ctx("pub", "eve", bot="pub", id=eve)["status"] == "ok"
    ctx.session("s_gone", "pub", status="deleted")
    gone = ctx.message("pub", "s_gone", "deleted session")
    assert ctx.ctx(bot="pub", id=gone) == NOT_FOUND


def test_8_trigger_is_never_anchor_or_neighbour(ctx):
    m = thread(ctx, n=3)
    resp = ctx.ctx(bot="pub", id=m[0], after=5, trigger_id=m[1])
    assert ids(resp, "after") == [m[2]]
    assert ctx.ctx(bot="pub", id=m[1], trigger_id=m[1]) == NOT_FOUND


def test_9_mismatched_session_anchor(ctx):
    cross = ctx.message("pub", "s_priv", "pub row in a priv session")
    other = ctx.message("pub", "s_pub2", "pub row in a pub2 session")
    assert ctx.ctx(bot="pub", id=cross) == NOT_FOUND
    assert ctx.ctx(bot="pub", id=other) == NOT_FOUND


def test_10_partial_author_anchor(ctx):
    for author in (("", "pub"), ("bot", ""), ("robot", "pub")):
        mid = ctx.message("pub", "s_pub", "partial", role="assistant", author=author)
        assert ctx.ctx(bot="pub", id=mid) == NOT_FOUND


def test_null_session_is_not_found(ctx):
    mid = ctx.message("pub", None, "legacy summary", role="assistant", author=("bot", "pub"))
    assert ctx.ctx(bot="pub", id=mid) == NOT_FOUND


def test_3_policy_flip_between_calls(ctx):
    delivery = ctx.message("pub", "s_pub", "from pub2", author=("bot", "pub2"))
    assert ctx.ctx(bot="pub", id=delivery)["anchor"]["speaker"] == "delivery"
    ctx.flag("pub2", False)
    assert ctx.ctx(bot="pub", id=delivery) == NOT_FOUND


# -- budget -----------------------------------------------------------------

def test_long_anchor_takes_half_the_budget_and_continues(ctx):
    body = "".join(f"{i:05d} \"q\"\n" for i in range(3000))  # escapes widen the wire form
    mid = ctx.message("pub", "s_pub", body)
    read, offset = "", 0
    while True:
        raw = ctx.ctx(bot="pub", id=mid, max_chars=2000, offset=offset, raw=True)
        resp = json.loads(raw)
        assert len(raw) <= 2000
        assert len(json.dumps(resp["anchor"]["content"], ensure_ascii=False)) <= 1000
        read += resp["anchor"]["content"]
        if resp["anchor"]["next_offset"] is None:
            assert resp["anchor"]["truncated"] is False
            break
        assert resp["anchor"]["truncated"] is True and resp["anchor"]["next_offset"] > offset
        offset = resp["anchor"]["next_offset"]
    assert read == body
    assert ctx.ctx(bot="pub", id=mid, offset=len(body) + 1) == INVALID


def test_neighbours_shrink_then_farthest_drop_and_anchor_stays(ctx):
    m = [ctx.message("pub", "s_pub", f"note {i} " + "word " * 200 + f"needle{i}", age=1000 - i * 10)
         for i in range(21)]
    anchor = m[10]
    roomy = ctx.ctx(bot="pub", id=anchor, before=2, after=2, max_chars=8000)
    assert all(len(n["snippet"]) <= 202 and n["truncated"] for n in roomy["neighbors"]["before"])
    raw = ctx.ctx(bot="pub", id=anchor, before=10, after=10, max_chars=1000, raw=True)
    tight = json.loads(raw)
    assert len(raw) <= 1000 and tight["anchor"]["id"] == anchor
    before, after = ids(tight, "before"), ids(tight, "after")
    assert all(len(n["snippet"]) <= 82 for n in tight["neighbors"]["before"] + tight["neighbors"]["after"])
    # The nearest survive: a contiguous window ending/starting at the anchor.
    assert before == m[10 - len(before):10] and after == m[11:11 + len(after)]
    assert len(before) + len(after) < 20
    # Dropped neighbours stay reachable through the cursors.
    assert tight["more_before"] and tight["more_after"]
    for max_chars in (1000, 1500, 2500, 4000, 8000):
        assert len(ctx.ctx(bot="pub", id=anchor, before=10, after=10, max_chars=max_chars,
                           raw=True)) <= max_chars


def test_focus_centres_snippets(ctx):
    before = ctx.message("pub", "s_pub", "x " * 300 + "zebra crossing" + " y" * 300, age=900)
    anchor = ctx.message("pub", "s_pub", "anchor", age=800)
    focused = ctx.ctx(bot="pub", id=anchor, focus=["zebra"])
    assert ids(focused, "before") == [before]
    assert "zebra" in focused["neighbors"]["before"][0]["snippet"]
    head = ctx.ctx(bot="pub", id=anchor)
    assert "zebra" not in head["neighbors"]["before"][0]["snippet"]


# -- cursors ----------------------------------------------------------------

def test_cursors_page_one_side_until_exhausted(ctx):
    m = thread(ctx, n=9)
    first = ctx.ctx(bot="pub", id=m[6], before=2, after=1)
    assert ids(first, "before") == m[4:6] and ids(first, "after") == [m[7]]
    page = ctx.ctx(cursor=first["more_before"], before=3)
    assert page["anchor"] == {"bot": "pub", "id": m[6], "session": "s_pub"}
    assert ids(page, "before") == m[1:4] and page["more_before"]
    last = ctx.ctx(cursor=page["more_before"], before=3)
    assert ids(last, "before") == [m[0]] and last["more_before"] is None
    tail = ctx.ctx(cursor=first["more_after"], after=5)
    assert ids(tail, "after") == [m[8]] and tail["more_after"] is None


def test_cursor_is_owner_bound_and_reauthorized(ctx):
    m = thread(ctx, n=5)
    first = ctx.ctx(bot="pub", id=m[2], before=1)
    assert ctx.ctx("pub2", cursor=first["more_before"]) == EXPIRED
    assert ctx.ctx("pub", "eve", cursor=first["more_before"]) == EXPIRED
    assert ctx.ctx(cursor="forged") == EXPIRED
    delivery = ctx.message("pub", "s_pub", "from pub2", author=("bot", "pub2"), age=5000)
    newer = thread(ctx, n=3, prefix="later")
    resp = ctx.ctx(bot="pub", id=delivery, after=1)
    assert resp["more_after"]
    ctx.flag("pub2", False)
    assert ctx.ctx(cursor=resp["more_after"]) == NOT_FOUND
    assert newer


# -- §5.3 card references ---------------------------------------------------

def find(w, caller="pub", **request) -> dict:
    return json.loads(w.finder.find(w.access(caller), FindRequest(**request)))


def test_card_ref_resolves_the_served_card_with_find_terms(ctx):
    m = thread(ctx, n=5)
    hit = ctx.message("pub", "s_pub", "x " * 200 + "TASK-901 lives here" + " y" * 200, age=1)
    served = find(ctx, query="TASK-901")
    assert served["best"][0]["ref"] == "c0.0" and served["best"][0]["id"] == hit
    resp = ctx.ctx(handle=served["handle"], card_ref="c0.0")
    assert resp["anchor"]["id"] == hit and ids(resp, "before") == m[2:5]
    # The served find's terms focus the neighbours by default.
    ctx.message("pub", "s_pub", "w " * 300 + "TASK-901 again" + " z" * 300, age=0.5)
    again = ctx.ctx(handle=served["handle"], card_ref="c0.0")
    assert "TASK-901" in again["neighbors"]["after"][0]["snippet"]


def test_2_insert_between_find_and_context_cannot_move_c0_0(ctx):
    ctx.session("s_a", "pub")
    first = ctx.message("pub", "s_a", "TASK-902 original", age=100)
    served = find(ctx, query="TASK-902")
    assert served["best"][0]["id"] == first
    ctx.session("s_b", "pub")
    ctx.message("pub", "s_b", "TASK-902 TASK-902 newer and better", age=1)
    assert find(ctx, query="TASK-902")["best"][0]["id"] != first  # a fresh find would move
    assert ctx.ctx(handle=served["handle"], card_ref="c0.0")["anchor"]["id"] == first


def test_3_policy_flip_makes_the_card_not_found(ctx):
    ctx.message("pub", "s_pub", "TASK-903 from pub2", author=("bot", "pub2"))
    served = find(ctx, query="TASK-903")
    ctx.flag("pub2", False)
    assert ctx.ctx(handle=served["handle"], card_ref="c0.0") == NOT_FOUND


def test_3_policy_flip_of_the_partition_makes_the_card_not_found(ctx):
    ctx.message("pub2", "s_pub2", "TASK-906 pub2 note")
    served = find(ctx, query="TASK-906")
    ctx.flag("pub2", False)
    assert ctx.ctx(handle=served["handle"], card_ref="c0.0") == NOT_FOUND


def test_4_other_owner_forged_or_expired_handle(ctx):
    ctx.message("pub", "s_pub", "TASK-904 note")
    served = find(ctx, query="TASK-904")
    assert ctx.ctx("pub2", handle=served["handle"], card_ref="c0.0") == EXPIRED
    assert ctx.ctx("pub", "eve", handle=served["handle"], card_ref="c0.0") == EXPIRED
    assert ctx.ctx(handle="forged", card_ref="c0.0") == EXPIRED
    now = [0.0]
    ctx.handles = ctx.finder.handles = HandleStore(ttl=60, clock=lambda: now[0])
    ctx.context.finds = ctx.handles
    fresh = find(ctx, query="TASK-904")
    assert ctx.ctx(handle=fresh["handle"], card_ref="c0.0")["status"] == "ok"
    now[0] = 61
    assert ctx.ctx(handle=fresh["handle"], card_ref="c0.0") == EXPIRED


def test_5_unserved_card_ref_is_not_found(ctx):
    ctx.message("pub", "s_pub", "TASK-905 note")
    served = find(ctx, query="TASK-905")
    assert ctx.ctx(handle=served["handle"], card_ref="c0.1") == NOT_FOUND
    assert ctx.ctx(handle=served["handle"], card_ref="c1.0") == NOT_FOUND
    assert ctx.ctx(handle=served["handle"], card_ref=None) == INVALID
