"""The UI cursor's Lua scripts, executed for real (TASK-1584).

``test_ui_stream_cursor.py`` models CLAIM/ADVANCE/SEED in Python, so an edit to
the Lua text itself can't fail it. Here the exact script constants run in
fakeredis's embedded Lua engine (lupa): no server, no network, every gate lane.

Contract pinned per script:

- CLAIM takes ownership, keeps stored cursors, and (re)sets the hash TTL.
- ADVANCE writes only while the token owns the hash, refreshes the TTL, and
  stores the given ids verbatim. A superseded or expired owner gets 0 and
  changes nothing: no field, no TTL, no resurrected hash.
- SEED only fills absent cursors, so it never moves a stored cursor in either
  direction, returns the authoritative fields, and refuses a superseded or
  expired owner the same way.

The Lua has no numeric comparison of stream ids: the reader only advances in
id order and a client resume rewinds just its own connection, which
``test_ui_stream_cursor.py`` covers.
"""

from __future__ import annotations

import asyncio

import fakeredis
import pytest

from agent_bridge.ui_stream import (
    ADVANCE_SCRIPT,
    CLAIM_SCRIPT,
    SEED_SCRIPT,
    UI_CURSOR_PREFIX,
    UiStreamReader,
)

KEY = f"{UI_CURSOR_PREFIX}probe-window"
TTL = 1800
STREAM_A = "events:probe-bot:probe-user"
STREAM_B = "events:probe-bot:other-user"


@pytest.fixture
def r():
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()


def _fields(raw) -> dict[str, str]:
    return dict(zip(raw[::2], raw[1::2]))


def _claim(r, token: str, ttl: int = TTL) -> dict[str, str]:
    return _fields(r.eval(CLAIM_SCRIPT, 1, KEY, token, ttl))


def _advance(r, token: str, *pairs: str, ttl: int = TTL):
    return r.eval(ADVANCE_SCRIPT, 1, KEY, token, ttl, *pairs)


def _seed(r, token: str, *pairs: str, ttl: int = TTL):
    return r.eval(SEED_SCRIPT, 1, KEY, token, ttl, *pairs)


def _ttl_is_full(r) -> bool:
    # Integer seconds may tick once between the script and this read.
    return TTL - 2 <= r.ttl(KEY) <= TTL


# CLAIM -----------------------------------------------------------------------


def test_claim_on_fresh_key_sets_owner_and_ttl(r):
    assert _claim(r, "tok-a") == {"owner": "tok-a"}
    assert _ttl_is_full(r)


def test_claim_takes_over_keeps_cursors_and_resets_ttl(r):
    r.hset(KEY, mapping={"owner": "tok-a", STREAM_A: "5-0"})
    r.expire(KEY, 30)

    assert _claim(r, "tok-b") == {"owner": "tok-b", STREAM_A: "5-0"}
    assert _ttl_is_full(r)


# ADVANCE ---------------------------------------------------------------------


def test_advance_by_owner_writes_ids_verbatim_and_refreshes_ttl(r):
    _claim(r, "tok-a")
    r.expire(KEY, 30)

    assert _advance(r, "tok-a", STREAM_A, "1700000000000-7", STREAM_B, "9-0") == 1
    assert r.hgetall(KEY) == {
        "owner": "tok-a",
        STREAM_A: "1700000000000-7",
        STREAM_B: "9-0",
    }
    assert _ttl_is_full(r)


def test_advance_without_pairs_is_owner_check_and_ttl_refresh(r):
    _claim(r, "tok-a")
    r.expire(KEY, 30)

    assert _advance(r, "tok-a") == 1
    assert r.hgetall(KEY) == {"owner": "tok-a"}
    assert _ttl_is_full(r)


def test_advance_by_superseded_owner_changes_nothing(r):
    _claim(r, "tok-a")
    _advance(r, "tok-a", STREAM_A, "5-0")
    _claim(r, "tok-b")
    r.expire(KEY, 30)

    assert _advance(r, "tok-a", STREAM_A, "9-0") == 0
    assert r.hgetall(KEY) == {"owner": "tok-b", STREAM_A: "5-0"}
    assert r.ttl(KEY) <= 30


def test_advance_after_expiry_does_not_resurrect_the_hash(r):
    assert _advance(r, "tok-a", STREAM_A, "9-0") == 0
    assert r.exists(KEY) == 0


# SEED ------------------------------------------------------------------------


def test_seed_fills_only_absent_cursors_and_returns_authoritative_fields(r):
    _claim(r, "tok-a")
    _advance(r, "tok-a", STREAM_A, "5-0")
    r.expire(KEY, 30)

    raw = _seed(r, "tok-a", STREAM_A, "9-0", STREAM_B, "3-0")

    expected = {"owner": "tok-a", STREAM_A: "5-0", STREAM_B: "3-0"}
    assert _fields(raw) == expected
    assert r.hgetall(KEY) == expected
    assert _ttl_is_full(r)


def test_seed_never_moves_a_stored_cursor_backwards(r):
    _claim(r, "tok-a")
    _advance(r, "tok-a", STREAM_A, "5-0")

    assert _fields(_seed(r, "tok-a", STREAM_A, "0-0"))[STREAM_A] == "5-0"


def test_seed_by_superseded_owner_is_refused_without_writes(r):
    _claim(r, "tok-a")
    _claim(r, "tok-b")
    r.expire(KEY, 30)

    assert _seed(r, "tok-a", STREAM_A, "3-0") == 0
    assert r.hgetall(KEY) == {"owner": "tok-b"}
    assert r.ttl(KEY) <= 30


def test_seed_after_expiry_does_not_resurrect_the_hash(r):
    assert _seed(r, "tok-a", STREAM_A, "3-0") == 0
    assert r.exists(KEY) == 0


# The reader's own argument order against the real scripts --------------------


def test_reader_round_trip_and_supersession_through_real_lua():
    async def scenario():
        redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        try:
            await redis.xadd(STREAM_A, {"payload": "{}"})
            ghost = UiStreamReader(redis, "probe-window", [STREAM_A], token="tok-a")
            await ghost.claim()
            tail = (await redis.xrevrange(STREAM_A, count=1))[0][0]
            assert await redis.hgetall(KEY) == {"owner": "tok-a", STREAM_A: tail}
            assert await ghost._advance({STREAM_A: "99-0"})

            live = UiStreamReader(redis, "probe-window", [STREAM_A], token="tok-b")
            await live.claim()
            assert live._cursors == {STREAM_A: "99-0"}

            assert not await ghost._advance({STREAM_A: "100-0"})
            assert ghost.superseded
            assert await redis.hgetall(KEY) == {"owner": "tok-b", STREAM_A: "99-0"}
        finally:
            await redis.aclose()

    asyncio.run(scenario())
