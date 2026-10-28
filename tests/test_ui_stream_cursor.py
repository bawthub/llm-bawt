"""UI fan-out reads by owner-fenced cursor, not consumer groups (TASK-1582).

A window reconnecting with the same consumer_id while its old upstream
connection was still alive used to share the ``ui:{consumer_id}`` consumer
group with that ghost, so the live connection received about half the events.
These tests pin the replacement contract:

- the newest connection owns the cursor; a superseded ghost yields nothing more
  and can't move the cursor, and the new connection gets every event;
- events published while no connection is open are delivered on resume;
- a connection creates no consumer group and leaves only a TTL'd cursor hash;
- a retired ``ui:*`` group migrates into the cursor and is destroyed only after
  that; a failed legacy lookup fails closed and keeps the group.
"""

from __future__ import annotations

import asyncio
import json

from redis.exceptions import ResponseError

from agent_bridge import ui_stream
from agent_bridge.publisher import GLOBAL_CONFIG_STREAM
from agent_bridge.subscriber import RedisSubscriber
from agent_bridge.ui_stream import UiStreamReader, unified_stream_keys
from tests import virtual_clock

STREAM = "events:probe-bot:probe-user"


class FakeRedis:
    """In-memory model of the Redis calls the UI reader makes."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.expires_at: dict[str, float] = {}
        self.consumers: dict[tuple[str, str], list[dict]] = {}
        self.pending_ids: dict[tuple[str, str], list[str]] = {}
        self.groups: dict[str, list[dict]] = {}
        self.destroyed: list[tuple[str, str]] = []
        self.created: list[tuple[str, str]] = []
        self.refused_tokens: list[str] = []
        self._seq = 0
        self._changed: asyncio.Condition | None = None

    def _cond(self) -> asyncio.Condition:
        if self._changed is None:
            self._changed = asyncio.Condition()
        return self._changed

    async def xadd(self, key: str, fields: dict, id: str | None = None) -> str:
        self._seq += 1
        msg_id = id or f"{1_700_000_000_000 + self._seq}-0"
        self.streams.setdefault(key, []).append((msg_id, dict(fields)))
        async with self._cond():
            self._cond().notify_all()
        return msg_id

    async def publish(self, n: int, key: str = STREAM) -> list[str]:
        ids = []
        for _ in range(n):
            seq = self._seq + 1
            ids.append(await self.xadd(key, {"payload": json.dumps({"seq": seq})}))
        return ids

    async def xrevrange(self, key: str, count: int | None = None):
        entries = list(reversed(self.streams.get(key, [])))
        return entries[:count] if count else entries

    def _after(self, streams: dict[str, str], count: int):
        out = []
        for key, cursor in streams.items():
            fresh = [
                (mid, f) for mid, f in self.streams.get(key, [])
                if ui_stream.stream_id_key(mid) > ui_stream.stream_id_key(cursor)
            ][:count]
            if fresh:
                out.append([key, fresh])
        return out

    async def xread(self, streams: dict[str, str], count: int, block: int):
        out = self._after(streams, count)
        if out or not block:
            return out
        async with self._cond():
            try:
                await asyncio.wait_for(
                    self._cond().wait_for(lambda: bool(self._after(streams, count))),
                    timeout=block / 1000,
                )
            except asyncio.TimeoutError:
                return []
        return self._after(streams, count)

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        assert numkeys == 1
        key, token, ttl, *pairs = keys_and_args
        now = asyncio.get_running_loop().time()
        if key in self.expires_at and self.expires_at[key] <= now:
            self.hashes.pop(key, None)
            self.expires_at.pop(key)
        h = self.hashes.setdefault(key, {})

        def touch():
            self.ttls[key] = int(ttl)
            self.expires_at[key] = now + int(ttl)

        if script == ui_stream.CLAIM_SCRIPT:
            h["owner"] = token
            touch()
            return [x for kv in h.items() for x in kv]
        if h.get("owner") != token:
            self.refused_tokens.append(token)
            if not h:
                self.hashes.pop(key)
            return 0
        if script == ui_stream.ADVANCE_SCRIPT:
            for i in range(0, len(pairs), 2):
                h[pairs[i]] = pairs[i + 1]
            touch()
            return 1
        if script == ui_stream.SEED_SCRIPT:
            for i in range(0, len(pairs), 2):
                h.setdefault(pairs[i], pairs[i + 1])
            touch()
            return [x for kv in h.items() for x in kv]
        raise AssertionError("unexpected script")

    def legacy_group(self, key: str, name: str, *, last_delivered: str, pending: list[str], idle_ms: int):
        self.groups.setdefault(key, []).append(
            {"name": name, "pending": len(pending), "consumers": 1, "last-delivered-id": last_delivered}
        )
        self.consumers[(key, name)] = [{"name": "reader", "idle": idle_ms}]
        self.pending_ids[(key, name)] = list(pending)

    async def xpending(self, key: str, name: str):
        ids = self.pending_ids.get((key, name), [])
        return {"pending": len(ids), "min": ids[0] if ids else None, "max": ids[-1] if ids else None}

    async def xinfo_consumers(self, key: str, name: str):
        return list(self.consumers.get((key, name), []))

    async def xinfo_groups(self, key: str):
        if key not in self.streams and key not in self.groups:
            raise ResponseError("no such key")
        return list(self.groups.get(key, []))

    async def xgroup_create(self, key: str, name: str, *args, **kwargs):
        self.created.append((key, name))
        self.groups.setdefault(key, []).append(
            {"name": name, "pending": 0, "consumers": 0, "last-delivered-id": "0-0"}
        )
        return True

    async def xgroup_destroy(self, key: str, name: str):
        self.groups[key] = [g for g in self.groups.get(key, []) if g["name"] != name]
        self.destroyed.append((key, name))
        return 1

    async def scan_iter(self, match: str, count: int = 100):
        prefix = match.rstrip("*")
        for key in list(self.streams) + list(self.groups):
            if key.startswith(prefix):
                yield key


def _seqs(events) -> list[int]:
    return [e["seq"] for e in events]


async def _take(gen, n: int) -> list[dict]:
    out = []
    async for ev in gen:
        if ev is not None:
            out.append(ev)
            if len(out) == n:
                break
    return out


def test_superseded_ghost_gets_nothing_and_new_connection_gets_every_event():
    async def scenario():
        redis = FakeRedis()
        ghost = UiStreamReader(redis, "window-1", [STREAM], token="ghost")
        await ghost.claim()
        ghost_seen: list[dict] = []

        async def run_ghost():
            async for ev in ghost.events(timeout_s=60):
                if ev is not None:
                    ghost_seen.append(ev)

        ghost_task = asyncio.create_task(run_ghost())
        await asyncio.sleep(0)  # ghost is now blocked inside XREAD

        live = UiStreamReader(redis, "window-1", [STREAM], token="live")
        await live.claim()
        ids = await redis.publish(100)

        live_gen = live.events(timeout_s=60)
        got = await _take(live_gen, 100)
        # The route resumes after each frame is sent; that persists #100.
        assert await live_gen.__anext__() is None
        await asyncio.wait_for(ghost_task, timeout=30)
        return redis, ghost, ghost_seen, got, ids

    redis, ghost, ghost_seen, got, ids = virtual_clock.run(scenario())
    assert _seqs(got) == list(range(1, 101))
    assert ghost_seen == []
    assert ghost.superseded
    assert "ghost" in redis.refused_tokens
    cursor = redis.hashes["ui:cursor:window-1"]
    assert cursor["owner"] == "live"
    assert cursor[STREAM] == ids[-1]


def test_ghost_parked_mid_batch_cannot_drop_events_for_the_new_connection():
    async def scenario():
        redis = FakeRedis()
        ghost = UiStreamReader(redis, "window-1", [STREAM], token="ghost")
        await ghost.claim()
        await redis.publish(10)
        ghost_gen = ghost.events(timeout_s=60)
        first = await ghost_gen.__anext__()  # handed off, then the socket freezes

        live = UiStreamReader(redis, "window-1", [STREAM], token="live")
        await live.claim()
        await redis.publish(10)

        # The frozen socket drains: the ghost's write is refused and it stops.
        rest = [ev async for ev in ghost_gen if ev is not None]
        got = await _take(live.events(timeout_s=60), 20)
        return first, rest, got, redis

    first, rest, got, redis = virtual_clock.run(scenario())
    assert first["seq"] == 1
    assert rest == []
    # At-least-once: the event whose hand-off was never confirmed is resent.
    assert _seqs(got) == list(range(1, 21))
    assert redis.hashes["ui:cursor:window-1"]["owner"] == "live"


def test_resume_after_gap_is_lossless_and_marks_catchup_as_replayed():
    async def scenario():
        redis = FakeRedis()
        first = UiStreamReader(redis, "window-1", [STREAM])
        await first.claim()
        await redis.publish(10)
        gen = first.events(timeout_s=60)
        before = await _take(gen, 10)
        assert await gen.__anext__() is None  # #10's frame was sent; idle tick
        await gen.aclose()  # clean disconnect

        await redis.publish(20)  # nobody connected

        second = UiStreamReader(redis, "window-1", [STREAM])
        await second.claim()
        gen2 = second.events(timeout_s=60)
        caught_up = await _take(gen2, 20)
        await redis.publish(5)
        live = await _take(gen2, 5)
        return before, caught_up, live

    before, caught_up, live = virtual_clock.run(scenario())
    assert _seqs(before) == list(range(1, 11))
    assert _seqs(caught_up) == list(range(11, 31))
    assert all(ev["_replayed"] for ev in caught_up)
    assert _seqs(live) == list(range(31, 36))
    assert not any(ev["_replayed"] for ev in live)


def test_reconnect_before_first_event_still_covers_the_gap():
    async def scenario():
        redis = FakeRedis()
        await redis.publish(3)  # history before the window ever connected
        first = UiStreamReader(redis, "window-1", [STREAM])
        await first.claim()  # starts at the tail, delivers nothing yet
        await redis.publish(4)
        second = UiStreamReader(redis, "window-1", [STREAM])
        await second.claim()
        return await _take(second.events(timeout_s=60), 4)

    assert _seqs(virtual_clock.run(scenario())) == [4, 5, 6, 7]


def test_events_from_several_streams_interleave_in_id_order():
    other = "events:other-bot:probe-user"

    async def scenario():
        redis = FakeRedis()
        reader = UiStreamReader(redis, "window-1", [STREAM, other])
        await reader.claim()
        for key in (STREAM, other, STREAM, other):
            await redis.publish(1, key)
        return await _take(reader.events(timeout_s=60), 4)

    assert _seqs(virtual_clock.run(scenario())) == [1, 2, 3, 4]


def test_idle_superseded_connection_exits_on_owner_check():
    async def scenario():
        redis = FakeRedis()
        ghost = UiStreamReader(redis, "window-1", [STREAM], owner_check_s=5.0)
        await ghost.claim()
        await UiStreamReader(redis, "window-1", [STREAM]).claim()
        ticks = [ev async for ev in ghost.events(timeout_s=600)]
        return ghost, ticks

    ghost, ticks = virtual_clock.run(scenario())
    assert ghost.superseded
    assert ticks and all(ev is None for ev in ticks)


def test_clean_close_leaves_no_consumer_group_only_a_ttl_cursor():
    async def scenario():
        redis = FakeRedis()
        sub = object.__new__(RedisSubscriber)
        sub._redis = redis
        reader = sub.ui_stream(["probe-bot"], "probe-user", "window-1", include_global=True)
        await reader.claim()
        await redis.publish(3)
        gen = reader.events(timeout_s=60)
        got = await _take(gen, 3)
        await gen.aclose()
        return redis, got

    redis, got = virtual_clock.run(scenario())
    assert _seqs(got) == [1, 2, 3]
    # The legacy ui:* set is closed: a fresh window creates no group at all.
    assert redis.created == [] and redis.destroyed == []
    assert redis.groups == {}
    assert set(redis.hashes) == {"ui:cursor:window-1"}
    assert redis.ttls["ui:cursor:window-1"] == ui_stream.UI_CURSOR_TTL_S
    assert set(redis.hashes["ui:cursor:window-1"]) == {"owner", STREAM, GLOBAL_CONFIG_STREAM}


def test_stream_keys_scope_each_window_to_its_user_plus_global_config():
    keys = unified_stream_keys(["al"], "user-a", include_global=True)
    assert keys == {"events:al:user-a": "al", GLOBAL_CONFIG_STREAM: "system"}
    assert GLOBAL_CONFIG_STREAM not in unified_stream_keys(["al"], "user-a")


def test_route_claims_before_hello_and_streams_from_the_cursor_reader(monkeypatch):
    from llm_bawt.service.routes import event_stream

    redis = FakeRedis()

    class Sub:
        def __init__(self, _url):
            self._redis = redis
            self.closed = False

        async def connect(self):
            return None

        ui_stream = RedisSubscriber.ui_stream

        async def close(self):
            self.closed = True

    monkeypatch.setattr(event_stream, "RedisSubscriber", Sub)

    async def scenario():
        gen = event_stream._unified_event_stream("redis://unused", ["probe-bot"], "probe-user", "window-1")
        hello = await gen.__anext__()
        # Claimed before hello: an event published now must be delivered.
        assert redis.hashes["ui:cursor:window-1"]["owner"]
        await redis.publish(1)
        frame = await gen.__anext__()
        await gen.aclose()
        return hello, frame

    hello, frame = virtual_clock.run(scenario())
    assert hello.startswith("event: hello")
    assert frame.startswith("event: event")
    data = json.loads(frame.split("data: ", 1)[1])
    assert data["seq"] == 1 and data["replayed"] is False


def test_legacy_group_migrates_into_the_cursor_on_reconnect_then_retires():
    async def scenario():
        redis = FakeRedis()
        ids = await redis.publish(10)
        # Before the restart the window had read 1-8 and acked 1-6.
        redis.legacy_group(STREAM, "ui:window-1", last_delivered=ids[7], pending=ids[6:8], idle_ms=600_000)
        sub = object.__new__(RedisSubscriber)
        sub._redis = redis
        reader = sub.ui_stream(["probe-bot"], "probe-user", "window-1")
        await reader.claim()
        got = await _take(reader.events(timeout_s=60), 4)
        return got, redis

    got, redis = virtual_clock.run(scenario())
    assert _seqs(got) == [7, 8, 9, 10]  # unacked + never-read, nothing skipped
    assert redis.groups[STREAM] == []  # migrated group retired by its window
    assert ("events:probe-bot:probe-user", "ui:window-1") in redis.destroyed


def test_migration_without_pending_resumes_after_last_delivered():
    async def scenario():
        redis = FakeRedis()
        ids = await redis.publish(5)
        redis.legacy_group(STREAM, "ui:window-1", last_delivered=ids[2], pending=[], idle_ms=1000)
        reader = UiStreamReader(redis, "window-1", [STREAM])
        await reader.claim()
        return await _take(reader.events(timeout_s=60), 2)

    assert _seqs(virtual_clock.run(scenario())) == [4, 5]


def test_ids_order_numerically_so_a_checkpoint_never_skips_a_lower_sequence():
    async def scenario():
        redis = FakeRedis()
        reader = UiStreamReader(redis, "window-1", [STREAM])
        await reader.claim()
        other = "events:other-bot:probe-user"
        await redis.xadd(other, {"payload": json.dumps({"seq": 10})}, id="1800000000000-10")
        await redis.xadd(STREAM, {"payload": json.dumps({"seq": 9})}, id="1800000000000-9")
        await redis.xadd(other, {"payload": json.dumps({"seq": 11})}, id="1800000000000-11")
        reader2 = UiStreamReader(redis, "window-2", [STREAM, other])
        reader2._cursors = {STREAM: "0-0", other: "0-0"}
        reader2._catchup_until = {}
        await redis.eval(ui_stream.CLAIM_SCRIPT, 1, reader2.cursor_key, reader2.token, 60)
        return await _take(reader2.events(timeout_s=60), 3)

    assert _seqs(virtual_clock.run(scenario())) == [9, 10, 11]
    assert ui_stream.stream_id_before("5-0") == "4-18446744073709551615"
    assert ui_stream.stream_id_before("5-3") == "5-2"


def test_quiet_live_connection_keeps_its_cursor_past_the_ttl():
    async def scenario():
        redis = FakeRedis()
        reader = UiStreamReader(redis, "window-1", [STREAM])
        await reader.claim()
        gen = reader.events(timeout_s=7200)
        loop = asyncio.get_running_loop()
        end = loop.time() + 2 * ui_stream.UI_CURSOR_TTL_S
        while loop.time() < end:
            assert await gen.__anext__() is None
        await redis.publish(1)
        got = await _take(gen, 1)
        return reader, got, redis

    reader, got, redis = virtual_clock.run(scenario())
    assert not reader.superseded
    assert _seqs(got) == [1]
    assert redis.hashes["ui:cursor:window-1"]["owner"] == reader.token


def test_window_away_past_the_ttl_restarts_at_the_tail():
    async def scenario():
        redis = FakeRedis()
        first = UiStreamReader(redis, "window-1", [STREAM])
        await first.claim()
        await asyncio.sleep(ui_stream.UI_CURSOR_TTL_S + 1)
        await redis.publish(3)  # missed; the client's long-gap reconcile covers it
        second = UiStreamReader(redis, "window-1", [STREAM])
        await second.claim()
        await redis.publish(1)
        return await _take(second.events(timeout_s=60), 1)

    assert _seqs(virtual_clock.run(scenario())) == [4]


def test_review_legacy_lookup_error_cannot_seed_past_pending():
    """Loopy's DESIGN r2 probe (receipt 20261027T173747Z-loopy-b9599a), kept as
    the regression: a transient lookup failure must not tail-seed past the
    group's pending and unread entries, and a retry must still migrate them."""
    async def scenario():
        redis = FakeRedis()
        ids = await redis.publish(5)
        redis.legacy_group(STREAM, "ui:window-1", last_delivered=ids[2], pending=ids[1:3], idle_ms=1000)
        original = redis.xinfo_groups

        async def failed_lookup(_key):
            raise ConnectionError("synthetic Redis read disconnect")

        redis.xinfo_groups = failed_lookup
        reader = UiStreamReader(redis, "window-1", [STREAM])
        failed = False
        try:
            await reader.claim()
        except ConnectionError:
            failed = True
        finally:
            redis.xinfo_groups = original
        assert failed, "a failed legacy lookup must fail the claim closed"
        assert STREAM not in redis.hashes[reader.cursor_key]
        assert [g["name"] for g in redis.groups[STREAM]] == ["ui:window-1"]
        assert redis.destroyed == []

        retry = UiStreamReader(redis, "window-1", [STREAM])
        await retry.claim()
        return await _take(retry.events(timeout_s=60), 4), redis

    got, redis = virtual_clock.run(scenario())
    assert _seqs(got) == [2, 3, 4, 5]
    assert redis.groups[STREAM] == []


def test_pending_lookup_failure_also_fails_closed_but_nogroup_is_absence():
    async def scenario():
        redis = FakeRedis()
        ids = await redis.publish(5)
        redis.legacy_group(STREAM, "ui:window-1", last_delivered=ids[2], pending=ids[1:3], idle_ms=1000)

        async def broken(_key, _name):
            raise ResponseError("LOADING Redis is loading the dataset in memory")

        redis.xpending = broken
        reader = UiStreamReader(redis, "window-1", [STREAM])
        try:
            await reader.claim()
            raise AssertionError("claim should fail closed")
        except ResponseError:
            pass
        assert STREAM not in redis.hashes[reader.cursor_key]
        assert redis.destroyed == []

        async def nogroup(_key, _name):
            raise ResponseError("NOGROUP No such key or consumer group")

        redis.xpending = nogroup
        absent = UiStreamReader(redis, "window-1", [STREAM])
        await absent.claim()  # group proven gone: start at the tail
        await redis.publish(1)
        return await _take(absent.events(timeout_s=60), 1)

    assert _seqs(virtual_clock.run(scenario())) == [6]


def test_route_fails_closed_without_hello_when_the_claim_fails(monkeypatch):
    from llm_bawt.service.routes import event_stream

    redis = FakeRedis()
    closed = []

    async def failed_lookup(_key):
        raise ConnectionError("synthetic")

    class Sub:
        def __init__(self, _url):
            self._redis = redis

        async def connect(self):
            return None

        ui_stream = RedisSubscriber.ui_stream

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(event_stream, "RedisSubscriber", Sub)

    async def scenario():
        ids = await redis.publish(2)
        redis.legacy_group(STREAM, "ui:window-1", last_delivered=ids[1], pending=[], idle_ms=1000)
        redis.xinfo_groups = failed_lookup
        gen = event_stream._unified_event_stream("redis://unused", ["probe-bot"], "probe-user", "window-1")
        return [frame async for frame in gen]

    frames = virtual_clock.run(scenario())
    assert len(frames) == 1 and frames[0].startswith("event: error")
    assert closed == [True]
    assert [g["name"] for g in redis.groups[STREAM]] == ["ui:window-1"]


def test_no_source_creates_a_group_on_a_ui_event_stream():
    """The legacy ``ui:*`` set is closed and finite (TASK-1582).

    Groups are created directly only on the bridge command streams and the
    history stream. Event streams get groups solely through
    ``RedisSubscriber.ensure_consumer_group``, which refuses ``ui:*`` names. A
    new ``xgroup_create`` anywhere in ``src`` has to be added here on purpose,
    so nothing quietly brings per-window groups back.
    """
    import ast
    from pathlib import Path

    allowed_streams = {"COMMANDS_STREAM", "HISTORY_STREAM"}
    guarded_helpers = {("subscriber.py", "ensure_consumer_group")}
    unexpected = []
    seen = 0
    for path in sorted(Path(__file__).resolve().parents[1].joinpath("src").rglob("*.py")):
        source = path.read_text()
        if "xgroup_create" not in source:
            continue  # parse only candidates; keeps this within the unit time budget
        tree = ast.parse(source, filename=str(path))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "xgroup_create"):
                    continue
                seen += 1
                stream = node.args[0] if node.args else None
                if isinstance(stream, ast.Name) and stream.id in allowed_streams:
                    continue
                if (path.name, fn.name) in guarded_helpers:
                    continue
                unexpected.append((path.name, fn.name, ast.unparse(node)))
    assert seen, "scan found no xgroup_create calls; the ratchet is not looking at src"
    assert unexpected == []


def test_event_stream_groups_refuse_retired_ui_names():
    async def scenario():
        redis = FakeRedis()
        sub = object.__new__(RedisSubscriber)
        sub._redis = redis
        try:
            await sub.ensure_consumer_group(STREAM, "ui:window-1")
        except ValueError:
            refused = True
        else:
            refused = False
        await sub.ensure_consumer_group(STREAM, "persist:tools")
        return refused, redis.created

    refused, created = virtual_clock.run(scenario())
    assert refused
    assert created == [(STREAM, "persist:tools")]


def test_tool_event_drain_still_creates_its_persist_group_and_delivers():
    """Regression: the persist:tools drain kept its group helper (Loopy, TASK-1582)."""

    async def scenario():
        redis = FakeRedis()
        sub = object.__new__(RedisSubscriber)
        sub._redis = redis
        event = {"_type": "tool_event", "tool_name": "probe"}
        acked = []
        reads = []
        listed = []

        async def list_streams():
            # Bounded: a broken loop (e.g. a missing helper raising each pass)
            # fails the asserts below instead of hanging.
            listed.append(True)
            if len(listed) > 3:
                raise asyncio.CancelledError
            return [STREAM]

        async def xreadgroup(group, consumer, streams, count, block):
            reads.append((group, consumer, dict(streams)))
            if len(reads) == 1:
                return [[STREAM, [("1-0", {"payload": json.dumps(event)})]]]
            raise asyncio.CancelledError

        async def xack(key, group, msg_id):
            acked.append((key, group, msg_id))

        sub.list_unified_streams = list_streams
        redis.xreadgroup = xreadgroup
        redis.xack = xack
        got = []
        try:
            await sub.drain_tool_events(got.append)
        except asyncio.CancelledError:
            pass
        return redis.created, reads, acked, got, event

    created, reads, acked, got, event = virtual_clock.run(scenario())
    assert created == [(STREAM, "persist:tools"), (STREAM, "persist:tools")]
    assert reads[0] == ("persist:tools", "writer", {STREAM: ">"})
    assert got == [event]
    assert acked == [(STREAM, "persist:tools", "1-0")]
