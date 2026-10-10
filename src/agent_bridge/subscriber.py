"""Redis Streams subscriber — consumer side counterpart to RedisPublisher.

Used by the main app to read events from Redis Streams instead of
the in-memory FanoutHub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

import redis.asyncio as aioredis
from redis.asyncio import BlockingConnectionPool

from .command_publisher import CommandPublisherMixin
from .events import AgentEvent
from .publisher import (
    COMMANDS_STREAM,
    EVENTS_STREAM_PREFIX,
    GLOBAL_CONFIG_STREAM,
    HISTORY_STREAM,
    RUN_STREAM_PREFIX,
    UNIFIED_EVENTS_PREFIX,
    UNIFIED_STREAM_MAXLEN,
)

logger = logging.getLogger(__name__)


class RedisSubscriber(CommandPublisherMixin):
    def __init__(self, redis_url: str) -> None:
        # socket_timeout=None: redis-py 8.0 changed the default to 5s, which
        # races our blocking XREADGROUP(block=5000) reads — every idle poll
        # would raise "Timeout reading from redis". Keep reads unbounded and
        # bound only the initial connect.
        # Stash the URL so long-lived holders (e.g. the SSE endpoint) can spin
        # up their own dedicated client per connection — see openclaw_ws.py.
        self._redis_url = redis_url
        self._redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=5,
        )
        # Dedicated, bounded pool for fire-and-forget writes (xadd / publish),
        # kept SEPARATE from the reader client above. Why a second pool:
        # ``self._redis`` runs long-lived blocking XREAD/XREADGROUP loops that
        # each pin a connection for a turn's whole duration. Mixing the
        # high-frequency, bursty publish path into that same pool let a single
        # fan-out — e.g. reaping N stale turns at once fires N simultaneous
        # ``publish_tool_event`` calls — spawn a fresh connection per in-flight
        # xadd. redis-py's default ConnectionPool never reaps idle connections
        # AND *raises* MaxConnectionsError (rather than waiting) at its cap
        # (100 since redis-py 8.0), so each burst ratcheted the open-connection
        # count up permanently until the app could no longer reach Redis at all
        # (even /health -> connection refused).
        #
        # BlockingConnectionPool WAITS (briefly — an xadd is sub-millisecond)
        # for a free connection instead of erroring, and the small cap means a
        # 34-at-once burst reuses ≤16 connections instead of opening 34 that
        # never close. Lazy: no connections until the first publish, so
        # reader-only instances (the per-SSE-connection subscribers) cost
        # nothing here.
        self._pub_redis = aioredis.Redis(
            connection_pool=BlockingConnectionPool.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=None,
                socket_connect_timeout=5,
                max_connections=16,
                timeout=5,
            )
        )
        self._connected = False

    async def connect(self) -> None:
        try:
            await self._redis.ping()
            self._connected = True
            logger.info("Redis subscriber connected")
        except Exception as e:
            logger.error("Redis subscriber connect failed: %s", e)

    async def subscribe(
        self,
        session_key: str,
        *,
        run_id: str | None = None,
        timeout_s: float = 300,
    ) -> AsyncIterator[AgentEvent]:
        """Stream events for a session from Redis Streams.

        If run_id is provided, yields events until RUN_COMPLETED for that run,
        then stops. Otherwise streams indefinitely until cancelled.
        """
        stream_key = f"{EVENTS_STREAM_PREFIX}{session_key}"
        last_id = "$"  # only new events
        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return

            # XREAD with block (in ms), max 1s blocks to allow timeout checks
            block_ms = min(int(remaining * 1000), 1000)
            try:
                results = await self._redis.xread(
                    {stream_key: last_id},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREAD error on %s", stream_key)
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    last_id = msg_id
                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                        event = AgentEvent.from_dict(data)
                    except Exception:
                        logger.debug("Failed to parse event from Redis: %s", payload_str[:200])
                        continue

                    yield event

                    # If tracking a specific run, stop on completion
                    if run_id and event.run_id == run_id:
                        from .events import AgentEventKind
                        if event.kind in (AgentEventKind.RUN_COMPLETED, AgentEventKind.ERROR):
                            return

    async def send_steer(
        self,
        *,
        session_key: str,
        message: str,
        message_id: str,
        backend: str,
        target_request_id: str | None = None,
        timeout_s: float = 10,
        request_id: str | None = None,
        priority: str = "now",
        origin: str = "system",
        escalate: bool = False,
        cancel: bool = False,
    ) -> dict:
        """Steer, escalate, or cancel an active run message and await its RPC result.

        ``priority="now"`` interrupts running tools; ``"next"`` delivers at the
        next tool boundary (claude-code-bridge only).
        """
        request_id = request_id or f"steer_{uuid.uuid4().hex}"
        fields: dict[str, str] = {
            "action": "chat.steer",
            "session_key": session_key,
            "message": message,
            "message_id": message_id,
            "backend": backend,
            "request_id": request_id,
            "priority": priority,
            "origin": origin,
        }
        if escalate and cancel:
            raise ValueError("Cannot escalate and cancel the same message")
        if escalate or cancel:
            fields["steer_action"] = "escalate" if escalate else "cancel"
        if target_request_id:
            fields["target_request_id"] = target_request_id

        dedupe_key = f"agent:steer:request:{request_id}"
        flat_fields: list[str] = []
        for key, value in fields.items():
            flat_fields.extend((str(key), str(value)))
        script = """
            local prior = redis.call('GET', KEYS[1])
            if prior then return {0, prior} end
            local args = {}
            for i = 2, #ARGV do table.insert(args, ARGV[i]) end
            local stream_id = redis.call(
                'XADD', KEYS[2], 'MAXLEN', '~', ARGV[1], '*', unpack(args)
            )
            redis.call('SET', KEYS[1], stream_id, 'EX', 604800)
            return {1, stream_id}
        """
        await self._pub_redis.eval(
            script, 2, dedupe_key, COMMANDS_STREAM, "1000", *flat_fields
        )

        stream_key = f"agent:rpc:{request_id}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        last_id = "0-0"
        while asyncio.get_event_loop().time() < deadline:
            remaining_ms = int(
                (deadline - asyncio.get_event_loop().time()) * 1000
            )
            if remaining_ms <= 0:
                break
            results = await self._redis.xread(
                {stream_key: last_id},
                count=1,
                block=min(remaining_ms, 2000),
            )
            if results:
                for _stream, messages in results:
                    for msg_id, response_fields in messages:
                        last_id = msg_id
                        payload_raw = response_fields.get("payload", "{}")
                        return json.loads(payload_raw)
        raise TimeoutError(f"chat.steer timed out after {timeout_s}s")

    async def send_tool_result(
        self,
        session_key: str,
        tool_use_id: str,
        result: str,
        *,
        backend: str | None = None,
        request_id: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> None:
        """Publish a chat.tool_result command for a deferred SDK tool call.

        ``answers`` (question text -> answer) is the tool's native input: the
        bridge allows the call with it. Without it, ``result`` is deny text.

        Routed by bridges that have pending AWAIT_TOOL_RESULT futures keyed by
        ``tool_use_id`` (currently only claude-code-bridge for AskUserQuestion).
        Fire-and-forget — the bridge resolves the future and the run resumes;
        the next deltas flow through the existing per-run event stream.
        """
        fields: dict = {
            "action": "chat.tool_result",
            "session_key": session_key,
            "tool_use_id": tool_use_id,
            "result": result,
        }
        if backend:
            fields["backend"] = backend
        if request_id:
            fields["request_id"] = request_id
        if answers:
            fields["answers_json"] = json.dumps(answers, ensure_ascii=False)
        await self._pub_redis.xadd(
            COMMANDS_STREAM,
            fields,
            maxlen=1000,
            approximate=True,
        )
        logger.info(
            "Sent chat.tool_result: session=%s tool_use_id=%s len=%d backend=%s",
            session_key, tool_use_id, len(result or ""), backend or "?",
        )

    async def send_approval_grant(
        self,
        *,
        session_key: str,
        grant_key: str,
        backend: str | None = None,
        ttl_seconds: int = 600,
        request_id: str | None = None,
    ) -> None:
        """Publish an approval.grant command so a bridge allows ONE gated call.

        Sent when a user approves a policy-gated tool (TASK-292). The bridge
        stores ``grant_key`` in an in-memory grant set (with ``ttl_seconds``
        expiry); when the model re-issues the identical tool call on the
        continuation turn the bridge computes the same key, finds the grant,
        consumes it, and allows the call exactly once. Fire-and-forget.
        """
        fields: dict = {
            "action": "approval.grant",
            "session_key": session_key,
            "grant_key": grant_key,
            "ttl_seconds": str(int(ttl_seconds)),
        }
        if backend:
            fields["backend"] = backend
        if request_id:
            fields["request_id"] = request_id
        await self._pub_redis.xadd(
            COMMANDS_STREAM, fields, maxlen=1000, approximate=True,
        )
        logger.info(
            "Sent approval.grant: session=%s grant_key=%s backend=%s ttl=%ds",
            session_key, grant_key[:12], backend or "?", ttl_seconds,
        )

    async def publish_approval_reload(self) -> int:
        """Broadcast a policy-bundle reload to every bridge (TASK-291, TASK-293).

        Bridges subscribe to ``approval:policies:reload`` and drop their cached
        bundle on any message, so an admin edit propagates without a restart
        and without waiting out the cache TTL.
        """
        # Return Redis's recipient count, including zero; never imply a bridge
        # installed the bundle. Let callers distinguish publication failures.
        recipients = await self._pub_redis.publish("approval:policies:reload", "1")
        logger.info("Published approval:policies:reload recipients=%s", recipients)
        return recipients

    async def send_rpc(
        self,
        method: str,
        params: dict,
        request_id: str,
        *,
        timeout_s: float = 15,
        backend: str | None = None,
    ) -> dict:
        """Send an RPC command to the bridge and wait for the result.

        If ``backend`` is provided it is included in the message so bridges
        that filter by backend (e.g. codex-bridge) can ACK and skip RPCs
        that aren't theirs. Bridges that don't filter (e.g. legacy claude-
        code-bridge) ignore the field.
        """
        import json as _json

        fields: dict = {
            "action": "rpc.call",
            "method": method,
            "params": _json.dumps(params, ensure_ascii=False),
            "request_id": request_id,
        }
        if backend:
            fields["backend"] = backend

        await self._pub_redis.xadd(
            COMMANDS_STREAM,
            fields,
            maxlen=1000,
            approximate=True,
        )

        # Wait for result on agent:rpc:{request_id}
        stream_key = f"agent:rpc:{request_id}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            remaining_ms = int((deadline - asyncio.get_event_loop().time()) * 1000)
            if remaining_ms <= 0:
                break
            results = await self._redis.xread(
                {stream_key: "0-0"}, count=1, block=min(remaining_ms, 2000),
            )
            if results:
                for _stream, messages in results:
                    for _msg_id, fields in messages:
                        payload_raw = fields.get("payload", "{}")
                        return _json.loads(payload_raw)
        raise TimeoutError(f"RPC {method} timed out after {timeout_s}s")

    async def subscribe_run(
        self,
        request_id: str,
        *,
        timeout_s: float = 600,
    ) -> AsyncIterator[AgentEvent]:
        """Subscribe to the per-run response stream published by the bridge.

        Yields events until the bridge signals 'done' or inactivity timeout.
        The timeout resets on each received event, so long-running tasks
        (e.g. backups) that continue producing events won't time out.
        """
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        last_id = "0"  # read from beginning (stream is created fresh per run)
        loop_time = asyncio.get_event_loop().time
        deadline = loop_time() + timeout_s
        logger.info(
            "subscribe_run started: request_id=%s timeout_s=%.0f stream=%s",
            request_id, timeout_s, stream_key,
        )

        while True:
            remaining = deadline - loop_time()
            if remaining <= 0:
                logger.warning("subscribe_run inactivity timeout for request_id=%s", request_id)
                raise TimeoutError(
                    f"bridge run {request_id} produced no event for {timeout_s:.0f}s"
                )

            block_ms = min(int(remaining * 1000), 2000)
            try:
                results = await self._redis.xread(
                    {stream_key: last_id},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREAD error on %s", stream_key)
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    last_id = msg_id
                    # Reset inactivity deadline on each event
                    deadline = loop_time() + timeout_s

                    # Check for done sentinel
                    if fields.get("done"):
                        return

                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                        event = AgentEvent.from_dict(data)
                    except Exception:
                        logger.debug("Failed to parse run event: %s", payload_str[:200])
                        continue

                    yield event

    # ---- Unified event stream (consumer groups) ----

    async def ensure_consumer_group(
        self,
        stream_key: str,
        group_name: str,
    ) -> None:
        """Idempotent XGROUP CREATE — creates the group starting from latest."""
        try:
            await self._redis.xgroup_create(
                stream_key, group_name, id="$", mkstream=True,
            )
        except Exception:
            pass  # group already exists

    async def ensure_groups(
        self,
        bot_id: str | list[str],
        user_id: str,
        consumer_id: str,
        *,
        include_global: bool = False,
    ) -> dict[str, str]:
        """Pre-create the unified-stream consumer group(s) for these bots.

        Returns the ``{stream_key: bot_id}`` mapping. Because the group is
        created at ``$`` (latest), any event published AFTER this call is
        captured and delivered on the next ``subscribe_group`` read — even if
        the caller hasn't started reading yet. This lets a caller create the
        group up-front and only then signal "subscribed" downstream, closing
        the race where events published between connect and group-create are
        lost. Idempotent, so ``subscribe_group`` re-invokes it harmlessly.
        """
        bot_ids = [bot_id] if isinstance(bot_id, str) else list(bot_id)
        group_name = f"ui:{consumer_id}"
        stream_keys: dict[str, str] = {}
        for bid in bot_ids:
            sk = f"{UNIFIED_EVENTS_PREFIX}{bid}:{user_id}"
            stream_keys[sk] = bid
        if include_global:
            stream_keys[GLOBAL_CONFIG_STREAM] = "system"
        for sk in stream_keys:
            await self.ensure_consumer_group(sk, group_name)
        return stream_keys

    async def subscribe_group(
        self,
        bot_id: str | list[str],
        user_id: str,
        consumer_id: str,
        *,
        timeout_s: float = 86400,
        include_global: bool = False,
    ) -> AsyncIterator[dict | None]:
        """Subscribe to unified event stream(s) via consumer groups.

        ``bot_id`` may be a single bot or a list of bots.  When multiple bots
        are given, events from ALL of them are interleaved and yielded.

        Two-phase read per stream:
        1. Replay pending (unacked) messages from previous connection
        2. Read new messages as they arrive

        Yields raw event dicts (not AgentEvent — unified stream carries
        tool events, OpenClaw events, and application events).
        """
        group_name = f"ui:{consumer_id}"
        consumer_name = "reader"

        # Ensure consumer groups exist for all streams (idempotent — the SSE
        # route may have pre-created them via ensure_groups before signaling
        # "hello"). Returns the {stream_key: bot_id} mapping used below.
        stream_keys = await self.ensure_groups(
            bot_id, user_id, consumer_id, include_global=include_global,
        )

        # Phase 1: replay pending (unacked) messages from all streams
        try:
            results = await self._redis.xreadgroup(
                group_name, consumer_name,
                {sk: "0" for sk in stream_keys},
                count=500,
            )
            if results:
                # Interleave messages from all streams by Redis message ID
                # (timestamp-based) so events aren't serialized per-stream.
                all_msgs: list[tuple[str, str, dict]] = []
                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        all_msgs.append((msg_id, stream_name, fields))
                all_msgs.sort(key=lambda x: x[0])

                for msg_id, stream_name, fields in all_msgs:
                    payload_str = fields.get("payload", "{}")
                    try:
                        data = json.loads(payload_str)
                    except Exception:
                        data = {"raw": payload_str}
                    data["_replayed"] = True
                    yield data
                    await self._redis.xack(stream_name, group_name, msg_id)
        except Exception:
            logger.exception("Error replaying pending messages for %s", list(stream_keys))

        # Phase 2: live events from all streams
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return

            block_ms = min(int(remaining * 1000), 2000)
            try:
                results = await self._redis.xreadgroup(
                    group_name, consumer_name,
                    {sk: ">" for sk in stream_keys},
                    count=100,
                    block=block_ms,
                )
            except Exception:
                logger.exception("Redis XREADGROUP error on %s", list(stream_keys))
                await asyncio.sleep(1)
                continue

            if not results:
                yield None  # keepalive tick — lets caller send SSE ping
                continue

            # Interleave messages from all streams by Redis message ID
            # so concurrent bot events are delivered in chronological order
            # instead of all-of-stream-A then all-of-stream-B.
            all_msgs: list[tuple[str, str, dict]] = []
            for stream_name, messages in results:
                for msg_id, fields in messages:
                    all_msgs.append((msg_id, stream_name, fields))
            all_msgs.sort(key=lambda x: x[0])

            for msg_id, stream_name, fields in all_msgs:
                deadline = asyncio.get_event_loop().time() + timeout_s
                payload_str = fields.get("payload", "{}")
                try:
                    data = json.loads(payload_str)
                except Exception:
                    data = {"raw": payload_str}
                yield data
                await self._redis.xack(stream_name, group_name, msg_id)

    async def publish_tool_event(
        self,
        bot_id: str,
        user_id: str,
        event: dict,
    ) -> str | None:
        """Publish a tool_start/tool_end event to the unified stream."""
        stream_key = f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}"
        try:
            fields = {"payload": json.dumps(event, ensure_ascii=False, default=str)}
            stream_id = await self._pub_redis.xadd(
                stream_key,
                fields,
                maxlen=UNIFIED_STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish tool event to %s", stream_key)
            return None

    async def publish_global_config_event(self, event: dict) -> str | None:
        """Fan out non-user-specific configuration changes to every UI window."""
        return await self.publish_tool_event("system", "__config__", event)

    async def latest_activity(
        self,
        bot_id: str,
        user_id: str,
        *,
        batch_size: int = 500,
        hard_cap: int = UNIFIED_STREAM_MAXLEN,
    ) -> dict | None:
        """Reconstruct the latest turn snapshot for a bot from the unified event
        stream — read-only and REALTIME (reflects an in-flight turn as it
        streams, unlike the turn-log DB which only lands at turn end).

        Returns the last user prompt, the streamed assistant text (full text +
        the current/last "bubble" — the segment since the most recent tool
        boundary), the turn id, and whether the turn is still streaming. Returns
        None if the stream has no turn we can anchor on.

        AGENT BOTS ONLY: chat bots don't emit ``text_delta`` into the stream, so
        their assistant text can't be reconstructed here — callers must not use
        this for chat bots.

        Pagination walks the stream newest→oldest in batches until it finds the
        anchoring ``turn_start`` or hits ``hard_cap`` (the stream's own maxlen),
        so a very long turn can't silently fall out of a fixed window.
        """
        stream_key = f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}"
        events: list[dict] = []
        max_id = "+"
        scanned = 0
        found_start = False
        try:
            while scanned < hard_cap and not found_start:
                batch = await self._pub_redis.xrevrange(
                    stream_key, max_id, "-", count=batch_size
                )
                if not batch:
                    break
                for _id, fields in batch:
                    payload = fields.get("payload")
                    if not payload:
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    events.append(ev)
                    if ev.get("_type") == "turn_start":
                        found_start = True
                scanned += len(batch)
                # Exclusive range for the next page (Redis 6.2+ "(" prefix).
                max_id = f"({batch[-1][0]}"
        except Exception:
            logger.exception("latest_activity read failed for %s", stream_key)
            return None

        if not events:
            return None

        # events are newest-first. The most recent turn_start anchors "the last
        # turn"; everything sharing its turn_id belongs to it.
        turn_start = next(
            (e for e in events if e.get("_type") == "turn_start"), None
        )
        if turn_start is None:
            return None
        turn_id = turn_start.get("turn_id")

        completed = any(
            e.get("_type") == "turn_complete" and e.get("turn_id") == turn_id
            for e in events
        )

        # Reconstruct assistant text: splice each text_delta at its text_offset
        # (the char count before that delta). Robust to gaps/overwrites.
        deltas = [
            e
            for e in events
            if e.get("_type") == "text_delta" and e.get("turn_id") == turn_id
        ]
        text = ""
        for e in sorted(deltas, key=lambda d: d.get("text_offset") or 0):
            off = e.get("text_offset")
            chunk = e.get("delta", "")
            if not isinstance(off, int):
                text += chunk
                continue
            if off > len(text):
                text += " " * (off - len(text))
            text = text[:off] + chunk + text[off + len(chunk):]
        full_text = text.rstrip()

        # Each tool_start cuts the assistant text into a new bubble. Split on
        # those offsets and take the last NON-EMPTY segment: while a tool is
        # running the trailing segment is empty (no text emitted yet), and for
        # agents that's most of the time — so "text after the final boundary"
        # would usually be blank. The last non-empty segment is the most recent
        # thing the bot actually said.
        boundaries = sorted(
            {
                e.get("text_offset")
                for e in events
                if e.get("_type") == "tool_event"
                and e.get("event") == "tool_start"
                and e.get("turn_id") == turn_id
                and isinstance(e.get("text_offset"), int)
            }
        )
        cuts = [0, *boundaries, len(full_text)]
        segments = [
            full_text[cuts[i]:cuts[i + 1]].strip() for i in range(len(cuts) - 1)
        ]
        non_empty = [s for s in segments if s]
        last_bubble = non_empty[-1] if non_empty else full_text.strip()

        return {
            "bot_id": bot_id,
            "user_id": user_id,
            "turn_id": turn_id,
            "streaming": not completed,
            "last_user_prompt": turn_start.get("content"),
            "assistant_message_id": turn_start.get("assistant_message_id"),
            "last_bubble": last_bubble,
            "full_text": full_text,
            "updated_at": events[0].get("ts"),
        }

    async def cleanup_stale_groups(
        self,
        stream_key: str,
        *,
        max_idle_ms: int = 1_800_000,
    ) -> int:
        """Destroy consumer groups that have been idle too long.

        Only targets ui:* groups (not system groups like llm-bawt).
        Returns the number of groups destroyed.
        """
        destroyed = 0
        try:
            groups = await self._redis.xinfo_groups(stream_key)
        except Exception:
            return 0

        for group in groups:
            name = group.get("name", "")
            if not name.startswith("ui:"):
                continue
            # Check last delivered ID and pending count
            # If no pending and idle > threshold, destroy
            consumers = group.get("consumers", 0)
            pending = group.get("pending", 0)
            if pending > 0:
                continue  # has unacked messages, keep alive
            # Check consumer idle time
            try:
                consumer_info = await self._redis.xinfo_consumers(stream_key, name)
                all_idle = all(
                    (c.get("idle", 0) > max_idle_ms) for c in consumer_info
                ) if consumer_info else (consumers == 0)
            except Exception:
                all_idle = consumers == 0

            if all_idle:
                try:
                    await self._redis.xgroup_destroy(stream_key, name)
                    destroyed += 1
                    logger.debug("Destroyed stale consumer group %s on %s", name, stream_key)
                except Exception:
                    logger.debug("Failed to destroy group %s on %s", name, stream_key)

        return destroyed

    async def drain_tool_events(
        self,
        callback,
        *,
        consumer_group: str = "persist:tools",
        consumer_name: str = "writer",
    ) -> None:
        """Continuously drain tool events from ALL unified streams.

        Uses a dedicated consumer group so persistence is independent
        of UI consumers. Calls ``callback(event_data)`` for each tool event.
        """
        while True:
            try:
                # Discover active unified streams
                streams = await self.list_unified_streams()
                if not streams:
                    await asyncio.sleep(5)
                    continue

                # Ensure consumer group on each stream
                for stream_key in streams:
                    await self.ensure_consumer_group(stream_key, consumer_group)

                # Build read dict
                read_streams = {sk: ">" for sk in streams}

                results = await self._redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    read_streams,
                    count=50,
                    block=5000,
                )
                if not results:
                    continue

                for stream_key, messages in results:
                    for msg_id, fields in messages:
                        payload_str = fields.get("payload", "{}")
                        try:
                            data = json.loads(payload_str)
                        except Exception:
                            await self._redis.xack(stream_key, consumer_group, msg_id)
                            continue

                        # Persist tool events (tool_start/tool_end) and, for
                        # agent turns, coalesced assistant text deltas — both
                        # need a durable path independent of the HTTP connection
                        # (TASK-286). turn_complete lets the sink release its
                        # per-turn text buffer. reasoning_delta (TASK-360/P4) is
                        # persisted the same way so a cold reload mid-turn
                        # recovers already-produced reasoning.
                        if data.get("_type") in ("tool_event", "text_delta", "reasoning_delta", "turn_complete"):
                            try:
                                callback(data)
                            except Exception:
                                logger.exception("Event persistence callback failed")
                        await self._redis.xack(stream_key, consumer_group, msg_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Tool event drain error")
                await asyncio.sleep(2)

    async def list_unified_streams(self) -> list[str]:
        """List all active unified event stream keys."""
        try:
            keys = []
            async for key in self._redis.scan_iter(match=f"{UNIFIED_EVENTS_PREFIX}*", count=100):
                keys.append(key)
            return keys
        except Exception:
            logger.exception("Failed to list unified streams")
            return []

    async def drain_history(
        self,
        callback,
        *,
        consumer_group: str = "llm-bawt",
        consumer_name: str = "main",
    ) -> None:
        """Continuously drain the agent:history stream and call callback(bot_id, role, content).

        Uses consumer groups so multiple main-app instances can share the load.
        """
        # Ensure consumer group exists
        try:
            await self._redis.xgroup_create(
                HISTORY_STREAM, consumer_group, id="0", mkstream=True
            )
        except Exception:
            pass  # group already exists

        while True:
            try:
                results = await self._redis.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {HISTORY_STREAM: ">"},
                    count=10,
                    block=5000,
                )
                if not results:
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        bot_id = fields.get("bot_id", "")
                        role = fields.get("role", "")
                        content = fields.get("content", "")
                        if bot_id and role and content:
                            try:
                                callback(bot_id, role, content)
                            except Exception:
                                logger.exception("History callback failed for bot=%s", bot_id)
                        # ACK the message
                        await self._redis.xack(HISTORY_STREAM, consumer_group, msg_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("History drain error")
                await asyncio.sleep(2)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:
            pass
        try:
            await self._pub_redis.aclose()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected
