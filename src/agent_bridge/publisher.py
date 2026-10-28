"""Redis Streams publisher for agent bridge events.

Events are published to a Redis Stream keyed by session:
    agent:events:{session_key}

History-persist commands are published to:
    agent:history

Consumers (main app) read via XREAD or consumer groups.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from .abort_contract import process_boot_id
from .events import AgentEvent, AgentEventKind
from .run_contract import (
    DONE_FIELD, RUN_TERMINAL_RETENTION_S, RunAccepted, RunTerminal, TeardownReport,
    run_terminal_key,
)

logger = logging.getLogger(__name__)

# Stream names
EVENTS_STREAM_PREFIX = "agent:events:"
HISTORY_STREAM = "agent:history"
COMMANDS_STREAM = "agent:commands"
RUN_STREAM_PREFIX = "agent:run:"
# Unified event stream for all bots: events:{bot_id}:{user_id}
UNIFIED_EVENTS_PREFIX = "events:"
# Shared bot-configuration notifications (no user-specific data in payloads).
GLOBAL_CONFIG_STREAM = f"{UNIFIED_EVENTS_PREFIX}system:__config__"

# Keep last 10k events per session stream (auto-trimmed)
STREAM_MAXLEN = 10_000
# Unified event streams are smaller — tool events + lifecycle only
UNIFIED_STREAM_MAXLEN = 5_000
# Run response streams are shorter-lived. This is a runaway BACKSTOP, not a
# turn-sizing cap: a single fat agent turn (fan-out sub-agents + long tool
# loops) can emit thousands of events, and a reconnecting client must be able
# to replay the whole turn from its head. A 5k count-cap would evict the head
# of a pathological turn mid-flight (TASK-366). Set generously so real turns
# are never truncated; abandoned streams are reclaimed by the sliding TTL
# below and by publish_run_done's shorter expire. The DB (TurnLog +
# ToolCallRecord) remains the durable backstop for anything beyond this.
RUN_STREAM_MAXLEN = 100_000
# Sliding TTL refreshed on every run event so an abandoned/crashed run stream
# self-cleans even if run_done never fires.
RUN_STREAM_TTL_SECONDS = 900
# Steer RPCs whose caller retries under the same request id: the subscriber's
# dedupe key lives 7 days, so the result must too, or a retry after 60s can
# neither re-send nor read the first answer (TASK-1251 deliveries, TASK-1233
# quiesce control). Every other RPC result is read once, within seconds.
DURABLE_RPC_RESULT_PREFIXES = ("steer_delivery_", "steer_quiesce_")
DURABLE_RPC_RESULT_TTL_S = 604800


class RedisPublisher:
    def __init__(self, redis_url: str, *, default_provider: str | None = None) -> None:
        # socket_timeout=None: redis-py 8.0 defaults to 5s, which breaks
        # long/blocking operations. Bound only the initial connect.
        self._redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=None,
            socket_connect_timeout=5,
        )
        self._connected = False
        # Bridges instantiate one publisher and pass their backend name here
        # ("claude-code", "codex"). Stamps every event at the
        # publish boundary so downstream consumers can dispatch on provider
        # without each call site remembering to set it.
        self._default_provider = default_provider
        try:
            self._redis.ping()
            self._connected = True
            logger.info("Redis publisher connected: %s", redis_url)
        except redis.ConnectionError as e:
            logger.error("Redis publisher connect failed: %s", e)

    def _stamp_provider(self, event: AgentEvent) -> AgentEvent:
        if event.provider is None and self._default_provider is not None:
            event.provider = self._default_provider
        return event

    def publish_event(self, event: AgentEvent) -> str | None:
        """Publish an event to the session's Redis Stream. Returns the stream ID."""
        if not self._connected:
            return None
        stream_key = f"{EVENTS_STREAM_PREFIX}{event.session_key}"
        self._stamp_provider(event)
        try:
            data = event.to_dict()
            # Redis Streams fields must be flat str->str
            fields = {"payload": json.dumps(data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish event to Redis stream %s", stream_key)
            return None

    def publish_history(self, bot_id: str, role: str, content: str) -> str | None:
        """Publish a history-persist command for the main app to consume."""
        if not self._connected:
            return None
        try:
            fields = {
                "bot_id": bot_id,
                "role": role,
                "content": content,
            }
            stream_id = self._redis.xadd(
                HISTORY_STREAM,
                fields,
                maxlen=1000,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish history command")
            return None

    def publish_run_event(self, request_id: str, event: AgentEvent) -> str | None:
        """Publish an event to a per-run response stream for the requesting client."""
        if not self._connected:
            if event.kind.value == "approval_decision":
                logger.warning("Bridge decision audit LOST: Redis disconnected request=%s event=%s",
                               request_id, event.event_id)
            return None
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        self._stamp_provider(event)
        try:
            data = event.to_dict()
            fields = {"payload": json.dumps(data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=RUN_STREAM_MAXLEN,
                approximate=True,
            )
            # Durable callback streams must survive app restarts and silent long
            # turns; ordinary interactive streams retain the short runaway TTL.
            ttl = 604800 if request_id.startswith("req_delivery_") else RUN_STREAM_TTL_SECONDS
            self._redis.expire(stream_key, ttl)
            return stream_id
        except Exception:
            logger.exception("Failed to publish run event to %s", stream_key)
            return None

    def publish_rpc_result(self, request_id: str, result: dict) -> None:
        """Publish an RPC call result to a short-lived response stream."""
        if not self._connected:
            return
        stream_key = f"agent:rpc:{request_id}"
        try:
            fields = {"payload": json.dumps(result, ensure_ascii=False, default=str)}
            # Reply identity (TASK-1232 P1): send_rpc ignores a reply stamped
            # by a backend other than the one it addressed. Stream fields, not
            # payload keys, so every existing payload shape is unchanged.
            if self._default_provider:
                fields["backend"] = self._default_provider
                fields["boot_id"] = process_boot_id()
            self._redis.xadd(stream_key, fields, maxlen=10)
            ttl = DURABLE_RPC_RESULT_TTL_S if request_id.startswith(DURABLE_RPC_RESULT_PREFIXES) else 60
            self._redis.expire(stream_key, ttl)
        except Exception:
            logger.exception("Failed to publish RPC result to %s", stream_key)

    def publish_run_done(self, request_id: str, *, teardown: TeardownReport) -> None:
        """Write the run's one terminal sentinel (TASK-1233 contract).

        The sentinel carries :meth:`RunTerminal.to_fields`: this bridge's
        backend and boot id plus what teardown did before publish. The same
        fields go to the run's terminal record, which outlives the stream by
        design (``run_contract.RUN_TERMINAL_RETENTION_S``). A publisher without
        a backend (never a deployed bridge) writes only the bare legacy
        sentinel, which readers treat as "no evidence".

        Durable inter-bot callbacks may need to reattach after an app restart,
        so their deterministic ``req_delivery_*`` run streams are retained for
        seven days. Ordinary interactive runs keep the existing five-minute TTL.
        """
        if not self._connected:
            return
        stream_key = f"{RUN_STREAM_PREFIX}{request_id}"
        terminal = (
            RunTerminal.here(request_id, self._default_provider, teardown)
            if self._default_provider else None
        )
        fields = terminal.to_fields() if terminal else {DONE_FIELD: "1"}
        ttl = 604800 if request_id.startswith("req_delivery_") else 300
        try:
            if terminal is not None:
                # The compact record first: a reader that sees the stream's
                # done can always find the record too.
                record_key = run_terminal_key(request_id)
                pipe = self._redis.pipeline(transaction=True)
                pipe.delete(record_key)
                pipe.hset(record_key, mapping=fields)
                pipe.expire(record_key, max(ttl, RUN_TERMINAL_RETENTION_S))
                pipe.execute()
            self._redis.xadd(stream_key, fields, maxlen=RUN_STREAM_MAXLEN, approximate=True)
            self._redis.expire(stream_key, ttl)
        except Exception:
            logger.exception("Failed to publish run done to %s", stream_key)
            return
        logger.info(
            "Run done request_id=%s exited=%s disconnect=%s signal=%s teardown_ms=%s",
            request_id, teardown.exited, teardown.disconnect, teardown.signal,
            teardown.elapsed_ms,
        )

    def publish_run_accepted(self, request_id: str, session_key: str) -> None:
        """Publish ``RUN_ACCEPTED`` naming the boot that admitted this send."""
        if not self._default_provider:
            return
        accepted = RunAccepted.here(request_id, self._default_provider)
        self.publish_run_event(request_id, AgentEvent(
            event_id=f"{request_id}:accepted", session_key=session_key, run_id=request_id,
            kind=AgentEventKind.RUN_ACCEPTED, origin="system", raw=accepted.to_dict(),
        ))

    def publish_unified_event(
        self,
        bot_id: str,
        user_id: str,
        event_data: dict[str, Any],
    ) -> str | None:
        """Publish an event to the unified event stream for a bot+user pair.

        Used for tool execution events and any application-level events that the UI consumes via consumer groups.
        """
        if not self._connected:
            return None
        stream_key = f"{UNIFIED_EVENTS_PREFIX}{bot_id}:{user_id}"
        try:
            fields = {"payload": json.dumps(event_data, ensure_ascii=False, default=str)}
            stream_id = self._redis.xadd(
                stream_key,
                fields,
                maxlen=UNIFIED_STREAM_MAXLEN,
                approximate=True,
            )
            return stream_id
        except Exception:
            logger.exception("Failed to publish unified event to %s", stream_key)
            return None

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._connected
