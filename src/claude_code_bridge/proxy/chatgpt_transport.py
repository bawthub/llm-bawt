"""ChatGPT Responses WS/Lite transport, based on Codex rust-v0.153.4.

Full-history response.create requests deliberately omit previous_response_id:
Claude SDK histories can branch and do not retain all upstream output items.
Only successful sockets are reusable; no transport-layer generation retries.
The adapter's existing output-aware retry policy owns all replay decisions.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_OID, uuid5

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

logger = logging.getLogger(__name__)
LITE_HEADER = "x-openai-internal-codex-responses-lite"
TURN_HEADER = "x-codex-turn-state"
WS_BETA = "responses_websockets=2026-02-06"
TERMINALS = {"response.completed", "response.incomplete"}
FAILURE_TERMINALS = {"response.failed", "response.error", "error"}
PRODUCTIVE_DELTAS = {
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
    "response.output_text.delta",
    "response.refusal.delta",
    "response.function_call_arguments.delta",
    "response.custom_tool_call_input.delta",
}
TOOL_ITEM_TYPES = {"function_call", "custom_tool_call"}
DEFAULT_PRODUCTIVE_IDLE_TIMEOUT = 90.0
DEFAULT_ATTEMPT_TIMEOUT = 240.0


def lite_request(body: dict, scope: str) -> dict:
    """Move tools/instructions into Lite input, without changing caller history."""
    body = copy.deepcopy(body)
    namespace = uuid5(NAMESPACE_OID, scope)
    tools = body.pop("tools", [])
    instructions = body.pop("instructions", "")
    prefix = [{
        "type": "additional_tools", "role": "developer", "tools": tools,
        "id": "at_" + str(uuid5(namespace, json.dumps(tools, separators=(",", ":")))),
    }]
    if instructions:
        prefix.append({
            "type": "message", "role": "developer",
            "id": "msg_" + str(uuid5(namespace, instructions)),
            "content": [{"type": "input_text", "text": instructions}],
        })
    body["input"] = prefix + body.get("input", [])
    # Codex strips detail from both message images and tool-result images.
    for item in body["input"]:
        for field in ("content", "output"):
            content = item.get(field)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        part.pop("detail", None)
    body["parallel_tool_calls"] = False
    body.setdefault("reasoning", {})["context"] = "all_turns"
    return body


def _event_object(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _event_object(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_event_object(v) for v in value]
    return value


@dataclass
class _Session:
    socket: object | None = None
    turn_state: str | None = None
    http_only: bool = False
    timer: asyncio.TimerHandle | None = None


class ChatGPTEventTimeout(TimeoutError):
    """A bounded Responses wait that the proxy retry loop already owns."""

    proxy_retry_owner = True

    def __init__(
        self,
        phase: str,
        *,
        transport: str,
        attempt: int,
        elapsed_seconds: float,
        productive_idle_seconds: float,
        fallback_transport: str | None,
    ) -> None:
        self.phase = phase
        self.transport = transport
        self.attempt = attempt
        self.elapsed_seconds = elapsed_seconds
        self.productive_idle_seconds = productive_idle_seconds
        self.fallback_transport = fallback_transport
        super().__init__(
            f"timeout waiting for {phase} ChatGPT Responses progress "
            f"(transport={transport}, attempt={attempt})"
        )


class ChatGPTResponsesTransport:
    """Bounded, exclusive socket leases scoped to account/conversation/turn/model.

    Parallel requests never share a socket. Idle leases expire automatically;
    cancellation or an unfinished response discards its lease immediately.
    """

    def __init__(self, *, idle_timeout: float = 240,
                 first_event_timeout: float = 60, keepalive: float = 60,
                 productive_idle_timeout: float = DEFAULT_PRODUCTIVE_IDLE_TIMEOUT,
                 attempt_timeout: float = DEFAULT_ATTEMPT_TIMEOUT,
                 max_connections: int = 32, connector=connect):
        self.idle_timeout = idle_timeout
        self.first_event_timeout = first_event_timeout
        self.productive_idle_timeout = productive_idle_timeout
        self.attempt_timeout = attempt_timeout
        self.keepalive = keepalive
        self.max_connections = max_connections
        self.connector = connector
        self._idle: dict[tuple, _Session] = {}
        self._slots = asyncio.Semaphore(max_connections)
        self._active: set[ChatGPTStream] = set()
        self._cleanup: set[asyncio.Task] = set()
        self._closed = False

    def _expire(self, key, session):
        if self._idle.get(key) is session:
            self._idle.pop(key)
            task = asyncio.create_task(self._disconnect(session))
            self._cleanup.add(task)
            task.add_done_callback(self._cleanup.discard)

    async def _disconnect(self, session):
        if session.timer:
            session.timer.cancel()
        if session.socket is not None:
            await session.socket.close()
            session.socket = None

    async def open(self, *, body, headers, bearer, base_url, context, http_client):
        if self._closed:
            raise RuntimeError("ChatGPT transport is closed")
        await asyncio.wait_for(self._slots.acquire(), timeout=15)
        # Never cache anonymous callers or use prompt hashes as security identity.
        scoped = bool(context and context.conversation_id and context.request_id)
        key = (base_url, headers.get("chatgpt-account-id"),
               hashlib.sha256(bearer.encode()).hexdigest(),
               context.conversation_id if scoped else None,
               context.request_id if scoped else None, body["model"])
        session = self._idle.pop(key, _Session()) if scoped else _Session()
        if session.timer:
            session.timer.cancel()
        stream = ChatGPTStream(
            self, key if scoped else None, session, context=context,
        )
        self._active.add(stream)
        try:
            request = lite_request(body, headers["session_id"])
            wire_headers = {**headers, "Authorization": f"Bearer {bearer}",
                            "OpenAI-Beta": WS_BETA, LITE_HEADER: "true"}
            if session.turn_state:
                wire_headers[TURN_HEADER] = session.turn_state
            if session.socket is not None:
                logger.info("chatgpt_transport transport=websocket model=%s reused=true", body["model"])
            if not session.http_only and session.socket is None:
                url = urlsplit(base_url.rstrip("/") + "/responses")
                ws_url = urlunsplit(url._replace(scheme="wss" if url.scheme == "https" else "ws"))
                try:
                    session.socket = await self.connector(
                        ws_url, additional_headers=wire_headers,
                        open_timeout=15, close_timeout=2, max_size=16 * 1024 * 1024,
                        ping_interval=20, ping_timeout=20,
                    )
                    response = session.socket.response
                    session.turn_state = session.turn_state or response.headers.get(TURN_HEADER)
                    logger.info("chatgpt_transport transport=websocket model=%s reused=false", body["model"])
                except InvalidStatus as exc:
                    # Codex's explicit fallback contract is HTTP 426. Never
                    # hide authentication, quota, bad payload or server errors.
                    response = exc.response
                    if response.status_code != 426:
                        raise httpx.HTTPStatusError(
                            f"WebSocket upgrade HTTP {response.status_code}",
                            request=httpx.Request("GET", ws_url),
                            response=httpx.Response(response.status_code, headers=response.headers),
                        ) from exc
                    session.http_only = True
                    logger.warning("chatgpt_transport transport=sse fallback=upgrade_426 model=%s", body["model"])
            if session.http_only:
                http_headers = {LITE_HEADER: "true"}
                if session.turn_state:
                    http_headers[TURN_HEADER] = session.turn_state
                # Lite adds fields not yet represented by the OpenAI SDK schema.
                from openai import AsyncStream
                from openai.types.responses import ResponseStreamEvent

                stream.http = await http_client.post(
                    "/responses", body=request, cast_to=object, stream=True,
                    stream_cls=AsyncStream[ResponseStreamEvent],
                    options={"headers": http_headers},
                )
                response = getattr(stream.http, "response", None)
                if response is not None:
                    session.turn_state = session.turn_state or response.headers.get(TURN_HEADER)
                stream.response = response
                stream.http_iterator = stream.http.__aiter__()
            else:
                metadata = request.setdefault("client_metadata", {})
                metadata["ws_request_header_x_openai_internal_codex_responses_lite"] = "true"
                if session.turn_state:
                    metadata["x-codex-turn-state"] = session.turn_state
                await asyncio.wait_for(session.socket.send(json.dumps({"type": "response.create", **request})), 60)
                stream.response = session.socket.response
            return stream
        except ConnectionClosed as exc:
            await stream.close()
            raise ConnectionError("ChatGPT WebSocket closed before response.create completed") from exc
        except BaseException:
            await stream.close()
            raise

    async def release(self, stream):
        self._active.discard(stream)
        try:
            if not stream.complete:
                # Keep the turn's routing token across a safe outer retry, but
                # never reuse a socket containing an unfinished generation.
                await self._disconnect(stream.session)
            if (stream.key and not self._closed
                    and stream.key not in self._idle
                    and len(self._idle) < self.max_connections):
                self._idle[stream.key] = stream.session
                stream.session.timer = asyncio.get_running_loop().call_later(
                    self.keepalive, self._expire, stream.key, stream.session,
                )
            else:
                await self._disconnect(stream.session)
        finally:
            self._slots.release()

    async def discard(self, stream):
        """Remove an incomplete attempt's retained routing lease entirely."""
        if stream.key and self._idle.get(stream.key) is stream.session:
            self._idle.pop(stream.key, None)
        await self._disconnect(stream.session)

    async def close(self):
        self._closed = True
        await asyncio.gather(*(s.close() for s in list(self._active)))
        idle, self._idle = self._idle, {}
        await asyncio.gather(*(self._disconnect(s) for s in idle.values()))
        if self._cleanup:
            await asyncio.gather(*self._cleanup)


class ChatGPTStream:
    def __init__(self, owner, key, session, *, context=None):
        self.owner, self.key, self.session = owner, key, session
        self.context = context
        self.http = self.http_iterator = self.response = None
        self.complete = self.closed = self.event_seen = False
        self.attempt = max(1, int(getattr(context, "attempt", 1) or 1))
        self.started_at = time.monotonic()
        self.last_transport_activity_at = self.started_at
        self.last_productive_activity_at = self.started_at
        self.recovery_reported = False

    @property
    def transport(self) -> str:
        return "sse" if self.http is not None else "websocket"

    @staticmethod
    def _item_type(event) -> str:
        item = getattr(event, "item", None)
        if isinstance(item, dict):
            return str(item.get("type") or "")
        return str(getattr(item, "type", "") or "")

    @classmethod
    def _is_productive(cls, event) -> bool:
        kind = str(getattr(event, "type", "") or "")
        if kind in TERMINALS or kind in FAILURE_TERMINALS:
            return True
        if kind in PRODUCTIVE_DELTAS:
            delta = getattr(event, "delta", None)
            return bool(delta) and not (
                isinstance(delta, str) and not delta.strip()
            )
        if kind in {"response.output_item.added", "response.output_item.done"}:
            return cls._item_type(event) in TOOL_ITEM_TYPES
        return False

    @property
    def fallback_transport(self) -> str | None:
        return "sse" if self.transport == "websocket" else None

    def _timeout(self, phase: str, now: float) -> ChatGPTEventTimeout:
        fallback = self.fallback_transport
        if self.transport == "websocket":
            # Preserve sticky turn state, but force the safe outer retry onto a
            # fresh HTTPS stream. The unfinished WebSocket is closed on release.
            self.session.http_only = True
        elapsed = max(0.0, now - self.started_at)
        productive_idle = max(0.0, now - self.last_productive_activity_at)
        logger.warning(
            "chatgpt_transport stall phase=%s transport=%s attempt=%d "
            "elapsed_s=%.1f productive_idle_s=%.1f fallback=%s",
            phase, self.transport, self.attempt, elapsed, productive_idle,
            fallback or "none",
        )
        return ChatGPTEventTimeout(
            phase,
            transport=self.transport,
            attempt=self.attempt,
            elapsed_seconds=elapsed,
            productive_idle_seconds=productive_idle,
            fallback_transport=fallback,
        )

    def _deadline(self, now: float) -> tuple[float, str]:
        transport_deadline = (
            self.started_at + self.owner.first_event_timeout
            if not self.event_seen
            else self.last_transport_activity_at + self.owner.idle_timeout
        )
        candidates = [
            (transport_deadline, "first" if not self.event_seen else "next"),
            (
                self.last_productive_activity_at
                + self.owner.productive_idle_timeout,
                "productive",
            ),
            (self.started_at + self.owner.attempt_timeout, "absolute"),
        ]
        return min(candidates, key=lambda candidate: candidate[0])

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed or self.complete:
            raise StopAsyncIteration
        now = time.monotonic()
        deadline, timeout_phase = self._deadline(now)
        timeout = deadline - now
        if timeout <= 0:
            raise self._timeout(timeout_phase, now)
        try:
            if self.http is not None:
                event = await asyncio.wait_for(anext(self.http_iterator), timeout)
            else:
                raw = await asyncio.wait_for(self.session.socket.recv(), timeout)
                payload = json.loads(raw)
                if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
                    raise ValueError("Invalid Responses WebSocket event")
                event = _event_object(payload)
            now = time.monotonic()
            self.event_seen = True
            self.last_transport_activity_at = now
            kind = getattr(event, "type", "")
            if kind == "response.metadata" and not self.session.turn_state:
                headers = getattr(event, "headers", None)
                if isinstance(headers, SimpleNamespace):
                    headers = vars(headers)
                if isinstance(headers, dict):
                    for name, value in headers.items():
                        if name.lower() == TURN_HEADER and isinstance(value, str):
                            self.session.turn_state = value
                            break
            if kind in TERMINALS:
                self.complete = True
            productive = self._is_productive(event)
            if productive:
                self.last_productive_activity_at = now
                if (
                    self.attempt > 1
                    and not self.recovery_reported
                    and self.context is not None
                    and kind not in FAILURE_TERMINALS
                ):
                    self.recovery_reported = True
                    self.context.report_status({
                        "state": "recovered",
                        "message": "Upstream connection recovered.",
                        "provider": self.context.provider,
                        "attempt": self.attempt,
                        "transport": self.transport,
                    })
            elif now >= self.started_at + self.owner.attempt_timeout:
                raise self._timeout("absolute", now)
            elif now >= (
                self.last_productive_activity_at
                + self.owner.productive_idle_timeout
            ):
                raise self._timeout("productive", now)
            return event
        except (ConnectionClosed, StopAsyncIteration) as exc:
            raise ConnectionError("stream closed before response.completed") from exc
        except asyncio.TimeoutError as exc:
            raise self._timeout(timeout_phase, time.monotonic()) from exc

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.http is not None:
                await self.http.close()
        finally:
            await self.owner.release(self)

    aclose = close

    async def discard(self):
        await self.owner.discard(self)
