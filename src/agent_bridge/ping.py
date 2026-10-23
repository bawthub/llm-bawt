"""The ``bridge.ping`` readiness RPC (TASK-1297, TASK-1293 C1 / Loopy M2).

A strict liveness proof for one bridge's *command consumer*: the same consumer
that receives ``chat.send`` answers this ping, so a positive reply shows that a
live process is reading the command stream for that backend at probe time.
It says nothing about inference, tools or future availability.

The shared :meth:`RedisSubscriber.send_rpc` is deliberately not used. It
accepts unstamped legacy replies and starts its deadline only after publishing,
both right for its callers and wrong here. :func:`probe_bridge` owns its own
exchange:

- one fresh ``request_id`` and ``nonce`` per probe, addressed to one backend;
- one budget covering connect, publish, read and cleanup;
- every reply checked by :func:`parse_ping_reply`, which requires the stamped
  stream identity and the payload to agree with each other and with the probe.

Request (``rpc.call`` on the commands stream)::

    method=bridge.ping  backend=<addressed>  request_id=ping_<hex>
    params={"v": 1, "nonce": <hex>, "expires_at": <epoch seconds>}

Reply payload (stream fields ``backend`` / ``boot_id`` are stamped by
``RedisPublisher.publish_rpc_result`` as for every RPC)::

    {"ok": true, "method": "bridge.ping", "v": 1, "backend": ..., "instance": <boot id>,
     "request_id": ..., "nonce": ...}

A bridge drops an expired ping unanswered. If a bridge starts late and finds a
backlog of old probes, nobody is listening for those replies anymore.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .abort_contract import process_boot_id
from .publisher import COMMANDS_STREAM

logger = logging.getLogger(__name__)

PING_METHOD = "bridge.ping"
PING_PROTOCOL_VERSION = 1
SUPPORTED_PING_VERSIONS = frozenset({PING_PROTOCOL_VERSION})
PING_BUDGET_S = 5.0
REQUEST_ID_PREFIX = "ping_"
_REPLY_STREAM_PREFIX = "agent:rpc:"
# Part of the budget kept back from reading so cleanup can still finish inside it.
_CLEANUP_RESERVE_S = 0.25
_MAX_READ_BLOCK_MS = 1000


class PingReplyError(ValueError):
    """A reply that is not a positive answer to this exact probe."""

    def __init__(self, reason: str, *, terminal: bool = False) -> None:
        super().__init__(reason)
        #: True when the addressed backend itself refused: waiting longer is pointless.
        self.terminal = terminal


@dataclass(frozen=True)
class PingProbe:
    """One probe's identity. Fresh per call; never reused."""

    backend: str
    request_id: str
    nonce: str
    expires_at: float

    @classmethod
    def new(cls, backend: str, *, budget_s: float, now: float | None = None) -> "PingProbe":
        if not backend or not backend.strip():
            raise ValueError("ping must be addressed to a backend")
        issued = time.time() if now is None else now
        return cls(
            backend=backend.strip(),
            request_id=f"{REQUEST_ID_PREFIX}{secrets.token_hex(16)}",
            nonce=secrets.token_hex(16),
            expires_at=issued + budget_s,
        )

    def command_fields(self) -> dict[str, str]:
        params = {"v": PING_PROTOCOL_VERSION, "nonce": self.nonce, "expires_at": self.expires_at}
        return {
            "action": "rpc.call",
            "method": PING_METHOD,
            "backend": self.backend,
            "request_id": self.request_id,
            "params": json.dumps(params),
        }

    @property
    def reply_stream(self) -> str:
        return f"{_REPLY_STREAM_PREFIX}{self.request_id}"


@dataclass(frozen=True)
class BridgeReadiness:
    """Probe outcome; the ``/v1/setup/agent-runtime`` body."""

    ready: bool
    backend: str
    instance: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "backend": self.backend, "instance": self.instance, "detail": self.detail}


# -- bridge side ------------------------------------------------------------


def answer_ping(backend: str, request_id: str, params: Any, *, now: float | None = None) -> dict[str, Any] | None:
    """The reply payload for a ping addressed to *backend*, or None to stay silent.

    Silence means the probe expired: its caller has already given up, so a reply
    could only be mistaken for something fresh. A malformed or
    unsupported-version request gets an explicit refusal, so the prober fails
    fast instead of timing out.
    """
    if not isinstance(params, Mapping):
        return {"ok": False, "error": "bridge.ping params must be an object"}
    expires_at = params.get("expires_at")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return {"ok": False, "error": "bridge.ping requires numeric expires_at"}
    if (time.time() if now is None else now) > expires_at:
        logger.info("Dropping expired ping %s", request_id)
        return None
    version = params.get("v")
    if version not in SUPPORTED_PING_VERSIONS:
        return {"ok": False, "error": f"unsupported bridge.ping version: {version!r}"}
    nonce = params.get("nonce")
    if not isinstance(nonce, str) or not nonce or not request_id:
        return {"ok": False, "error": "bridge.ping requires nonce and request_id"}
    return {
        "ok": True,
        "method": PING_METHOD,
        "v": PING_PROTOCOL_VERSION,
        "backend": backend,
        "instance": process_boot_id(),
        "request_id": request_id,
        "nonce": nonce,
    }


# -- app side ---------------------------------------------------------------


def parse_ping_reply(fields: Mapping[str, Any], probe: PingProbe) -> BridgeReadiness:
    """Accept only a positive, identified, versioned answer to *probe*.

    Raises :class:`PingReplyError` for anything else. The error is ``terminal``
    when the addressed backend itself answered with a refusal, such as an
    unknown method (a bridge that predates ping) or an error.
    """
    stamped_backend = str(fields.get("backend") or "").strip()
    stamped_boot = str(fields.get("boot_id") or "").strip()
    if not stamped_backend or not stamped_boot:
        raise PingReplyError("malformed ping reply: missing backend/boot identity")
    if stamped_backend != probe.backend:
        raise PingReplyError(f"reply from wrong backend {stamped_backend!r}")
    try:
        payload = json.loads(fields.get("payload") or "")
    except (TypeError, ValueError):
        raise PingReplyError("malformed ping reply: payload is not JSON", terminal=True) from None
    if not isinstance(payload, dict):
        raise PingReplyError("malformed ping reply: payload is not an object", terminal=True)
    if payload.get("ok") is not True:
        error = str(payload.get("error") or "no error given")[:200]
        raise PingReplyError(f"{probe.backend} bridge rejected ping: {error}", terminal=True)

    def _field(name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise PingReplyError(f"malformed ping reply: missing {name}", terminal=True)
        return value

    if payload.get("method") != PING_METHOD:
        raise PingReplyError("malformed ping reply: not a bridge.ping answer", terminal=True)
    version = payload.get("v")
    if isinstance(version, bool) or version not in SUPPORTED_PING_VERSIONS:
        raise PingReplyError(f"malformed ping reply: unsupported version {version!r}", terminal=True)
    if _field("backend") != probe.backend:
        raise PingReplyError("malformed ping reply: payload backend disagrees with stamp", terminal=True)
    instance = _field("instance")
    if instance != stamped_boot:
        raise PingReplyError("malformed ping reply: instance disagrees with stamped boot id", terminal=True)
    # Not terminal: a replayed or foreign reply says nothing about whether our own
    # answer is still on its way.
    if _field("request_id") != probe.request_id or _field("nonce") != probe.nonce:
        raise PingReplyError("stale or replayed ping reply")
    return BridgeReadiness(
        ready=True,
        backend=probe.backend,
        instance=instance,
        detail=f"{probe.backend} bridge answered ping v{version}",
    )


RedisFactory = Callable[[], Any]


async def probe_bridge(
    redis_factory: RedisFactory,
    backend: str,
    *,
    budget_s: float = PING_BUDGET_S,
) -> BridgeReadiness:
    """Ping *backend*'s command consumer. Never raises; never exceeds *budget_s*.

    ``redis_factory`` returns an async Redis client with ``decode_responses=True``.
    Constructing it, publishing and reading all count against the budget, and so
    does cleanup: the reply stream is deleted, and an unanswered command is
    removed from the stream.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    probe = PingProbe.new(backend, budget_s=budget_s)
    state: dict[str, Any] = {"phase": "connect", "last_rejection": None, "outcome": None}

    def _not_ready(detail: str) -> BridgeReadiness:
        return BridgeReadiness(ready=False, backend=probe.backend, instance=None, detail=detail)

    def _no_reply() -> BridgeReadiness:
        rejection = state["last_rejection"]
        suffix = f" (ignored: {rejection})" if rejection else ""
        return _not_ready(f"no reply from the {probe.backend} bridge within {budget_s:.1f}s{suffix}")

    async def _exchange() -> BridgeReadiness:
        client = redis_factory()
        command_id: str | None = None
        outcome: BridgeReadiness | None = None
        try:
            state["phase"] = "publish"
            command_id = await client.xadd(COMMANDS_STREAM, probe.command_fields(), maxlen=1000, approximate=True)
            state["phase"] = "read"
            read_until = deadline - _CLEANUP_RESERVE_S
            last_id = "0-0"
            while outcome is None:
                remaining_ms = int((read_until - loop.time()) * 1000)
                if remaining_ms <= 0:
                    break
                results = await client.xread(
                    {probe.reply_stream: last_id}, count=10, block=min(remaining_ms, _MAX_READ_BLOCK_MS),
                )
                for _stream, messages in results or ():
                    for msg_id, fields in messages:
                        last_id = msg_id
                        try:
                            outcome = parse_ping_reply(fields, probe)
                        except PingReplyError as exc:
                            state["last_rejection"] = str(exc)
                            if exc.terminal:
                                outcome = _not_ready(str(exc))
                            else:
                                logger.warning("Ignoring ping reply for %s: %s", probe.request_id, exc)
                        if outcome is not None:
                            break
                    if outcome is not None:
                        break
            state["outcome"] = outcome or _no_reply()
            return state["outcome"]
        finally:
            # ``phase`` keeps naming the step that was cut short; cleanup is not a phase
            # that decides the outcome.
            await _cleanup(client, probe, command_id if outcome is None or not outcome.ready else None)

    try:
        async with asyncio.timeout_at(deadline):
            return await _exchange()
    except TimeoutError:
        if state["outcome"] is not None:  # the deadline cut cleanup short, not the answer itself
            return state["outcome"]
        if state["phase"] in {"connect", "publish"}:
            return _not_ready(f"Redis unavailable: timed out during {state['phase']}")
        return _no_reply()
    except Exception as exc:  # noqa: BLE001 — every transport failure is "not ready"
        if state["phase"] in {"connect", "publish"}:
            return _not_ready(f"Redis unavailable: {type(exc).__name__}: {str(exc)[:160]}")
        logger.warning("Ping %s failed during %s: %s", probe.request_id, state["phase"], exc)
        return _not_ready(f"Redis unavailable: {type(exc).__name__}: {str(exc)[:160]}")


async def _cleanup(client: Any, probe: PingProbe, unanswered_command_id: str | None) -> None:
    """Best-effort and bounded by the caller's deadline; ``expires_at`` covers what it misses."""
    try:
        await client.delete(probe.reply_stream)
        if unanswered_command_id:
            await client.xdel(COMMANDS_STREAM, unanswered_command_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ping %s cleanup failed: %s", probe.request_id, exc)
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            try:
                result = close()
                if isinstance(result, Awaitable):
                    await result
            except Exception:  # noqa: BLE001
                pass
