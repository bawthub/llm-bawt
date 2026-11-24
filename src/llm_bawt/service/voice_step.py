"""Read one proven post-injection assistant step; never replay accepted speech."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .delivery_reply import resolve_delivery_reply

VOICE_STEP_TIMEOUT_SECONDS = 60.0
MAX_STEP_UNITS = 65536


@dataclass(frozen=True)
class VoiceReply:
    text: str
    animation: str | None = None


@dataclass
class VoiceStep:
    """Fail-closed UTF-16 frontier for a single ordered consumption barrier."""

    binding: dict
    message_id: str
    bot_id: str
    user_id: str
    offset: int | None = None
    parts: list[str] = field(default_factory=list)
    units: int = 0

    def accept(self, event: dict) -> tuple[bool, str | None]:
        """Return (finished, text); None means no provable spoken answer."""
        if (event.get("turn_id") != self.binding["turn_id"]
                or event.get("bot_id") != self.bot_id or event.get("user_id") != self.user_id):
            return False, None
        kind = event.get("_type")
        if kind == "steer_lifecycle":
            exact = (event.get("message_id") == self.message_id
                     and event.get("run_request_id") == self.binding["run_request_id"]
                     and event.get("steer_rpc_id") == self.binding["steer_rpc_id"])
            if exact and event.get("state") == "started":
                position = event.get("text_offset")
                if type(position) is not int or position < 0:
                    return True, None
                if self.offset is None:
                    self.offset = position
                # Duplicate started must not reset the collected segment.
                return False, None
            if exact and event.get("state") in {"cancelled", "rejected", "uncertain"}:
                return True, None
            if self.offset is not None and event.get("state") == "started" and not exact:
                return True, None  # Another injection makes attribution ambiguous.
        boundary = ((kind == "tool_event" and event.get("event") == "tool_start"
                     and not event.get("parent_tool_use_id")) or kind == "turn_complete")
        if boundary and self.offset is not None:
            if event.get("text_offset") != self.offset:
                return True, None  # Lost tail delta, even if earlier text was valid.
            if kind == "turn_complete" and event.get("status") not in {"ok", "completed"}:
                return True, None
            text = "".join(self.parts).strip()
            if not text and kind == "tool_event":
                # TASK-2428: an agent usually runs a tool before it answers.
                # A tool start before any spoken text is not the step's end;
                # keep reading until the answer's text reaches a boundary.
                return False, None
            return True, text or None
        if kind == "turn_complete":
            return True, None  # Finished without a consumption barrier.
        if kind != "text_delta" or self.offset is None:
            return False, None
        start, delta = event.get("text_offset"), event.get("delta")
        if type(start) is not int or start < 0 or not isinstance(delta, str) or start > self.offset:
            return True, None
        try:
            raw = delta.encode("utf-16-le")
            end = start + len(raw) // 2
            if end <= self.offset:
                return False, None
            tail = raw[(self.offset - start) * 2:].decode("utf-16-le")
        except UnicodeError:
            return True, None
        self.units += end - self.offset
        if self.units > MAX_STEP_UNITS:
            return True, None
        self.parts.append(tail)
        self.offset = end
        return False, None


@dataclass
class FreshTurnText:
    """Contiguous UTF-16 frontier of one fresh turn's live text (TASK-2612).

    The turn's own text_delta events are released as they arrive so a call
    speaks each block while the agent keeps working. Any gap or malformed
    delta stops live release; the caller repairs from the persisted reply.
    """

    turn_id: str
    bot_id: str
    user_id: str
    offset: int = 0
    spoken: str = ""

    def accept(self, event: dict) -> tuple[bool, str]:
        """Return (stop, new_text)."""
        if (event.get("turn_id") != self.turn_id
                or event.get("bot_id") != self.bot_id or event.get("user_id") != self.user_id):
            return False, ""
        kind = event.get("_type")
        if kind == "turn_complete":
            return True, ""
        if kind != "text_delta":
            return False, ""
        start, delta = event.get("text_offset"), event.get("delta")
        if type(start) is not int or start < 0 or not isinstance(delta, str) or start > self.offset:
            return True, ""
        try:
            raw = delta.encode("utf-16-le")
            end = start + len(raw) // 2
            if end <= self.offset:
                return False, ""
            tail = raw[(self.offset - start) * 2:].decode("utf-16-le")
        except UnicodeError:
            return True, ""
        if end > MAX_STEP_UNITS:
            return True, ""
        self.offset = end
        self.spoken += tail
        return False, tail


FRESH_DURABLE_CHECK_SECONDS = 1.0


async def _stream_fresh_turn(reader: Any, fresh: FreshTurnText, on_text: Callable[[str], None],
                             *, deadline: float,
                             still_running: Callable[[], Awaitable[bool]]) -> None:
    """Release live text; stop when Redis ends it or the durable delivery settles.

    A missed terminal event must not hide a persisted reply, so the durable
    delivery is re-checked at least every FRESH_DURABLE_CHECK_SECONDS.
    """
    loop = asyncio.get_running_loop()
    next_check = loop.time() + FRESH_DURABLE_CHECK_SECONDS
    async with aclosing(reader.events(timeout_s=max(0, deadline - loop.time()))) as events:
        async for event in events:
            if loop.time() >= next_check or event is None:
                if not await still_running():
                    return
                next_check = loop.time() + FRESH_DURABLE_CHECK_SECONDS
            if event is None:
                continue
            stop, text = fresh.accept(event)
            if text:
                on_text(text)
            if stop:
                return


def _unspoken_remainder(text: str, spoken: str) -> str | None:
    """Text still owed after live release; None if the two disagree."""
    if not spoken:
        return text
    return text[len(spoken):] if text.startswith(spoken) else None


async def observe_voice_step(service: Any, request: Any, receipt: dict, reader: Any,
                             *, deadline: float,
                             on_text: Callable[[str], None] | None = None) -> VoiceReply | None:
    """Wait for durable identity, then consume the cursor claimed BEFORE admission.

    Lifecycle may precede delivery settlement; Redis retains those events while
    the exact injected message/session is verified. No whole-turn reply fallback.
    Closing this observer never closes the shared Redis client or aborts a run.
    """
    if reader is None and receipt.get("busy") is True:
        return None
    loop = asyncio.get_running_loop()
    deliveries = getattr(getattr(service, "_inter_bot_dispatcher", None), "store", None)
    if deliveries is None or not receipt.get("delivery_id"):
        return None
    bot = receipt.get("bot_id") or request.bot_id or service._default_bot
    user = receipt.get("user_id") or request.user or service.config.DEFAULT_USER
    fresh: FreshTurnText | None = None
    try:
        async with asyncio.timeout_at(deadline):
            while True:
                resolved = await asyncio.to_thread(
                    resolve_delivery_reply, deliveries, service._turn_log_store, receipt["delivery_id"],
                    bot_id=bot, user_id=user, user_message_id=receipt["user_message_id"],
                    include_voice_binding=True,
                )
                if resolved is None:
                    return None
                binding = resolved.get("voice_binding")
                if binding and binding.get("run_request_id") and binding.get("steer_rpc_id"):
                    if reader is None:
                        return None
                    break
                if (not binding and resolved.get("status") == "DELIVERED"
                        and resolved.get("voice_direct_reply") is True):
                    # A verified fresh turn is the request's OWN answer. Route
                    # by durable delivery, not the advisory busy snapshot: an
                    # idle admission may become a steer before claim (or vice
                    # versa). Never return a steered target's whole response.
                    text = resolved.get("response_text") or ""
                    if fresh is not None:
                        text = _unspoken_remainder(text, fresh.spoken)
                        if text is None:
                            return None  # Never re-speak or splice a mismatched reply.
                    return VoiceReply(text, resolved.get("animation")) if text else None
                if resolved.get("status") not in {"QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"}:
                    return None
                if (fresh is None and on_text is not None and reader is not None and not binding
                        and resolved.get("voice_direct_reply") is True
                        and resolved.get("status") in {"DISPATCHING", "IN_FLIGHT"}
                        and isinstance(resolved.get("turn_id"), str) and resolved["turn_id"]):
                    # The request's own fresh turn is running: speak its text as
                    # it streams, then wait for the persisted reply's remainder.
                    fresh = FreshTurnText(resolved["turn_id"], bot, user)
                    async def still_running() -> bool:
                        current = await asyncio.to_thread(
                            resolve_delivery_reply, deliveries, service._turn_log_store,
                            receipt["delivery_id"], bot_id=bot, user_id=user,
                            user_message_id=receipt["user_message_id"], include_voice_binding=True,
                        )
                        return bool(current) and current.get("status") in {
                            "QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"}

                    await _stream_fresh_turn(reader, fresh, on_text, deadline=deadline,
                                             still_running=still_running)
                    continue
                await asyncio.sleep(min(0.25, max(0, deadline - loop.time())))
            step = VoiceStep(binding, receipt["user_message_id"], bot, user)
            async with aclosing(reader.events(timeout_s=max(0, deadline - loop.time()))) as events:
                async for event in events:
                    if event is not None:
                        done, text = step.accept(event)
                        if done:
                            # A step can end at a tool boundary while its target
                            # turn continues. That turn's animation is not a
                            # classification of this injected speech segment.
                            return VoiceReply(text) if text else None
    except Exception:
        # Database/reader failures are observation failures, not failed work.
        # CancelledError still propagates; never swallow call cancellation.
        return None
    return None
