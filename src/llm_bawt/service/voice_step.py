"""Read one proven post-injection assistant step; never replay accepted speech."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from .delivery_reply import resolve_delivery_reply

VOICE_STEP_TIMEOUT_SECONDS = 60.0
MAX_STEP_UNITS = 65536


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
            return True, "".join(self.parts).strip() or None
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


async def observe_voice_step(service: Any, request: Any, receipt: dict, reader: Any,
                             *, deadline: float) -> str | None:
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
                    return resolved.get("response_text") or None
                if resolved.get("status") not in {"QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"}:
                    return None
                await asyncio.sleep(min(0.25, max(0, deadline - loop.time())))
            step = VoiceStep(binding, receipt["user_message_id"], bot, user)
            async with aclosing(reader.events(timeout_s=max(0, deadline - loop.time()))) as events:
                async for event in events:
                    if event is not None:
                        done, text = step.accept(event)
                        if done:
                            return text
    except Exception:
        # Database/reader failures are observation failures, not failed work.
        # CancelledError still propagates; never swallow call cancellation.
        return None
    return None
