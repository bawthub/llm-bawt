"""Validation and preflight normalization for one Claude bridge send."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import COMMANDS_STREAM

from .compact_policy import apply_compact_default, is_bare_compact
from .compact_prompt_lookup import fetch_compact_prompt
from .send_request import SendRequest

logger = logging.getLogger("claude_code_bridge.bridge")


@dataclass(slots=True)
class SendPreflight:
    request: SendRequest
    message: str
    reset_requested: bool


async def prepare_send(
    bridge: Any,
    fields: dict,
    msg_id: str,
    async_redis: Any,
) -> SendPreflight | None:
    """Validate a send and run its one-time ``/new`` preprocessing."""
    request = SendRequest.from_fields(fields)
    if not request.request_id or not request.message:
        logger.warning("Invalid send command: missing request_id or message")
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        return None

    if not request.model:
        error = (
            "Claude Code bridge: missing 'model' field for "
            f"bot={request.bot_slug or '?'} session={request.session_key}. "
            "Set the bot's Model (default_model) to a claude-code catalog "
            "entry on the bot's profile."
        )
        logger.error(error)
        bridge._publish_event(
            request.request_id,
            request.session_key,
            1,
            kind=AgentEventKind.ERROR,
            text=error,
        )
        bridge._publisher.publish_run_done(request.request_id)
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        return None

    if request.trigger_message_id:
        bridge._trigger_message_ids[request.request_id] = request.trigger_message_id

    reset_requested = (
        not request.explicit_thread
        and request.message.lstrip().startswith("/new")
    )
    message = await bridge._preprocess_new_command(
        request.message,
        explicit_thread=request.explicit_thread,
        bot_slug=request.bot_slug,
        session_key=request.session_key,
        request_id=request.request_id,
        model=request.model,
        context_window=request.bot_context_window,
        inject_messages=request.inject_messages,
        thread_session_id=request.thread_session_id,
        msg_id=msg_id,
        async_redis=async_redis,
    )
    if message is None:
        return None
    # Only a bare command needs a registry request. Custom instructions and
    # ordinary messages remain byte-for-byte untouched, even during an outage.
    guidance = None
    if is_bare_compact(message):
        try:
            guidance = await asyncio.wait_for(
                asyncio.to_thread(fetch_compact_prompt, getattr(bridge, "_app_api_url", "")),
                timeout=2.0,
            )
        except TimeoutError:
            logger.warning("Compaction prompt lookup exceeded 2s; using CLI defaults")
    return SendPreflight(
        request=request,
        message=apply_compact_default(message, guidance),
        reset_requested=reset_requested,
    )
