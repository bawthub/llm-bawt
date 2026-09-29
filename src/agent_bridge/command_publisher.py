"""Redis chat.send command serialization and idempotent publication."""

from __future__ import annotations

import inspect
import json
import logging

from .publisher import COMMANDS_STREAM

logger = logging.getLogger(__name__)


class CommandPublisherMixin:
    """Publish normalized bridge commands through a subscriber's write client."""

    async def send_command(
        self,
        session_key: str,
        message: str,
        request_id: str,
        attachments: list | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        bot_id: str | None = None,
        trigger_message_id: str | None = None,
        effort: str | None = None,
        max_turns: int | None = None,
        subagent_model: str | None = None,
        disallowed_tools: list[str] | None = None,
        inject_messages: list | None = None,
        context_window: int | None = None,
        compact_threshold: int | None = None,
        responses_transport: str | None = None,
        mcp_tool_timeout_ms: int | None = None,
        thread_session_id: str | None = None,
        thread_resume_id: str | None = None,
        explicit_thread: bool = False,
        task_turn_capability: str | None = None,
        skill_bundle: str | None = None,
    ) -> None:
        """Publish one idempotent ``chat.send`` command to the bridge stream."""
        fields: dict = {
            "action": "chat.send",
            "session_key": session_key,
            "message": message,
            "request_id": request_id,
        }
        if skill_bundle is not None:
            from .skill_registry import _name

            fields["skill_bundle"] = _name(skill_bundle)
        if attachments:
            fields["attachments"] = json.dumps(attachments, ensure_ascii=False)
        if system_prompt:
            fields["system_prompt"] = system_prompt
        if model:
            fields["model"] = model
        if backend:
            fields["backend"] = backend
        if bot_id:
            fields["bot_id"] = bot_id
        if trigger_message_id:
            fields["trigger_message_id"] = trigger_message_id
        if effort:
            fields["effort"] = effort
        if max_turns is not None:
            fields["max_turns"] = str(max_turns)
        if subagent_model:
            fields["subagent_model"] = subagent_model
        if context_window is not None and context_window > 0:
            fields["context_window"] = str(context_window)
        if compact_threshold is not None and compact_threshold > 0:
            fields["compact_threshold"] = str(compact_threshold)
        if responses_transport:
            fields["responses_transport"] = responses_transport
        if mcp_tool_timeout_ms is not None and mcp_tool_timeout_ms > 0:
            fields["mcp_tool_timeout_ms"] = str(mcp_tool_timeout_ms)
        if disallowed_tools is not None:
            fields["disallowed_tools"] = json.dumps(
                disallowed_tools, ensure_ascii=False
            )
        if inject_messages is not None:
            fields["inject_messages"] = json.dumps(
                inject_messages, ensure_ascii=False
            )
        if thread_session_id:
            fields["thread_session_id"] = thread_session_id
            if thread_resume_id:
                fields["thread_resume_id"] = thread_resume_id
            if explicit_thread:
                fields["explicit_thread"] = "1"
        if task_turn_capability:
            fields["task_turn_capability"] = task_turn_capability

        dedupe_key = f"agent:command:request:{request_id}"
        flat_fields: list[str] = []
        for key, value in fields.items():
            flat_fields.extend((str(key), str(value)))
        script = """
            local prior = redis.call('GET', KEYS[1])
            if prior then return {0, prior} end
            local args = {}
            for i = 3, #ARGV do table.insert(args, ARGV[i]) end
            local stream_id = redis.call(
                'XADD', KEYS[2], 'MAXLEN', '~', ARGV[1], '*', unpack(args)
            )
            local ttl = 86400
            if string.sub(ARGV[2], 1, 13) == 'req_delivery_' then ttl = 604800 end
            redis.call('SET', KEYS[1], stream_id, 'EX', ttl)
            return {1, stream_id}
        """
        eval_fn = getattr(self._pub_redis, "eval", None)
        if callable(eval_fn):
            eval_result = eval_fn(
                script,
                2,
                dedupe_key,
                COMMANDS_STREAM,
                "1000",
                request_id,
                *flat_fields,
            )
            if inspect.isawaitable(eval_result):
                published, stream_id = await eval_result
            else:
                published, stream_id = eval_result
        else:
            stream_id = await self._pub_redis.xadd(
                COMMANDS_STREAM, fields, maxlen=1000, approximate=True
            )
            published = 1
        logger.debug(
            "%s command: request_id=%s stream_id=%s session=%s attachments=%d",
            "Sent" if int(published) else "Reused",
            request_id,
            stream_id,
            session_key,
            len(attachments or []),
        )
