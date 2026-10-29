"""Redis chat.send command serialization and idempotent publication."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping, Sequence

from .continuity import SEED_FRAMING_FIELD, SeedFraming
from .publisher import COMMANDS_STREAM

logger = logging.getLogger(__name__)

# How long a request id stays published-once (the dedupe key's TTL). A bridge's
# request tombstone lives exactly as long, since that is the window in which the
# same request id can't be published again (TASK-1232 P2).
COMMAND_DEDUPE_TTL_S = 86400
DELIVERY_REQUEST_PREFIX = "req_delivery_"
DELIVERY_COMMAND_DEDUPE_TTL_S = 604800

# TASK-1318: ``system_prompt_delivery`` for a session-pinned prompt on a bridge
# that stores the prompt in thread history (codex). "now": the pin moved this
# turn, deliver it even on a resumed thread. "cold_start": deliver it only when
# the bridge opens a fresh thread. Absent: the historical every-turn delivery.
SYSTEM_PROMPT_DELIVERY_NOW = "now"
SYSTEM_PROMPT_DELIVERY_COLD_START = "cold_start"
SYSTEM_PROMPT_DELIVERY_MODES = frozenset({SYSTEM_PROMPT_DELIVERY_NOW, SYSTEM_PROMPT_DELIVERY_COLD_START})


def command_dedupe_ttl_s(request_id: str) -> int:
    """The dedupe TTL that :meth:`CommandPublisherMixin.send_command` applies."""
    if request_id.startswith(DELIVERY_REQUEST_PREFIX):
        return DELIVERY_COMMAND_DEDUPE_TTL_S
    return COMMAND_DEDUPE_TTL_S


class CommandPublisherMixin:
    """Publish normalized bridge commands through a subscriber's write client."""

    async def send_command(
        self,
        session_key: str,
        message: str,
        request_id: str,
        attachments: list | None = None,
        system_prompt: str | None = None,
        system_prompt_delivery: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        bot_id: str | None = None,
        trigger_message_id: str | None = None,
        effort: str | None = None,
        max_turns: int | None = None,
        subagent_model: str | None = None,
        disallowed_tools: list[str] | None = None,
        inject_messages: list | None = None,
        inject_seed_framing: SeedFraming | None = None,
        bridge_reset_action: str | None = None,
        context_window: int | None = None,
        compact_threshold: int | None = None,
        responses_transport: str | None = None,
        codex_transport: str | None = None,
        mcp_tool_timeout_ms: int | None = None,
        question_answer_wait_seconds: int | None = None,
        prompt_cache_ttl: str | None = None,
        thread_session_id: str | None = None,
        thread_resume_id: str | None = None,
        explicit_thread: bool = False,
        task_turn_capability: str | None = None,
        skill_bundle: str | None = None,
        access_path: str | None = None,
        access_path_provider: str | None = None,
        skill_selection: Mapping | None = None,
        injected_skills: Sequence[str] | None = None,
    ) -> None:
        """Publish one idempotent ``chat.send`` command to the bridge stream.

        ``access_path`` is the resolved catalog endpoint's access-path key
        (TASK-1295). It is published whenever it is not None, so a blank or
        unknown value reaches the bridge and is rejected there rather than
        being dropped into legacy credential selection.
        ``access_path_provider`` is the proxy provider the catalog derives
        from that path; the bridge refuses a proxy turn whose model prefix
        differs from it.
        """
        fields: dict = {
            "action": "chat.send",
            "session_key": session_key,
            "message": message,
            "request_id": request_id,
        }
        if skill_bundle is not None:
            from .skill_registry import _name

            fields["skill_bundle"] = _name(skill_bundle)
        if skill_selection is not None:  # TASK-1309: validated before it is published
            from .skill_selection import SELECTION_FIELD, ManagedSelection, SkillSelectionError

            if skill_bundle is not None:
                raise SkillSelectionError("a skill bundle and a managed selection are exclusive")
            fields[SELECTION_FIELD] = ManagedSelection.from_manifest(skill_selection).to_field()
        if injected_skills:  # TASK-1691: skills whose bodies the system prompt carries
            from .skill_selection import INJECTED_FIELD, injected_field

            fields[INJECTED_FIELD] = injected_field(injected_skills)
        if attachments:
            fields["attachments"] = json.dumps(attachments, ensure_ascii=False)
        if system_prompt:
            fields["system_prompt"] = system_prompt
        if system_prompt_delivery is not None:
            if system_prompt_delivery not in SYSTEM_PROMPT_DELIVERY_MODES:
                raise ValueError("Unsupported system prompt delivery mode")
            fields["system_prompt_delivery"] = system_prompt_delivery
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
        if codex_transport:  # TASK-1243: codex exec or app-server
            fields["codex_transport"] = codex_transport
        if mcp_tool_timeout_ms is not None and mcp_tool_timeout_ms > 0:
            fields["mcp_tool_timeout_ms"] = str(mcp_tool_timeout_ms)
        if question_answer_wait_seconds is not None:
            fields["question_answer_wait_seconds"] = str(question_answer_wait_seconds)
        if prompt_cache_ttl:
            fields["prompt_cache_ttl"] = str(prompt_cache_ttl)
        if disallowed_tools is not None:
            fields["disallowed_tools"] = json.dumps(
                disallowed_tools, ensure_ascii=False
            )
        if inject_messages is not None:
            fields["inject_messages"] = json.dumps(
                inject_messages, ensure_ascii=False
            )
            # TASK-1579: registry framing rides only with a non-empty seed.
            if inject_messages and inject_seed_framing is not None:
                fields[SEED_FRAMING_FIELD] = inject_seed_framing.to_field()
        if bridge_reset_action:
            supported = (
                backend in {"claude-code", "codex"}
                and bridge_reset_action in {"reset_only", "reset_follow_up"}
            )
            if not supported:
                raise ValueError("Unsupported agent bridge reset action")
            fields["bridge_reset_action"] = bridge_reset_action
        if thread_session_id:
            fields["thread_session_id"] = thread_session_id
            if thread_resume_id:
                fields["thread_resume_id"] = thread_resume_id
            if explicit_thread:
                fields["explicit_thread"] = "1"
        if task_turn_capability:
            fields["task_turn_capability"] = task_turn_capability
        if access_path is not None:
            fields["access_path"] = access_path
        if access_path_provider:
            fields["access_path_provider"] = access_path_provider

        dedupe_key = f"agent:command:request:{request_id}"
        flat_fields: list[str] = []
        for key, value in fields.items():
            flat_fields.extend((str(key), str(value)))
        script = f"""
            local prior = redis.call('GET', KEYS[1])
            if prior then return {{0, prior}} end
            local args = {{}}
            for i = 3, #ARGV do table.insert(args, ARGV[i]) end
            local stream_id = redis.call(
                'XADD', KEYS[2], 'MAXLEN', '~', ARGV[1], '*', unpack(args)
            )
            local ttl = {COMMAND_DEDUPE_TTL_S}
            if string.sub(ARGV[2], 1, {len(DELIVERY_REQUEST_PREFIX)}) == '{DELIVERY_REQUEST_PREFIX}' then ttl = {DELIVERY_COMMAND_DEDUPE_TTL_S} end
            redis.call('SET', KEYS[1], stream_id, 'EX', ttl)
            return {{1, stream_id}}
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
