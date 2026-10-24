"""One agent reset preparation sequence for streaming and nonstreaming turns.

Admission and durable reset-policy rotation already happened under the target
lock. This step assembles continuity before a typed /new rotates its durable
thread, then resolves the exact successor for the provider command.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

from agent_bridge.maintenance import is_new_command

from .intent import Continuity, Origin


@dataclass(frozen=True)
class AgentResetPlan:
    """Execution semantics; durable caller provenance remains on the delivery."""

    continuity: Continuity
    predecessor_session_id: str | None
    rotate_here: bool

    @classmethod
    def from_request(cls, request: Any, user_prompt: str,
                     thread_binding: dict | None) -> AgentResetPlan | None:
        policy = getattr(request, "inter_bot_session_policy", None)
        if policy and policy != "continue":
            modes = {
                "reset_retain_history": Continuity.RETAIN,
                "reset_without_history": Continuity.CLEAN,
            }
            if policy not in modes:
                raise ValueError(f"unsupported reset policy: {policy}")
            return cls(modes[policy],
                       getattr(request, "inter_bot_seed_session_id", None), False)
        if (is_new_command(user_prompt)
                and not (thread_binding or {}).get("explicit_thread")):
            return cls(Continuity.CONFIGURED,
                       (thread_binding or {}).get("thread_session_id"), True)
        return None


@dataclass(frozen=True)
class PreparedNewSession:
    seed_messages: list | None
    thread_binding: dict | None
    rotated_id: str | None
    bridge_reset_action: str | None = None
    bridge_prompt: str | None = None
    # The one scope this configured /new summarized and seeded with; None when
    # no configured /new ran or the bot setting could not be resolved.
    history_scope: str | None = None


@dataclass(frozen=True)
class PreparedChatSession:
    """A claimed stateless-chat reset, bound to its durable successor."""

    session_id: str
    follow_up: str | None


class NewSessionExecution:
    @staticmethod
    def prepare_chat(
        service: Any, llm_bawt: Any, request: Any, *, bot_id: str,
        user_id: str, user_prompt: str,
    ) -> PreparedChatSession | None:
        """Summarize and rotate only a claimed continuous chat command.

        The same durable delivery ID fences rotation and receipt. A selected
        archived thread treats /new as text, just like the agent path.
        """
        if not is_new_command(user_prompt) or getattr(request, "session_id", None):
            return None
        delivery_id = getattr(request, "inter_bot_delivery_id", None)
        if not delivery_id or not getattr(request, "inter_bot_claim_token", None):
            raise ValueError("chat /new requires a claimed durable delivery")
        from ..agent_new_rotation import delivery_new_predecessor
        from ...memory.session_reset_lineage import SessionResetLineage

        backend = getattr(getattr(llm_bawt, "history_manager", None), "_db_backend", None)
        get_session = getattr(backend, "get_session", None)
        if not callable(get_session):
            raise ValueError("chat /new cannot inspect its keyed successor")
        already_rotated = get_session(SessionResetLineage.delivery_successor_id(delivery_id)) is not None
        predecessor = delivery_new_predecessor(
            llm_bawt, bot_id=bot_id, user_id=user_id, delivery_id=delivery_id,
        )
        if predecessor is None and not already_rotated:
            getter = getattr(backend, "get_active_session", None)
            if not callable(getter):
                raise ValueError("chat /new cannot resolve its outgoing thread")
            active = getter(bot_id=bot_id, user_id=user_id)
            predecessor = (active or {}).get("id")
        if predecessor and not already_rotated:
            service._maybe_summarize_on_new(
                llm_bawt, bot_id, user_prompt,
                thread_binding={"thread_session_id": predecessor},
            )
        successor = service._rotate_chat_session(
            llm_bawt, bot_id, idempotency_key=delivery_id,
            working_memory_mode=NewSessionExecution.working_memory_mode(service, delivery_id),
        )
        if successor != SessionResetLineage.delivery_successor_id(delivery_id):
            raise RuntimeError("chat /new did not bind its keyed successor")
        return PreparedChatSession(
            session_id=successor,
            follow_up=NewSessionExecution.typed_follow_up(user_prompt),
        )

    @staticmethod
    def working_memory_mode(service: Any, delivery_id: str | None) -> str:
        """The successor mode of a durable /new, from its own persisted intent.

        Read like ``scheduled_history_scope``: only the server-stamped
        ``new_session_intent`` decides (docs/WORKING_MEMORY.md 10.1), never a
        request field. A /new without a delivery, or a bare harness client
        without a durable store, is a human /new and pins (N1).
        """
        from ...working_memory.session_state import PINNED
        from .lifecycle import NewSessionLifecycle

        if not delivery_id:
            return PINNED
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None:
            return PINNED
        record = dispatcher.store.get(delivery_id)
        if record is None:
            raise ValueError("durable /new delivery is missing")
        return NewSessionLifecycle.working_memory_mode(record.metadata, record.session_policy)

    @staticmethod
    def scheduled_history_scope(service: Any, delivery_id: str | None) -> str | None:
        """A scheduled /new's continuity override, from its own durable intent.

        The scope comes only from the server-stamped ``new_session_intent``
        policy snapshot persisted at admission, never from request fields, so
        a later parameter edit or a public caller cannot change what an
        admitted reset carries. Every non-scheduler /new returns ``None``.
        """
        if not delivery_id:
            return None
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None:
            # Bare harness clients have no durable store; deployed /new
            # deliveries are always dispatched through one.
            return None
        record = dispatcher.store.get(delivery_id)
        if record is None:
            raise ValueError("durable /new delivery is missing")
        raw = record.metadata.get("new_session_intent")
        if not isinstance(raw, dict) or raw.get("origin") != Origin.SCHEDULER:
            return None
        from .lifecycle import NewSessionLifecycle

        parameters = NewSessionLifecycle._decode(raw).scheduler_parameters
        return parameters.history_scope_override if parameters else None

    @staticmethod
    def typed_follow_up(user_prompt: str) -> str | None:
        """Parse an actual /new command, never a /newspaper-style task."""
        stripped = (user_prompt or "").lstrip()
        if not is_new_command(stripped):
            return None
        return stripped[len("/new"):].strip() or None

    @staticmethod
    def bridge_messages(
        messages: list, bridge_prompt: str | None, original_command: str,
    ) -> list:
        """Keep the visible command and any model-only prefix/suffix separate."""
        if bridge_prompt is None:
            return messages
        if not messages or getattr(messages[-1], "role", None) != "user":
            raise ValueError("typed reset follow-up has no outbound user message")
        task = copy(messages[-1])
        start = task.content.find(original_command)
        if start < 0:
            raise ValueError("typed reset follow-up lost its original command")
        task.content = (task.content[:start] + bridge_prompt
                        + task.content[start + len(original_command):])
        return [*messages[:-1], task]

    @staticmethod
    def prepare_agent(
        service: Any, llm_bawt: Any, request: Any, *, bot_id: str,
        user_prompt: str, model_alias: str, thread_binding: dict | None,
        delivery_id: str | None, backend_name: str,
        prompt_record: Any = None, is_agent: bool = True,
    ) -> PreparedNewSession:
        plan = AgentResetPlan.from_request(request, user_prompt, thread_binding)
        typed_task = (
            NewSessionExecution.typed_follow_up(user_prompt)
            if delivery_id and getattr(request, "inter_bot_typed_follow_up", False)
            and plan and plan.rotate_here else None
        )
        if getattr(request, "inter_bot_typed_follow_up", False) and typed_task is None:
            raise ValueError("typed reset follow-up has invalid admission provenance")
        if typed_task is not None and backend_name not in {"claude-code", "codex"}:
            raise ValueError("typed reset follow-up needs a supported agent bridge")
        if plan and backend_name == "codex" and plan.continuity is Continuity.RETAIN:
            raise ValueError("native Codex cannot retain reset history")
        # Resolve one concrete scope per configured /new and hand the same value
        # to the summarize gate, the seed and the continuity receipt, so a
        # concurrent bot-setting edit cannot split them. Only a scheduled reset
        # can override the bot's scope. If the bot setting cannot be resolved to
        # a canonical scope, None keeps each consumer's established fallback.
        reset_scope = None
        if plan and plan.continuity is Continuity.CONFIGURED and not prompt_record:
            from ...setting_definitions import HISTORY_SCOPES
            from ...utils.history import effective_history_scope

            reset_scope = NewSessionExecution.scheduled_history_scope(service, delivery_id)
            try:
                resolved = effective_history_scope(llm_bawt.config_resolver, reset_scope)
            except Exception:
                resolved = None
            if resolved in HISTORY_SCOPES:
                reset_scope = resolved
        # Both configured /new and retained bot resets summarize the outgoing
        # thread before building a seed. The latter was already rotated at claim,
        # so point the summarizer at the fenced predecessor rather than the new
        # active session. Clean resets never inject either raw or summary context.
        if (plan and plan.continuity is not Continuity.CLEAN
                and (plan.rotate_here or plan.predecessor_session_id)):
            predecessor = (
                {"thread_session_id": plan.predecessor_session_id}
                if plan.predecessor_session_id else thread_binding
            )
            service._maybe_summarize_on_new(
                llm_bawt, bot_id, "/new", thread_binding=predecessor,
                history_scope=reset_scope,
            )
        from ..routes.history import build_context_seed, maybe_build_session_seed

        if prompt_record:
            from ..prompt_execution import build_prompt_seed

            seed = build_prompt_seed(service, llm_bawt, thread_binding)
        elif plan and plan.continuity is Continuity.RETAIN:
            seed = (
                build_context_seed(
                    bot_id, model_alias, service,
                    session_id=plan.predecessor_session_id, llm_bawt=llm_bawt,
                ).get("messages", [])
                if plan.predecessor_session_id else []
            )
        elif plan and plan.continuity is Continuity.CLEAN:
            seed = []
        else:
            seed = maybe_build_session_seed(
                llm_bawt, bot_id, model_alias, user_prompt, service,
                thread_binding=thread_binding,
                history_scope=reset_scope,
            )
        if (is_agent and plan and backend_name == "codex"
                and plan.rotate_here and (user_prompt.strip() == "/new" or typed_task)
                and seed):
            raise ValueError("native Codex cannot seed configured reset history")
        if typed_task and backend_name == "codex":
            seed = []
        rotated = None
        if is_agent and (plan is None or plan.rotate_here):
            rotated = service._maybe_rotate_agent_session(
                llm_bawt, bot_id, user_prompt, thread_binding=thread_binding,
                **({"delivery_id": delivery_id} if delivery_id else {}),
            )
        if is_agent and plan and plan.rotate_here:
            from ..agent_new_rotation import bind_new_successor

            thread_binding = bind_new_successor(
                service, llm_bawt, rotated_id=rotated,
                delivery_id=delivery_id, backend_name=backend_name,
            )
        # The provider must not infer a policy reset from the follow-up text.
        # This action rides only the request-local bridge command, never the
        # shared bot config or model-visible prompt. Legacy /new-with-follow-up
        # retains its existing textual path until its reply contract migrates.
        bridge_action = None
        if is_agent and plan:
            standalone_policy = bool(
                getattr(request, "inter_bot_delivery_id", None)
                and getattr(request, "inter_bot_reset_only", False)
                and not plan.rotate_here
            )
            bare_command = plan.rotate_here and user_prompt.strip() == "/new"
            if (standalone_policy or bare_command) and backend_name in {"claude-code", "codex"}:
                bridge_action = "reset_only"
                if backend_name == "codex":
                    # Native Codex can prove an empty fresh binding only; refuse
                    # configured history above rather than silently dropping it.
                    seed = []
            elif (typed_task is not None or not plan.rotate_here) and backend_name in {"claude-code", "codex"}:
                bridge_action = "reset_follow_up"
        return PreparedNewSession(seed, thread_binding, rotated, bridge_action, typed_task,
                                  reset_scope)
