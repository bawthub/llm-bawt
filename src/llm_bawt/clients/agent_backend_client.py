"""LLM client shell for agent backends (e.g. OpenClaw).

This wraps an :class:`AgentBackend` as a standard :class:`LLMClient` so that
the entire pipeline (history, memory, extraction, summarization, tool-call
SSE events) flows through the normal ``ServiceLLMBawt`` path.

The model alias for these virtual models is the backend name itself
(e.g. ``openclaw``).  A synthetic model definition with
``type: "agent_backend"`` is injected into ``config.defined_models`` at
service startup for each bot that declares ``agent_backend``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, List

from ..utils.config import Config
from ..models.message import Message
from .base import LLMClient

logger = logging.getLogger(__name__)


def _attachments_from_content_parts(
    content_parts: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Convert a :class:`Message`'s multimodal ``content_parts`` into the
    agent-backend ``attachments`` contract.

    ``content_parts`` are OpenAI image_url parts
    (``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}``,
    produced by ``prepare_messages_for_query``); the agent bridge wants
    ``{"mimeType": <mime>, "content": <naked-b64>}``.  Keeping this conversion
    here lets ``Message.content_parts`` be the single source of truth for
    images on the agent path — non-image parts and malformed / non-base64
    data URLs are skipped.
    """
    out: list[dict[str, str]] = []
    for part in content_parts or []:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        url = (part.get("image_url") or {}).get("url", "")
        if not isinstance(url, str) or not url.startswith("data:"):
            continue
        try:
            header, payload = url.split(",", 1)
            mime = header.split(":", 1)[1].split(";", 1)[0]
        except (IndexError, ValueError):
            continue
        if payload:
            out.append({"mimeType": mime, "content": payload})
    return out


class AgentBackendClient(LLMClient):
    """LLMClient that delegates to an external agent backend.

    ``query()`` runs the backend's ``chat()`` coroutine synchronously
    (via ``asyncio.run`` or the running loop's executor) and returns the
    text response.  Structured metadata — tool calls, model info, usage —
    is stashed in ``last_result`` for the caller to inspect after the
    query completes.
    """

    SUPPORTS_STREAMING = True

    def __init__(
        self,
        backend_name: str,
        config: Config,
        bot_config: dict[str, Any] | None = None,
        model_definition: dict | None = None,
    ):
        model_definition = model_definition or {
            "type": "agent_backend",
            "backend": backend_name,
        }
        super().__init__(
            model=backend_name,
            config=config,
            model_definition=model_definition,
        )
        self._backend_name = backend_name
        self._bot_config: dict[str, Any] = bot_config or {}
        self._backend = self._load_backend(backend_name)

        # Populated after each query()
        self.last_result: Any | None = None

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def query(
        self,
        messages: List[Message],
        plaintext_output: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send the last user message to the agent backend."""
        # Extract the last user message as the prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt = msg.content or ""
                break

        if not prompt:
            logger.warning("AgentBackendClient.query() called with no user message")
            return ""

        # TASK-501: non-streaming seed history to push to the bridge (mirrors
        # stream_raw's inject_messages). Threaded into the config _chat_full
        # hands the backend.
        inject_messages = kwargs.pop("inject_messages", None)
        # TASK-252: request-local per-thread SDK binding (explicit-thread turn).
        thread_binding = kwargs.pop("thread_binding", None)
        bridge_request_id = kwargs.pop("bridge_request_id", None)
        bridge_timeout_seconds = kwargs.pop("bridge_timeout_seconds", None)
        bridge_event_callback = kwargs.pop("bridge_event_callback", None)
        # TASK-1015: same request-local identity the streaming path threads.
        # Without turn_execution the bridge request id was never persisted, so
        # Stop/steer on a non-stream turn could not target it.
        request_local = {
            key: value
            for key, value in {
                "turn_execution": kwargs.pop("turn_execution", None),
                "task_turn_capability": kwargs.pop("task_turn_capability", None),
            }.items()
            if value is not None
        }

        # Run the async backend call synchronously.
        # If there's already a running loop we schedule via run_in_executor.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    self._chat_full(
                        prompt,
                        inject_messages,
                        thread_binding,
                        bridge_request_id,
                        bridge_timeout_seconds,
                        bridge_event_callback,
                        request_local,
                    ),
                ).result()
        else:
            result = asyncio.run(
                self._chat_full(
                    prompt,
                    inject_messages,
                    thread_binding,
                    bridge_request_id,
                    bridge_timeout_seconds,
                    bridge_event_callback,
                    request_local,
                )
            )

        self.last_result = result
        return result.text if hasattr(result, "text") else str(result)

    def get_styling(self) -> tuple[str | None, str]:
        return self.bot_name, "cyan"

    def stream_raw(self, messages: List[Message], **kwargs: Any) -> Iterator[str | dict[str, Any]]:
        """Stream from backend when supported; fallback to one-shot query."""
        prompt = ""
        last_user_msg: Message | None = None
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_msg = msg
                content = msg.content
                if isinstance(content, list):
                    prompt = "".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                else:
                    prompt = content or ""
                break

        if not prompt:
            logger.warning("AgentBackendClient.stream_raw() called with no user message")
            return

        # Image attachments. The explicit ``attachments`` kwarg is the fast
        # path (chat_streaming passes the resolved user_attachments straight
        # through), but the user Message's ``content_parts`` is the
        # authoritative carrier: derive attachments from it whenever the kwarg
        # is absent, so images can't silently vanish if a caller builds a
        # multimodal Message without also threading the kwarg through. Both
        # originate from the same user_attachments list, so this never
        # double-counts — it only recovers the dropped case.
        attachments: list = kwargs.pop("attachments", None) or []
        if not attachments and last_user_msg is not None:
            attachments = _attachments_from_content_parts(last_user_msg.content_parts)
            if attachments:
                logger.debug(
                    "stream_raw: recovered %d image attachment(s) from "
                    "Message.content_parts (no attachments kwarg passed)",
                    len(attachments),
                )
        # Frontend-supplied user-message UUID (or "local-user-*" placeholder).
        # Bridges stamp this on every emitted tool event so the frontend can
        # bucket tool activity under the originating user message without
        # relying on the brittle turn_id / activeStreamMessageId fallback chain.
        trigger_message_id: str | None = kwargs.pop("trigger_message_id", None)

        # TASK-501: history the app pre-assembled for a fresh SDK session
        # (seed). Merged into config so the bridge receives it in the Redis
        # command and seeds WITHOUT calling back to /v1/history/context-seed.
        inject_messages = kwargs.pop("inject_messages", None)
        # TASK-252: request-local per-thread SDK binding (explicit-thread
        # turn). Merged into the per-call config COPY below — never written to
        # the shared _bot_config, so concurrent turns cannot cross-bind.
        thread_binding = kwargs.pop("thread_binding", None)
        # TASK-701: opaque trusted current-turn capability. Request-local only;
        # never persist it on shared bot config or expose its identifiers.
        task_turn_capability = kwargs.pop("task_turn_capability", None)
        bridge_request_id = kwargs.pop("bridge_request_id", None)

        # Extract system prompt from messages for backends that support it
        # (e.g. claude-code bridge). Merge into config so the backend can
        # pass it through in the Redis command.
        config = self._with_question_wait(dict(self._bot_config))
        system_parts = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content if isinstance(msg.content, str) else "")
        if system_parts:
            config["system_prompt"] = "\n\n".join(p for p in system_parts if p)
        if inject_messages is not None:
            config["inject_messages"] = inject_messages
        if thread_binding:
            config.update(thread_binding)
        if task_turn_capability:
            config["task_turn_capability"] = task_turn_capability
        if bridge_request_id:
            config["request_id"] = bridge_request_id

        execution = kwargs.pop("turn_execution", None)
        if execution is not None:
            config["turn_execution"] = execution

        if hasattr(self._backend, "stream_raw"):
            backend_kwargs: dict[str, Any] = {"attachments": attachments}
            # Only forward trigger_message_id to backends that accept it
            # (OpenClaw + subclasses do; older custom backends may not).
            try:
                import inspect
                sig = inspect.signature(self._backend.stream_raw)
                if "trigger_message_id" in sig.parameters:
                    backend_kwargs["trigger_message_id"] = trigger_message_id
            except (TypeError, ValueError):
                # Builtins / C-extensions may not have a signature; pass it
                # anyway and let the backend ignore unknown kwargs if it can.
                backend_kwargs["trigger_message_id"] = trigger_message_id
            try:
                for item in self._backend.stream_raw(prompt, config, **backend_kwargs):
                    yield item
            finally:
                # Interrupted/error streams still finalize a structured backend
                # result with usage-so-far. Preserve it even when iteration raises
                # before the normal post-loop assignment (TASK-304).
                if hasattr(self._backend, "get_last_stream_result"):
                    self.last_result = self._backend.get_last_stream_result()
            return

        response = self.query(messages, plaintext_output=True, **kwargs)
        if response:
            yield response

    # effective_context_window / effective_max_tokens: intentionally NOT
    # overridden (TASK-609). The old flat 128_000 / 16_384 constants were a
    # third, model-blind budget authority; all bots are agent bots, so the
    # chat-turn path always hit them. The base LLMClient properties now delegate
    # to config.get_model_context_window / get_model_max_tokens (catalog-driven),
    # giving agent clients the correct per-model window/capability too.

    # ------------------------------------------------------------------
    # Tool call helpers (used by background_service streaming path)
    # ------------------------------------------------------------------

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Return tool calls from the last query as dicts for SSE broadcast.

        Each dict has ``name``, ``arguments``, and ``display_name``.
        """
        result = self.last_result
        if result is None:
            return []

        tool_calls = getattr(result, "tool_calls", [])
        return [
            {
                "name": tc.name,
                "display_name": getattr(tc, "display_name", tc.name),
                "arguments": tc.arguments,
                "result": getattr(tc, "result", None),
            }
            for tc in tool_calls
        ]

    def get_upstream_model(self) -> str:
        """Return the actual model used by the upstream agent (e.g. gpt-5.3-codex)."""
        result = self.last_result
        if result is None:
            return self._backend_name
        return getattr(result, "model", "") or self._backend_name

    def get_token_usage(self) -> dict[str, Any] | None:
        """Return normalized usage from the last structured backend result."""
        result = self.last_result
        usage = getattr(result, "usage", None) if result is not None else None
        return dict(usage) if isinstance(usage, dict) and usage else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _with_question_wait(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply the global setting to this Claude Code turn's private config."""
        if getattr(self, "_backend_name", None) != "claude-code":
            return config
        from ..runtime_setting_resolution import resolve_global_runtime_setting

        return {
            **config,
            "question_answer_wait_seconds": int(resolve_global_runtime_setting(
                self.config, "question_answer_wait_seconds",
            )),
        }

    async def _chat_full(
        self,
        prompt: str,
        inject_messages: list | None = None,
        thread_binding: dict | None = None,
        bridge_request_id: str | None = None,
        bridge_timeout_seconds: float | None = None,
        bridge_event_callback=None,
        request_local: dict | None = None,
    ) -> Any:
        """Call the backend's ``chat_full`` (or fall back to ``chat``)."""
        # TASK-501: merge the seed into the config the backend forwards to the
        # bridge (chat_full delegates to stream_raw, which reads
        # config["inject_messages"]). Copy so we never mutate shared _bot_config.
        # TASK-252: thread_binding rides the same copy — request-local, never
        # written to the shared _bot_config.
        config = self._bot_config
        if (
            inject_messages is not None
            or thread_binding
            or bridge_request_id
            or bridge_timeout_seconds
            or bridge_event_callback is not None
            or request_local
        ):
            config = {**self._bot_config, **(request_local or {})}
            if inject_messages is not None:
                config["inject_messages"] = inject_messages
            if thread_binding:
                config.update(thread_binding)
            if bridge_request_id:
                config["request_id"] = bridge_request_id
            if bridge_timeout_seconds:
                config["timeout_seconds"] = max(1, int(bridge_timeout_seconds))
            if bridge_event_callback is not None:
                config["event_callback"] = bridge_event_callback
        config = self._with_question_wait(config)
        if hasattr(self._backend, "chat_full"):
            return await self._backend.chat_full(prompt, config)
        # Fallback for backends that only implement chat()
        text = await self._backend.chat(prompt, config)
        # Wrap in a minimal result-like object
        from types import SimpleNamespace

        return SimpleNamespace(text=text, tool_calls=[], model="", duration_ms=0)

    @staticmethod
    def _load_backend(name: str):
        """Load the agent backend by name."""
        from ..agent_backends import get_backend

        backend = get_backend(name)
        if backend is None:
            raise ValueError(f"Agent backend '{name}' not found")
        return backend
