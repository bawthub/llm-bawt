"""Request-local automation instance using the ordinary chat implementation."""
from __future__ import annotations

from .request_model_execution import BoundAgentLLMBawt, build_bound_instance
from ..utils.history import HistoryManager


class PromptHistoryManager(HistoryManager):
    """Strict automation pool: no global summaries or continuous fallback."""

    extract_memory = True

    def add_message(self, role, content, **kwargs):
        if role in {"user", "assistant"} and not self.extract_memory:
            kwargs["extract_memory"] = False
        return super().add_message(role, content, **kwargs)

    def load_history(self, since_minutes=None, session_id=None):
        self.messages = []
        if session_id is None:
            # Construction must not load the personal conversation at all.
            return
        loader = getattr(self._db_backend, "load_session_scoped", None)
        if not callable(loader):
            raise ValueError("Automation requires persistent scoped history")
        messages = loader(since_minutes=since_minutes, session_id=session_id)
        if messages is None:
            raise ValueError("Automation scoped history is unavailable")
        # Existing thread loaders intentionally include global rolling summaries
        # for interactive hydration. Automation must carry only its own thread.
        self.messages = [message for message in messages if message.session_id == session_id]


class PromptLLMBawt(BoundAgentLLMBawt):
    """A bound endpoint with strictly schedule-scoped history."""

    def _history_manager_type(self):
        return PromptHistoryManager

    def _init_history(self):
        # Automation already requires the app's persistent DB and an explicit
        # thread. Use a request-owned adapter, not the cached MCP history writer.
        from ..memory.postgresql_short_term import PostgreSQLShortTermManager

        self.history_manager = PromptHistoryManager(
            client=self.client, config=self.config, bot_id=self.bot_id,
            settings_getter=self._resolve_setting,
            db_backend=PostgreSQLShortTermManager(
                self.config, bot_id=self.bot_id, user_id=self.user_id,
            ),
        )
        self.load_history()


def prompt_delivery_record(service, request):
    """Identify automation only from the already validated durable receipt."""
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if not delivery_id or dispatcher is None:
        return None
    record = dispatcher.store.get(delivery_id)
    if record and (getattr(record, "metadata", None) or {}).get("prompt_schedule"):
        payload = dispatcher.store.payload(record.id) or {}
        if not request.session_id or not request.model:
            raise ValueError("Scheduled delivery requires its explicit thread and bound model")
        if any(getattr(request, key, None) != payload.get(key) for key in (
                "session_id", "model", "user", "bot_id", "augment_memory", "extract_memory")):
            raise ValueError("Scheduled delivery does not match its immutable request")
        if [message.model_dump(include={"role", "content"}) for message in request.messages] != payload.get("messages"):
            raise ValueError("Scheduled delivery prompt does not match its immutable request")
        return record
    return None


def build_prompt_seed(service, instance, binding):
    """Hydrate only this automation thread, without touching a personal instance."""
    from .routes.history_seed import build_context_seed

    if binding.get("thread_resume_id"):
        return None
    return build_context_seed(
        instance.bot_id, instance.resolved_model_alias, service,
        session_id=binding["thread_session_id"], llm_bawt=instance,
    )["messages"]


def create_prompt_instance(service, request):
    from .prompt_capabilities import resolve_prompt_target

    model = resolve_prompt_target(service, request.bot_id, request.model, request.user)
    return build_bound_instance(service, request, model, instance_type=PromptLLMBawt)
