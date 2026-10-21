"""Request-owned agent instances for bound endpoint execution.

This is shared by isolated scheduled prompts and routed personal deliveries;
schedule-only history restrictions remain in prompt_execution. Neither path
reads/writes the mutable interactive model override or changes a bot default.
"""
from __future__ import annotations

from ..bot_types import agent_backend_for_model_def, durable_agent_backend
from ..bots import BotManager
from ..model_catalog import resolve_model_config
from .core import ServiceLLMBawt


class BoundAgentLLMBawt(ServiceLLMBawt):
    """A disposable request-local bot with one resolved catalog endpoint."""

    def _get_model_lifecycle(self, config):
        return None

    def _agent_model_ref(self):
        return self.resolved_model_alias


def resolve_execution_endpoint(service, bot_id: str, model: str, user_id: str) -> str:
    """Revalidate the immutable endpoint against the current bot/backend."""
    if not user_id or not bot_id or not model:
        raise ValueError("Bound execution needs user, bot, and model")
    bot = BotManager(service.config).get_bot(bot_id)
    if not bot or not getattr(bot, "enabled", True):
        raise ValueError("Bound execution bot is missing or disabled")
    backend = durable_agent_backend(bot)
    if not backend:
        raise ValueError("Bound execution requires a durable agent backend")
    catalog = service.config.ensure_model_catalog()
    if catalog is None:
        raise ValueError("Bound execution model catalog unavailable")
    endpoint = catalog.resolve_endpoint(model, harness=bot.harness)
    actual = agent_backend_for_model_def(resolve_model_config(
        service.config, endpoint.ref, harness=bot.harness, default={},
    ))
    if actual != backend:
        raise ValueError("Bound execution endpoint no longer matches backend")
    return endpoint.ref


def build_bound_instance(service, request, model: str, instance_type=BoundAgentLLMBawt):
    """Construct a disposable instance after its producer validates the model."""
    instance = instance_type(
        resolved_model_alias=model, config=service.config.model_copy(deep=True),
        bot_id=request.bot_id, user_id=request.user,
        local_mode=not request.augment_memory,
    )
    instance.history_manager.extract_memory = request.extract_memory
    return model, instance


def create_bound_instance(service, request):
    model = resolve_execution_endpoint(
        service, request.bot_id, request.model, request.user,
    )
    return build_bound_instance(service, request, model)


def routed_delivery_record(service, request):
    """Trust only a claimed delivery and its already-persisted request envelope."""
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if not delivery_id or dispatcher is None:
        return None
    record = dispatcher.store.get(delivery_id)
    routed = (record.metadata or {}).get("message_route") if record else None
    if not isinstance(routed, dict):
        return None
    payload = dispatcher.store.payload(delivery_id) or {}
    binding = routed.get("session_binding")
    if (request.bot_id != record.target_bot_id or request.user != routed.get("owner_user_id")
            or request.model != routed.get("model_ref")
            or (request.inter_bot_session_policy or "continue") != record.session_policy
            or request.model != payload.get("model")
            or request.session_id != payload.get("session_id")
            or (isinstance(binding, dict) and request.session_id != binding.get("session_id"))
            or not request.messages or request.messages[-1].content != record.message
            or payload.get("messages") != [{"role": "user", "content": record.message}]
            or request.augment_memory is not True or request.extract_memory is not True
            or payload.get("prefer_steer") is not False):
        raise ValueError("Routed delivery differs from its bound request")
    return record
