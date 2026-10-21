"""Select request-local bound execution before ordinary cached-instance fallback."""
from __future__ import annotations

from .prompt_execution import create_prompt_instance, prompt_delivery_record
from .request_model_execution import create_bound_instance, routed_delivery_record


def select_stream_execution(service, request, bot_id: str, local_mode: bool):
    prompt_record = prompt_delivery_record(service, request)
    route_record = routed_delivery_record(service, request)
    if prompt_record and route_record:
        raise ValueError("Delivery cannot be both scheduled and routed")
    if prompt_record:
        model, instance = create_prompt_instance(service, request)
        return model, [], instance, prompt_record
    if route_record:
        model, instance = create_bound_instance(service, request)
        return model, [], instance, None
    model, warnings = service._resolve_request_model(request.model, bot_id, local_mode)
    return model, warnings, None, None
