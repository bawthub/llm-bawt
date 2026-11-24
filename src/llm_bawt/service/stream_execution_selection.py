"""Select request-local bound execution before ordinary cached-instance fallback."""
from __future__ import annotations

from .prompt_execution import create_prompt_instance, prompt_delivery_record
from .request_model_execution import create_bound_instance, routed_delivery_record
from .strict_reader_execution import create_strict_instance, is_strict_request, refuse_stamped_ordinary


def select_stream_execution(service, request, bot_id: str, local_mode: bool):
    # The strict named reader is decided first so none of the paths below
    # can touch the shared bot registry for it.
    if is_strict_request(service, bot_id):
        model, instance = create_strict_instance(service, request)
        return model, [], instance, None
    # A row admitted under strict mode stays fenced after disable/retarget.
    refuse_stamped_ordinary(service, request, bot_id)
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
