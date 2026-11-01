"""Optional accepted-session markers for every canonical history page."""
from ..logging import get_service_logger

log = get_service_logger(__name__)


def hydrate_resumptions_for_page(service, bot_id: str, messages: list[dict],
                                user_id: str | None = None) -> dict[str, dict]:
    if not any(m.get("id") and m.get("role") == "user" for m in messages):
        return {}
    try:
        from ...media.assets import _build_engine
        from ..session_resumption import for_messages

        engine = _build_engine(service.config)
        if engine is None:
            return {}
        return for_messages(engine, bot_id=bot_id, messages=messages, user_id=user_id)
    except Exception as exc:
        # Rolling upgrade: absent receipt table omits markers, not the timeline.
        # Never bootstrap or repair schema on history reads.
        log.warning("Failed to enrich accepted-session markers: %s", exc)
        return {}
