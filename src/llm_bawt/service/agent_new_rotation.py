"""Provenance of an already-rotated durable /new retry.

The session successor is deterministic for the delivery. Its predecessor link
and delivery key are written in the same transaction as the rotation, so a
retry can seed from the original archived thread instead of the new one.
"""
from __future__ import annotations

from ..memory.session_reset_lineage import SessionResetLineage


def delivery_new_session_id(delivery_id: str) -> str:
    return SessionResetLineage.delivery_successor_id(delivery_id)


def prepare_delivery_new_binding(
    service, llm_bawt, request, *, bot_id: str, user_id: str,
    user_prompt: str, thread_binding: dict | None, backend_name: str,
) -> tuple[dict | None, str | None]:
    """Use the archived outgoing thread as retry seed source, never the new one."""
    if (not (user_prompt or "").lstrip().startswith("/new")
            or (thread_binding or {}).get("explicit_thread")
            or getattr(request, "inter_bot_session_policy", None)):
        return thread_binding, None
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if not delivery_id:
        return thread_binding, None
    predecessor = delivery_new_predecessor(
        llm_bawt, bot_id=bot_id, user_id=user_id, delivery_id=delivery_id,
    )
    if predecessor:
        thread_binding = service._resolve_thread_sdk_key(
            llm_bawt, {"thread_session_id": predecessor}, backend_name,
        )
    return thread_binding, delivery_id


def bind_new_successor(
    service, llm_bawt, *, rotated_id: str | None | bool,
    delivery_id: str | None, backend_name: str,
) -> dict | None:
    """A retry's archived successor, not some later active thread, owns its reply."""
    if delivery_id and rotated_id and getattr(llm_bawt, "history_manager", None):
        return service._resolve_thread_sdk_key(
            llm_bawt, {"thread_session_id": str(rotated_id)}, backend_name,
        )
    return service._resolve_active_thread_binding(llm_bawt)


def delivery_new_predecessor(llm_bawt, *, bot_id: str, user_id: str, delivery_id: str) -> str | None:
    manager = getattr(llm_bawt, "history_manager", None)
    backend = getattr(manager, "_db_backend", None)
    # Bare harness test clients have no history backend; deployed agent requests
    # require one, and their admission already fails closed without its DB.
    if backend is None:
        return None
    getter = getattr(backend, "get_session", None)
    if not callable(getter):
        raise ValueError("Durable /new cannot inspect its session successor")
    row = getter(delivery_new_session_id(delivery_id))
    if row is None:
        return None
    metadata = SessionResetLineage.metadata(row.get("session_metadata"))
    if (row.get("bot_id") != bot_id or row.get("user_id") != user_id
            or row.get("status") == "deleted"
            or metadata.get("rotation_delivery_id") != delivery_id):
        raise ValueError("Durable /new successor belongs to another delivery")
    predecessors = metadata.get("reset_predecessor_ids")
    if (not isinstance(predecessors, list) or len(predecessors) > 1
            or (predecessors and (not isinstance(predecessors[0], str) or not predecessors[0]))):
        raise ValueError("Durable /new predecessor is ambiguous")
    return predecessors[0] if predecessors else None
