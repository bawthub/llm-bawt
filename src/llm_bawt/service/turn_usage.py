"""Live provider-neutral usage checkpoint, event construction, and persistence encoding."""
from __future__ import annotations

import json
import time
from typing import Any


def access_path_snapshot(catalog: Any, model_ref: str | None) -> dict[str, Any] | None:
    """Facts of the access path that served ``model_ref`` (``model@access_path``).

    Frozen into the turn's usage so a turn keeps showing the route that
    actually served it even after the catalog is edited. Facts only — display
    labels are formatted by the frontend's single access-path formatter.
    """
    endpoint_by_ref = getattr(catalog, "endpoint_by_ref", None)
    endpoint = endpoint_by_ref(model_ref) if callable(endpoint_by_ref) else None
    if endpoint is None:
        return None
    access = endpoint.access_path
    return {
        "key": access.key,
        "vendor": access.vendor,
        "protocol": access.protocol,
        "auth_mechanism": access.auth_mechanism,
        "engine_kind": access.engine_kind,
    }


def encode_token_usage(
    token_usage: dict | None,
    *,
    model_ref: str | None,
    catalog: Any,
) -> str | None:
    """Serialize a turn's usage for ``turn_logs.token_usage_json``.

    The one write-side encoder: every turn-log writer funnels through it, so
    each persisted snapshot carries ``access_path`` unless the caller already
    supplied one.
    """
    if not isinstance(token_usage, dict):
        return None
    if "access_path" not in token_usage:
        snapshot = access_path_snapshot(catalog, model_ref)
        if snapshot is not None:
            token_usage = {**token_usage, "access_path": snapshot}
    return json.dumps(token_usage, ensure_ascii=False, default=str)


class TurnUsageCoordinator:
    """Own latest-snapshot persistence and public live usage payloads."""

    def __init__(self, turn_log_store: Any) -> None:
        self._turn_log_store = turn_log_store

    def capture(
        self,
        *,
        turn_id: str,
        trigger_message_id: str | None,
        bot_id: str,
        user_id: str,
        model: str,
        token_usage: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Checkpoint a partial snapshot and return its unified event envelope."""
        if not isinstance(token_usage, dict) or not token_usage:
            return None
        # Post-DONE context enrichment carries the complete final usage. Live
        # iteration checkpoints remain partial; never downgrade a final snapshot.
        status = "final" if token_usage.get("usage_status") == "final" else "partial"
        snapshot = {**token_usage, "usage_status": status}
        self._turn_log_store.update_turn(turn_id=turn_id, token_usage=snapshot)
        return {
            "_type": "turn_usage",
            "turn_id": turn_id,
            "trigger_message_id": trigger_message_id,
            "bot_id": bot_id,
            "user_id": user_id,
            "model": model,
            "token_usage": snapshot,
            "ts": time.time(),
        }

    @staticmethod
    def http_chunk(event: dict[str, Any]) -> dict[str, Any]:
        """Return the metadata-only OpenAI-compatible initiating-stream chunk."""
        return {
            "object": "chat.completion.usage",
            "turn_id": event["turn_id"],
            "trigger_message_id": event.get("trigger_message_id"),
            "model": event.get("model"),
            "token_usage": event["token_usage"],
        }
