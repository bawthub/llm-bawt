"""Live provider-neutral usage checkpoint, event construction, and persistence encoding."""
from __future__ import annotations

import json
import logging
import math
import time
from decimal import Decimal, InvalidOperation
from typing import Any

log = logging.getLogger(__name__)


def classify_turn_cost(usage: dict[str, Any]) -> dict[str, Any]:
    """Give dollars their frozen route's meaning, without treating auth as billing.

    Subscription equivalents are retained for audit, never as spend. Unknown
    routes cannot authorize a dollar claim; local inference has no token bill.
    """
    result = dict(usage)
    access = usage.get("access_path")
    access = access if isinstance(access, dict) else {}
    billing_mode = access.get("billing_mode")
    if "billing_mode" not in access:
        # Older frozen snapshots name the serving path without its billing
        # capability. Consult only that path's static owner, never auth or the
        # bot's current model. Explicit frozen values remain authoritative.
        from .providers.registry import access_path_billing_mode

        billing_mode = access_path_billing_mode(access.get("key"))
    raw = usage.get("total_cost_usd")
    cost = None
    if isinstance(raw, (int, float, Decimal, str)) and not isinstance(raw, bool):
        try:
            amount = Decimal(str(raw))
            if amount.is_finite() and amount >= 0:
                candidate = float(amount)
                # Persist a JSON number: both the live parser and recorded
                # frontend cost require numbers, not Decimal's string fallback.
                if math.isfinite(candidate) and (candidate > 0 or amount == 0):
                    cost = candidate
        except (InvalidOperation, ValueError, OverflowError):
            pass
    if access.get("vendor") == "local" or access.get("engine_kind") in ("llama-cpp", "vllm", "ollama"):
        kind, cost = "none", 0.0
    elif billing_mode == "subscription":
        kind = "notional" if cost is not None else "none"
    elif billing_mode == "metered":
        explicit = usage.get("cost_kind")
        if explicit in ("notional", "none", "unknown"):
            kind = explicit
            if kind != "notional":
                cost = None
        else:
            kind = "billed" if cost is not None else "unknown"
    else:
        kind, cost = "unknown", None
    result.update(cost_kind=kind, total_cost_usd=cost)
    return result


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
    # Keep this shared write-side module independent of provider imports at load
    # time: adapters can themselves depend on service/turn-log code.
    from .providers.registry import access_path_billing_mode

    access = endpoint.access_path
    return {
        "key": access.key,
        "vendor": access.vendor,
        "protocol": access.protocol,
        "auth_mechanism": access.auth_mechanism,
        "engine_kind": access.engine_kind,
        "billing_mode": access_path_billing_mode(access.key),
    }


def catalog_turn_cost(usage: dict[str, Any], catalog: Any, model_ref: str | None) -> float | None:
    """API-equivalent turn cost from the serving endpoint's catalog rates, or None.

    Prices only the bridge's per-request sums in ``turn_request_totals``
    (``input_tokens`` includes cached tokens; ``output_tokens`` is measured
    per request, so interrupted turns count every request). Top-level counts
    are never mixed in: on interrupted turns they cover one request only.
    Catalog rates are the standard (short-context) tier, so a turn whose
    largest request exceeds the model's pricing threshold stays unpriced.
    Missing rates or totals stay unpriced, never $0.
    """
    totals = usage.get("turn_request_totals")
    endpoint_by_ref = getattr(catalog, "endpoint_by_ref", None)
    endpoint = endpoint_by_ref(model_ref) if callable(endpoint_by_ref) else None
    pricing = getattr(endpoint, "pricing", None)
    if not isinstance(totals, dict) or not pricing:
        return None
    if "output_tokens" not in totals or "max_request_input_tokens" not in totals:
        return None
    if totals.get("unmeasured_requests") != 0:  # absent or >0: some request's usage is unknown
        return None
    # OAuth endpoints with only base rates estimate the non-read input at the
    # fresh rate. This lower bound excludes the write premium (TASK-2503).
    oauth_base_rates = endpoint.access_path.key == "openai-oauth" and "cache_write" not in pricing
    if endpoint.access_path.key == "openai-oauth" and not oauth_base_rates:
        # Require positive evidence even when an older bridge omits the field.
        unknown_writes = totals.get("unpriced_cache_write_requests")
        if type(unknown_writes) is not int or unknown_writes != 0:
            return None
    from ..model_catalog import effective_pricing_threshold
    from .usage.background_spend import catalog_cost

    try:
        cache_read = int(totals.get("cache_read_tokens") or 0)
        cache_write = int(totals.get("cache_creation_tokens") or 0)
        fresh = int(totals.get("input_tokens") or 0) - cache_read - cache_write
        output = int(totals["output_tokens"])
        peak = int(totals["max_request_input_tokens"])
    except (TypeError, ValueError):
        return None
    if fresh < 0:
        return None
    if oauth_base_rates:
        fresh += cache_write
        cache_write = 0  # price the base input, not the unknown premium
    threshold = effective_pricing_threshold(getattr(endpoint.model, "pricing_threshold", None))
    if threshold is not None and peak > threshold:
        return None
    cost = catalog_cost({
        "input_tokens": fresh,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output,
    }, dict(pricing))
    return float(cost) if cost is not None else None


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
    if token_usage.get("total_cost_usd") is None:
        priced = catalog_turn_cost(token_usage, catalog, model_ref)
        if priced is not None:
            token_usage = {**token_usage, "total_cost_usd": priced, "cost_source": "catalog_pricing"}
    return json.dumps(classify_turn_cost(token_usage), ensure_ascii=False, default=str)


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
        # TASK-1385: a mid-run usage checkpoint is telemetry. If it fails (e.g.
        # a turn_logs row-lock timeout), the bridge run is still going; raising
        # here would end the app's turn as an error the bridge never confirmed.
        # The next checkpoint or finalization rewrites the snapshot.
        try:
            self._turn_log_store.update_turn(turn_id=turn_id, token_usage=snapshot)
        except Exception as err:
            log.warning("Usage checkpoint for turn %s not persisted (%s: %s); turn continues",
                        turn_id, type(err).__name__, err)
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
