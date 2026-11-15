"""Measured usage for one direct-chat turn, including every tool-loop request."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from typing import Any


_active_turn: ContextVar[tuple[object, DirectTurnUsage] | None] = ContextVar(
    "direct_turn_usage", default=None,
)


@contextmanager
def direct_turn_usage_scope():
    """Keep a worker's usage private, including no-inference and error paths."""
    token = _active_turn.set(None)
    try:
        yield
    finally:
        _active_turn.reset(token)


def begin_direct_turn_usage(client: Any) -> None:
    _active_turn.set((client, DirectTurnUsage(
        client.model_definition, context_window=client.effective_context_window,
        max_output_tokens=client.effective_max_tokens,
    )))


def current_direct_turn_usage(client: Any) -> DirectTurnUsage | None:
    active = _active_turn.get()
    # Recall/maintenance clients in this context are not the chat-turn client.
    return active[1] if active is not None and active[0] is client else None


class DirectTurnUsage:
    def __init__(self, definition: dict, *, context_window: int | None = None,
                 max_output_tokens: int | None = None) -> None:
        # Reuse the maintenance path's provider normalization and catalog rates;
        # a missing rate is unknown, not a made-up model-name fallback.
        from ..service.providers.registry import access_path_billing_mode

        self.pricing = definition.get("pricing")
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.access = {
            "key": definition.get("access_path"),
            "vendor": definition.get("access_vendor"),
            "protocol": definition.get("protocol"),
            "auth_mechanism": definition.get("auth_mechanism"),
            "engine_kind": definition.get("engine_kind"),
            "billing_mode": access_path_billing_mode(definition.get("access_path")),
        }
        self.last: dict | None = None
        self.cost = Decimal(0)
        self.priced = True
        self.sources: set[str] = set()
        self.output = 0
        self.reasoning = 0
        self.requests = 0
        self.input = 0
        self.cache_read = 0
        self.cache_write = 0
        self.with_cache = 0

    def record(self, response: Any) -> None:
        from ..service.usage.background_spend import call_cost, normalize_usage

        self.requests += 1
        previously_priced = self.priced
        # A malformed/unmeasured call cannot leave a partial sum looking final.
        self.priced = False
        self.last = None
        usage = normalize_usage(response)
        if usage is None or "input_tokens" not in usage:
            return
        self.last = usage
        self.output += usage["output_tokens"]
        self.reasoning += usage["reasoning_tokens"]
        self.input += usage["input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"]
        self.cache_read += usage["cache_read_tokens"]
        self.cache_write += usage["cache_write_tokens"]
        self.with_cache += int(usage["cache_read_tokens"] > 0)
        cost, source = call_cost(usage, self.pricing)
        if cost is not None:
            self.priced = previously_priced
            self.cost += cost
            self.sources.add(source)

    def snapshot(self) -> dict | None:
        from ..service.turn_usage import classify_turn_cost

        if self.requests == 0:
            return None
        usage = self.last or {}
        snapshot = {
            "input_tokens": usage.get("input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_tokens", 0),
            "cache_creation_tokens": usage.get("cache_write_tokens", 0),
            "output_tokens": self.output,
            "reasoning_tokens": self.reasoning,
            "reasoning_source": "exact",
            "usage_status": "final" if self.last is not None else "totals_only",
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "access_path": dict(self.access),
            "total_cost_usd": float(self.cost) if self.priced else None,
            "cost_source": ("catalog_pricing" if "catalog_pricing" in self.sources else "provider_reported")
            if self.priced else None,
            "turn_request_totals": {
                "requests": self.requests, "input_tokens": self.input,
                "cache_read_tokens": self.cache_read, "cache_creation_tokens": self.cache_write,
                "requests_with_cache_read": self.with_cache,
            },
        }
        if self.last is not None:
            snapshot["resident_tokens"] = usage["input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"]
            snapshot["resident_source"] = "final_iteration_total_input"
        return classify_turn_cost(snapshot)
