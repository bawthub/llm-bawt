"""Background (maintenance) call spend into the balance ledger (TASK-1733).

Chat turns are ledgered from their turn logs. Background clients (summaries,
memory extraction, profile maintenance, utility/recall completions) write no
turn log, so without this their spend never reached ``api_balance_ledger`` and a
prepaid balance estimate ignored nearly all xAI cost.

:func:`install_spend_recorder` is the one hook: the background client factory
installs a :class:`BackgroundSpendRecorder` as the client's ``usage_observer``,
and every client hands it each completed response. The recorder normalizes the
provider usage, prices the call (provider-reported cost when present, else the
endpoint's catalog pricing, else unpriced) and charges it through
``balance.charge_ledger_entry`` with ``source='maintenance'``.

An entry id is derived from the provider's response id, so a replayed response
never charges twice. Turn-log repair (``backfill_ledger``) only rewrites
``turn_logs`` rows and ``_recount`` sums the whole ledger, so repair neither drops
nor double-counts maintenance entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .balance import SOURCE_MAINTENANCE, _CENT_FRACTION, balance_adapter_for_path, charge_ledger_entry

logger = logging.getLogger(__name__)

#: xAI reports ``cost_in_usd_ticks``: 1 USD = 10^10 ticks.
XAI_TICKS_PER_USD = Decimal(10) ** 10
_PER_MILLION = Decimal(1_000_000)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() and number >= 0 else None


def normalize_usage(response: Any) -> dict | None:
    """Provider-neutral usage of one completed response, or None if it has none.

    Reads the three protocol shapes the background clients speak: OpenAI chat
    completions, Responses and Anthropic Messages. ``input_tokens`` is the
    uncached input, so the four token counts never overlap.
    """
    raw = _as_dict(getattr(response, "usage", None) if not isinstance(response, dict) else response.get("usage"))
    if not raw:
        return None
    details_in = _as_dict(raw.get("prompt_tokens_details") or raw.get("input_tokens_details"))
    details_out = _as_dict(raw.get("completion_tokens_details") or raw.get("output_tokens_details"))
    if "prompt_tokens" in raw:  # chat completions: cached tokens are inside prompt_tokens
        cache_read = _int(details_in.get("cached_tokens"))
        input_tokens = max(_int(raw.get("prompt_tokens")) - cache_read, 0)
        output_tokens = _int(raw.get("completion_tokens"))
        cache_write = 0
    elif "cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw:  # Anthropic Messages
        cache_read = _int(raw.get("cache_read_input_tokens"))
        cache_write = _int(raw.get("cache_creation_input_tokens"))
        input_tokens = _int(raw.get("input_tokens"))
        output_tokens = _int(raw.get("output_tokens"))
    elif "input_tokens" in raw or "output_tokens" in raw:  # Responses: cached inside input_tokens
        cache_read = _int(details_in.get("cached_tokens"))
        input_tokens = max(_int(raw.get("input_tokens")) - cache_read, 0)
        output_tokens = _int(raw.get("output_tokens"))
        cache_write = 0
    else:
        input_tokens = None  # unrecognized shape: no token counts, so never catalog-priced
    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage.update({
            "input_tokens": input_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "output_tokens": output_tokens,
            "reasoning_tokens": _int(details_out.get("reasoning_tokens")),
        })
    reported = None
    ticks = _decimal(raw.get("cost_in_usd_ticks"))
    if ticks is not None:
        reported = ticks / XAI_TICKS_PER_USD
    else:
        reported = _decimal(raw.get("cost"))  # OpenRouter: USD credits
    if reported is not None:
        usage["reported_cost_usd"] = str(reported.quantize(_CENT_FRACTION))
    return usage


def catalog_cost(usage: dict, pricing: Any) -> Decimal | None:
    """Cost from the endpoint's catalog rates (USD per 1M tokens), or None.

    Every token class the call used must have a rate; a partial rate card is
    not a price. Reasoning tokens are billed inside ``output_tokens``.
    """
    if not isinstance(pricing, dict) or "input_tokens" not in usage:
        return None  # no recognized token counts: unpriced, never $0
    total = Decimal(0)
    for tokens_key, rate_key in (("input_tokens", "input"), ("output_tokens", "output"),
                                 ("cache_read_tokens", "cache_read"), ("cache_write_tokens", "cache_write")):
        tokens = usage.get(tokens_key) or 0
        if not tokens:
            continue
        rate = _decimal(pricing.get(rate_key))
        if rate is None:
            return None
        total += Decimal(tokens) * rate / _PER_MILLION
    return total.quantize(_CENT_FRACTION)


def call_cost(usage: dict, pricing: Any) -> tuple[Decimal | None, str | None]:
    """``(cost, cost_source)``: provider-reported first, then catalog, else unpriced."""
    reported = _decimal(usage.get("reported_cost_usd"))
    if reported is not None:
        return reported.quantize(_CENT_FRACTION), "provider_reported"
    priced = catalog_cost(usage, pricing)
    return (priced, "catalog_pricing") if priced is not None else (None, None)


def _entry_id(path: str, response: Any) -> str:
    response_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    if isinstance(response_id, str) and response_id:
        digest = hashlib.sha256(f"{path}\0{response_id}".encode()).hexdigest()[:40]
        return f"bg:{digest}"
    return f"bg:{uuid.uuid4().hex}"  # no provider id: a unique call, charged once


def _default_engine():
    from ..dependencies import get_service

    return get_service()._turn_log_store.engine


class BackgroundSpendRecorder:
    """``usage_observer`` for one background endpoint's client."""

    def __init__(self, model_def: dict, *, engine_getter: Callable[[], Any] = _default_engine) -> None:
        self.path = str(model_def.get("access_path") or "")
        self.model = model_def.get("model_key") or model_def.get("model_id")
        self.pricing = model_def.get("pricing")
        self._engine_getter = engine_getter

    def __call__(self, response: Any) -> None:
        adapter = balance_adapter_for_path(self.path)
        if adapter is None:
            return  # local or subscription path: no per-token bill to track
        usage = normalize_usage(response)
        if usage is None:
            # Billed but unmeasured: surfaces as an unpriced call, never as $0.
            usage = {}
        cost, cost_source = call_cost(usage, self.pricing)
        entry = {**usage, "model": self.model, "cost_source": cost_source}
        engine = self._engine_getter()
        if engine is None or engine.dialect.name != "postgresql":
            return
        with engine.begin() as conn:
            charge_ledger_entry(
                conn, entry_id=_entry_id(self.path, response), adapter=adapter, path=self.path,
                created_at=datetime.now(timezone.utc), cost=cost, source=SOURCE_MAINTENANCE,
                usage_json=json.dumps(entry, default=str),
            )
        if cost is None:
            logger.warning("Unpriced background call on %s (%s): no reported cost or catalog pricing",
                           self.path, self.model)


def install_spend_recorder(client: Any, model_def: dict) -> Any:
    """Attach the spend recorder to a freshly built background client."""
    client.usage_observer = BackgroundSpendRecorder(model_def)
    return client
