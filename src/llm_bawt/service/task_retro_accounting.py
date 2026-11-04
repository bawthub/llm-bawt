"""Deterministic measured usage for task retros; no resident-context sums."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any


def nonnegative(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and number >= 0 else None


def counter(value: Any) -> int | None:
    number = nonnegative(value)
    return int(number) if number is not None and number == number.to_integral_value() else None


def usage_facts(raw: str | dict | None) -> dict:
    """Null means unmeasured, never zero. SDK session_cost_usd is never spend.

    turn_request_totals.input_tokens includes cache reads and writes. Top-level
    input/cache values describe resident context/last request, NOT turn spend.
    Fresh input includes cache creation; writes are also provided as a subset.
    """
    try:
        usage = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        usage = None
    usage = usage if isinstance(usage, dict) else {}
    path = usage.get("access_path")
    path = path if isinstance(path, dict) else {}
    totals = usage.get("turn_request_totals")
    totals = totals if isinstance(totals, dict) else {}
    total, read, write = (counter(totals.get(key)) for key in
                          ("input_tokens", "cache_read_tokens", "cache_creation_tokens"))
    valid_input = (total is not None and read is not None and write is not None
                   and read + write <= total)
    cost = nonnegative(usage.get("total_cost_usd"))
    return {
        "cost_usd": str(cost) if cost is not None else None,
        "fresh_input_tokens": total - read if valid_input else None,
        "cache_read_tokens": read if valid_input else None,
        "cache_write_tokens": write if valid_input else None,
        "output_tokens": counter(usage.get("output_tokens")),
        "model_requests": counter(totals.get("requests")),
        "usage_status": usage.get("usage_status") if isinstance(usage.get("usage_status"), str) else None,
        "billing_mode": path.get("billing_mode") if isinstance(path.get("billing_mode"), str) else "unknown",
        "vendor": path.get("vendor") if isinstance(path.get("vendor"), str) else None,
    }


def usage_totals(facts: list[dict]) -> dict:
    """Partial sums carry independent per-metric measurement coverage."""
    result = {"turns": len(facts), "measured_turns": {}, "coverage_pct": {}}
    for key in ("cost_usd", "fresh_input_tokens", "cache_read_tokens",
                "cache_write_tokens", "output_tokens", "model_requests"):
        values = [nonnegative(row.get(key)) for row in facts]
        measured = [value for value in values if value is not None]
        result[key] = ((str(sum(measured, Decimal(0))) if key == "cost_usd"
                        else int(sum(measured))) if measured else None)
        result["measured_turns"][key] = len(measured)
        result["coverage_pct"][key] = round(100 * len(measured) / len(facts), 1) if facts else None
    return result


def estimated_quota_share(task_usd, fleet_usd, utilization_pct) -> float | None:
    """Percentage POINTS of quota, only for an already matched quota window.

    Caller must supply recorded Anthropic-subscription API-equivalent spend
    from exactly the snapshot's quota-week start through its fetched_at cutoff.
    This is an estimate, not measured provider task-level quota consumption.
    """
    task, fleet, utilization = (nonnegative(v) for v in (task_usd, fleet_usd, utilization_pct))
    if (task is None or fleet is None or utilization is None or fleet <= 0
            or task > fleet or utilization > 100):
        return None
    return float(task / fleet * utilization)
