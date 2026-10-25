"""Prepaid API balances: one anchor/counter row per provider plus a per-turn ledger.

Every pay-per-token usage adapter declares ``balance_mode``:

* ``"anchor"``: the user enters "$X left as of now". ``remaining = anchor_usd -
  spent_usd`` is one row read, never a scan or a cache.
* ``"provider"``: the provider's own balance API is authoritative. Turns are
  still ledgered (for spend detail), but nothing is counted down locally.

``spent_usd`` moves only inside the transaction that finalizes a turn's cost
(``record_turn_charge``, called from ``TurnLogStore.update_turn``). The ledger
holds the cost already charged for each turn, so a replayed finalize changes
nothing and a revised cost charges only the difference. Only turns created
after ``anchor_at`` whose usage carries a frozen ``access_path`` count: a model
alias is catalog intent, not proof of the credential used.

A counted turn with no recorded cost is never charged a guess. It increments
``unpriced_turns`` and the balance reports as unknown until it is set again.

The turn-log scan (``backfill_ledger``) exists only for schema creation,
migration and repair. It is never on the read path.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import text

from .canonical import UsageBalance

logger = logging.getLogger(__name__)

BALANCE_ANCHOR = "anchor"
BALANCE_PROVIDER = "provider"
_CENT_FRACTION = Decimal("0.00000001")  # NUMERIC(18,8): compare at stored precision


def bootstrap_balance_schema(conn) -> None:
    """Create the ledger and counter tables; backfill the ledger once on creation."""
    if getattr(getattr(conn, "dialect", None), "name", None) != "postgresql":
        return
    ledger_is_new = conn.execute(text("SELECT to_regclass('public.api_balance_ledger') IS NULL")).scalar_one()
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS api_balance_ledger (
            turn_id VARCHAR(128) PRIMARY KEY,
            provider VARCHAR(64) NOT NULL,
            access_path_key VARCHAR(64) NOT NULL,
            turn_created_at TIMESTAMPTZ NOT NULL,
            cost_usd NUMERIC(18, 8),
            updated_at TIMESTAMPTZ NOT NULL
        )
    """))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_api_balance_ledger_provider_created"
        " ON api_balance_ledger (provider, turn_created_at)"
    ))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS api_balances (
            provider VARCHAR(64) PRIMARY KEY,
            anchor_usd NUMERIC(18, 8) NOT NULL CHECK (anchor_usd > 0),
            anchor_at TIMESTAMPTZ NOT NULL,
            spent_usd NUMERIC(18, 8) NOT NULL DEFAULT 0,
            unpriced_turns INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL
        )
    """))
    if ledger_is_new:
        backfill_ledger(conn)
    _migrate_legacy_anchors(conn)


def balance_adapter_for_path(key: Any):
    """The balance-capable usage adapter that owns an access path, or None."""
    from . import usage_adapter_for_access_path

    adapter = usage_adapter_for_access_path(key if isinstance(key, str) else None)
    return adapter if adapter is not None and adapter.balance_mode else None


def _usage(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _frozen_path(usage: dict | None) -> str | None:
    frozen = usage.get("access_path") if usage else None
    key = frozen.get("key") if isinstance(frozen, dict) else None
    return key if isinstance(key, str) and key else None


def _cost(usage: dict) -> Decimal | None:
    raw = usage.get("total_cost_usd")
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        cost = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return cost.quantize(_CENT_FRACTION) if cost.is_finite() and cost >= 0 else None


def _upsert_ledger(conn, *, turn_id: str, provider: str, path: str,
                   created_at: datetime, cost: Decimal | None) -> None:
    conn.execute(text("""
        INSERT INTO api_balance_ledger (turn_id, provider, access_path_key, turn_created_at, cost_usd, updated_at)
        VALUES (:turn, :provider, :path, :created, :cost, :now)
        ON CONFLICT (turn_id) DO UPDATE SET cost_usd = EXCLUDED.cost_usd, updated_at = EXCLUDED.updated_at
    """), {"turn": turn_id, "provider": provider, "path": path, "created": created_at,
           "cost": cost, "now": datetime.now(timezone.utc)})


def record_turn_charge(conn, *, turn_id: str, created_at: datetime, token_usage_json: Any) -> None:
    """Ledger a terminal turn's cost and move its provider's counter by the change.

    Runs inside the caller's turn-log transaction, which holds the turn row
    lock, so the prior-cost read cannot race another finalize of this turn.
    """
    if conn.dialect.name != "postgresql":
        return
    usage = _usage(token_usage_json)
    path = _frozen_path(usage)
    adapter = balance_adapter_for_path(path)
    if adapter is None:
        return
    provider = adapter.provider
    cost = _cost(usage)

    def prior_row():
        return conn.execute(text("SELECT cost_usd FROM api_balance_ledger WHERE turn_id=:t"),
                            {"t": turn_id}).first()

    prior = prior_row()
    if prior is not None and prior.cost_usd == cost:
        return  # replayed finalize: already charged exactly this, so take no lock
    # Lock the counter before the ledger row (repair takes them in this order
    # too), then re-read: a repair may have ledgered this turn meanwhile.
    counted = adapter.balance_mode == BALANCE_ANCHOR and conn.execute(text(
        "SELECT 1 FROM api_balances WHERE provider=:p AND anchor_at < :created FOR UPDATE"
    ), {"p": provider, "created": created_at}).first() is not None
    prior = prior_row()
    if prior is not None and prior.cost_usd == cost:
        return
    _upsert_ledger(conn, turn_id=turn_id, provider=provider, path=path, created_at=created_at, cost=cost)
    if not counted:
        return
    prior_cost = prior.cost_usd if prior is not None else None
    was_unpriced = prior is not None and prior_cost is None
    conn.execute(text("""
        UPDATE api_balances
        SET spent_usd = spent_usd + :delta, unpriced_turns = unpriced_turns + :unpriced, updated_at = :now
        WHERE provider = :p
    """), {"p": provider, "delta": (cost or Decimal(0)) - (prior_cost or Decimal(0)),
           "unpriced": int(cost is None) - int(was_unpriced), "now": datetime.now(timezone.utc)})


def backfill_ledger(conn, *, provider: str | None = None, since: datetime | None = None) -> int:
    """Rebuild ledger rows from terminal turn logs (creation, migration, repair only)."""
    from . import _ACCESS_PATH_ADAPTERS

    paths = sorted(key for key, adapter in _ACCESS_PATH_ADAPTERS.items()
                   if adapter.balance_mode and (provider is None or adapter.provider == provider))
    if not paths:
        return 0
    hints = " OR ".join(f"token_usage_json LIKE :hint{i}" for i in range(len(paths)))
    params: dict[str, Any] = {f"hint{i}": f'%"{key}"%' for i, key in enumerate(paths)}
    params["since"] = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
    rows = conn.execute(text(f"""
        SELECT id, created_at, token_usage_json FROM turn_logs
        WHERE ended_at IS NOT NULL AND created_at > :since AND ({hints})
    """), params)
    count = 0
    for turn_id, created_at, raw in rows:
        usage = _usage(raw)
        path = _frozen_path(usage)
        adapter = balance_adapter_for_path(path)
        if adapter is None or path not in paths:
            continue
        _upsert_ledger(conn, turn_id=turn_id, provider=adapter.provider, path=path,
                       created_at=created_at, cost=_cost(usage))
        count += 1
    return count


def _recount(conn, provider: str) -> None:
    conn.execute(text("""
        UPDATE api_balances b SET
            spent_usd = COALESCE((SELECT SUM(l.cost_usd) FROM api_balance_ledger l
                                  WHERE l.provider = b.provider AND l.turn_created_at > b.anchor_at), 0),
            unpriced_turns = (SELECT COUNT(*) FROM api_balance_ledger l
                              WHERE l.provider = b.provider AND l.turn_created_at > b.anchor_at
                                AND l.cost_usd IS NULL),
            updated_at = :now
        WHERE b.provider = :p
    """), {"p": provider, "now": datetime.now(timezone.utc)})


def _migrate_legacy_anchors(conn) -> None:
    """Move a runtime-setting anchor (TASK-1461 r2) into the counter, once."""
    from . import _REGISTRY

    if conn.execute(text("SELECT to_regclass('public.runtime_settings') IS NULL")).scalar_one():
        return
    for adapter in list(_REGISTRY.values()):
        key = adapter.legacy_balance_setting
        if not key:
            continue
        raw = conn.execute(text(
            "SELECT value_json FROM runtime_settings WHERE scope_type='global' AND key=:k"
        ), {"k": key}).scalar_one_or_none()
        if raw is None:
            continue
        value = _usage(raw) or {}
        try:
            amount = Decimal(str(value.get("amount_usd")))
            anchor_at = datetime.fromisoformat(str(value.get("as_of")))
        except (InvalidOperation, TypeError, ValueError):
            amount, anchor_at = Decimal(0), None
        if anchor_at is not None and anchor_at.tzinfo is not None and amount.is_finite() and amount > 0:
            conn.execute(text("""
                INSERT INTO api_balances (provider, anchor_usd, anchor_at, spent_usd, unpriced_turns, updated_at)
                VALUES (:p, :amount, :at, 0, 0, :now) ON CONFLICT (provider) DO NOTHING
            """), {"p": adapter.provider, "amount": amount, "at": anchor_at, "now": datetime.now(timezone.utc)})
            _recount(conn, adapter.provider)
        else:
            logger.warning("Dropping invalid legacy balance anchor for %s", adapter.provider)
        conn.execute(text("DELETE FROM runtime_settings WHERE scope_type='global' AND key=:k"), {"k": key})


@dataclass(frozen=True)
class BalanceRow:
    provider: str
    anchor_usd: Decimal
    anchor_at: datetime
    spent_usd: Decimal
    unpriced_turns: int
    updated_at: datetime


class ApiBalanceStore:
    """Reads and writes the per-provider anchor/counter rows."""

    def __init__(self, engine) -> None:
        if engine is None:
            raise RuntimeError("balance database unavailable")
        self.engine = engine

    def read(self, provider: str) -> BalanceRow | None:
        with self.engine.connect() as conn:
            row = conn.execute(text(
                "SELECT provider, anchor_usd, anchor_at, spent_usd, unpriced_turns, updated_at"
                " FROM api_balances WHERE provider=:p"
            ), {"p": provider}).first()
        return BalanceRow(*row) if row is not None else None

    def set_anchor(self, provider: str, amount: Decimal) -> BalanceRow:
        """Anchor = amount as of now (app clock, like turn created_at); counter restarts."""
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO api_balances (provider, anchor_usd, anchor_at, spent_usd, unpriced_turns, updated_at)
                VALUES (:p, :amount, :now, 0, 0, :now)
                ON CONFLICT (provider) DO UPDATE SET anchor_usd = EXCLUDED.anchor_usd,
                    anchor_at = EXCLUDED.anchor_at, spent_usd = 0, unpriced_turns = 0,
                    updated_at = EXCLUDED.updated_at
            """), {"p": provider, "amount": amount, "now": now})
        return BalanceRow(provider, amount, now, Decimal(0), 0, now)

    def clear(self, provider: str) -> bool:
        with self.engine.begin() as conn:
            return conn.execute(text("DELETE FROM api_balances WHERE provider=:p"),
                                {"p": provider}).rowcount > 0

    def repair(self, provider: str) -> BalanceRow | None:
        """Re-ledger turns since the anchor from turn logs, then recount (repair/tests only)."""
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT anchor_at FROM api_balances WHERE provider=:p FOR UPDATE"),
                               {"p": provider}).first()
            if row is None:
                return None
            backfill_ledger(conn, provider=provider, since=row.anchor_at)
            _recount(conn, provider)
        return self.read(provider)

    def spend(self, provider: str, *, since: datetime, starts: tuple[datetime, ...]) -> tuple[Decimal, ...]:
        """Ledgered cost after ``since`` within each window start (indexed, no turn-log scan)."""
        sums = ", ".join(
            f"COALESCE(SUM(cost_usd) FILTER (WHERE turn_created_at >= :s{i}), 0)" for i in range(len(starts)))
        params: dict[str, Any] = {f"s{i}": start for i, start in enumerate(starts)}
        params.update({"p": provider, "since": since, "floor": min(starts)})
        with self.engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT {sums} FROM api_balance_ledger
                WHERE provider = :p AND turn_created_at > :since AND turn_created_at >= :floor
            """), params).one()
        return tuple(Decimal(value) for value in row)


def anchored_balance(row: BalanceRow | None, connected_at: datetime | None) -> tuple[UsageBalance | None, str | None]:
    """The canonical balance for an anchor row, or (None, reason) when it can't be trusted."""
    if row is None:
        return None, None
    if connected_at is not None and row.anchor_at < connected_at:
        # A replaced key may bill another account: never keep counting silently.
        return None, "API key changed after the balance was set; set the balance again"
    if row.unpriced_turns:
        return None, (f"{row.unpriced_turns} API turn(s) since the balance was set have no recorded cost;"
                      " set the balance again")
    return UsageBalance(
        amount_usd=float(row.anchor_usd), remaining_usd=float(row.anchor_usd - row.spent_usd),
        used_usd=float(row.spent_usd), as_of=int(row.anchor_at.timestamp()),
        source="bawt_recorded_estimate", updated_at=row.updated_at.timestamp(),
    ), None


def balance_update_for_turn(engine, token_usage_json: Any) -> dict | None:
    """``turn_complete.balance_update`` for a finished turn on a balance provider."""
    path = _frozen_path(_usage(token_usage_json))
    adapter = balance_adapter_for_path(path)
    if adapter is None:
        return None
    update: dict[str, Any] = {"provider": adapter.provider, "access_path_key": path}
    if adapter.balance_mode == BALANCE_PROVIDER:
        # Only the provider knows what is left: readers fetch it after this.
        return {**update, "balance": None, "updated_at": time.time(), "refresh": True}
    row = ApiBalanceStore(engine).read(adapter.provider)
    balance, _reason = anchored_balance(row, adapter.credential_connected_at())
    updated = row.updated_at.timestamp() if row is not None else time.time()
    return {**update, "balance": balance.model_dump() if balance else None, "updated_at": updated}
