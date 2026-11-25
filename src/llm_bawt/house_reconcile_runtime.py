"""Bounded JSON control used inside the app by the host's house reconciler.

No Docker access or host-side database credentials. Reservations share the
assignment transaction lock; occupancy comes from the existing runtime rules.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

from sqlalchemy import delete, insert

from agent_bridge.houses import commands_stream, normalize_house
from .house_retirement import RESERVATIONS, reservations
from .house_tenancy import HouseState, _tenants, lock_house_assignments

_RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")


def operate(
    store: Any, read_state: Callable[[], HouseState], action: str,
    houses: list[str], reconcile_id: str,
) -> dict[str, Any]:
    if not isinstance(action, str) or action not in {"inspect", "reserve", "release", "cancel"}:
        raise ValueError("unknown reconciliation action")
    if not isinstance(reconcile_id, str) or not _RUN_ID.fullmatch(reconcile_id):
        raise ValueError("invalid reconcile id")
    if not isinstance(houses, list) or len(houses) > 16:
        raise ValueError("at most 16 houses are allowed")
    if any(not isinstance(h, str) or not h or normalize_house(h) != h for h in houses):
        raise ValueError("invalid house id")
    if len(set(houses)) != len(houses):
        raise ValueError("duplicate house id")
    with store.engine.begin() as conn:
        lock_house_assignments(conn)
        held = _tenants(conn)
        state = read_state()  # unreadable Redis aborts, never implies empty
        if state.pool is None:
            raise ValueError("runtime house pool unavailable")
        owned = reservations(conn)
        targets = houses
        if action == "inspect" and not houses:
            targets = sorted(set(state.pool) | set(owned))
            if len(targets) > 16:
                raise ValueError("runtime pool exceeds the supported house count")
        rows = []
        for house in targets:
            reasons = []
            if held.get(house):
                reasons.append("assigned")
            if commands_stream(house) in state.busy_streams:
                reasons.append("live_or_pending_work")
            if owned.get(house) not in (None, reconcile_id):
                reasons.append("another_reconcile")
            rows.append({
                "house": house, "reasons": reasons,
                "reservation": owned.get(house),
                "configured": state.pool is not None and house in state.pool,
                "live": sorted(state.leases.get(house, {})),
                "preview_host": state.preview_hosts.get(house),
            })
        if action == "inspect":
            return {"houses": rows}
        if action == "reserve":
            if any(row["reasons"] for row in rows):
                raise ValueError("removal refused: assigned, busy, or reserved house")
            for house in houses:
                if house not in owned:
                    conn.execute(insert(RESERVATIONS).values(
                        house=house, reconcile_id=reconcile_id,
                    ))
        else:
            for row in rows:
                house = row["house"]
                if house not in owned:
                    continue  # an already-completed release is idempotent
                if owned[house] != reconcile_id:
                    raise ValueError("reservation belongs to another reconciliation")
                if action == "release" and (row["configured"] or row["reasons"]):
                    raise ValueError("remove from the advertised pool and drain before release")
                if action == "cancel" and (
                    not row["configured"] or not row["live"]
                ):
                    raise ValueError("restore and verify the retained house before cancellation")
            for house in houses:
                conn.execute(delete(RESERVATIONS).where(
                    RESERVATIONS.c.house == house,
                    RESERVATIONS.c.reconcile_id == reconcile_id,
                ))
        return {"action": action, "houses": houses, "reconcile_id": reconcile_id}


def main() -> int:
    try:
        raw = sys.stdin.read(8193)
        if len(raw) > 8192:
            raise ValueError("request too large")
        request = json.loads(raw)
        if not isinstance(request, dict) or request.keys() != {
            "action", "houses", "reconcile_id",
        }:
            raise ValueError("invalid request fields")
        from .house_pool_config import configured_pool
        from .house_tenancy import _stream_busy, read_house_state
        from .runtime_settings import BotProfileStore
        from .utils.config import Config
        import redis

        config = Config()
        # Run the profile store's own idempotent bootstrap (bot_profiles,
        # house_retirements, ...) under its advisory lock. The host runs this
        # CLI right after the app reports healthy, which does not prove the
        # app has bootstrapped those tables yet (TASK-2637).
        store = BotProfileStore(config)
        engine = store.engine
        if engine is None:
            raise RuntimeError("database unavailable")
        client = redis.Redis.from_url(
            config.REDIS_URL, decode_responses=True,
            socket_timeout=3, socket_connect_timeout=3,
        )
        def read_state() -> HouseState:
            state = read_house_state(client, configured_pool())
            # A stopped surplus house may have lost its lease while unread or
            # pending sends remain. Reuse the same rule for every target.
            targeted_busy = frozenset(
                commands_stream(house) for house in request["houses"]
                if _stream_busy(client, commands_stream(house))
            )
            return replace(state, busy_streams=state.busy_streams | targeted_busy)
        try:
            result = operate(
                SimpleNamespace(engine=engine), read_state, **request,
            )
        finally:
            client.close()
        print(json.dumps(result))
        return 0
    except Exception as exc:
        # Underlying connection/driver errors can include credentials. Only
        # our validation messages are safe to include in the host receipt.
        detail = str(exc) if type(exc) is ValueError else type(exc).__name__
        print(json.dumps({"error": detail}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
