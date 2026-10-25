"""Writes that ride a terminal turn update, inside ``update_turn``'s transaction.

``TurnLogStore.update_turn`` calls this for every update of a turn that has
ended, holding the turn row lock. Each effect must therefore be idempotent for
a replayed update and must only act on what changed.
"""

from __future__ import annotations


def record_terminal_effects(session, row) -> None:
    from .turn_context_snapshot import record_turn_context
    from .turn_termination import interrupt_unresolved_tools
    from .usage.balance import record_turn_charge

    interrupt_unresolved_tools(session, row.id, row.ended_at)
    session.flush()
    record_turn_context(session, row)
    # TASK-1461: the API balance counter moves in the same commit as the cost.
    record_turn_charge(session.connection(), turn_id=row.id, created_at=row.created_at,
                       token_usage_json=row.token_usage_json)
