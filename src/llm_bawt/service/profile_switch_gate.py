"""A profile save may not move an agent bot onto a held runtime (TASK-1233 S3).

While a system gate holds a runtime, that runtime admits nothing new: not a
turn, and not a bot switched onto it. The save takes the bot's admission lock
before its row write (the order every ``bot_profiles`` writer under
``lock_target`` already uses) and asks the gates, in the same transaction,
about the row it is about to commit, so a gate activating concurrently either
sees the committed switch or the switch sees the gate.

The question is the ``reserve`` actor's: a switch is refused exactly when a
new turn on the switched-to runtime would be (409 ``system_held``, 503
``system_frozen``). A save that keeps the runtime (any edit of a bot already
held) or moves the bot out of a held scope is admitted; the gate's roster is
pinned at activation, so leaving the scope never drops a member.
"""
from __future__ import annotations

from ..runtime_settings import BotProfile, ProfileRuntime
from .turn_admission import lock_target, refuse_if_gated
from .turn_gates import GateActor

PROFILE_SWITCH_ACTOR = GateActor("reserve")


class ProfileSwitchGate:
    """The guard ``BotProfileStore.upsert`` runs for route saves."""

    def lock(self, conn, slug: str) -> None:
        lock_target(conn, slug)

    def check(self, conn, previous: ProfileRuntime | None, row: BotProfile) -> None:
        """Raise ``SystemGateRefused`` when this save moves ``row`` onto a gated runtime."""
        if previous == ProfileRuntime.of(row):
            return
        refuse_if_gated(conn, row.slug, PROFILE_SWITCH_ACTOR)
