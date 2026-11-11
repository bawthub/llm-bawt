"""The "in the room" freeze command field (TASK-1692).

The app publishes :data:`ROOM_FREEZE_FIELD` on a ``chat.send`` while an
operator freeze with ``admit_user`` covers the bot. The Claude bridge hands it
to the CLI as :data:`ROOM_FREEZE_ENV`, and its git hook denies commit/push
while the value is :data:`LOCKED`.
"""
from __future__ import annotations

ROOM_FREEZE_FIELD = "room_freeze"
ROOM_FREEZE_ENV = "LLM_BAWT_ROOM_FREEZE"
LOCKED = "locked"  # no commit/push this turn
ASKED = "asked"    # the user's message on this turn asks to commit/push/ship
ROOM_FREEZE_STATES = frozenset({LOCKED, ASKED})

__all__ = ["ASKED", "LOCKED", "ROOM_FREEZE_ENV", "ROOM_FREEZE_FIELD", "ROOM_FREEZE_STATES"]
