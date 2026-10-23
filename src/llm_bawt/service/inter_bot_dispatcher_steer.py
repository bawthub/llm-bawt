"""Steer generations for durable inter-bot deliveries (TASK-1251).

The bridge dedupes a steer RPC on its request id and replays the first cached
result for seven days (``agent_bridge.subscriber.send_steer``). A delivery that
always steered with one deterministic id therefore replayed its first
definitive rejection forever: a retry while its target turn was still starting
re-read the cached ``no_active_run``, and a later steer into a newer turn was
demoted again behind that turn.

A generation ``k`` names the steer RPC. It advances only in the requeue that
records ``k``'s own definitive rejection before transport acceptance, which
proves ``k`` never reached a run, so a fresh id can never deliver the message
twice. Once the bridge accepts ``k`` it is frozen: every replay reuses the
accepted id. A crash before that requeue replays ``k``, which returns the
cached rejection, and the advance is recorded then.

The one exception is proven non-execution (TASK-1300): when the bridge later
reports that accepted command ``k`` was swept, discarded or refused before it
ran, with proof (``cli_terminal`` or ``sweep_receipt``), the row reopens at
``k+1`` (``llm_bawt.inter_bot_delivery_steer_fence``). An ``uncertain`` outcome
never advances ``k``.

``k`` and the per-target not-ready count live in
``inter_bot_deliveries.metadata_json`` (no schema change).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_bridge.steer_contract import STEER_UNFRAMED

STEER_REQUEST_PREFIX = "steer_delivery_"
# ``delivery-<uuid hex>`` ids (request_identity); generation 0 has no suffix.
_REQUEST_ID_RE = re.compile(rf"{STEER_REQUEST_PREFIX}([0-9a-f]{{32}})(?:_g([1-9][0-9]*))?")

# App-side: the target row has no bridge session yet, so no RPC was published
# and the current generation is still unused.
APP_RUN_NOT_READY = "Active bridge run is not ready"

# The bridge's chat.steer raises these before the message reaches the CLI
# (claude_code_bridge/command_ops.py). While the target turn is still open
# they mean "its run has not registered yet", not "this turn rejected you".
BRIDGE_RUN_NOT_READY = frozenset({"no_active_run", "active_run_mismatch"})

# Every definitive non-acceptance before transport acceptance. The two
# maintenance errors and ``steer_unframed`` (TASK-1300: no app framing for a
# non-system steer) are raised by ActiveRun.steer before anything is injected;
# the last two are app-side and publish no RPC.
DEFINITIVE_NOT_ACCEPTED = BRIDGE_RUN_NOT_READY | frozenset({
    "maintenance_active",
    "maintenance_command_requires_turn",
    STEER_UNFRAMED,
    "Turn is no longer active",
    "Turn not found",
})

# Turn rows whose run may still register with the bridge.
OPEN_TURN_STATUSES = frozenset({"reserved", "pending", "streaming"})

# After this many not-ready generations against one target turn, the next
# definitive rejection demotes to ``steer_rejected``: today's zero-cost wait
# for that turn to end. Each generation costs one bridge command plus two
# seven-day Redis keys. 1+2+4+8+15*16 s is about 4.25 minutes of waiting.
NOT_READY_BOUND = 20
NOT_READY_BACKOFF_CAP_SECONDS = 15.0


@dataclass(frozen=True)
class SteerGeneration:
    """The steer RPC identity a claimed delivery uses, read from its metadata."""

    delivery_id: str
    generation: int = 0
    not_ready_target: str | None = None
    not_ready_count: int = 0

    @classmethod
    def of(cls, record: Any) -> "SteerGeneration":
        metadata = getattr(record, "metadata", None) or {}
        not_ready = metadata.get("steer_not_ready")
        if not isinstance(not_ready, dict):
            not_ready = {}
        return cls(
            delivery_id=record.id,
            generation=_non_negative(metadata.get("steer_generation")),
            not_ready_target=str(not_ready.get("target_turn_id") or "") or None,
            not_ready_count=_non_negative(not_ready.get("count")),
        )

    @property
    def request_id(self) -> str:
        """The bridge RPC id. Generation 0 keeps the pre-TASK-1251 id."""
        base = f"{STEER_REQUEST_PREFIX}{self.delivery_id.removeprefix('delivery-')}"
        return base if self.generation == 0 else f"{base}_g{self.generation}"

    @staticmethod
    def parse_request_id(request_id: str | None) -> tuple[str, int] | None:
        """``(delivery_id, generation)`` named by a delivery steer RPC id.

        The inverse of :attr:`request_id`; None for any other id (a user or
        system steer, or a malformed one), so it never names the wrong row.
        """
        match = _REQUEST_ID_RE.fullmatch(request_id or "")
        if match is None:
            return None
        return f"delivery-{match.group(1)}", int(match.group(2) or 0)

    def not_ready_count_for(self, target_turn_id: str | None) -> int:
        """Not-ready generations already spent on ``target_turn_id``."""
        if target_turn_id and target_turn_id == self.not_ready_target:
            return self.not_ready_count
        return 0

    def advance(self, *, not_ready_target: str | None = None) -> dict[str, Any]:
        """The ``metadata_merge`` for the requeue recording this generation's
        definitive rejection. With ``not_ready_target`` it also counts one more
        not-ready generation against that turn; the count restarts at 1 when
        the target changes."""
        merge: dict[str, Any] = {"steer_generation": self.generation + 1}
        if not_ready_target:
            merge["steer_not_ready"] = {
                "target_turn_id": not_ready_target,
                "count": self.not_ready_count_for(not_ready_target) + 1,
            }
        return merge

    @staticmethod
    def backoff_seconds(not_ready_count: int) -> float:
        """Delay before the next generation: 1, 2, 4, 8, then 15 s."""
        exponent = min(max(int(not_ready_count), 1) - 1, 4)
        return min(NOT_READY_BACKOFF_CAP_SECONDS, float(2 ** exponent))


def _non_negative(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
