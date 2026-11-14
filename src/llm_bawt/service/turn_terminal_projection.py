"""Retry-safe delivery of a committed terminal, never provider execution."""
from __future__ import annotations

import asyncio

from agent_bridge.terminal_publish import TerminalPublishResult, publish_terminal

from .turn_terminal_history import project_terminal_history
from .turn_terminal_protocol import TurnTerminalProtocol


def terminal_message_rows(engine, bot_id: str):
    """Open existing canonical storage without a memory/session initializer."""
    from ..memory.message_store import MESSAGES_PARENT, MessageRowStore
    from ..memory.postgresql import get_message_table_pg, partition_name

    # Table metadata only: no schema provisioning, active-session resolution,
    # history cache mutation, extraction, compaction or model configuration.
    return MessageRowStore(
        engine, get_message_table_pg(bot_id), partition_name(MESSAGES_PARENT, bot_id),
    )


async def project_committed_terminal(
    protocol: TurnTerminalProtocol, turn_id: str, redis, *, rows_for_bot=None,
) -> TerminalPublishResult | None:
    """Claim, project immutable history, publish atomically, then record ACK.

    Any failure leaves durable retry work behind. History inserts tolerate a
    crash before their DB receipt; Lua tolerates a crash before its DB receipt.
    A lost lease never gains authority merely because publication succeeded.
    No live UI deltas, approvals, usage ingestion or inference are replayed.
    """
    lease = await asyncio.to_thread(protocol.claim, turn_id, projection=True)
    if lease is None:
        return None
    projection = await asyncio.to_thread(protocol.load_projection, lease)
    snapshot = projection.snapshot
    completion = snapshot["completion"]
    if not projection.history_projected:
        if snapshot.get("history_disposition") == "unbound":
            # Validation of the explicit empty no-history disposition still
            # runs, but even a row-store initializer is unnecessary.
            await asyncio.to_thread(project_terminal_history, snapshot, None)
        else:
            rows = await asyncio.to_thread(
                rows_for_bot if rows_for_bot is not None else
                lambda bot: terminal_message_rows(protocol.engine, bot),
                completion["bot_id"],
            )
            if rows.engine is not protocol.engine:
                raise ValueError("terminal history and protocol must share the canonical engine")
            await asyncio.to_thread(project_terminal_history, snapshot, rows)
        await asyncio.to_thread(protocol.mark_history_projected, lease)
    result = await publish_terminal(
        redis, bot_id=completion["bot_id"], user_id=completion["user_id"], turn_id=turn_id,
        publication_deadline_epoch=projection.publication_deadline_epoch,
        completion=completion, tts_tail=snapshot.get("tts_tail"),
    )
    await asyncio.to_thread(protocol.record_publication, lease, result=result)
    return result
