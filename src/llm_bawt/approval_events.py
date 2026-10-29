"""Safe approval lifecycle projection; no prompt, tool input or response leaves the ledger."""

from __future__ import annotations

from .approval_models import REQ_APPROVED, REQ_PENDING, ToolApprovalRequest
from .system_events import SystemEventEmitter


def append_approval_transition(emitter: SystemEventEmitter | None, conn,
                               row: ToolApprovalRequest) -> None:
    if emitter is None:
        return
    emitter.append(
        conn, event_key=f"approval:{row.id}:{row.status}",
        kind="approval", source="approval",
        user_id=row.user_id if row.user_id not in ("", "unknown") else None,
        tone=("info" if row.status == REQ_PENDING else
              "success" if row.status == REQ_APPROVED else "error"),
        summary=f"Tool approval · {row.status}",
        bot_id=row.bot_id, actor_type="bot", actor_id=row.bot_id,
        subject_type="approval", subject_id=row.id,
    )
