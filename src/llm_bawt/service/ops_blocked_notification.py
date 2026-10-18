"""Non-terminal ops warnings through the existing durable delivery outbox."""
from __future__ import annotations

import hashlib
import json

from ..bot_identities import SYSTEM_BOT_ID
from .approval_continuation_session import approval_origin_session


class OpsBlockedNotifier:
    """One warning per approved job/error, without consuming its final receipt."""

    def __init__(self, service):
        self.service = service

    def notify(self, row, result: dict) -> None:
        if result.get("state") != "queued" or not result.get("error_text"):
            return
        dispatcher = getattr(self.service, "_inter_bot_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("Ops blocked notification delivery dispatcher unavailable")
        # Keep the identity and payload stable across sweeps/restarts. Binding to
        # the original thread avoids injecting this into unrelated active work.
        error = str(result["error_text"])
        job_id = str(result["job_id"])
        message = (
            "[Ops preflight blocked — not a completion receipt]\n"
            + json.dumps({"job_id": job_id, "operation": result.get("operation"),
                          "state": "queued", "error_text": error}, sort_keys=True)
            + "\nThe approved job is still queued because preflight failed. No action "
            "was submitted by that preflight attempt. Do not submit another job or "
            "re-issue ops_run. Report the blocker; inspect this same job with "
            "ops_job_status. The existing job may run after prerequisites recover. "
            "Its terminal receipt will be delivered separately."
        )
        record, duplicate = dispatcher.store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=row.bot_id,
            message=message,
            idempotency_key=f"ops-blocked:{row.id}:{job_id}:{hashlib.sha256(error.encode()).hexdigest()}",
            message_kind="OPS_BLOCKED",
            metadata={"approval_request_id": row.id, "ops_job_id": job_id},
            payload={
                "bot_id": row.bot_id, "user": row.user_id,
                "session_id": approval_origin_session(row),
                "messages": [{"role": "user", "content": message}],
                "prefer_steer": False, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
        )
        if not duplicate:
            dispatcher.announce_admitted(record)
        else:
            dispatcher.wake(row.bot_id)
