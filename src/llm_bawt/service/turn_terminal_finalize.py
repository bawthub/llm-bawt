"""Normal-execution terminal preparation, separate from headless projection."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from .turn_stream_publish import TurnStreamPublishMixin
from .turn_terminal_projection import project_committed_terminal


class NonstreamTerminalMetadata(TurnStreamPublishMixin):
    """Normal-run metadata and existing approval/question effects, never replay."""

    def __init__(self, svc, request, *, turn_id, trigger_message_id, bot_id, user_id, loop):
        self.ctx = SimpleNamespace(
            svc=svc, request=request, turn_log_id=turn_id, trigger_message_id=trigger_message_id,
            bot_id=bot_id, user_id=user_id, loop=loop,
            _redis_sub=getattr(svc, "_redis_subscriber", None), _tool_event_coordinator=None,
            publish_lock=asyncio.Lock(), _approval_handled=set(), _await_handled=set(),
            approval_id_holder=[None], approval_persist_failed_holder=[None], question_id_holder=[None],
        )
        self.attachments = []
        self.reasoning = ""
        self.token_usage = None
        self.cancelled = False

    def observe(self, item, *, session_id):
        event = item.get("event")
        if event == "attachments":
            self.attachments.extend(
                ref for ref in item.get("attachments", []) if isinstance(ref, dict)
            )
        elif event == "reasoning":
            self.reasoning += item.get("text") or ""
        elif event == "token_usage" and isinstance(item.get("token_usage"), dict):
            self.token_usage = item["token_usage"]
        elif event == "turn_cancelled":
            self.cancelled = True
        elif event == "approval_required":
            self.ctx.request = SimpleNamespace(session_id=session_id)
            self._persist_publish_approval(item)
        elif event == "await_tool_result":
            self._persist_publish_await(item)

    def frozen_fields(self):
        question = self.ctx.question_id_holder[0]
        if question:
            row = self.ctx.svc._pending_question_store.get(question)
            if row is not None and row.live_deadline and row.status in ("answered", "skipped"):
                question = None
        return {
            "question_id": question,
            "approval_id": self.ctx.approval_id_holder[0],
            "approval_persist_failed": self.ctx.approval_persist_failed_holder[0],
        }

    def frozen_attachments(self):
        if not self.attachments:
            return None
        try:
            return self._enrich_attachment_refs(self.attachments) or list(self.attachments)
        except Exception:
            return list(self.attachments)


def prepare_terminal_content(
    svc, llm_bawt, *, response_text: str, tool_context: str,
    tool_call_details: list[dict], token_usage: dict | None,
) -> dict:
    """Finish local output before commit; never persist history or accounting.

    A projector must never call this helper: client state and adapters belong
    to normal execution, not to replay of the already frozen winning bytes.
    """
    adapter = getattr(llm_bawt, "adapter", None)
    if adapter is not None:
        response_text = adapter.clean_output(response_text)
    extracted = svc._extract_agent_backend_tool_calls(llm_bawt=llm_bawt)
    calls = tool_call_details or extracted
    if not tool_context and calls:
        tool_context = "\n\n".join(
            f"[{call.get('tool') or call.get('name', 'unknown')}]\n{call.get('result', '')}"
            for call in calls
        )
    return {
        "response_text": response_text,
        "tool_context": tool_context,
        "token_usage": svc._resolve_turn_token_usage(llm_bawt, token_usage),
    }


def commit_nonstream_terminal(
    svc, llm_bawt, follower, *, status: str, end_reason: str,
    response_text: str | None = None, tool_context: str = "",
    tool_call_details: list[dict] | None = None, token_usage: dict | None = None,
    attachments: list[dict] | None = None, reasoning: str | None = None,
    error_text: str | None = None, completion_fields: dict | None = None,
):
    """The real nonstream worker commits once, before any history writer."""
    row = svc._turn_log_store.get_turn(follower.lease.turn_id)
    if row is None:
        raise ValueError("terminal turn disappeared before finalization")
    if row.terminal_snapshot_json is not None:
        # Use the primitive's locked winner return, including an earlier Stop
        # or setup-failure commit. No stale client preparation is necessary.
        result = follower.protocol.commit(
            follower.lease, status=status, end_reason=end_reason,
        )
    else:
        follower.check()
        if row.terminal_protocol_state == "binding":
            # Pre-inference cancellation/setup failure: no fabricated history,
            # no previous cached backend result, no usage from an earlier turn.
            if response_text or tool_context or tool_call_details or token_usage or attachments or reasoning:
                raise ValueError("unbound nonstream turn unexpectedly contains inference output")
            content = {"response_text": "", "tool_context": "", "token_usage": None}
            attachments = reasoning = None
        else:
            content = prepare_terminal_content(
                svc, llm_bawt,
                response_text=response_text if response_text is not None else (row.response_text or ""),
                tool_context=tool_context, tool_call_details=tool_call_details or [],
                token_usage=token_usage,
            )
        fields = dict(completion_fields or {})
        try:
            fields["changed_files"] = svc._turn_log_store.changed_files_summary(row.id)
        except Exception:
            fields["changed_files"] = None
        result = follower.protocol.commit(
            follower.lease, status=status, end_reason=end_reason, **content,
            attachments=attachments, reasoning=reasoning, error_text=error_text,
            completion_fields=fields,
        )
    follower.close()
    return result


async def project_execution_terminal(protocol, turn_id: str, subscriber, *, publish_lock=None):
    """Use the existing bounded write pool, after earlier queued live events."""
    if subscriber is None:
        raise RuntimeError("terminal projection requires the service Redis publisher")
    if publish_lock is not None:
        async with publish_lock:
            return await project_committed_terminal(protocol, turn_id, subscriber._pub_redis)
    return await project_committed_terminal(protocol, turn_id, subscriber._pub_redis)


def project_execution_terminal_from_worker(
    protocol, turn_id: str, subscriber, loop, *, publish_lock=None,
):
    """Await publication ACK from a real worker, never the HTTP event loop.

    A timeout leaves the committed snapshot recoverable. Cancelling the await
    cannot undo an in-flight immutable DB write or acknowledged Lua publication;
    either crash window is safe to retry from its durable receipt.
    """
    future = asyncio.run_coroutine_threadsafe(
        project_execution_terminal(protocol, turn_id, subscriber, publish_lock=publish_lock),
        loop,
    )
    try:
        return future.result(timeout=30)
    except TimeoutError:
        future.cancel()
        raise
