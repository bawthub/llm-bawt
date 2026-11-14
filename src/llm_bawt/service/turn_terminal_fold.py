"""Pure content fold for an already-proven retained run; no replay effects.

Does not choose a terminal status or persist anything. The recovery coordinator
must reconcile interaction/reset receipts and durable tools before committing.
Usage is snapshot data, never ledger ingestion. Missing/trimmed replay is not
accepted here: its caller must use durable partials, not append a replay suffix.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from agent_bridge.abort_contract import abort_cause_of
from agent_bridge.events import AgentEvent, AgentEventKind as Kind
from agent_bridge.terminal_replay import ReplayDisposition, RetainedRun


@dataclass(frozen=True)
class ReplayContent:
    response_text: str
    reasoning: str
    tools: tuple[dict, ...]
    token_usage: dict | None
    attachments: tuple[dict, ...]
    assistant_done: bool
    cancelled: bool
    abort_cause: str | None
    error_text: str | None
    interactions: tuple[AgentEvent, ...]
    reset_ack: bool


class IncompleteReplay(ValueError):
    """Use the durable-partial error path, never promote deltas to success."""


def fold_retained_run(replay: RetainedRun) -> ReplayContent:
    """Rebuild from the aggregate, without model, tool, DB or UI calls.

    Retention proof cannot detect a producer's failed XADD. Exactly one full
    ASSISTANT_DONE is the text/usage/attachments authority, not the delta sum.
    ERROR-only, absent or ambiguous aggregates use the durable-partial error
    path instead. Tool completeness still requires durable-record reconciliation.
    """
    if replay.disposition is not ReplayDisposition.COMPLETE or replay.terminal is None:
        raise IncompleteReplay("content fold requires proven complete retained replay")
    aggregates = [event for event in replay.events if event.kind is Kind.ASSISTANT_DONE]
    if len(aggregates) != 1 or not isinstance(aggregates[0].text, str):
        raise IncompleteReplay("recovery requires one full assistant aggregate")
    aggregate = aggregates[0]
    reasoning: list[str] = []
    tools: dict[str, dict] = {}
    attachments: dict[str, dict] = {}
    interactions: list[AgentEvent] = []
    token_usage = None
    assistant_done = cancelled = reset_ack = False
    error_text = None

    for event in replay.events:
        kind = event.kind
        if kind is Kind.REASONING_DELTA:
            reasoning.append(event.text or "")
        elif kind is Kind.ASSISTANT_DONE:
            assistant_done = True
            cancelled = cancelled or (event.raw or {}).get("end_reason") == "aborted"
            reset_ack = reset_ack or bool((event.raw or {}).get("new_session_ack"))
        elif kind in (Kind.TOOL_START, Kind.TOOL_END):
            # Anonymous observations are kept separately rather than guessing
            # a name/LIFO pairing across concurrent calls. Durable reconciliation
            # can match the SDK id; a replay event id is a stable fallback.
            key = event.tool_use_id or event.event_id
            call = tools.setdefault(key, {"tool_use_id": event.tool_use_id, "event_id": event.event_id})
            if event.tool_name:
                call["name"] = event.tool_name
            if event.tool_arguments is not None:
                call["arguments"] = deepcopy(event.tool_arguments)
            if kind is Kind.TOOL_END:
                call.update(result=deepcopy(event.tool_result),
                            tool_result_payload=deepcopy(event.tool_result_payload),
                            is_error=event.tool_error, result_observed=True,
                            attachments=deepcopy(event.attachments))
        elif kind is Kind.ERROR:
            error_text = event.text or "Bridge run failed."
        elif kind in (Kind.APPROVAL_REQUIRED, Kind.AWAIT_TOOL_RESULT):
            # Evidence only: the caller reads existing receipts, never invokes
            # the live interception/persistence handlers a second time.
            interactions.append(deepcopy(event))
        if kind in (Kind.ASSISTANT_DONE, Kind.USAGE_UPDATE, Kind.ERROR) and event.token_usage:
            token_usage = deepcopy(event.token_usage)
        if kind is Kind.ASSISTANT_DONE:
            for ref in event.attachments or ():
                if isinstance(ref, dict) and ref.get("asset_id"):
                    attachments[str(ref["asset_id"])] = deepcopy(ref)
    return ReplayContent(
        response_text=aggregate.text, reasoning="".join(reasoning),
        tools=tuple(tools.values()), token_usage=token_usage,
        attachments=tuple(attachments.values()), assistant_done=assistant_done,
        cancelled=cancelled, abort_cause=abort_cause_of(aggregate.raw) if cancelled else None,
        error_text=error_text,
        interactions=tuple(interactions), reset_ack=reset_ack,
    )
