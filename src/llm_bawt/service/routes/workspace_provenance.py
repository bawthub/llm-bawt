"""Bounded live-workspace provenance endpoint (TASK-729)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import bindparam, text
from sqlmodel import Session, select

from ..changed_files_store import ChangedFilesStore, TurnChangedFile
from ..dependencies import get_turn_log_store
from ..schemas_workspace_provenance import (
    WorkspaceCandidateResponse,
    WorkspaceFileAttributionResponse,
    WorkspaceProvenanceRequest,
    WorkspaceProvenanceResponse,
    WorkspaceTurnResponse,
)
from ...memory.message_store import MESSAGES_PARENT
from ..turn_logs import TurnLog
from ..workspace_provenance import (
    CurrentWorkspaceFile,
    LineChange,
    TurnFileSnapshot,
    attribute_workspace,
)

router = APIRouter()
MAX_PROVENANCE_ROWS = 2000


def _source_ids(row: TurnChangedFile) -> tuple[str, ...]:
    if not row.source_tool_call_ids:
        return ()
    try:
        values = json.loads(row.source_tool_call_ids)
    except (TypeError, ValueError):
        return ()
    return tuple(str(value) for value in values) if isinstance(values, list) else ()


def _session_ids_by_turn(session: Session, rows: list[TurnChangedFile]) -> dict[str, str]:
    """Resolve each turn's conversation session through its trigger message.

    ``<bot>_messages`` is the only table in the conversation chain carrying
    ``session_id`` (turn logs deliberately do not), so one batched lookup on
    ``(bot_id, trigger_message_id)`` is the canonical turn → session path.
    """
    pairs = {
        (row.bot_id, row.trigger_message_id): row.turn_id
        for row in rows
        if row.bot_id and row.trigger_message_id
    }
    if not pairs:
        return {}
    statement = text(
        f"SELECT bot_id, id, session_id FROM {MESSAGES_PARENT} "
        "WHERE id IN :ids AND bot_id IN :bots AND session_id IS NOT NULL"
    ).bindparams(bindparam("ids", expanding=True), bindparam("bots", expanding=True))
    result = session.connection().execute(statement, {
        "ids": sorted({message_id for _, message_id in pairs}),
        "bots": sorted({bot_id for bot_id, _ in pairs}),
    })
    sessions: dict[str, str] = {}
    for bot_id, message_id, session_id in result:
        turn_id = pairs.get((bot_id, message_id))
        if turn_id:
            sessions[turn_id] = session_id
    return sessions


@router.post(
    "/v1/workspace-provenance",
    response_model=WorkspaceProvenanceResponse,
    tags=["Debug"],
)
def workspace_provenance(payload: WorkspaceProvenanceRequest) -> WorkspaceProvenanceResponse:
    turn_store = get_turn_log_store()
    if turn_store.engine is None:
        raise HTTPException(status_code=503, detail="Turn provenance DB unavailable")

    changed_store = ChangedFilesStore(turn_store.engine)
    try:
        rows, truncated = changed_store.recent_rows(
            user_id=payload.user_id,
            since_hours=payload.lookback_hours,
            limit=MAX_PROVENANCE_ROWS,
        )
        content = changed_store.content_for_rows(rows)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Changed-file provenance unavailable") from error

    turn_ids = sorted({row.turn_id for row in rows})
    with Session(turn_store.engine) as session:
        turn_rows = list(session.exec(
            select(TurnLog).where(TurnLog.id.in_(turn_ids))  # type: ignore[attr-defined]
        ).all()) if turn_ids else []
        try:
            session_ids = _session_ids_by_turn(session, rows)
        except Exception:
            # Session links are a navigation nicety; attribution must not fail on them.
            session_ids = {}
    turns_by_id = {row.id: row for row in turn_rows}

    snapshots: list[TurnFileSnapshot] = []
    for row in rows:
        sides = content.get(row.id or -1)
        before_text = sides[0].text if sides else ""
        after_text = sides[1].text if sides else ""
        turn = turns_by_id.get(row.turn_id)
        snapshots.append(TurnFileSnapshot(
            turn_id=row.turn_id,
            trigger_message_id=row.trigger_message_id,
            bot_id=row.bot_id,
            user_id=row.user_id,
            created_at=row.created_at,
            repo_key=row.repo_key,
            repo_label=row.repo_label,
            path=row.path,
            old_path=row.old_path,
            change_kind=row.change_kind,
            before_sha256=row.before_sha256,
            after_sha256=row.after_sha256,
            before_text=before_text,
            after_text=after_text,
            binary=row.binary,
            truncated=row.truncated,
            source_tool_call_ids=_source_ids(row),
            prompt=turn.user_prompt if turn else None,
        ))

    current_files = [CurrentWorkspaceFile(
        repo_id=item.repo_id,
        repo_key=item.repo_key,
        repo_aliases=tuple(item.repo_aliases),
        path=item.path,
        staged=item.staged,
        status=item.status,
        old_text=item.old_text,
        new_text=item.new_text,
        binary=item.binary,
        truncated=item.truncated,
        changes=tuple(LineChange(
            kind=change.kind,
            text=change.text,
            hunk=change.hunk,
            line=change.line,
        ) for change in item.changes) or None,
    ) for item in payload.files]
    attributions = attribute_workspace(current_files, snapshots)

    response_files = [WorkspaceFileAttributionResponse(
        repo_id=item.repo_id,
        repo_key=item.repo_key,
        path=item.path,
        staged=item.staged,
        ownership=item.ownership,
        unmatched_changes=item.unmatched_changes,
        candidates=[WorkspaceCandidateResponse(
            turn_id=candidate.turn_id,
            trigger_message_id=candidate.trigger_message_id,
            bot_id=candidate.bot_id,
            session_id=session_ids.get(candidate.turn_id),
            created_at=candidate.created_at,
            confidence=candidate.confidence,
            matched_changes=candidate.matched_changes,
            dirty_changes=candidate.dirty_changes,
            candidate_changes=candidate.candidate_changes,
            dirty_coverage=candidate.dirty_coverage,
            candidate_coverage=candidate.candidate_coverage,
            dirty_hunks=candidate.dirty_hunks,
            exact_transition=candidate.exact_transition,
            final_snapshot=candidate.final_snapshot,
            source_tool_call_ids=list(candidate.source_tool_call_ids),
            prompt=candidate.prompt,
            truncated=candidate.truncated,
        ) for candidate in item.candidates],
    ) for item in attributions]

    selected_turn_ids = {
        candidate.turn_id
        for item in attributions
        for candidate in item.candidates
    }
    turn_summaries = [WorkspaceTurnResponse(
        turn_id=snapshot.turn_id,
        trigger_message_id=snapshot.trigger_message_id,
        bot_id=snapshot.bot_id,
        session_id=session_ids.get(snapshot.turn_id),
        created_at=snapshot.created_at,
        prompt=snapshot.prompt,
    ) for snapshot in snapshots if snapshot.turn_id in selected_turn_ids]
    unique_turns = {
        summary.turn_id: summary
        for summary in sorted(turn_summaries, key=lambda item: item.created_at, reverse=True)
    }

    return WorkspaceProvenanceResponse(
        generated_at=datetime.now(timezone.utc),
        status_fingerprint=payload.status_fingerprint,
        lookback_hours=payload.lookback_hours,
        files=response_files,
        turns=list(unique_turns.values()),
        warnings=["Provenance result capped at recent rows"] if truncated else [],
        truncated=truncated,
    )
