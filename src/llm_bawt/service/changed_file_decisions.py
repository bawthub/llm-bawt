"""User intent for captured changes, independent of read-time Git evidence."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import text
from sqlmodel import Session, select

from .changed_files_store import ChangedFilesStore, TurnChangedFile


def snapshot_version(row: TurnChangedFile) -> str:
    """Stable on retries; changes when a captured snapshot changes in place."""
    values = [row.before_sha256, row.after_sha256, row.change_kind, row.old_path,
              bool(row.binary), bool(row.truncated), row.additions, row.deletions]
    # Binary, truncated and metadata-only captures cannot prove byte equality.
    if row.binary or row.truncated or not (row.before_sha256 or row.after_sha256):
        values.append(row.source_tool_call_ids)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class ChangedFileDecisionConflict(ValueError):
    """A selected capture changed while the user was looking at it."""


class ChangedFileDecisionsStore(ChangedFilesStore):
    def set_decision(
        self, *, session_id: str | None, bot_id: str, user_id: str,
        files: list[dict[str, Any]], ignored: bool,
    ) -> tuple[str | None, int]:
        """Atomically update exact owned snapshots, never future edits or Git."""
        resolved = self._resolve_session_id(
            session_id=session_id, anchor_turn_id=files[0]["turn_id"] if files else None,
            bot_id=bot_id, user_id=user_id,
        )
        if self.engine is None or not resolved:
            return None, 0
        selected = {(f["turn_id"], f["repo_key"], f["path"]): f for f in files}
        with Session(self.engine) as session, session.begin():
            for (turn_id, repo_key, path), descriptor in selected.items():
                row = session.exec(
                    select(TurnChangedFile)
                    .where(TurnChangedFile.turn_id == turn_id)
                    .where(TurnChangedFile.repo_key == repo_key)
                    .where(TurnChangedFile.path == path)
                    .where(TurnChangedFile.bot_id == bot_id)
                    .where(TurnChangedFile.user_id == user_id)
                    .where(text(
                        "EXISTS (SELECT 1 FROM messages AS m "
                        "WHERE m.bot_id = turn_changed_files.bot_id "
                        "AND m.id = turn_changed_files.trigger_message_id "
                        "AND m.session_id = :decision_session_id)"
                    ).bindparams(decision_session_id=resolved))
                    .with_for_update()
                ).first()
                if row is None:
                    raise LookupError("Changed file not found in conversation")
                version = snapshot_version(row)
                if version != descriptor["snapshot_version"]:
                    raise ChangedFileDecisionConflict("Changes updated; refresh and try again")
                row.ignored_version = version if ignored else None
                row.restored_version = None if ignored else version
                session.add(row)
        return resolved, len(selected)
