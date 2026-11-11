"""Immutable commit → signed authoring turn receipts (TASK-1462).

The helper supplies Git facts, never the bot or turn. This is advisory
attribution under the existing six-hour bearer capability, not proof of
active-turn liveness or an independent Git signature.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, select
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel

from llm_bawt.task_turn_context import TaskTurnContext
from llm_bawt.utils.schema import SchemaBootstrapGuard

MAX_COMMIT_BYTES = 131_072
_schema = SchemaBootstrapGuard()


class CommitAuthoringTurn(SQLModel, table=True):
    __tablename__ = "commit_authoring_turns"

    repo_key: str = Field(sa_column=Column(String(128), primary_key=True))
    commit_sha: str = Field(sa_column=Column(String(64), primary_key=True, index=True))
    turn_id: str = Field(sa_column=Column(String(128), nullable=False))
    bot_id: str = Field(sa_column=Column(String(128), nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class CommitAttributionConflict(ValueError):
    """An existing commit receipt cannot be reassigned."""


def ensure_schema(engine: Engine) -> None:
    _schema.run(
        engine, "commit_authoring_turns",
        lambda conn: SQLModel.metadata.create_all(bind=conn, tables=[CommitAuthoringTurn.__table__]),
    )


def validate_commit(commit_sha: str, commit_base64: str, bot_id: str) -> str:
    """Validate a bounded Git object and its signed-bot author, return canonical SHA."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit_sha):
        raise ValueError("A full Git commit SHA is required")
    try:
        raw = base64.b64decode(commit_base64, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid commit object encoding") from None
    if not raw or len(raw) > MAX_COMMIT_BYTES:
        raise ValueError("Commit object is empty or too large")
    sha = commit_sha.lower()
    object_bytes = b"commit " + str(len(raw)).encode("ascii") + b"\0" + raw
    algorithm = hashlib.sha1 if len(sha) == 40 else hashlib.sha256
    if algorithm(object_bytes).hexdigest() != sha:
        raise ValueError("Commit object does not match its SHA")
    # Read only actual header lines (not a message's lookalike author line).
    authors = [line for line in raw.split(b"\n\n", 1)[0].splitlines() if line.startswith(b"author ")]
    expected = rb"author " + re.escape(f"{bot_id} <{bot_id}@bots.bawt.local>".encode())
    if len(authors) != 1 or not re.fullmatch(expected + rb" \d+ [+-]\d{4}", authors[0]):
        raise ValueError("Commit author does not match the signed turn")
    return sha


def record_commit(
    engine: Engine, context: TaskTurnContext, repo_key: str, commit_sha: str,
) -> None:
    """Insert once; retries may repeat the same binding but never replace it."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    ensure_schema(engine)
    table = CommitAuthoringTurn.__table__
    insert = sqlite_insert if engine.dialect.name == "sqlite" else pg_insert
    values = {
        "repo_key": repo_key, "commit_sha": commit_sha,
        "turn_id": context.turn_id, "bot_id": context.bot_id,
        "created_at": datetime.now(timezone.utc),
    }
    with engine.begin() as conn:
        conn.execute(insert(table).values(**values).on_conflict_do_nothing(
            index_elements=["repo_key", "commit_sha"],
        ))
        row = conn.execute(select(table.c.turn_id, table.c.bot_id).where(
            table.c.repo_key == repo_key, table.c.commit_sha == commit_sha,
        )).one()
        if tuple(row) != (context.turn_id, context.bot_id):
            raise CommitAttributionConflict("Commit already belongs to a different signed turn")


def authoring_turns(engine: Engine, revision: str) -> list[tuple[str, str]]:
    """At most two matches: ambiguity must not silently choose an author.

    Reviews currently have an unqualified targetRevision, so any ambiguity
    across repositories is also unresolved. No prose, branch or tag guessing.
    """
    ensure_schema(engine)
    table = CommitAuthoringTurn.__table__
    with engine.connect() as conn:
        rows = conn.execute(select(table.c.turn_id, table.c.bot_id).where(
            table.c.commit_sha.startswith(revision.lower()),
        ).limit(2)).all()
    return [tuple(row) for row in rows]
