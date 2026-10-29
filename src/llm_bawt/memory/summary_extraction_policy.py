"""Check durable source-row consent before extracting from summary prose."""
from __future__ import annotations

import json

from sqlalchemy import bindparam, text


def _metadata(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def summary_allows_extraction(backend, summary_id: str) -> bool:
    """A mixed summary is indivisible: skip it if any source opted out.

    Check canonical provenance rather than mutable schedule settings. Missing
    source rows/provenance cannot prove consent and therefore fail closed. This
    also covers user content copied into a summary and extraction after restart.
    """
    table = backend._messages_table_name
    with backend.engine.connect() as conn:
        value = conn.execute(text(
            f"SELECT summary_metadata FROM {table} WHERE id=:id AND role='summary'"
        ), {"id": summary_id}).scalar_one_or_none()
        metadata = _metadata(value)
        if metadata.get("extract_memory") is False:
            return False
        ids = metadata.get("message_ids")
        if not isinstance(ids, list) or not ids:
            return False
        rows = conn.execute(text(
            f"SELECT id, summary_metadata FROM {table} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)), {"ids": ids}).mappings().all()
        return (len(rows) == len(set(ids)) and all(
            _metadata(row["summary_metadata"]).get("extract_memory") is not False
            for row in rows
        ))


def summary_has_human_source(backend, summary_id: str) -> bool:
    """True when any source row of the summary was authored by a human user.

    Summary prose of a bot-only session holds no user facts (TASK-1727), so
    extracting from it only spends a call. Authorship is the durable column,
    via the same classifier extraction uses; missing provenance fails closed.
    """
    from .speaker_attribution import human_user_turns_present

    table = backend._messages_table_name
    with backend.engine.connect() as conn:
        value = conn.execute(text(
            f"SELECT summary_metadata FROM {table} WHERE id=:id AND role='summary'"
        ), {"id": summary_id}).scalar_one_or_none()
        ids = _metadata(value).get("message_ids")
        if not isinstance(ids, list) or not ids:
            return False
        rows = conn.execute(text(
            f"SELECT role, content, author_entity_type, author_entity_id FROM {table} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)), {"ids": ids}).mappings().all()
        return human_user_turns_present([dict(r) for r in rows])
