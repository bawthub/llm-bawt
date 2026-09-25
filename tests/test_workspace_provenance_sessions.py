"""Turn → session resolution for workspace provenance links."""

from types import SimpleNamespace

from sqlalchemy import text
from sqlmodel import Session, create_engine

from llm_bawt.service.routes.workspace_provenance import _session_ids_by_turn


def _row(turn_id: str, bot_id: str | None, trigger: str | None):
    return SimpleNamespace(turn_id=turn_id, bot_id=bot_id, trigger_message_id=trigger)


def test_resolves_session_through_trigger_message_per_bot() -> None:
    engine = create_engine("sqlite://")
    with Session(engine) as session:
        conn = session.connection()
        conn.execute(text("CREATE TABLE messages (bot_id TEXT, id TEXT, session_id TEXT)"))
        conn.execute(text(
            "INSERT INTO messages VALUES "
            "('snark', 'm1', 's-snark'), ('loopy', 'm1', 's-loopy'), ('snark', 'm2', NULL)"
        ))
        result = _session_ids_by_turn(session, [
            _row("t1", "snark", "m1"),
            _row("t2", "loopy", "m1"),   # same message id, different bot
            _row("t3", "snark", "m2"),   # no session recorded
            _row("t4", None, "m1"),      # no bot → not resolvable
            _row("t5", "snark", None),   # no trigger message
        ])
    assert result == {"t1": "s-snark", "t2": "s-loopy"}


def test_no_resolvable_rows_skips_query() -> None:
    engine = create_engine("sqlite://")
    with Session(engine) as session:
        assert _session_ids_by_turn(session, [_row("t1", None, None)]) == {}
