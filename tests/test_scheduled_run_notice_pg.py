"""Private-PG additive upgrade receipt; never uses a live database."""
import pytest
from sqlalchemy import text

from llm_bawt.service.scheduled_run_notice import upgrade_notice_columns
from llm_bawt.service.turn_logs import TurnLogStore
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_old_occurrence_upgrade_is_idempotent_without_backfill():
    engine = TurnLogStore(require_test_db_config()).engine
    with engine.begin() as conn:
        # A transaction-local temp table shadows the private app table; no
        # shared schema or application state is dropped or rewritten.
        conn.execute(text("CREATE TEMP TABLE prompt_occurrences (run_id VARCHAR PRIMARY KEY) ON COMMIT DROP"))
        conn.execute(text("INSERT INTO prompt_occurrences(run_id) VALUES ('old-completed')"))
        upgrade_notice_columns(conn, "postgresql")
        upgrade_notice_columns(conn, "postgresql")
        row = conn.execute(text("SELECT run_id, main_notice_json, main_notice_turn_id FROM prompt_occurrences")).one()
        assert tuple(row) == ("old-completed", None, None)
        conn.execute(text("UPDATE prompt_occurrences SET main_notice_json=CAST(:receipt AS JSON)"),
                     {"receipt": '{"reply_excerpt":"bounded"}'})
        assert conn.execute(text("SELECT main_notice_json FROM prompt_occurrences")).scalar_one() == {"reply_excerpt": "bounded"}
