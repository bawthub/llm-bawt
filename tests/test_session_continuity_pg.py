"""JSONB/CAS proof using the session-search private PostgreSQL fixture."""
import json
import pytest
from sqlalchemy import text

from llm_bawt.memory.session_continuity import SessionContinuity
from tests import test_session_search_pg

pg_repository = test_session_search_pg.repository
pytestmark = pytest.mark.integration


def test_atomic_capture_receipts_and_unrelated_metadata(pg_repository):
    engine = pg_repository.engine
    repository = SessionContinuity(engine)
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO sessions (id,bot_id,user_id,status,session_metadata)
            VALUES ('continuity-test','test-bot','owner','active',
                '{"title":"Keep me","agent_session_keys":{"test":"keep-key"}}'::jsonb)"""))
    messages = [{"source_id": "m1", "role": "user", "content": "secret source body", "timestamp": 1}]
    first = repository.manifest(messages, settings={"history_scope": "inline"}, model="test-model")
    assert not repository.record("continuity-test", "test-bot", "other-owner", first)
    assert not repository.record("continuity-test", "other-bot", "owner", first)
    assert repository.record("continuity-test", "test-bot", "owner", first)
    assert not repository.acknowledge("continuity-test", "test-bot", "0" * 64, restored=True)
    assert not repository.acknowledge("continuity-test", "other-bot", first["payload_digest"], restored=True)
    assert repository.acknowledge("continuity-test", "test-bot", first["payload_digest"], restored=False)
    # Retry may replace a failed/prepared capture, but a late old receipt cannot confirm it.
    changed = [{**messages[0], "content": "new source body"}]
    second = repository.manifest(changed, settings={"history_scope": "inline"}, model="test-model")
    assert repository.record("continuity-test", "test-bot", "owner", second)
    assert not repository.acknowledge("continuity-test", "test-bot", first["payload_digest"], restored=True)
    assert repository.acknowledge("continuity-test", "test-bot", second["payload_digest"], restored=True)
    assert not repository.acknowledge("continuity-test", "test-bot", second["payload_digest"], restored=False)
    assert not repository.record("continuity-test", "test-bot", "owner", first)
    with engine.connect() as conn:
        metadata = conn.execute(text("SELECT session_metadata FROM sessions WHERE id='continuity-test'" )).scalar_one()
    assert metadata["title"] == "Keep me"
    assert metadata["agent_session_keys"] == {"test": "keep-key"}
    assert metadata["continuity"]["status"] == "restored"
    assert metadata["continuity"]["source_message_ids"] == ["m1"]
    assert "source body" not in json.dumps(metadata)
