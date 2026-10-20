"""Independent TASK-1188 regression cases from integration review."""
import pytest
from sqlalchemy import text

from llm_bawt.service.session_lifecycle.admission import accept_session_binding
from llm_bawt.service.session_lifecycle.intent import NewSessionResult
from llm_bawt.service.turn_admission import AdmissionFenceLost
from tests.test_session_lifecycle_admission import engine, pointer  # noqa: F401


def test_cancelled_newer_accepted_binding_still_fences_older_worker(engine):
    accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick", session_id="second")
    with engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET status='cancelled', ended_at=4 WHERE id='new'"))
    with pytest.raises(AdmissionFenceLost, match="superseded"):
        accept_session_binding(engine, turn_id="old", bot_id="fixture", user_id="nick", session_id="first")
    assert pointer(engine) == "second"


def test_completed_reset_requires_receipt_evidence_not_only_stage_label():
    with pytest.raises(ValueError, match="receipt"):
        NewSessionResult(outcome="completed", stage="receipt", new_session_id="successor")
