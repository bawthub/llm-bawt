"""Immutable terminal projection does not change generic native upsert."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from unittest.mock import Mock
import uuid

import pytest
from sqlalchemy import select, update

from llm_bawt.memory.message_store import TerminalMessageConflict
from llm_bawt.service.turn_terminal_history import project_terminal_history
from tests.test_message_row_timestamp import rows

__all__ = ["rows"]


def snapshot(session="pinned-session", bot="test-bot"):
    assistant_id = str(uuid.uuid4())
    return {
        "version": 1, "turn_id": "test-turn",
        "assistant_message_id": assistant_id, "tools_message_id": str(uuid.uuid4()),
        "history_session_id": session, "history_timestamp": 123.25,
        "tool_context": "[Read]\nfrozen result",
        "response_text": "frozen answer", "reasoning": "frozen thought",
        "attachments": [{"asset_id": "frozen-image"}],
        "completion": {"bot_id": bot},
    }


def all_rows(store):
    with store.engine.connect() as connection:
        return list(connection.execute(select(store.table).order_by(
            store.table.c.timestamp,
        )).mappings().all())


def check_retry(store, session="pinned-session", bot="test-bot"):
    frozen = snapshot(session, bot)
    baseline = all_rows(store)
    project_terminal_history(frozen, store)
    original = all_rows(store)
    project_terminal_history(deepcopy(frozen), store)
    assert all_rows(store) == original
    assert len(original) == len(baseline) + 2
    projected = [item for item in original if item["id"] in {
        frozen["assistant_message_id"], frozen["tools_message_id"],
    }]
    assert [item["role"] for item in projected] == ["system", "assistant"]
    assert {item["session_id"] for item in projected} == {session}
    assert projected[1]["content"] == "frozen answer"
    assert projected[1]["reasoning"] == "frozen thought"
    assert projected[1]["attachments"] == [{"asset_id": "frozen-image"}]
    assert projected[1]["author_entity_id"] == bot
    assert [item for item in original if item not in projected] == baseline
    return frozen


def test_repeated_projection_keeps_exact_pair_and_session(rows):
    check_retry(rows)


@pytest.mark.parametrize("field,value", [
    ("response_text", "late answer"), ("reasoning", "late thought"),
    ("history_session_id", "latest-session"), ("history_timestamp", 999),
    ("attachments", []),
])
def test_conflicting_terminal_retries_never_overwrite(rows, field, value):
    frozen = snapshot()
    project_terminal_history(frozen, rows)
    original = all_rows(rows)
    altered = {**frozen, field: value}
    with pytest.raises(TerminalMessageConflict):
        project_terminal_history(altered, rows)
    assert all_rows(rows) == original


def test_pair_rolls_back_if_assistant_identity_collides(rows):
    frozen = snapshot()
    rows.upsert(message_id=frozen["assistant_message_id"], role="user",
                content="unrelated user", timestamp=100, session_id="other")
    with pytest.raises(TerminalMessageConflict):
        project_terminal_history(frozen, rows)
    stored = all_rows(rows)
    assert len(stored) == 1 and stored[0]["content"] == "unrelated user"


def test_summary_flags_can_advance_without_breaking_terminal_retry(rows):
    frozen = snapshot()
    project_terminal_history(frozen, rows)
    with rows.engine.begin() as connection:
        connection.execute(update(rows.table).values(
            processed=True, summary_metadata={"summary": "new maintenance metadata"},
        ))
    project_terminal_history(frozen, rows)
    assert all(item["processed"] for item in all_rows(rows))
    assert all(item["summary_metadata"] == {"summary": "new maintenance metadata"} for item in all_rows(rows))


def test_reasoning_only_terminal_is_not_lost_and_no_empty_tools_row(rows):
    frozen = {**snapshot(), "response_text": "", "tool_context": ""}
    project_terminal_history(frozen, rows)
    assert len(all_rows(rows)) == 1
    assert all_rows(rows)[0]["reasoning"] == "frozen thought"


def unbound_snapshot():
    frozen = snapshot(session=None)
    frozen.update(
        history_disposition="unbound", response_text="", reasoning=None,
        tool_context="", attachments=None,
    )
    frozen["completion"].update(status="error", session_id=None)
    return frozen


def test_unbound_failure_is_an_explicit_no_history_projection_without_session_resolution():
    rows = Mock()
    frozen = unbound_snapshot()
    project_terminal_history(frozen, rows)
    project_terminal_history(deepcopy(frozen), rows)
    rows.engine.begin.assert_not_called()
    rows.insert_terminal.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("history_session_id", "invented-session"), ("response_text", "invented answer"),
    ("reasoning", "invented thought"), ("tool_context", "[Bash]\nunbound result"),
    ("attachments", [{"asset_id": "unbound"}]), ("tts_tail", {"delta": "unbound"}),
    ("completion", {"status": "completed", "session_id": None}),
    ("completion", {"status": "error", "session_id": "invented-session"}),
])
def test_unbound_projection_cannot_silently_drop_inference_content(field, value):
    frozen = {**unbound_snapshot(), field: value}
    rows = Mock()
    with pytest.raises(ValueError, match="empty pre-inference"):
        project_terminal_history(frozen, rows)
    rows.engine.begin.assert_not_called()


def test_bound_projection_cannot_fall_back_from_a_missing_pinned_session(rows):
    with pytest.raises(ValueError, match="pinned session"):
        project_terminal_history({**snapshot(), "history_session_id": None}, rows)
    assert all_rows(rows) == []


def test_generic_upsert_keeps_its_mutable_contract(rows):
    rows.upsert(message_id="generic", role="assistant", content="partial", timestamp=1)
    rows.upsert(message_id="generic", role="assistant", content="final", timestamp=2)
    assert all_rows(rows)[0]["content"] == "final"


class TestPostgresTerminalHistory:
    pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

    @pytest.fixture
    def storage_rows(self, rig, real_storage):
        return real_storage.get_backend(rig.bot)._message_rows

    def test_partitioned_terminal_pair_is_idempotent(self, storage_rows, rig):
        check_retry(storage_rows, rig.session_id, rig.bot)

    def test_concurrent_projectors_write_one_pair(self, storage_rows, rig):
        frozen = snapshot(rig.session_id, rig.bot)
        baseline = all_rows(storage_rows)
        barrier = threading.Barrier(2)

        def project():
            barrier.wait(timeout=5)
            project_terminal_history(frozen, storage_rows)

        with ThreadPoolExecutor(max_workers=2) as pool:
            one, two = pool.submit(project), pool.submit(project)
            one.result(timeout=10)
            two.result(timeout=10)
        stored = all_rows(storage_rows)
        assert len(stored) == len(baseline) + 2
        assert {item["id"] for item in stored} == {
            *(item["id"] for item in baseline),
            frozen["assistant_message_id"], frozen["tools_message_id"],
        }


# Reuse the production partition provisioning fixture, not a mock row store.
from tests.test_message_native_upsert_pg import real_storage, rig  # noqa: E402

__all__ += ["real_storage", "rig"]
