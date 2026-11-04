"""TASK-1151: bots_send_message returns a compact receipt without losing state."""

from llm_bawt.mcp_server.delivery_receipt_compact import compact_send_receipt


def _full():
    return {
        "delivery_id": "delivery-1", "status": "QUEUED", "sender_bot_id": "caid",
        "author": {"entity_type": "bot", "entity_id": "caid"}, "target_bot_id": "gavel",
        "user_message_id": "m1", "turn_id": "t1", "target_turn_id": None,
        "reply_mode": "async", "session_policy": "continue", "reset_status": None,
        "overflow_recovery_count": 0, "attempt_count": 0, "max_attempts": 5,
        "idempotency_key": None, "task_id": "TASK-1", "metadata": {"_failure_origin": {}},
        "created_at": "2026-11-04T00:00:00", "updated_at": "2026-11-04T00:00:00",
        "available_at": "2026-11-04T00:00:00", "last_error": None, "revision": 0,
        "duplicate": False, "success": True, "queued": True, "mode": "async",
        "dispatched": True, "note": "durable",
    }


def test_drops_null_internal_and_default_fields():
    out = compact_send_receipt(_full())
    for key in ("author", "metadata", "revision", "updated_at",
                "available_at", "target_turn_id", "idempotency_key", "last_error",
                "attempt_count", "max_attempts", "session_policy", "duplicate"):
        assert key not in out
    for key in ("delivery_id", "status", "sender_bot_id", "target_bot_id", "turn_id",
                "task_id", "success", "dispatched", "reply_mode", "note", "created_at"):
        assert key in out
    assert "mode" not in out and "queued" not in out  # duplicates of reply_mode/status
    assert out["full_receipt"] == "bots_delivery_get(delivery_id)"


def test_keeps_non_default_state_that_changes_the_next_action():
    out = compact_send_receipt({**_full(), "duplicate": True, "attempt_count": 2,
                                "last_error": "boom", "steer_uncertain": True,
                                "session_policy": "reset_retain_history"})
    assert out["duplicate"] is True and out["attempt_count"] == 2
    assert out["last_error"] == "boom" and out["steer_uncertain"] is True
    assert out["session_policy"] == "reset_retain_history"


def test_refusals_without_delivery_pass_through_untouched():
    refusal = {"success": False, "dispatched": False, "error": "target_inactive", "x": None}
    assert compact_send_receipt(refusal) is refusal


def test_keeps_mode_and_queued_when_they_differ():
    out = compact_send_receipt({**_full(), "mode": "waited", "status": "RUNNING"})
    assert out["mode"] == "waited" and out["queued"] is True
