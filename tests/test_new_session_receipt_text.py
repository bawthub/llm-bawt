"""The app-authored /new receipt cites only bridge-acknowledged continuity."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION, reset_confirmation


def _ts(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def _restored(**overrides) -> dict:
    continuity = {
        "status": "restored", "summary_count": 0, "message_count": 40,
        "approx_tokens": 11034, "oldest_timestamp": _ts("2026-10-21T03:28:00"),
        "newest_timestamp": _ts("2026-10-21T21:14:00"),
    }
    return {"continuity": {**continuity, **overrides}}


@pytest.mark.parametrize("metadata", [
    None, {}, {"continuity": None}, {"continuity": "restored"},
    {"continuity": {"status": "prepared", "message_count": 40}},
    {"continuity": {"status": "failed", "message_count": 40}},
    {"continuity": {"status": "empty", "message_count": 0}},
])
def test_unacknowledged_or_absent_continuity_keeps_the_plain_confirmation(metadata):
    assert reset_confirmation(metadata) == RESET_CONFIRMATION


def test_restored_inline_history_reports_messages_without_a_zero_summary_clause():
    assert reset_confirmation(_restored()) == (
        "Session reset. Seeded 40 recent messages (~11.0k tokens), from 2026-10-21. "
        "Continuity restored."
    )


def test_restored_summaries_and_messages_report_both_buckets_and_span():
    metadata = _restored(summary_count=1, message_count=1, approx_tokens=812,
                         oldest_timestamp=_ts("2026-09-29T12:00:00"))
    assert reset_confirmation(metadata) == (
        "Session reset. Seeded 1 summary record + 1 recent message (~812 tokens), "
        "spanning 2026-09-29 → 2026-10-21. Continuity restored."
    )


def test_restored_summaries_only_and_missing_span_stay_factual():
    metadata = _restored(summary_count=3, message_count=0, approx_tokens=2100,
                         oldest_timestamp=None, newest_timestamp=None)
    assert reset_confirmation(metadata) == (
        "Session reset. Seeded 3 summary records (~2.1k tokens). Continuity restored."
    )


@pytest.mark.parametrize("counts", [
    {"summary_count": 0, "message_count": 0},
    {"summary_count": True, "message_count": "40"},
    {"summary_count": -1, "message_count": None},
])
def test_restored_record_without_countable_sources_is_not_described(counts):
    assert reset_confirmation(_restored(**counts)) == RESET_CONFIRMATION


def test_bridge_ack_and_app_receipt_share_one_formatter():
    """TASK-1750: the bridge ack and app receipt can't drift (no zero bucket)."""
    from agent_bridge.continuity import format_seed_ack

    stats = {**_restored()["continuity"], "seeded": True}
    assert format_seed_ack(stats) == reset_confirmation(_restored())
    assert format_seed_ack({"clean_start": True}) == RESET_CONFIRMATION
