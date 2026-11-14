"""Result-size budgets for the MCP list tools (TASK-2249).

Each fixture is one representative, fully populated row as the source sends it
today. A new field that fattens every row should fail here instead of shipping;
raise a budget only deliberately, with the reason in the commit.
"""

from __future__ import annotations

import json

import llm_bawt.mcp_server.server  # noqa: F401  registers tools; import first
from llm_bawt.mcp_server import inter_bot_tools, task_tools, working_memory_tools


def _chars(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


FULL_BOT = {
    "slug": "fixture-coder", "name": "Fixture Coder", "bot_type": "agent",
    "description": "Implements requested changes and verifies them end to end.",
    "default_model": "fixture-model", "agent_backend": "claude-code",
    "tags": ["coder"], "capabilities": ["implement"],
    "model": "fixture-model@fixture-path", "provider": "vendor-a",
    "quota": "unknown", "quota_checked_at": None, "same_provider_as_caller": True,
    "state": {
        "turn": {"kind": "executing", "turn_id": "turn-0123456789abcdef",
                 "started_at": "2026-11-14T04:40:00+00:00",
                 "task": {"task_ref": "TASK-1234", "title": "A long representative task title for sizing",
                          "source": "association"}},
        "last_turn": {"turn_id": "turn-fedcba9876543210", "ended_at": "2026-11-14T04:30:00+00:00",
                      "status": "completed", "end_reason": "done",
                      "task": {"task_ref": "TASK-1234", "title": "A long representative task title for sizing",
                               "source": "association"}},
        "queued_deliveries": {"count": 2, "oldest_at": "2026-11-14T04:41:00+00:00"},
        "gated": False, "gate": None, "pending_approvals": 1,
        "session": {"id": "6a22871a-e177-5027-be43-c3fdb6c9998d", "started_at": "2026-11-14T01:00:00+00:00",
                    "prompt_stale": {"since": "2026-11-14T02:00:00+00:00",
                                     "changed_at": "2026-11-14T02:00:00+00:00", "sections": ["tags", "skills"]}},
        "location": {"runtime": "house", "house": "h1"},
    },
}

#: ~80 tokens per bot (JSON punctuation tokenizes densely; 4 chars/token is pessimistic).
BOT_ROW_BUDGET = 340


def test_a_compact_bot_row_fits_its_budget_and_keeps_the_routing_facts():
    row = inter_bot_tools._compact_bot(FULL_BOT)

    assert _chars(row) <= BOT_ROW_BUDGET, row
    assert row == {
        "slug": "fixture-coder", "tags": ["coder"], "capabilities": ["implement"], "provider": "vendor-a",
        "turn": {"kind": "executing", "task": "TASK-1234", "started_at": "2026-11-14T04:40:00+00:00"},
        "last_turn": {"ended_at": "2026-11-14T04:30:00+00:00", "task": "TASK-1234"},
        "queued": 2, "pending_approvals": 1, "location": "h1", "prompt_stale": True,
    }


def test_a_compact_bot_keeps_gate_and_unreadable_state():
    gated = {**FULL_BOT, "state": {**FULL_BOT["state"], "gated": True, "unavailable": ["session"],
                                    "gate": {"id": 7, "mode": "hold", "owner_kind": "ops",
                                             "reason": "restart", "expires_at": None}}}
    row = inter_bot_tools._compact_bot(gated)
    assert row["gated"] is True
    assert row["gate"] == {"mode": "hold", "owner_kind": "ops", "reason": "restart"}
    assert row["unavailable"] == ["session"]

    unread = {**FULL_BOT, "state": {**FULL_BOT["state"], "gated": None, "gate": None}}
    assert inter_bot_tools._compact_bot(unread)["gated"] is None  # unknown is never "not gated"


BOARD_ROW = {
    "shortId": "TASK-1234", "title": "A long representative task title for sizing the board row",
    "status": "WAITING", "priority": "HIGH", "agentBotId": "fixture-coder", "focusRank": None,
    "archivedAt": None, "updatedAt": "2026-11-14T04:40:00.000Z", "blockedReason": None,
    "waitingOn": "RESTART", "waitingReason": "x" * 600, "waitingSince": "2026-11-14T04:00:00.000Z",
}
BOARD_ROW_BUDGET = 420


def test_a_board_row_fits_its_budget():
    reply = task_tools._compact_board({"lanes": [{"id": "doing", "rows": [BOARD_ROW]}]})
    row = reply["lanes"][0]["rows"][0]

    assert _chars(row) <= BOARD_ROW_BUDGET, row
    assert {"focusRank", "archivedAt", "blockedReason"}.isdisjoint(row)
    assert len(row["waitingReason"]) == task_tools.BOARD_WAITING_REASON_CHARS
    assert row["shortId"] == "TASK-1234" and row["waitingOn"] == "RESTART"


WM_ITEM = {
    "id": "3ae317b6-3c00-48f8-bc65-487b4f6d8311", "revision": 2, "status": "active", "kind": "personal",
    "objective": "A representative objective of moderate length for sizing", "next_action": None,
    "blockers": [], "checkpoint": None, "priority": None, "refs": [],
    "initiative_task_id": None, "canonical_task_id": None, "task_short_id": None, "task_title": None,
    "task_status": None, "created_at": "2026-11-14T04:00:00+00:00", "updated_at": "2026-11-14T04:10:00+00:00",
    "archived_at": None, "removed_at": None,
}
WM_ITEM_BUDGET = 260


def test_a_working_memory_item_fits_its_budget():
    listing = working_memory_tools._without_empty({"version": 3, "items": [WM_ITEM]})
    item = listing["items"][0]

    assert _chars(item) <= WM_ITEM_BUDGET, item
    assert listing["version"] == 3
    assert item["id"] == WM_ITEM["id"] and item["revision"] == 2
    assert None not in item.values()
