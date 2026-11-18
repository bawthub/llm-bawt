"""TASK-2432: the per-turn assigned-cards block."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bawt import assigned_cards_notice as acn
from llm_bawt.chat_prompt_defaults import ASSIGNED_CARDS
from llm_bawt.prompt_template_text import render_template


def row(n, status, bot="al", title="Do the thing", **extra):
    return {"shortId": f"TASK-{n}", "status": status, "agentBotId": bot, "title": title, **extra}


ROWS = [
    row(5, "QUEUED"),
    row(4, "REVIEW"),
    row(3, "WAITING", waitingOn="RELEASE"),
    row(2, "IN_PROGRESS"),
    row(1, "COMPLETED"),
    row(6, "WAITING", bot="vex"),
    row(7, "IN_PROGRESS", bot="vex"),
    row(8, "BLOCKED"),
    row(9, "QUEUED", archivedAt="2026-11-18T00:00:00Z"),
]


def test_card_lines_order_status_and_waiting_kind():
    assert acn.card_lines(ROWS, "al") == [
        "- TASK-2 IN_PROGRESS: Do the thing",
        "- TASK-3 WAITING RELEASE: Do the thing",
        "- TASK-8 BLOCKED: Do the thing",
        "- TASK-4 REVIEW: Do the thing",
        "- TASK-5 QUEUED: Do the thing",
    ]


def test_card_lines_cap_and_title_trim():
    rows = [row(n, "QUEUED", title="x" * 200) for n in range(1, 20)]
    lines = acn.card_lines(rows, "al")
    assert len(lines) == acn.MAX_LINES + 1
    assert lines[-1] == f"- +{19 - acn.MAX_LINES} more"
    assert lines[0].endswith("…") and len(lines[0].split(": ", 1)[1]) == acn.TITLE_CHARS


def test_fleet_line():
    assert acn.fleet_line(ROWS) == "al TASK-2 W1; vex TASK-7 W1"
    assert acn.fleet_line([row(1, "WAITING", bot="nova")]) == "nova idle W1"
    assert acn.fleet_line([]) == "no open assigned cards"


def test_render_default_body_coordinator_and_not():
    cards, fleet = acn.render(ROWS, "vex", coordinator=False)
    assert fleet == ""
    text = render_template(ASSIGNED_CARDS, {"cards": cards, "fleet": fleet})
    assert "- TASK-7 IN_PROGRESS" in text and "Fleet" not in text
    cards, fleet = acn.render([], "snark", coordinator=True)
    text = render_template(ASSIGNED_CARDS, {"cards": cards, "fleet": fleet})
    assert "- none\nFleet (bot, IN_PROGRESS card, W=waiting): no open assigned cards\n" in text


def test_should_send():
    decision = SimpleNamespace(pin_turn_id="p1")
    mark = {"pin_turn_id": "p1", "compaction_gen": 0, "digest": "d"}
    assert acn.should_send(None, decision, 0, "d")
    assert not acn.should_send(mark, decision, 0, "d")
    assert acn.should_send(mark, decision, 1, "d")  # compacted
    assert acn.should_send(mark, decision, 0, "e")  # list changed
    assert acn.should_send(mark, SimpleNamespace(pin_turn_id="p2"), 0, "d")  # new session


def test_digest_changes_with_assignment():
    a = acn.digest(*acn.render(ROWS, "al", False))
    b = acn.digest(*acn.render(ROWS + [row(10, "QUEUED")], "al", False))
    assert a != b


@pytest.fixture
def stubs(monkeypatch):
    calls = {}
    monkeypatch.setattr("llm_bawt.working_memory.policy.is_protected", lambda bot, engine: bot == "mira")
    monkeypatch.setattr(acn, "board_rows", lambda: ROWS)
    monkeypatch.setattr(acn, "is_coordinator", lambda engine, bot: False)
    monkeypatch.setattr(acn, "_claim", lambda engine, sid, decision, current: calls.setdefault("claim", current) is not None)
    monkeypatch.setattr(acn, "render_block", lambda bot, cards, fleet: f"BLOCK {cards}")
    return calls


def test_notice_sends_block(stubs):
    out = acn.assigned_cards_notice(object(), bot_id="al", session_id="s", decision=SimpleNamespace(pin_turn_id="p"))
    assert out.startswith("BLOCK - TASK-2 IN_PROGRESS")


@pytest.mark.parametrize("kwargs", [
    {"bot_id": "mira"}, {"eligible": False}, {"session_id": ""}, {"decision": None},
])
def test_notice_skips(stubs, kwargs):
    base = {"bot_id": "al", "session_id": "s", "decision": SimpleNamespace(pin_turn_id="p"), "eligible": True}
    assert acn.assigned_cards_notice(object(), **{**base, **kwargs}) is None
    assert "claim" not in stubs


def test_notice_failure_is_swallowed(stubs, monkeypatch):
    def boom():
        raise RuntimeError("board down")
    monkeypatch.setattr(acn, "board_rows", boom)
    assert acn.assigned_cards_notice(object(), bot_id="al", session_id="s", decision=SimpleNamespace(pin_turn_id="p")) is None
