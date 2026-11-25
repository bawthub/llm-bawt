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
    assert acn.fleet_line(ROWS, {"al": "idle", "vex": "reserved"}) == "al idle TASK-2 W1; vex reserved TASK-7 W1"
    waiting = [row(1, "WAITING", bot="worker")]
    assert acn.fleet_line(waiting, {"worker": "busy"}) == "worker busy — W1"
    assert acn.fleet_line(waiting, {}) == "worker unknown — W1"
    assert acn.fleet_line([], {}) == "no open assigned cards"


def test_render_default_body_coordinator_and_not():
    cards, fleet = acn.render(ROWS, "vex", coordinator=False)
    assert fleet == ""
    text = render_template(ASSIGNED_CARDS, {"cards": cards, "fleet": fleet})
    assert "- TASK-7 IN_PROGRESS" in text and "Fleet" not in text
    cards, fleet = acn.render([], "snark", coordinator=True)
    text = render_template(ASSIGNED_CARDS, {"cards": cards, "fleet": fleet})
    assert "- none\nFleet (bot, live turn, IN_PROGRESS card, W=waiting): no open assigned cards\n" in text


@pytest.mark.parametrize("unavailable", [False, True])
def test_coordinator_reads_live_turns_for_visible_board_bots(stubs, monkeypatch, unavailable):
    from llm_bawt.bot_tags.facade import TagSnapshot
    from llm_bawt.service.turn_admission import TargetState

    rows = [row(12, "WAITING", bot="worker"), row(13, "QUEUED", bot="private-worker")]
    tags = TagSnapshot(tags={"worker": frozenset(), "private-worker": frozenset({"private"})})
    monkeypatch.setattr(acn, "board_rows", lambda: rows)
    monkeypatch.setattr(acn, "is_coordinator", lambda *_: True)
    monkeypatch.setattr("llm_bawt.bot_tags.facade.BotTags.snapshot", lambda _: tags)
    monkeypatch.setattr(acn, "render_block", lambda _bot, cards, fleet: cards + fleet)

    def peek(_engine, bots):
        assert bots == ["worker"]
        if unavailable:
            raise RuntimeError("turn log unavailable")
        return {"worker": TargetState(({"status": "streaming"},))}

    monkeypatch.setattr("llm_bawt.service.turn_admission.peek_targets", peek)
    out = acn.assigned_cards_notice(object(), bot_id="coordinator", session_id="s", decision=object())
    status = "unknown" if unavailable else "busy"
    assert out is not None and f"worker {status} — W1" in out
    assert "private-worker unknown — W0" in out


def test_own_card_after_many_others_is_kept():
    rows = [row(n, "QUEUED", bot="vex") for n in range(1, 61)] + [row(99, "QUEUED")]
    assert acn.card_lines(rows, "al") == ["- TASK-99 QUEUED: Do the thing"]


def test_other_open_statuses_trail():
    rows = [row(1, "BUG"), row(2, "REFINED"), row(3, "PLANNING"), row(4, "QUEUED"), row(5, "CANCELLED")]
    assert [line.split(":")[0] for line in acn.card_lines(rows, "al")] == [
        "- TASK-4 QUEUED", "- TASK-3 PLANNING", "- TASK-2 REFINED", "- TASK-1 BUG",
    ]


def test_board_request_has_no_row_limit(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"lanes": [{"rows": [row(1, "QUEUED")]}]}

    def get(url, params, auth, timeout):
        seen.update(params)
        return Resp()

    monkeypatch.setattr("httpx.get", get)
    monkeypatch.setattr("llm_bawt.bawthub_auth.service_auth", lambda: None)
    monkeypatch.setattr(acn, "_board_cache", None)
    assert acn.board_rows() == [row(1, "QUEUED")]
    assert "limit" not in seen and seen["updated"] == "all"


@pytest.fixture
def stubs(monkeypatch):
    calls = {}
    monkeypatch.setattr("llm_bawt.working_memory.policy.is_protected", lambda bot, engine: bot == "mira")
    monkeypatch.setattr(acn, "board_rows", lambda: ROWS)
    monkeypatch.setattr(acn, "is_coordinator", lambda engine, bot: False)
    def block(bot, cards, fleet):
        calls["rendered"] = calls.get("rendered", 0) + 1
        return f"BLOCK {cards}"
    monkeypatch.setattr(acn, "render_block", block)
    return calls


def test_notice_sends_block_every_turn(stubs):
    for _ in range(3):  # unchanged list, same session: still sent each turn
        out = acn.assigned_cards_notice(object(), bot_id="al", session_id="s", decision=SimpleNamespace(pin_turn_id="p"))
        assert out.startswith("BLOCK - TASK-2 IN_PROGRESS")
    assert stubs["rendered"] == 3


@pytest.mark.parametrize("kwargs", [
    {"bot_id": "mira"}, {"eligible": False}, {"session_id": ""}, {"decision": None},
])
def test_notice_skips(stubs, kwargs):
    base = {"bot_id": "al", "session_id": "s", "decision": SimpleNamespace(pin_turn_id="p"), "eligible": True}
    assert acn.assigned_cards_notice(object(), **{**base, **kwargs}) is None
    assert "rendered" not in stubs


def test_notice_failure_is_swallowed(stubs, monkeypatch):
    def boom():
        raise RuntimeError("board down")
    monkeypatch.setattr(acn, "board_rows", boom)
    assert acn.assigned_cards_notice(object(), bot_id="al", session_id="s", decision=SimpleNamespace(pin_turn_id="p")) is None
