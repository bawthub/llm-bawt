"""Private PostgreSQL tests for the session system-prompt pin (TASK-1318)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql import MESSAGES_PARENT, PARENT_TABLES, ensure_bot_partitions, partition_name
from llm_bawt.service.system_prompt_pin import (
    BOUNDARY_KEY,
    COLD_START,
    COMPACTION,
    LEGACY_ADOPT,
    MISSING_ROW,
    PENDING_KEY,
    PIN_KEY,
    RESET_SKILLS_KEY,
    SHA_MISMATCH,
    SystemPromptPinStore,
    prompt_sha256,
)
from llm_bawt.service.turn_termination import request_payload_with_audit
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

COMPACT_OK = {"context_action": "compact", "context_action_outcome": "success"}


class Rig:
    def __init__(self, config):
        from llm_bawt.service.turn_logs import TurnLogStore

        self.config = config
        self.turns = TurnLogStore(config)
        self.engine = self.turns.engine
        self.store = SystemPromptPinStore(self.engine)
        self.bot = f"testpin{uuid.uuid4().hex[:12]}"
        self.user = f"owner-{uuid.uuid4().hex}"
        self.session_id = str(uuid.uuid4())
        self.table = partition_name(MESSAGES_PARENT, self.bot)
        self.clock = datetime.now(timezone.utc) - timedelta(hours=1)
        with self.engine.begin() as conn:
            ensure_bot_partitions(conn, self.bot)
            conn.execute(text("""
                INSERT INTO sessions (id, bot_id, user_id, status, started_at)
                VALUES (:id, :bot, :user, 'active', now() - interval '2 hours')
            """), {"id": self.session_id, "bot": self.bot, "user": self.user})

    def add_turn(self, prompt="hello", usage=None):
        self.clock += timedelta(minutes=1)
        message_id = str(uuid.uuid4())
        turn_id = f"turn-{uuid.uuid4().hex}"
        with self.engine.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {self.table} (bot_id, id, role, content, timestamp, session_id)
                VALUES (:bot, :id, 'user', :content, 1, :session)
            """), {"bot": self.bot, "id": message_id, "content": prompt, "session": self.session_id})
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                       user_prompt, trigger_message_id, token_usage_json)
                VALUES (:id, :created, '/test', true, :bot, :user, 'ok', :prompt, :message, :usage)
            """), {
                "id": turn_id, "created": self.clock, "bot": self.bot, "user": self.user,
                "prompt": prompt, "message": message_id,
                "usage": json.dumps(usage) if usage is not None else None,
            })
        return turn_id

    def resolve(self, turn_id, rendered, *, cold=False, sections=None, dispatched=True, resume=None,
                skill_manifest=None, injected_skills=()):
        """Resolve a turn; ``dispatched`` models the bridge's first output confirming it."""
        decision = self.store.resolve(
            session_id=self.session_id, turn_id=turn_id, storage_bot=self.bot,
            rendered=rendered, cold_start=cold, sections=sections, resume_id=resume,
            skill_manifest=skill_manifest, injected_skills=injected_skills,
        )
        if dispatched and decision is not None and decision.is_boundary:
            assert self.store.confirm(session_id=self.session_id, turn_id=turn_id)
        return decision

    def metadata(self):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT session_metadata FROM sessions WHERE id=:sid"), {"sid": self.session_id},
            ).scalar_one() or {}

    def turn_request(self, turn_id):
        with self.engine.connect() as conn:
            return json.loads(conn.execute(
                text("SELECT request_json FROM turn_logs WHERE id=:id"), {"id": turn_id},
            ).scalar_one())

    def cleanup(self):
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": self.bot})
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, self.bot)}"))
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": self.bot})


@pytest.fixture
def rig():
    rig = Rig(require_test_db_config())
    try:
        yield rig
    finally:
        rig.cleanup()


def test_cold_start_pins_and_resumed_turns_send_pinned_bytes(rig):
    first = rig.add_turn()
    pinned = rig.resolve(first, "V1", cold=True)
    assert (pinned.action, pinned.reason, pinned.is_boundary, pinned.text) == ("pinned", COLD_START, True, "V1")
    assert rig.turn_request(first)[PIN_KEY]["text"] == "V1"

    pointer = rig.store.get(rig.session_id)
    assert pointer["turn_id"] == first and pointer["sha256"] == prompt_sha256("V1")
    assert pointer["snapshot_unverified"] is False and pointer["reason"] == COLD_START

    # A mid-session prompt edit is recorded as drift, never sent.
    second = rig.add_turn()
    kept = rig.resolve(second, "V2")
    assert (kept.action, kept.text, kept.pin_turn_id, kept.is_boundary) == ("kept", "V1", first, False)
    assert kept.drift
    audit = rig.turn_request(second)[PIN_KEY]
    assert audit["pin_turn_id"] == first and audit["rendered_sha256"] == prompt_sha256("V2")
    assert "text" not in audit
    assert rig.store.pinned_text(rig.session_id) == "V1"


def test_new_session_boundary_moves_the_pin(rig):
    rig.resolve(rig.add_turn(), "V1", cold=True)
    after_new = rig.add_turn("/new hi")
    moved = rig.resolve(after_new, "V2", cold=True)
    assert (moved.reason, moved.text) == (COLD_START, "V2")
    assert rig.store.get(rig.session_id)["turn_id"] == after_new


def test_first_turn_after_manual_compaction_repins(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    compact = rig.add_turn("/compact")
    # The compaction call itself still runs on the old snapshot.
    assert rig.resolve(compact, "V2").text == "V1"
    with rig.engine.begin() as conn:  # the usage lands when the /compact turn completes
        conn.execute(text("UPDATE turn_logs SET token_usage_json=:u WHERE id=:id"),
                     {"u": json.dumps(COMPACT_OK), "id": compact})
    after = rig.add_turn()
    moved = rig.resolve(after, "V3")
    assert (moved.action, moved.reason, moved.text, moved.snapshot_unverified) == ("pinned", COMPACTION, "V3", False)
    # The next turn keeps the new pin: the /compact turn predates it.
    assert rig.resolve(rig.add_turn(), "V4").text == "V3"


def test_auto_compaction_and_failed_compaction_keep_the_pin(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    rig.add_turn("Message from bot 'x': long review", usage=COMPACT_OK)  # CLI auto-compact
    rig.add_turn("/compact", usage={"context_action": "compact", "context_action_outcome": "failed"})
    rig.add_turn("/compactness check", usage=COMPACT_OK)  # not the /compact command
    kept = rig.resolve(rig.add_turn(), "V2")
    assert (kept.action, kept.text, kept.pin_turn_id) == ("kept", "V1", first)


def test_compaction_in_another_session_is_ignored(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    # Same bot, another session: its /compact must not move this pin.
    session_id, other = rig.session_id, str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'archived', now() - interval '2 hours')
        """), {"id": other, "bot": rig.bot, "user": rig.user})
    rig.session_id = other
    rig.add_turn("/compact", usage=COMPACT_OK)
    rig.session_id = session_id
    assert rig.resolve(rig.add_turn(), "V2").action == "kept"


def test_deleted_pin_row_falls_back_to_a_verified_repin(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    with rig.engine.begin() as conn:  # delivery recovery / purge
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": first})
    turn = rig.add_turn()
    moved = rig.resolve(turn, "V2")
    assert (moved.action, moved.reason, moved.text, moved.snapshot_unverified) == ("pinned", MISSING_ROW, "V2", True)
    pointer = rig.store.get(rig.session_id)
    assert pointer["turn_id"] == turn and pointer["snapshot_unverified"] is True
    # The unverified flag rides later kept turns until the next real boundary.
    later = rig.resolve(rig.add_turn(), "V2")
    assert later.action == "kept" and later.snapshot_unverified
    assert rig.resolve(rig.add_turn(), "V3", cold=True).snapshot_unverified is False


def test_tampered_pin_text_is_not_sent(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    payload = rig.turn_request(first)
    payload[PIN_KEY]["text"] = "edited by hand"
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"),
                     {"rj": json.dumps(payload), "id": first})
    moved = rig.resolve(rig.add_turn(), "V2")
    assert (moved.reason, moved.text) == (SHA_MISMATCH, "V2")
    assert rig.store.pinned_text(rig.session_id) == "V2"


def test_legacy_session_adopts_current_render_unverified(rig):
    decision = rig.resolve(rig.add_turn(), "V1", sections={"working_memory": {"v": 1, "version": 1}})
    assert (decision.reason, decision.is_boundary, decision.snapshot_unverified) == (LEGACY_ADOPT, True, True)
    assert decision.pinned_sections is None and "sections" not in rig.store.get(rig.session_id)


def test_pointer_sections_record_what_the_harness_holds(rig):
    held = {"working_memory": {"v": 1, "version": 3, "state": "ok"}}
    edited = {"working_memory": {"v": 1, "version": 4, "state": "ok"}}
    first = rig.add_turn()
    assert rig.resolve(first, "V1", cold=True, sections=held).pinned_sections == held
    # A kept turn reports the pinned sections, not this render's.
    assert rig.resolve(rig.add_turn(), "V2", sections=edited).pinned_sections == held
    # Losing the pin row keeps the pointer's sections: the CLI still holds V1.
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": first})
    repinned = rig.resolve(rig.add_turn(), "V2", sections=edited)
    assert repinned.reason == MISSING_ROW and repinned.pinned_sections == held
    assert rig.store.get(rig.session_id)["sections"] == held
    # A verified boundary takes the new render's sections.
    assert rig.resolve(rig.add_turn(), "V3", cold=True, sections=edited).pinned_sections == edited
    assert rig.store.get(rig.session_id)["sections"] == edited


@pytest.mark.parametrize("damage", ["missing_row", "missing_text", "sha_mismatch"])
def test_proven_compaction_beats_a_damaged_pin(rig, damage):
    held = {"working_memory": {"v": 1, "version": 3, "state": "ok"}}
    now = {"working_memory": {"v": 1, "state": "unavailable"}}
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True, sections=held)
    rig.add_turn("/compact", usage=COMPACT_OK)
    payload = rig.turn_request(first)
    with rig.engine.begin() as conn:
        if damage == "missing_row":
            conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": first})
        else:
            if damage == "missing_text":
                del payload[PIN_KEY]["text"]
            else:
                payload[PIN_KEY]["text"] = "edited by hand"
            conn.execute(text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"),
                         {"rj": json.dumps(payload), "id": first})
    moved = rig.resolve(rig.add_turn(), "V2", sections=now)
    assert (moved.reason, moved.text, moved.snapshot_unverified) == (COMPACTION, "V2", False)
    assert moved.pinned_sections == now and moved.candidate_sections is None
    assert rig.store.get(rig.session_id)["sections"] == now


def test_unverified_repin_carries_candidate_sections_until_a_verified_boundary(rig):
    held = {"working_memory": {"v": 1, "version": 3, "state": "ok"}}
    render = {"working_memory": {"v": 1, "version": 4, "state": "ok"}}
    first = rig.add_turn()
    assert rig.resolve(first, "V1", cold=True, sections=held).candidate_sections is None
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": first})
    repinned = rig.resolve(rig.add_turn(), "V2", sections=render)
    assert (repinned.pinned_sections, repinned.candidate_sections) == (held, render)
    kept = rig.resolve(rig.add_turn(), "V3", sections={"working_memory": {"v": 1, "version": 9}})
    assert (kept.action, kept.pinned_sections, kept.candidate_sections) == ("kept", held, render)
    verified = rig.resolve(rig.add_turn(), "V4", cold=True, sections=render)
    assert (verified.pinned_sections, verified.candidate_sections) == (render, None)
    assert "candidate_sections" not in rig.store.get(rig.session_id)
    assert rig.resolve(rig.add_turn(), "V5").candidate_sections is None


def test_legacy_adopt_records_only_candidate_sections(rig):
    render = {"working_memory": {"v": 1, "version": 2, "state": "ok"}}
    adopted = rig.resolve(rig.add_turn(), "V1", sections=render)
    assert (adopted.reason, adopted.pinned_sections, adopted.candidate_sections) == (LEGACY_ADOPT, None, render)
    kept = rig.resolve(rig.add_turn(), "V2")
    assert (kept.pinned_sections, kept.candidate_sections) == (None, render)


def test_pending_boundary_sections_surface_only_after_confirm(rig):
    held = {"working_memory": {"v": 1, "version": 3, "state": "ok"}}
    rig.resolve(rig.add_turn(), "V1", cold=True, sections=held)
    rig.resolve(rig.add_turn("/new"), "V2", cold=True, dispatched=False,
                sections={"working_memory": {"v": 1, "version": 4, "state": "ok"}})
    assert rig.resolve(rig.add_turn(), "V2").pinned_sections == held


def test_pin_text_survives_backfill_and_rearm(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    # The worker's prepared-messages backfill runs after the pin write.
    rig.turns.update_turn(turn_id=first, request_payload={"messages": [{"role": "system", "content": "V1"}]})
    payload = rig.turn_request(first)
    assert payload[PIN_KEY]["text"] == "V1" and payload["messages"][0]["content"] == "V1"
    # A delivery re-arm starts a new attempt on the same row; the pin text is an audit.
    rearmed = request_payload_with_audit(json.dumps(payload), {"messages": []}, rearm=True)
    assert json.loads(rearmed)[PIN_KEY]["text"] == "V1"
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"), {"rj": rearmed, "id": first})
    again = rig.resolve(first, "V2")  # the re-armed attempt resumes on its own pin
    assert (again.action, again.text) == ("kept", "V1")
    assert rig.turn_request(first)[PIN_KEY]["text"] == "V1"


def test_pin_move_is_staged_until_the_bridge_confirms_it(rig):
    first = rig.add_turn()
    rig.store.resolve(session_id=rig.session_id, turn_id=first, storage_bot=rig.bot,
                      rendered="V1", cold_start=True)
    assert rig.store.get(rig.session_id) is None
    assert rig.metadata()[PENDING_KEY]["turn_id"] == first
    assert rig.turn_request(first)[PIN_KEY]["pending"] is True

    assert rig.store.confirm(session_id=rig.session_id, turn_id=first)
    assert rig.store.get(rig.session_id)["turn_id"] == first
    assert PENDING_KEY not in rig.metadata()
    assert "pending" not in rig.turn_request(first)[PIN_KEY]
    assert rig.store.confirm(session_id=rig.session_id, turn_id=first) is False  # idempotent


def test_codex_repin_cancelled_before_dispatch_is_redelivered(rig):
    """Codex r1: a boundary turn that never reached the bridge must not leave
    later resumed turns on a pin the thread never received."""
    from codex_bridge._bridge_helpers import system_prompt_for_attempt
    from llm_bawt.agent_backends.codex import CodexBackend

    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": first})

    cancelled_turn = rig.add_turn()
    cancelled = rig.resolve(cancelled_turn, "V2", dispatched=False)
    assert cancelled.reason == MISSING_ROW and cancelled.is_boundary
    assert rig.turn_request(cancelled_turn)[PIN_KEY]["pending"] is True  # logs never claim it took effect

    resumed = rig.resolve(rig.add_turn(), "V3")
    # The boundary is decided again and takes this turn's render: V2 never reached the thread.
    assert (resumed.reason, resumed.is_boundary, resumed.text) == (MISSING_ROW, True, "V3")
    delivery = CodexBackend.system_prompt_delivery(None, resumed.is_boundary)
    assert system_prompt_for_attempt(resumed.text, delivery, resuming=True) == "V3"
    assert rig.store.confirm(session_id=rig.session_id, turn_id=cancelled_turn) is False  # superseded


def test_unconfirmed_boundaries_are_decided_again(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    # /new cancelled before the bridge reset the session: the harness still holds V1.
    rig.resolve(rig.add_turn("/new hi"), "V2", cold=True, dispatched=False)
    kept = rig.resolve(rig.add_turn(), "V2")
    assert (kept.action, kept.text, kept.pin_turn_id) == ("kept", "V1", first)
    assert PENDING_KEY not in rig.metadata()  # the stale staged move is dropped

    # The first turn after /compact fails before dispatch: the next turn re-pins.
    rig.add_turn("/compact", usage=COMPACT_OK)
    rig.resolve(rig.add_turn(), "V3", dispatched=False)
    moved = rig.resolve(rig.add_turn(), "V4")
    assert (moved.reason, moved.text) == (COMPACTION, "V4")
    assert rig.store.get(rig.session_id)["sha256"] == prompt_sha256("V4")


def test_missing_turn_or_session_records_nothing(rig):
    assert rig.resolve("turn-missing", "V1", cold=True) is None
    turn = rig.add_turn()
    assert rig.store.resolve(
        session_id=str(uuid.uuid4()), turn_id=turn, storage_bot=rig.bot, rendered="V1", cold_start=True,
    ) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT request_json FROM turn_logs WHERE id=:id"), {"id": turn}).scalar_one() is None


# ── bare reset boundary (Snark live finding on /new) ────────────────────────

WM = {"working_memory": {"v": 1, "version": 6, "state": "ok"}}


def _rotate_to_successor(rig):
    """A bare /new archives the thread and binds a fresh successor session."""
    predecessor, successor = rig.session_id, str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status='archived' WHERE id=:id"), {"id": predecessor})
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now())
        """), {"id": successor, "bot": rig.bot, "user": rig.user})
    rig.session_id = successor
    return predecessor, successor


def _bare_new(rig, resume=None):
    """The reset_only turn: marks the bound session, pins and audits nothing."""
    turn = rig.add_turn("/new")
    assert rig.store.mark_reset_boundary(session_id=rig.session_id, turn_id=turn, resume_id=resume)
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT request_json FROM turn_logs WHERE id=:id"),
                            {"id": turn}).scalar_one() is None
    return turn


def test_bare_new_then_seeded_resume_pins_cold_start_on_the_successor(rig):
    rig.resolve(rig.add_turn(), "V1", cold=True)
    predecessor, successor = _rotate_to_successor(rig)
    reset_turn = _bare_new(rig)
    assert rig.metadata()[BOUNDARY_KEY]["turn_id"] == reset_turn
    with rig.engine.connect() as conn:  # the marker is on the successor, never the archived thread
        old = conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:id"),
                           {"id": predecessor}).scalar_one() or {}
    assert BOUNDARY_KEY not in old and old[PIN_KEY]["sha256"] == prompt_sha256("V1")

    # The bridge bound a seed transcript; the first real turn resumes it.
    first = rig.add_turn("whats in your todo")
    decision = rig.resolve(first, "V2", sections=WM, resume="seed-41a1")
    assert (decision.reason, decision.is_boundary, decision.snapshot_unverified) == (COLD_START, True, False)
    assert (decision.pinned_sections, decision.candidate_sections) == (WM, None)
    pointer = rig.store.get(successor)
    assert pointer["turn_id"] == first and pointer["sections"] == WM
    assert "candidate_sections" not in pointer and pointer["snapshot_unverified"] is False
    assert BOUNDARY_KEY not in rig.metadata()  # consumed by the confirm
    kept = rig.resolve(rig.add_turn(), "V3", resume="seed-41a1")
    assert (kept.action, kept.text) == ("kept", "V2")


def test_unconfirmed_first_turn_after_bare_new_is_decided_again(rig):
    _rotate_to_successor(rig)
    _bare_new(rig)
    cancelled = rig.add_turn()
    assert rig.resolve(cancelled, "V2", resume="seed", dispatched=False).reason == COLD_START
    assert BOUNDARY_KEY in rig.metadata() and rig.store.get(rig.session_id) is None
    retried = rig.resolve(rig.add_turn(), "V3", sections=WM, resume="seed")
    assert (retried.reason, retried.text, retried.snapshot_unverified) == (COLD_START, "V3", False)
    assert BOUNDARY_KEY not in rig.metadata()


def test_same_session_bare_reset_moves_an_existing_pin(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True, resume="old")
    _bare_new(rig, resume="old")  # inter_bot_reset_only: no rotation, same session
    moved = rig.resolve(rig.add_turn(), "V2", sections=WM, resume="reseeded")
    assert (moved.action, moved.reason, moved.text, moved.snapshot_unverified) == ("pinned", COLD_START, "V2", False)
    assert rig.store.get(rig.session_id)["sections"] == WM


def test_reset_that_never_ran_keeps_the_pin_and_drops_the_marker(rig):
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True, resume="old")
    _bare_new(rig, resume="old")
    kept = rig.resolve(rig.add_turn(), "V2", resume="old")  # still the old SDK session
    assert (kept.action, kept.text, kept.pin_turn_id) == ("kept", "V1", first)
    assert BOUNDARY_KEY not in rig.metadata()


def test_bare_reset_marker_beats_a_proven_compaction(rig):
    rig.resolve(rig.add_turn(), "V1", cold=True, resume="old")
    rig.add_turn("/compact", usage=COMPACT_OK)
    _bare_new(rig, resume="old")
    moved = rig.resolve(rig.add_turn(), "V2", sections=WM, resume="reseeded")
    assert (moved.reason, moved.snapshot_unverified) == (COLD_START, False)
    assert (moved.pinned_sections, moved.candidate_sections) == (WM, None)


def test_bare_reset_marker_needs_a_session(rig):
    turn = rig.add_turn("/new")
    assert rig.store.mark_reset_boundary(session_id=str(uuid.uuid4()), turn_id=turn, resume_id=None) is False


def test_reset_skill_facts_land_on_the_session_and_need_it(rig):
    facts = {"refs": ["llm-bawt"], "tokens": 4200, "turn_id": "t"}
    assert rig.store.record_reset_skills(session_id=rig.session_id, facts=facts) is True
    assert rig.metadata()[RESET_SKILLS_KEY] == facts
    assert rig.store.record_reset_skills(session_id=str(uuid.uuid4()), facts=facts) is False


# ── TASK-1309: the skill selection is pinned with the bytes ─────────────────

def _manifest(*refs):
    return {"harness": "claude-code", "injected": [],
            "entries": [{"ref": ref, "mode": "available", "source": "ambient"} for ref in refs],
            "manifest_sha": uuid.uuid4().hex}


def test_pinned_skills_move_only_at_a_boundary(rig):
    first_skills, edited, after_new = _manifest("alpha"), _manifest("alpha", "beta"), _manifest("gamma")
    first = rig.add_turn()
    pinned = rig.resolve(first, "V1", cold=True, skill_manifest=first_skills)
    assert pinned.skill_manifest == first_skills
    assert rig.turn_request(first)[PIN_KEY]["skill_manifest"] == first_skills

    # Assignments edited mid-session: the kept turn replays the pinned skills.
    second = rig.add_turn()
    kept = rig.resolve(second, "V1", skill_manifest=edited)
    assert (kept.action, kept.skill_manifest) == ("kept", first_skills)
    assert "skill_manifest" not in rig.turn_request(second)[PIN_KEY]

    moved = rig.resolve(rig.add_turn("/new hi"), "V2", cold=True, skill_manifest=after_new)
    assert (moved.action, moved.skill_manifest) == ("pinned", after_new)
    assert rig.resolve(rig.add_turn(), "V2", skill_manifest=edited).skill_manifest == after_new


def test_injected_skills_are_pinned_with_the_bytes(rig):
    # TASK-1691: kept turns replay the pin's injected skills, not this render's.
    first = rig.add_turn()
    pinned = rig.resolve(first, "V1", cold=True, injected_skills=("git-commits", "bad ref"))
    assert pinned.injected_skills == ("git-commits",)
    assert rig.turn_request(first)[PIN_KEY]["injected_skills"] == ["git-commits"]
    kept = rig.resolve(rig.add_turn(), "V1", injected_skills=())
    assert (kept.action, kept.injected_skills) == ("kept", ("git-commits",))
    moved = rig.resolve(rig.add_turn("/new hi"), "V2", cold=True)
    assert moved.injected_skills == ()


def test_a_session_pinned_unmanaged_stays_unmanaged_until_its_boundary(rig):
    # Also a pin taken before skills were pinned: its record has no manifest.
    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True)
    assert "skill_manifest" not in rig.turn_request(first)[PIN_KEY]
    kept = rig.resolve(rig.add_turn(), "V1", skill_manifest=_manifest("alpha"))
    assert (kept.action, kept.skill_manifest) == ("kept", None)


def test_a_corrupt_pinned_selection_fails_the_turn(rig):
    from agent_bridge.skill_selection import SkillSelectionError

    first = rig.add_turn()
    rig.resolve(first, "V1", cold=True, skill_manifest=_manifest("alpha"))
    payload = rig.turn_request(first)
    payload[PIN_KEY]["skill_manifest"] = {"entries": [{"ref": "../escape"}], "manifest_sha": "x"}
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:rj WHERE id=:id"),
                     {"rj": json.dumps(payload), "id": first})
    second = rig.add_turn()
    with pytest.raises(SkillSelectionError):
        rig.resolve(second, "V1", skill_manifest=_manifest("alpha"))
    # Nothing was recorded for the failed turn, and the pin did not move.
    with rig.engine.connect() as conn:
        raw = conn.execute(text("SELECT request_json FROM turn_logs WHERE id=:id"),
                           {"id": second}).scalar_one()
    assert raw is None or PIN_KEY not in json.loads(raw)
    assert rig.store.get(rig.session_id)["turn_id"] == first
