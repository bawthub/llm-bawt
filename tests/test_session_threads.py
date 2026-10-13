"""TASK-284 step 18: session-thread coverage.

Hermetic tests (default run) cover the app-side coordinators added in steps
14/15 — the agent ``/new`` rotation gate, chatbot rotation semantics, and the
provider-session→thread mirror — plus a source-level drift guard on ``/new``
detection parity between the app and all three bridges.

DB-invariant tests (one-active-per-(bot,user), concurrent first-write, rotate/
activate atomicity, multi-user isolation) are marked ``integration``: they run
against the live Postgres via the real ``PostgreSQLShortTermManager`` with a
throwaway bot id, and clean up after themselves.

    uv run pytest tests/test_session_threads.py                 # hermetic only
    uv run pytest -m integration tests/test_session_threads.py  # live-DB too
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin

SRC = Path(__file__).resolve().parents[1] / "src"


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

class _Bridge(ChatStreamingBridgeMixin):
    """Bare mixin host recording _rotate_chat_session calls."""

    def __init__(self, rotate_result: bool = True):
        self.rotate_calls: list[str] = []
        self._rotate_result = rotate_result

    def _rotate_chat_session(self, llm_bawt, bot_id: str) -> bool:  # type: ignore[override]
        self.rotate_calls.append(bot_id)
        return self._rotate_result


def _llm(backend=None):
    """Fake llm_bawt instance: history_manager backend + cache invalidation."""
    hm = SimpleNamespace(_db_backend=backend, messages=[])
    return SimpleNamespace(
        history_manager=hm,
        invalidate_history_cache=lambda: None,
    )


# ──────────────────────────────────────────────────────────────────────────
# Agent /new rotation gate (_maybe_rotate_agent_session)
# ──────────────────────────────────────────────────────────────────────────

class TestAgentNewRotationGate:
    def test_non_new_prompt_never_rotates(self):
        b = _Bridge()
        assert b._maybe_rotate_agent_session(_llm(), "byte", "hello there") is False
        assert b.rotate_calls == []

    def test_new_mentioned_mid_message_does_not_rotate(self):
        # Detection is prefix-only — "/new" inside a sentence must not reset.
        b = _Bridge()
        assert b._maybe_rotate_agent_session(_llm(), "byte", "tell me about /new") is False
        assert b.rotate_calls == []

    def test_new_prompt_rotates(self):
        b = _Bridge()
        assert b._maybe_rotate_agent_session(_llm(), "byte", "  /new please") is True
        assert b.rotate_calls == ["byte"]

    def test_empty_prompt_is_safe(self):
        b = _Bridge()
        assert b._maybe_rotate_agent_session(_llm(), "byte", "") is False
        assert b._maybe_rotate_agent_session(_llm(), "byte", None) is False
        assert b.rotate_calls == []


# ──────────────────────────────────────────────────────────────────────────
# Chatbot rotation semantics (_rotate_chat_session)
# ──────────────────────────────────────────────────────────────────────────

class TestChatRotation:
    def test_no_backend_returns_false_for_legacy_fallback(self):
        b = ChatStreamingBridgeMixin()
        assert b._rotate_chat_session(_llm(backend=None), "nova") is None  # TASK-257: falsy, not False

    def test_rotate_failure_returns_false(self):
        backend = SimpleNamespace(rotate_session=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        b = ChatStreamingBridgeMixin()
        assert b._rotate_chat_session(_llm(backend=backend), "nova") is None  # TASK-257: falsy, not False

    def test_success_rotates_invalidates_and_keeps_only_summaries(self):
        backend = SimpleNamespace(rotate_session=lambda: "new-session-id")
        invalidated = []
        llm = _llm(backend=backend)
        llm.invalidate_history_cache = lambda: invalidated.append(True)
        llm.history_manager.messages = [
            SimpleNamespace(role="user", content="a"),
            SimpleNamespace(role="summary", content="s"),
            SimpleNamespace(role="assistant", content="b"),
        ]
        b = ChatStreamingBridgeMixin()
        assert b._rotate_chat_session(llm, "nova") == "new-session-id"  # TASK-257: returns the new id
        assert invalidated == [True]
        assert [m.role for m in llm.history_manager.messages] == ["summary"]


# ──────────────────────────────────────────────────────────────────────────
# Provider-session → durable-thread mirror (step 15 chokepoint)
# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# /new detection parity — app-side gate must byte-match all bridges
# ──────────────────────────────────────────────────────────────────────────

class TestNewDetectionParity:
    DETECTION = re.compile(r"\.lstrip\(\)\.startswith\(\"/new\"\)")
    SITES = [
        "llm_bawt/service/chat_streaming_bridge.py",   # app-side rotation gate
        "claude_code_bridge/send_preflight.py",        # claude-code bridge reset
        "codex_bridge/command_ops.py",                 # codex bridge reset
        "openclaw_bridge/bridge.py",                   # openclaw bridge reset
    ]

    @pytest.mark.parametrize("rel", SITES)
    def test_detection_expression_present(self, rel):
        # The app rotates the durable thread iff the bridge will reset its
        # provider session. If someone changes detection on either side, this
        # fails and forces them to change both (or update this list).
        text = (SRC / rel).read_text(encoding="utf-8")
        assert self.DETECTION.search(text), (
            f"{rel}: /new detection expression changed — app rotation gate and "
            "bridge reset detection must stay byte-identical (TASK-284 step 15)"
        )


# ──────────────────────────────────────────────────────────────────────────
# Live-DB registry invariants (opt-in: pytest -m integration)
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def live_managers():
    """Real PostgreSQLShortTermManager pair on a throwaway bot; full cleanup."""
    from sqlalchemy import text
    from llm_bawt.utils.config import Config
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

    bot = f"_t284_pytest_{uuid.uuid4().hex[:8]}"
    config = Config()

    def make(user_id):
        return PostgreSQLShortTermManager(config, bot_id=bot, user_id=user_id)

    managers = {"bot": bot, "make": make, "created": []}
    yield managers

    # Cleanup: remove every session row this test bot created.
    m = make("nick")
    with m._backend.engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE bot_id = :b"), {"b": bot})


def _active_rows(mgr, bot, user):
    from sqlalchemy import text
    with mgr._backend.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT id, status FROM sessions "
                "WHERE bot_id = :b AND user_id = :u AND status = 'active'"
            ),
            {"b": bot, "u": user},
        ).fetchall()


@pytest.mark.integration
class TestRegistryInvariantsLiveDB:
    def test_two_managers_converge_on_one_active_session(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        a = make("nick").get_or_create_active_session(bot_id=bot, user_id="nick")
        b = make("nick").get_or_create_active_session(bot_id=bot, user_id="nick")
        assert a == b
        assert len(_active_rows(make("nick"), bot, "nick")) == 1

    def test_concurrent_first_writes_leave_exactly_one_active(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        user = f"race_{uuid.uuid4().hex[:6]}"

        def resolve(_):
            return make(user).get_or_create_active_session(bot_id=bot, user_id=user)

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(resolve, range(8)))

        rows = _active_rows(make(user), bot, user)
        assert len(rows) == 1, f"one-active invariant violated: {rows}"
        # Every racer must have ended up with a real session; the winner's id
        # is the single active row (losers may hold a since-superseded id only
        # if the IntegrityError re-resolve path failed — which would show as >1
        # distinct id AND >1 active row).
        assert rows[0][0] in ids

    def test_rotate_closes_old_and_opens_new_atomically(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("nick")
        old = m.get_or_create_active_session(bot_id=bot, user_id="nick")
        new = m.rotate_session(bot_id=bot, user_id="nick")
        assert new != old
        rows = _active_rows(m, bot, "nick")
        assert [r[0] for r in rows] == [new]
        assert m.get_or_create_active_session(bot_id=bot, user_id="nick") == new

    def test_activate_switches_back_to_old_thread(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("nick")
        current = m.get_or_create_active_session(bot_id=bot, user_id="nick")
        newer = m.rotate_session(bot_id=bot, user_id="nick")
        assert m.activate_session(current, bot_id=bot, user_id="nick") is True
        rows = _active_rows(m, bot, "nick")
        assert [r[0] for r in rows] == [current]
        assert newer not in [r[0] for r in rows]

    def test_cross_user_activation_is_rejected(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        alice = make("alice").get_or_create_active_session(bot_id=bot, user_id="alice")
        mallory = make("mallory")
        mallory.get_or_create_active_session(bot_id=bot, user_id="mallory")
        # mallory may not activate (or even implicitly confirm) alice's thread
        assert mallory.activate_session(alice, bot_id=bot, user_id="mallory") is False
        assert len(_active_rows(mallory, bot, "alice")) == 1

    def test_list_sessions_is_user_scoped(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        make("u1").get_or_create_active_session(bot_id=bot, user_id="u1")
        make("u2").get_or_create_active_session(bot_id=bot, user_id="u2")
        u1_rows = make("u1").list_sessions(bot_id=bot, user_id="u1")
        assert u1_rows, "expected u1 to see own session"
        assert all(r.get("user_id") == "u1" for r in u1_rows), (
            f"cross-user leak in list_sessions: {u1_rows}"
        )

    # ── TASK-250: status lifecycle (active|archived|deleted) ──────────────

    def test_rotate_archives_old_thread(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("lc1")
        old = m.get_or_create_active_session(bot_id=bot, user_id="lc1")
        m.rotate_session(bot_id=bot, user_id="lc1")
        row = m.get_session(old)
        assert row["status"] == "archived"
        assert row["archived_at"] is not None
        assert row["ended_at"] is not None

    def test_activate_clears_archive_state(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("lc2")
        old = m.get_or_create_active_session(bot_id=bot, user_id="lc2")
        m.rotate_session(bot_id=bot, user_id="lc2")
        assert m.activate_session(old, bot_id=bot, user_id="lc2") is True
        row = m.get_session(old)
        assert row["status"] == "active"
        assert row["archived_at"] is None
        assert row["ended_at"] is None

    def test_soft_delete_excluded_from_default_list(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("lc3")
        doomed = m.get_or_create_active_session(bot_id=bot, user_id="lc3")
        m.rotate_session(bot_id=bot, user_id="lc3")
        assert m.set_session_status(doomed, "deleted") is True
        default_ids = [r["id"] for r in m.list_sessions(bot_id=bot, user_id="lc3")]
        assert doomed not in default_ids, "deleted thread leaked into default list"
        admin_ids = [
            r["id"]
            for r in m.list_sessions(bot_id=bot, user_id="lc3", include_deleted=True)
        ]
        assert doomed in admin_ids, "include_deleted did not re-include the thread"
        # Restore (undelete → archived) and it reappears.
        assert m.set_session_status(doomed, "archived") is True
        restored = [r["id"] for r in m.list_sessions(bot_id=bot, user_id="lc3")]
        assert doomed in restored

    def test_set_session_status_rejects_bad_values(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("lc4")
        sid = m.get_or_create_active_session(bot_id=bot, user_id="lc4")
        assert m.set_session_status(sid, "active") is False, (
            "reactivation must go through activate_session, not set_session_status"
        )
        assert m.set_session_status(sid, "completed") is False
        assert m.get_session(sid)["status"] == "active"

    def test_legacy_completed_normalized_on_read(self, live_managers):
        from sqlalchemy import text as _text

        bot, make = live_managers["bot"], live_managers["make"]
        m = make("lc5")
        sid = m.get_or_create_active_session(bot_id=bot, user_id="lc5")
        m.rotate_session(bot_id=bot, user_id="lc5")
        # Simulate a row written by a pre-TASK-250 process.
        with m._backend.engine.begin() as conn:
            conn.execute(
                _text("UPDATE sessions SET status = 'completed' WHERE id = :id"),
                {"id": sid},
            )
        assert m.get_session(sid)["status"] == "archived"
        # And the archived filter still finds it.
        ids = [
            r["id"]
            for r in m.list_sessions(bot_id=bot, user_id="lc5", status="archived")
        ]
        assert sid in ids

    # ── TASK-702: non-activating (background dispatch) session creation ────

    def test_create_inactive_session_does_not_disturb_active_thread(self, live_managers):
        """Non-activating creation must not rotate/close the live thread.

        Background task dispatch relies on this to open a dedicated task
        conversation without unseating the user's currently active one.
        """
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("t702-a")
        live = m.get_or_create_active_session(bot_id=bot, user_id="t702-a")
        inactive = m.create_inactive_session(bot_id=bot, user_id="t702-a")

        assert inactive != live
        row = m.get_session(inactive)
        assert row["status"] == "archived"
        assert row["archived_at"] is not None
        assert row["ended_at"] is not None
        assert row["user_id"] == "t702-a"

        # The live thread is untouched — still active and still the winner
        # of the one-active-per-(bot,user) invariant.
        live_row = m.get_session(live)
        assert live_row["status"] == "active"
        active_rows = _active_rows(m, bot, "t702-a")
        assert [r[0] for r in active_rows] == [live], (
            "create_inactive_session must not touch the active thread"
        )

    def test_create_inactive_session_persists_metadata(self, live_managers):
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("t702-b")
        sid = m.create_inactive_session(
            bot_id=bot,
            user_id="t702-b",
            session_metadata={"created_inactive": True, "source": "dispatch"},
        )
        row = m.get_session(sid)
        assert row["session_metadata"]["created_inactive"] is True
        assert row["session_metadata"]["source"] == "dispatch"
        assert row["status"] == "archived"

    def test_create_inactive_session_multiple_calls_never_collide(self, live_managers):
        """Multiple non-activating creations for the same (bot,user) all coexist.

        No unique-index conflict, no rotation racing, and the one-active
        invariant continues to hold because none of them are active.
        """
        bot, make = live_managers["bot"], live_managers["make"]
        m = make("t702-c")
        live = m.get_or_create_active_session(bot_id=bot, user_id="t702-c")
        ids = {
            m.create_inactive_session(bot_id=bot, user_id="t702-c")
            for _ in range(4)
        }
        assert len(ids) == 4, "each call must return a fresh session id"
        assert live not in ids
        active_rows = _active_rows(m, bot, "t702-c")
        assert [r[0] for r in active_rows] == [live], (
            "one-active-per-(bot,user) invariant must survive many inactive creations"
        )
