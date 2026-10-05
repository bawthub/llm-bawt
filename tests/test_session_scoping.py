"""TASK-251 — the four scoped-read proofs (explicit session_id ONLY).

THE design principle (locked by Nick, 2026-07-21):
    Continuous message history is the substrate. Threads are an optional
    lens over it. A request WITHOUT session_id gets the continuous pool —
    session boundaries invisible. Scoped reads happen ONLY when a request
    carries an explicit session_id.

Proofs (all live-DB, `pytest -m integration`):
    1. User msg persists to the REQUESTED session_id
    2. Assistant + tool msgs persist to the same session_id
    3. Context assembly for a session_id-carrying turn loads only that
       thread's raw (+ the bot's rolling summaries)
    4. REGRESSION GUARD: a request WITHOUT session_id gets the continuous
       pool — messages from every thread, boundaries invisible

Plus hermetic dispatch-layer checks that the per-turn override is set
fresh on every turn (a cached instance must never leak a prior turn's
thread override into a continuous request).
"""

import re
import uuid
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


# ──────────────────────────────────────────────────────────────────────────
# Live-DB fixtures (mirrors test_session_threads.py conventions)
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def scoped_env():
    """Throwaway bot with TWO threads of real messages + a summary row.

    Layout built once per class:
      thread_a: user/assistant pair  ("alpha" turns)
      thread_b: user/assistant pair  ("beta" turns)   <- left ACTIVE
      one summary row (continuous, bot-wide)
    """
    from sqlalchemy import text
    from llm_bawt.utils.config import Config
    from llm_bawt.mcp_server.client import get_memory_client

    bot = f"_t251_pytest_{uuid.uuid4().hex[:8]}"
    user = "nick"
    config = Config()
    client = get_memory_client(config=config, bot_id=bot, user_id=user)
    manager = client.get_short_term_manager()  # _MCPShortTermManager adapter
    pg = client  # MemoryClient (embedded mode) — the raw write/read surface

    def _pg_manager():
        # Direct PostgreSQL manager for row-level assertions/cleanup.
        from llm_bawt.memory.postgresql import PostgreSQLShortTermManager
        return PostgreSQLShortTermManager(config, bot_id=bot, user_id=user)

    direct = _pg_manager()

    thread_a = direct.get_or_create_active_session(bot_id=bot, user_id=user)
    pg.add_message(role="user", content="alpha question", session_id=thread_a)
    pg.add_message(role="assistant", content="alpha answer", session_id=thread_a)
    thread_b = direct.rotate_session(bot_id=bot, user_id=user)
    pg.add_message(role="user", content="beta question", session_id=thread_b)
    pg.add_message(role="assistant", content="beta answer", session_id=thread_b)
    # A continuous rolling summary (summaries are bot-wide, provenance only).
    pg.add_message(role="summary", content="rolling summary husk", session_id=thread_a)

    env = {
        "bot": bot,
        "user": user,
        "config": config,
        "client": client,
        "manager": manager,
        "pg": pg,
        "direct": direct,
        "thread_a": thread_a,
        "thread_b": thread_b,
    }
    yield env

    with direct._backend.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM messages WHERE bot_id = :b"), {"b": bot}
        )
        conn.execute(text("DELETE FROM sessions WHERE bot_id = :b"), {"b": bot})


def _rows_for(direct, bot, session_id):
    from sqlalchemy import text
    with direct._backend.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT role, content, session_id FROM messages "
                "WHERE bot_id = :b AND session_id = :s ORDER BY timestamp"
            ),
            {"b": bot, "s": session_id},
        ).fetchall()


@pytest.mark.integration
class TestScopedPersistenceProofs:
    """Proofs 1 & 2 — explicit-session_id writes land on the REQUESTED thread."""

    def test_proof1_user_msg_persists_to_requested_thread(self, scoped_env):
        e = scoped_env
        # thread_b is ACTIVE; write explicitly to thread_a (continue-old-thread).
        e["pg"].add_message(
            role="user", content="proof1 user msg", session_id=e["thread_a"]
        )
        rows = _rows_for(e["direct"], e["bot"], e["thread_a"])
        assert any(r.content == "proof1 user msg" for r in rows), (
            "user msg did not land on the requested (non-active) thread"
        )
        # And it must NOT have leaked onto the active thread.
        active_rows = _rows_for(e["direct"], e["bot"], e["thread_b"])
        assert not any(r.content == "proof1 user msg" for r in active_rows)

    def test_proof2_assistant_and_tool_persist_to_same_thread(self, scoped_env):
        e = scoped_env
        e["pg"].add_message(
            role="assistant", content="proof2 assistant msg",
            session_id=e["thread_a"],
        )
        e["pg"].add_message(
            role="system", content="[Tool Results]\nproof2 tool msg",
            session_id=e["thread_a"],
        )
        rows = _rows_for(e["direct"], e["bot"], e["thread_a"])
        contents = [r.content for r in rows]
        assert "proof2 assistant msg" in contents
        assert "[Tool Results]\nproof2 tool msg" in contents
        active_rows = _rows_for(e["direct"], e["bot"], e["thread_b"])
        assert not any("proof2" in r.content for r in active_rows)

    def test_manager_adapter_forwards_session_override(self, scoped_env):
        """The direct PG manager adapter honors the explicit override too."""
        e = scoped_env
        # direct manager's current session is thread_b (active) — override to a.
        e["direct"].add_message(
            "user", "adapter override msg", session_id=e["thread_a"]
        )
        rows = _rows_for(e["direct"], e["bot"], e["thread_a"])
        assert any(r.content == "adapter override msg" for r in rows)


@pytest.mark.integration
class TestScopedContextProofs:
    """Proofs 3 & 4 — context pool scoping through HistoryManager."""

    def _history_manager(self, env):
        from llm_bawt.utils.history import HistoryManager

        class _NullClient:  # console shim for file-fallback error paths
            pass

        return HistoryManager(
            client=_NullClient(),
            config=env["config"],
            db_backend=env["manager"],
            bot_id=env["bot"],
        )

    def test_proof3_scoped_load_is_thread_raw_plus_summaries(self, scoped_env):
        e = scoped_env
        hm = self._history_manager(e)
        hm.load_history(session_id=e["thread_a"])
        raw = [m for m in hm.messages if m.role in ("user", "assistant", "system")]
        summaries = [m for m in hm.messages if m.role == "summary"]
        # Only thread_a raw content — nothing from thread_b.
        assert raw, "scoped load returned no raw messages"
        assert all("beta" not in (m.content or "") for m in raw), (
            f"thread_b raw leaked into thread_a's scoped pool: "
            f"{[(m.role, m.content) for m in raw]}"
        )
        assert any("alpha" in (m.content or "") for m in raw)
        # Rolling summaries ride along (continuity — bot-wide by design).
        assert summaries, "rolling summaries missing from scoped pool"

    def test_proof4_default_load_is_continuous_across_threads(self, scoped_env):
        """THE regression guard: no session_id -> continuous pool."""
        e = scoped_env
        hm = self._history_manager(e)
        hm.load_history()  # no session_id — the primary mode
        contents = [(m.content or "") for m in hm.messages]
        joined = "\n".join(contents)
        assert "alpha question" in joined, (
            "REGRESSION: continuous load lost pre-rotation thread_a messages — "
            "session boundaries must be invisible without explicit session_id"
        )
        assert "beta question" in joined, (
            "continuous load lost active-thread messages"
        )

    def test_scoped_then_default_does_not_stick(self, scoped_env):
        """A scoped load followed by a default load returns to continuous."""
        e = scoped_env
        hm = self._history_manager(e)
        hm.load_history(session_id=e["thread_a"])
        assert all("beta" not in (m.content or "") for m in hm.messages)
        hm.load_history()
        joined = "\n".join((m.content or "") for m in hm.messages)
        assert "beta question" in joined and "alpha question" in joined


# ──────────────────────────────────────────────────────────────────────────
# Hermetic dispatch-layer guards (no DB)
# ──────────────────────────────────────────────────────────────────────────

class TestDispatchSetsOverrideFresh:
    """Both dispatch paths must set _session_id_override EVERY turn.

    A cached LLMBawt instance survives across turns; if a path ever stops
    assigning the override, a scoped turn would leak its thread into the
    next continuous turn. Source-level guard, mirroring the TASK-284
    detection-parity pattern.
    """

    ASSIGN = re.compile(r"llm_bawt\._session_id_override\s*=")
    SITES = [
        "llm_bawt/service/turn_stream_worker.py",   # streaming dispatch
        "llm_bawt/service/chat_nonstream.py",   # non-streaming dispatch
    ]

    @pytest.mark.parametrize("rel", SITES)
    def test_override_assigned_every_turn(self, rel):
        text = (SRC / rel).read_text(encoding="utf-8")
        assert self.ASSIGN.search(text), (
            f"{rel}: _session_id_override is no longer assigned per-turn — "
            "a cached instance will leak a scoped thread into continuous "
            "requests (TASK-251)"
        )

    def test_override_defaults_to_none_on_instance(self):
        text = (SRC / "llm_bawt/core/base.py").read_text(encoding="utf-8")
        assert "self._session_id_override: str | None = None" in text

    def test_prepare_messages_honors_override(self):
        text = (SRC / "llm_bawt/service/core.py").read_text(encoding="utf-8")
        assert "load_history(session_id=_thread_override)" in text
        assert "invalidate_history_cache()" in text, (
            "scoped pool must be invalidated so the next continuous turn "
            "reloads the continuous pool"
        )

    def test_default_load_history_takes_continuous_branch(self):
        """load_history() without session_id must never call the scoped loader."""
        text = (SRC / "llm_bawt/utils/history.py").read_text(encoding="utf-8")
        assert "if session_id and self._db_backend:" in text, (
            "scoped branch must be gated on an explicit session_id"
        )
