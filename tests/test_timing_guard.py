"""No new hand-rolled pg_stat_activity polling or fixed wall-clock asserts (TASK-1457).

See tests/timing_guard.py for why each pattern fails on a slower CI runner and
what to write instead.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

from tests.timing_guard import scan_source, scan_tree

TESTS = Path(__file__).resolve().parent
# The helper is the one sanctioned poller; the guard and its samples name the patterns.
SKIP = frozenset({"tests/pg_lock_wait.py", "tests/timing_guard.py", "tests/test_timing_guard.py"})


def _scan(source: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.kind.split(";")[0])
            for finding in scan_source(textwrap.dedent(source), "tests/sample.py")]


def test_no_test_hand_rolls_lock_polling_or_asserts_a_wall_clock_limit():
    findings = scan_tree(TESTS, SKIP)
    assert not findings, (
        "Timing patterns that pass on a fast machine and fail on CI (see tests/timing_guard.py). "
        "Wait with tests/pg_lock_wait.wait_blocked_by; prove promptness with an event, barrier, "
        "fake clock or the call's own timeout result. A justified exception carries "
        "`# timing-ok: <reason>`.\n" + "\n".join(map(str, findings)))


def test_flags_an_elapsed_bound_through_assigned_names():
    assert _scan("""
        import time
        def test_x():
            started = time.monotonic()
            run()
            elapsed = time.monotonic() - started
            assert elapsed < 0.5
            assert time.perf_counter() - started <= 1
            assert 2 > elapsed
            assert outcome.elapsed_ms < 5000
    """) == [(7, "asserts wall-clock time under a fixed limit"),
             (8, "asserts wall-clock time under a fixed limit"),
             (9, "asserts wall-clock time under a fixed limit"),
             (10, "asserts wall-clock time under a fixed limit")]


def test_flags_a_deadline_assert_but_not_a_deadline_loop_or_lower_bound():
    assert _scan("""
        import time
        def test_x():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                poll()
            started = time.monotonic()
            assert time.monotonic() - started >= 0.1
            assert time.monotonic() < deadline
            assert count < 3
    """) == [(9, "asserts wall-clock time under a fixed limit")]


def test_a_bracket_a_comparison_with_now_or_data_built_from_a_timestamp_is_not_flagged():
    assert _scan("""
        import time
        def test_x():
            before = time.time()
            stamp = rotate()
            assert before <= stamp <= time.time()
            assert token.expires_at > time.time()
            now = time.time()
            session = build(now + 20)
            assert savings(session) > savings_short
            assert stamp <= before + 0.05
    """) == [(11, "asserts wall-clock time under a fixed limit")]


def test_flags_pg_stat_activity_sql_but_not_a_docstring():
    assert _scan('''
        def test_x(conn):
            """Explains why pg_stat_activity snapshots per transaction."""
            conn.execute(text("""
                SELECT 1 FROM pg_stat_activity WHERE wait_event_type = 'Lock'
            """))
            conn.execute(text("SELECT 1 FROM pg_locks WHERE NOT granted"))
    ''') == [(4, "hand-rolled pg_stat_activity query"), (7, "hand-rolled pg_locks query")]


def test_an_exemption_needs_a_reason_and_sits_on_or_above_the_statement():
    assert _scan("""
        import time
        def test_x(conn):
            started = time.monotonic()
            assert time.monotonic() - started < 9  # timing-ok: proves no 30s hang
            # timing-ok: one-shot read of this backend's own row, not a poll
            conn.execute("SELECT state FROM pg_stat_activity WHERE pid = pg_backend_pid()")
            assert time.monotonic() - started < 9  # timing-ok:
    """) == [(8, "asserts wall-clock time under a fixed limit")]
