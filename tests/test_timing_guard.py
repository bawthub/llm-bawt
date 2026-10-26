"""No new hand-rolled pg_stat_activity polling or fixed wall-clock asserts (TASK-1457).

See tests/timing_guard.py for why each pattern fails on a slower CI runner and
what to write instead.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

from tests.timing_guard import (
    report_findings,
    scan_clock_imports,
    scan_source,
    scan_thread_waits,
    scan_tree,
)

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


def _waits(source: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.kind)
            for finding in scan_thread_waits(textwrap.dedent(source), "tests/sample.py")]


def test_d4_flags_short_thread_waits_but_not_loop_waits_or_lookups():
    assert _waits("""
        async def test_x(event, thread, queue, future, mapping, tasks):
            assert event.wait(0.5)
            assert queue.get(timeout=1) == 1
            assert queue.get(True, 2) == 1
            assert future.result(3) == 1
            assert mapping.get(1) is None
            assert session.get(Row, 1) is None
            assert kwargs.get("delay", 1) == 1
            assert ", ".join(parts)
            assert event.wait(5)
            await asyncio.wait(tasks, timeout=0.1)
            await asyncio.wait_for(event.wait(), 0.1)
            await loop_event.wait()
    """) == [(3, ".wait(0.5) thread wait under 5s"), (4, ".get(1) thread wait under 5s"),
             (5, ".get(2) thread wait under 5s"), (6, ".result(3) thread wait under 5s")]


def test_d4_flags_a_silent_wait_or_join_but_not_a_checked_one():
    assert _waits("""
        def test_x(event, thread, other, barrier, proc):
            event.wait(timeout=30)
            barrier.wait(timeout=30)
            thread.join(30)
            other.join(30)
            assert not other.is_alive()
            assert event.wait(30)
            thread.join()
            if True:
                proc.join(30)
            assert done
    """) == [(3, ".wait(timeout) result ignored; a timeout passes silently"),
             (5, ".join(timeout) without an is_alive() check after it"),
             (11, ".join(timeout) without an is_alive() check after it")]


def test_d4_knows_a_barrier_by_its_constructor():
    assert _waits("""
        import threading
        start = threading.Barrier(2)
        def test_x(self, ready):
            start.wait(timeout=10)
            self.gate = Barrier(3)
            self.gate.wait(10)
            ready.wait(10)
    """) == [(8, ".wait(timeout) result ignored; a timeout passes silently")]


def test_d4_honours_a_timing_ok_reason():
    assert _waits("""
        def test_x(event):
            event.wait(0.1)  # timing-ok: proves the event is NOT set in time
            # timing-ok: short poll inside a bounded loop
            assert event.wait(0.1) is False
    """) == []


def test_d7_flags_clock_functions_bound_by_name():
    findings = scan_clock_imports(textwrap.dedent("""
        from time import monotonic, sleep
        from time import perf_counter as clock
        import time
    """), "src/sample.py")
    assert [(finding.line, finding.kind) for finding in findings] == [
        (2, "from time import monotonic; use time.monotonic()"),
        (3, "from time import perf_counter; use time.perf_counter()")]


def test_report_findings_scans_tests_for_d4_and_src_for_d7(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a(e):\n    e.wait(1)\nfrom time import time\n")
    (tmp_path / "src" / "a.py").write_text("from time import time\ndef f(e):\n    e.wait(1)\n")
    report = report_findings(tmp_path)
    assert [str(finding) for finding in report["D7"]] == ["src/a.py:1: from time import time; use time.time()"]
    assert {finding.path for finding in report["D4"]} == {"tests/test_a.py"}
