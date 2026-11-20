"""Tests for the turn-integrity guard predicate (TASK-306 Section B).

``_is_unbacked_work_claim`` flags a bridge/agent turn that finalized a
non-trivial narration with ZERO tool calls — the "claims work, no tool_use"
bug. Pure predicate, so this needs no DB, no mixin, no live turn.

Skips cleanly if llm_bawt isn't importable (run outside the app container).
Runnable under pytest, or standalone: ``python tests/test_turn_integrity_guard.py``.
"""

from __future__ import annotations

try:
    from llm_bawt.service.turn_lifecycle import (
        _MIN_NARRATION_CHARS_FOR_GUARD,
        _is_unbacked_work_claim,
    )

    _OK = True
    _SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001
    _OK = False
    _SKIP_REASON = f"llm_bawt deps unavailable ({exc}); run in app container"


_LONG = "I committed the fix and pushed it to origin/master, all done."  # >= 40 chars


def test_flags_bridge_turn_with_narration_and_no_tools():
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=_LONG
    ) is True


def test_not_flagged_when_tools_present():
    assert _is_unbacked_work_claim(
        is_bridge_turn=True,
        tool_call_details=[{"tool": "bash", "result": "ok"}],
        response_text=_LONG,
    ) is False


def test_not_flagged_for_native_turn():
    # Native-LLM chat replies legitimately have text and no tools.
    assert _is_unbacked_work_claim(
        is_bridge_turn=False, tool_call_details=[], response_text=_LONG
    ) is False


def test_not_flagged_for_trivial_narration():
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text="ok"
    ) is False


def test_not_flagged_for_empty_or_whitespace():
    for text in ("", "   \n\t  ", None):
        assert _is_unbacked_work_claim(
            is_bridge_turn=True, tool_call_details=[], response_text=text
        ) is False


def test_boundary_is_inclusive():
    exact = "x" * _MIN_NARRATION_CHARS_FOR_GUARD
    below = "x" * (_MIN_NARRATION_CHARS_FOR_GUARD - 1)
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=exact
    ) is True
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=below
    ) is False


def test_whitespace_is_stripped_before_length_check():
    padded = "   " + ("x" * _MIN_NARRATION_CHARS_FOR_GUARD) + "   "
    # Leading/trailing space must not push a trivial reply over the threshold,
    # nor drop a real one under it — length is measured on the stripped text.
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=padded
    ) is True
    short_padded = "          x          "  # lots of space, 1 real char
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=short_padded
    ) is False


def test_none_tool_details_treated_as_empty():
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=None, response_text=_LONG
    ) is True


class _FakeConn:
    def __init__(self, acked):
        self._acked = acked

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params):
        sql = str(statement)
        assert "new_session_provider_ack" in sql and "turn_id" in sql
        hit = params["turn"] in self._acked

        class _Result:
            def first(self_inner):
                return (1,) if hit else None

        return _Result()


class _FakeEngine:
    def __init__(self, acked):
        self.acked = set(acked)

    def connect(self):
        return _FakeConn(self.acked)


def _with_engine(engine, fn):
    import llm_bawt.utils.db as db

    original = db.get_shared_engine
    db.get_shared_engine = lambda config=None, *a, **k: engine
    try:
        return fn()
    finally:
        db.get_shared_engine = original


_RESET = "Session reset. Ready for a new conversation."


def test_reset_text_alone_is_still_a_flaggable_claim():
    # TASK-2535: the confirmation prose is >= the threshold, so the predicate
    # flags it; the exemption must come from recorded evidence, not text.
    assert _is_unbacked_work_claim(
        is_bridge_turn=True, tool_call_details=[], response_text=_RESET
    ) is True


def test_bridge_acknowledged_reset_turn_is_exempt():
    from llm_bawt.service.turn_lifecycle import _turn_has_reset_ack

    assert _with_engine(_FakeEngine({"turn-reset"}),
                        lambda: _turn_has_reset_ack(object(), "turn-reset")) is True


def test_model_imitating_reset_text_is_not_exempt():
    # Same text on a turn with no recorded provider ack stays checked.
    from llm_bawt.service.turn_lifecycle import _turn_has_reset_ack

    assert _with_engine(_FakeEngine({"turn-reset"}),
                        lambda: _turn_has_reset_ack(object(), "turn-imitation")) is False


def test_reset_ack_lookup_fails_closed_to_flagging():
    from llm_bawt.service.turn_lifecycle import _turn_has_reset_ack

    class _Broken:
        def connect(self):
            raise RuntimeError("db down")

    assert _with_engine(_Broken(), lambda: _turn_has_reset_ack(object(), "t")) is False
    assert _with_engine(None, lambda: _turn_has_reset_ack(object(), "t")) is False
    assert _with_engine(_FakeEngine({""}), lambda: _turn_has_reset_ack(object(), "")) is False


if __name__ == "__main__":
    import sys
    import traceback

    if not _OK:
        print(f"SKIP test_turn_integrity_guard: {_SKIP_REASON}")
        sys.exit(0)

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(fns)} total)")
    sys.exit(1 if failed else 0)
