"""Pure §4.2 classification and §4.3 projection (TASK-2480)."""
from llm_bawt.delivery.state import EventKind as E, ItemKind as I, RunClass as R, classify, project

C, A, S, REJ, DONE, NR, UNC = E.CLAIMED, E.ACCEPTED, E.STARTED, E.REJECTED, E.DONE, E.NOT_RUN, E.UNCERTAIN


def test_never_claimed_is_not_run():
    assert classify([], run_ended=True) is R.NOT_RUN


def test_rejected_without_started_is_not_run():
    assert classify([C, A, REJ], run_ended=True) is R.NOT_RUN


def test_missing_evidence_is_never_proof():
    assert classify([C], run_ended=True) is R.UNCERTAIN
    assert classify([C, A], run_ended=True) is R.UNCERTAIN


def test_started_and_finished_ran():
    assert classify([C, A, S], run_ended=True, finished=True) is R.RAN


def test_started_then_torn_down_is_uncertain():
    assert classify([C, A, S], run_ended=True, finished=False) is R.UNCERTAIN


def test_live_until_run_ends():
    assert classify([C, A], run_ended=False) is R.LIVE
    assert classify([C, A, S], run_ended=False) is R.LIVE


def test_conflicts():
    assert classify([S, REJ], run_ended=True) is R.CONFLICT
    assert classify([S, NR], run_ended=True) is R.CONFLICT
    assert classify([DONE, UNC], run_ended=True) is R.CONFLICT


def test_projection_rows():
    assert project(I.ADDITION, [S, REJ]).last_error == "outcome_unknown: evidence_conflict"  # 1
    assert project(I.ADDITION, [C, A, S]).status == "DELIVERED"  # 2
    assert project(I.USER_ADDITION, [S]).status == "DELIVERED"
    assert project(I.START, [C, S, DONE], done_ok=True).status == "DELIVERED"  # 3
    assert project(I.ADDITION, [C, A], cancelled=True).status == "CANCELLED"  # 4
    assert project(I.START, [C], failed_reason="max attempts").last_error == "max attempts"  # 5
    p = project(I.ADDITION, [C, A], unknown_reason="stopped")  # 6
    assert (p.status, p.last_error) == ("FAILED", "outcome_unknown: stopped")
    assert project(I.ADDITION, [C, A], live=True) is None  # 7-8 keep today's status
    assert project(I.ADDITION, []).status == "QUEUED"  # 9


def test_accepted_addition_is_not_delivered():
    # The root producing output proves nothing about an accepted addition.
    assert project(I.ADDITION, [C, A], live=True) is None


def test_only_an_output_delta_is_an_execution_frame():
    """TASK-2480 (Codex r2): the early keepalive and setup failures are not STARTED."""
    import json as _json

    from llm_bawt.service.inter_bot_dispatcher import is_execution_frame

    def chunk(obj):
        return f"data: {_json.dumps(obj)}\n\n"

    delta = lambda d: chunk({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": d}]})  # noqa: E731
    assert not is_execution_frame(": connected\n\n")
    assert not is_execution_frame("data: [DONE]\n\n")
    assert not is_execution_frame(chunk({"object": "service.warning", "warnings": ["request_failed: x"]}))
    assert not is_execution_frame(delta({"role": "assistant"}))
    assert not is_execution_frame(chunk({"object": "chat.completion.chunk",
                                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    assert is_execution_frame(delta({"content": "hi"}))
    assert is_execution_frame(delta({"tool_calls": [{"id": "t"}]}))
