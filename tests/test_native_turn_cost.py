"""TASK-1054: native Claude turn cost is a per-turn delta, not the session total.

The CLI's ``ResultMessage.total_cost_usd`` is the running SESSION total (restored
on resume from the transcript's last ``cost-state`` record), so the bridge must
subtract the pre-turn baseline before reporting ``total_cost_usd``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import claude_code_bridge.send_result as send_result_module
from agent_bridge.continuity import SeedFraming
from claude_code_bridge._bridge_helpers import _read_session_ledger
from claude_code_bridge.send_result import ClaudeResultMixin
from claude_code_bridge.send_usage import (
    ClaudeUsageMixin,
    native_turn_cost_usd,
    turn_session_ledger,
)
from claude_code_bridge.session_ops import ClaudeSessionMixin

MODEL = "claude-opus-4-6"
SID = "11111111-2222-3333-4444-555555555555"


class _NativeHarness(ClaudeResultMixin, ClaudeUsageMixin):
    def __init__(self):
        self.events = []

    @staticmethod
    def _model_provider_prefix(_model):
        return None  # direct Anthropic: no proxy price, CLI cost is authoritative

    def _publish_event(self, request_id, session_key, seq, **kwargs):
        self.events.append(kwargs)


def _result(total):
    return SimpleNamespace(
        usage={
            "input_tokens": 10,
            "cache_read_input_tokens": 50_000,
            "cache_creation_input_tokens": 2_000,
            "output_tokens": 300,
        },
        model_usage={MODEL: {"contextWindow": 200_000, "maxOutputTokens": 32_000}},
        total_cost_usd=total,
        text="done",
        content=[],
    )


def _baseline(resume_id):
    return turn_session_ledger(resume_id).cost_baseline_usd


def _usage(harness, total, baseline):
    usage, _, _ = harness._compute_result_usage(
        _result(total),
        actual_model=MODEL,
        model=MODEL,
        bot_context_window=None,
        latest_assistant_usage=None,
        latest_stream_usage=None,
        cost_baseline_usd=baseline,
    )
    return usage


@pytest.fixture
def transcripts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / ".claude" / "projects" / "-home-bridge-dev"
    project.mkdir(parents=True)

    def write(session_id, lines):
        path = project / f"{session_id}.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        return path

    return write


def _cost_state(session_id, total):
    return {"type": "cost-state", "sessionId": session_id, "totalCostUSD": total}


def test_successive_session_totals_report_the_turn_delta(transcripts):
    """Snark's fixture: session totals $3.00 then $3.50 → this turn cost $0.50."""
    transcripts(SID, [{"type": "user"}, _cost_state(SID, 3.00)])
    baseline = _baseline(SID)
    assert baseline == 3.00

    usage = _usage(_NativeHarness(), 3.50, baseline)

    assert usage["total_cost_usd"] == pytest.approx(0.50)
    assert usage["session_cost_usd"] == 3.50


def test_fresh_session_turn_cost_is_the_whole_total():
    assert _baseline(None) == 0.0
    usage = _usage(_NativeHarness(), 0.86, 0.0)
    assert usage["total_cost_usd"] == pytest.approx(0.86)


_FRAMING = SeedFraming(briefing="briefing", ack="ack")


class _SeedWriter(ClaudeSessionMixin):
    _cwd = "/home/bridge/dev"


def test_seed_transcript_baseline_is_zero_on_a_later_request(transcripts):
    """Snark P2: bare ``/new`` writes the seed in one request; the NEXT ordinary
    request resumes it via ``thread_resume_id`` with no in-memory seed
    provenance. The real seed bytes alone must prove a known-zero baseline."""
    _SeedWriter()._write_seed_transcript(SID, [{"role": "user", "content": "hi"}], _FRAMING)

    baseline = _baseline(SID)  # fresh call: no request-local state

    assert baseline == 0.0
    assert _usage(_NativeHarness(), 0.42, baseline)["total_cost_usd"] == pytest.approx(0.42)


def test_seed_then_billed_turn_uses_the_recorded_cost_state(transcripts):
    path = transcripts(SID, [])
    _SeedWriter()._write_seed_transcript(SID, [{"role": "user", "content": "hi"}], _FRAMING)
    with open(path, "a") as fh:
        fh.write(json.dumps(_billed_assistant()) + "\n")
        fh.write(json.dumps(_cost_state(SID, 0.42)) + "\n")
    assert _baseline(SID) == 0.42


def _billed_assistant():
    return {"type": "assistant", "message": {"model": MODEL, "content": []}}


def test_first_turn_start_is_proven_by_the_transcript_alone(transcripts):
    """TASK-1442: bare /new seeds in its own request, so the first executed
    turn is a resume. Only a never-billed transcript claims the first turn."""
    assert turn_session_ledger(None).first_turn_start is None  # cold path marks itself
    assert turn_session_ledger(SID).first_turn_start is None  # no transcript: unknown

    path = transcripts(SID, [])
    _SeedWriter()._write_seed_transcript(SID, [{"role": "user", "content": "hi"}], _FRAMING)
    assert turn_session_ledger(SID).first_turn_start == "seeded"

    with open(path, "a") as fh:
        fh.write(json.dumps(_billed_assistant()) + "\n")
    assert turn_session_ledger(SID).first_turn_start is None

    transcripts(SID, [{"type": "user"}, _cost_state(SID, 0.0)])
    assert turn_session_ledger(SID).first_turn_start == "clean"


def test_resume_without_recorded_baseline_is_unknown_not_session_total(transcripts):
    # The CLI billed this session (real-model assistant entry) but left no
    # cost-state, e.g. a pre-cost-state CLI version.
    transcripts(SID, [{"type": "user"}, _billed_assistant()])
    baseline = _baseline(SID)
    assert baseline is None

    usage = _usage(_NativeHarness(), 7.84, baseline)

    # Unknown → null so the UI estimates, never the $7.84 session total.
    assert usage["total_cost_usd"] is None
    assert usage["session_cost_usd"] == 7.84


def test_missing_transcript_on_resume_is_unknown(transcripts):
    assert _baseline(SID) is None


def test_total_below_baseline_is_unknown():
    assert native_turn_cost_usd(1.228, 6.321) is None
    assert native_turn_cost_usd(2.0, 2.0 + 1e-12) == 0.0
    assert native_turn_cost_usd(None, 0.0) is None


def test_cost_state_reader_takes_last_record_for_this_session_only(transcripts):
    transcripts(SID, [
        _cost_state(SID, 0.86),
        {"type": "assistant", "message": {"content": "talks about \"cost-state\""}},
        _cost_state("other-session", 99.0),
        _cost_state(SID, 1.06),
        {"type": "user"},
    ])
    assert _read_session_ledger(SID).cost_baseline_usd == 1.06


def test_double_computation_on_one_result_is_idempotent(transcripts):
    # send_handler computes result usage once for the interrupted snapshot and
    # again in finalization; both must see the same turn delta.
    harness = _NativeHarness()
    first = _usage(harness, 3.50, 3.00)
    second = _usage(harness, 3.50, 3.00)
    assert first["total_cost_usd"] == second["total_cost_usd"] == pytest.approx(0.50)


def test_successful_compaction_reports_turn_delta_not_session_total(monkeypatch):
    harness = _NativeHarness()
    monkeypatch.setattr(
        send_result_module,
        "_read_latest_compact_metadata",
        lambda _session: {"preTokens": 120_000, "postTokens": 9_000},
    )
    asyncio.run(harness._finalize_result_message(
        _result(5.40),
        request_id="req-compact",
        session_key="caid:nick",
        seq=0,
        text_parts=[],
        assistant_snapshot_text="",
        api_retry_count=0,
        api_last_error=None,
        actual_model=MODEL,
        model=MODEL,
        bot_context_window=None,
        latest_assistant_usage=None,
        latest_stream_usage=None,
        cost_baseline_usd=5.00,
        native_context_usage=None,
        compact_status="success",
        compact_error_msg=None,
        turn_session_id=SID,
        turn_screenshot_assets=[],
    ))
    usage = harness.events[-1]["token_usage"]
    assert usage["context_action"] == "compact"
    assert usage["total_cost_usd"] == pytest.approx(0.40)
    assert usage["session_cost_usd"] == 5.40
