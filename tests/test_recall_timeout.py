"""TASK-1688: only synthesis timeouts degrade into bounded authorized quotes."""
import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_bawt.recall.budgets import compact_json, tokens
from llm_bawt.recall.contracts import Evidence, EvidenceBatch, RecallError, RecallRequest
from llm_bawt.recall.service import RecallService
from tests.test_isolated_recall import SCOPE, Model, Source, request
from tests.test_recall_ledger import Sink


class TimeoutModel(Model):
    async def synthesize(self, *args):
        raise RecallError("timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RecallError("timeout"), TimeoutError("PRIVATE provider detail")])
@pytest.mark.parametrize("budget", [400, 1000, 1600])
async def test_timeout_returns_cited_bounded_partial_not_model_answer(error, budget):
    class Failed(Model):
        async def synthesize(self, *args):
            raise error

    rows = [replace(Source().rows[0], id=f"m{i}", text=(f"Record {i}: relay background " * 20),
                    start_char=120, source_bot="peer", session_id="s1") for i in range(6)]
    result = await RecallService([Source(rows)], Failed()).recall(request(output_tokens=budget), SCOPE)
    assert result["status"] == "partial" and result["degraded"] is True
    assert result["error_code"] == "timeout" and result["handle"]
    assert 1 <= len(result["findings"]) <= 6
    assert "degraded" in result["uncertainty"] and "not selected answers" in result["uncertainty"]
    assert "PRIVATE" not in compact_json(result)
    assert result["truncated"]
    assert tokens(compact_json(result)) == result["metrics"]["parent_tokens"] <= budget
    assert result["metrics"]["model_calls"] == 1 and result["metrics"]["utility_output_tokens"] == 0
    for finding in result["findings"]:
        row = next(r for r in rows if r.id == finding["id"])
        assert finding["quote"] in row.text and finding["source"] == "messages"
        assert finding["start_char"] == row.start_char
        assert finding["truncated"] is True


@pytest.mark.asyncio
async def test_timeout_uses_retrieval_rank_not_oldest_first_and_retains_ledger():
    rows = [replace(Source().rows[0], id=f"m{i}", text=f"Evidence {i}", at=f"2026-10-{28-i:02d}T00:00:00Z")
            for i in range(6)]
    sink = Sink()
    service = RecallService([Source(rows)], TimeoutModel(), ledger=sink)
    result = await service.recall(request(output_tokens=1600), SCOPE)
    await service.drain_ledger()
    assert [f["id"] for f in result["findings"]] == [r.id for r in rows]
    assert result["coverage"]["messages"]["records"] == 6
    assert sink.writes[0][0] == result["handle"]
    assert len(sink.writes[0][3]) == 6
    assert all(e["cited"] and e["admitted"] for e in sink.writes[0][3])


@pytest.mark.asyncio
async def test_leading_whitespace_cannot_hide_the_only_nonblank_quote():
    row = replace(Source().rows[0], text=" " * 450 + "A real receipt.", start_char=25)
    packet = await RecallService([Source([row])], TimeoutModel()).recall(request(), SCOPE)
    assert packet["status"] == "partial"
    assert packet["findings"][0]["quote"] == "A real receipt."
    assert packet["findings"][0]["start_char"] == 475


@pytest.mark.asyncio
@pytest.mark.parametrize("with_valid", [False, True])
async def test_whitespace_only_rows_never_become_rejected_fallback_quotes(with_valid):
    row = Source().rows[0]
    blank = replace(row, id="blank", text=" \n\t ")
    class Forced(TimeoutModel):
        def __init__(self):
            super().__init__()
            self.attempts = 0
        async def synthesize(self, *args):
            self.attempts += 1
            return await super().synthesize(*args)
    model = Forced()
    packet = await RecallService([Source([blank, row] if with_valid else [blank])], model).recall(request(), SCOPE)
    assert packet["status"] == ("partial" if with_valid else "no_match")
    assert packet["rejected_findings"] == 0
    assert model.attempts == int(with_valid)
    assert [finding["id"] for finding in packet["findings"]] == ([row.id] if with_valid else [])


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_swallowed_as_a_synthesis_timeout():
    class Cancelled(Model):
        async def synthesize(self, *args):
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await RecallService([Source()], Cancelled()).recall(request(), SCOPE)


@pytest.mark.asyncio
async def test_timeout_handle_refreshes_with_same_authority_and_scope():
    source = Source()
    service = RecallService([source], TimeoutModel())
    first = await service.recall(request(), SCOPE)
    assert "cool glass" not in repr(service.handles._entries)
    source.rows = [replace(source.rows[0], text="Fresh correction: not deployed.")]
    followup = request(handle=first["handle"])
    denied = await service.recall(followup, replace(SCOPE, session_id="other"))
    assert denied["error_code"] == "invalid_handle" and len(source.calls) == 1
    refreshed = await service.recall(followup, SCOPE)
    assert refreshed["status"] == "partial"
    assert refreshed["findings"][0]["quote"] == source.rows[0].text
    assert len(source.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["model_unavailable", "empty_response", "invalid_synthesis"])
async def test_model_failures_after_retrieval_degrade_to_quotes(code):
    """TASK-1751: a failed utility model keeps the retrieved, authorized evidence."""
    class Failed(Model):
        async def synthesize(self, *args):
            raise RecallError(code)
    result = await RecallService([Source()], Failed()).recall(request(), SCOPE)
    assert result["status"] == "partial" and result["degraded"] is True
    assert result["error_code"] == code and result["handle"]
    assert result["findings"] and "Synthesis failed; degraded" in result["uncertainty"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not json", '{"findings": "nope"}', "x " * 5000])
async def test_unusable_model_output_degrades_instead_of_discarding(raw):
    class Bad(Model):
        async def synthesize(self, *args):
            return raw
    result = await RecallService([Source()], Bad()).recall(request(), SCOPE)
    assert result["status"] == "partial" and result["error_code"] == "invalid_synthesis"
    assert result["findings"] and result["metrics"]["utility_output_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["busy", "unauthorized", "input_budget"])
async def test_other_model_failures_do_not_gain_a_fallback(code):
    class Failed(Model):
        async def synthesize(self, *args):
            raise RecallError(code)
    result = await RecallService([Source()], Failed()).recall(request(), SCOPE)
    assert result["status"] == "error" and result["error_code"] == code
    assert result["findings"] == [] and result["handle"] is None


@pytest.mark.asyncio
async def test_partial_coverage_kept_and_no_protected_rows_reach_fallback():
    good = Source(EvidenceBatch(Source().rows, withheld=1))
    bad = Source(error=RecallError("timeout"))
    bad.source = "memories"
    packet = await RecallService([good, bad], TimeoutModel()).recall(
        RecallRequest(query="relay", sources=("messages", "memories")), SCOPE)
    assert packet["status"] == "partial" and len(packet["findings"]) == 1
    assert packet["coverage"]["messages"]["status"] == "partial"
    assert packet["coverage"]["memories"]["status"] == "unavailable"
    denied = Source([replace(Source().rows[0], source_bot="excluded")])
    packet = await RecallService([good, denied], TimeoutModel()).recall(request(), SCOPE)
    assert packet["status"] == "error" and packet["error_code"] == "unauthorized"
    assert not packet["findings"]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [400, 1000, 1600])
async def test_24_retrieved_records_are_bounded_not_all_dumped(budget):
    sources = []
    for kind in ("messages", "memories", "tasks", "tool_evidence"):
        source = Source([Evidence(kind, f"{kind}{i}", SCOPE.bot_id, SCOPE.user_id, f"relay background {i}")
                         for i in range(6)])
        source.source = kind
        sources.append(source)
    packet = await RecallService(sources, TimeoutModel()).recall(
        RecallRequest(query="relay", sources=tuple(s.source for s in sources), output_tokens=budget), SCOPE)
    assert packet["status"] == "partial"
    assert sum(c["records"] for c in packet["coverage"].values()) == 24
    assert 1 <= len(packet["findings"]) <= 6
    assert packet["truncated"] and "omitted" in packet["uncertainty"]
    assert tokens(compact_json(packet)) <= budget


@pytest.mark.asyncio
async def test_remaining_synthesis_deadline_cancels_model_but_preserves_retrieval(monkeypatch):
    import llm_bawt.recall.service as module
    # Advance only the service's clock to exhaust its synthesis allowance;
    # cancellation is deterministic, not a sleep/timing assertion.
    clock = [time.monotonic()]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    cancelled = asyncio.Event()
    class Slow(Model):
        async def synthesize(self, *args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
    class Retrieved(Source):
        async def search(self, *args):
            clock[0] += 30
            return self.rows
    packet = await RecallService([Retrieved()], Slow()).recall(request(), SCOPE)
    assert packet["status"] == "partial" and packet["findings"] and cancelled.is_set()


@pytest.mark.asyncio
async def test_adapter_passes_its_earlier_deadline_after_scope_lookup(monkeypatch):
    from llm_bawt.mcp_server import recall_tools
    captured = []
    clock = [time.monotonic()]
    async def scope(_):
        clock[0] += 3
        return SCOPE
    service = RecallService([Source()], TimeoutModel())
    original = service.recall
    async def recall(*args, deadline):
        captured.append(deadline)
        return await original(*args, deadline=deadline)
    monkeypatch.setattr(service, "recall", recall)
    monkeypatch.setattr(recall_tools, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(recall_tools, "current_task_turn_context", lambda: object())
    monkeypatch.setattr(recall_tools, "_runtime", lambda: (SimpleNamespace(scope=scope), service))
    packet = json.loads(await recall_tools.context_recall(request()))
    assert packet["status"] == "partial" and packet["findings"]
    assert captured == [clock[0] - 3 + recall_tools.MAX_SECONDS]
