"""Hermetic authority, packet and resource-boundary regression tests."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_bawt.recall.budgets import BoundedWorker, compact_json, tokens
from llm_bawt.recall.contracts import Evidence, RecallError, RecallRequest, RecallScope, SourceProvider, Synthesizer
from llm_bawt.recall.handles import HandleStore
from llm_bawt.recall.service import RecallService

SCOPE = RecallScope("reviewer", "operator", "session-a", frozenset({"excluded"}), frozenset({"reviewer", "peer"}))


class Source(SourceProvider):
    source = "messages"

    def __init__(self, rows=None, error=None):
        self.rows = rows if rows is not None else [Evidence("messages", "m1", "reviewer", "operator", "The relay background is cool glass.")]
        self.error = error
        self.calls = []

    async def search(self, query, scope, task_ids):
        self.calls.append((query, scope, task_ids))
        if self.error:
            raise self.error
        return self.rows


class Model(Synthesizer):
    def __init__(self, raw=None):
        self.raw = raw
        self.calls = []

    async def synthesize(self, query, evidence, output_tokens):
        self.calls.append((query, evidence, output_tokens))
        if self.raw is not None:
            return self.raw
        return compact_json({"findings": [{"source": evidence[0].source, "id": evidence[0].id,
                                           "quote": evidence[0].text[:100]}]})


def request(**kwargs):
    return RecallRequest(query="relay background", sources=("messages",), **kwargs)


@pytest.mark.asyncio
async def test_grounded_compact_packet_exact_count_and_no_transcript():
    source, model = Source(), Model()
    result = await RecallService([source], model).recall(request(), SCOPE)
    assert result["status"] == "ok"
    assert result["findings"][0]["quote"] == source.rows[0].text
    assert result["metrics"]["parent_tokens"] == tokens(compact_json(result)) <= 1000
    assert len(model.calls) == 1
    assert "untrusted" in result["uncertainty"]
    assert "raw" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,code", [
    ("SECRET raw source fallback", "invalid_synthesis"),
    ('{"findings":[],"prose":"SECRET"}', "invalid_synthesis"),
    ('{"findings":[{"source":"messages","id":"outside","quote":"SECRET"}]}', "ungrounded_synthesis"),
    ('{"findings":[{"source":"messages","id":"m1","quote":"invented"}]}', "ungrounded_synthesis"),
    ("x" * 17000, "invalid_synthesis"),
    ('{"findings":[{"source":"messages","id":"m1","quote":" "}]}', "ungrounded_synthesis"),
])
async def test_invalid_or_hallucinated_synthesis_has_no_raw_fallback(raw, code):
    result = await RecallService([Source()], Model(raw)).recall(request(), SCOPE)
    assert {k: v for k, v in result.items() if k != "coverage"} == {
        "status": "error", "error_code": code, "findings": [], "handle": None}
    assert "SECRET" not in compact_json(result)


@pytest.mark.asyncio
async def test_failure_distinct_from_no_match_and_irrelevant_evidence():
    model = Model()
    no_match = await RecallService([Source([])], model).recall(request(), SCOPE)
    assert no_match["status"] == "no_match" and not model.calls
    failed = await RecallService([Source(error=RuntimeError("SECRET"))], model).recall(request(), SCOPE)
    assert failed["status"] == "error" and "SECRET" not in compact_json(failed)
    insufficient = await RecallService([Source()], Model('{"findings":[]}')).recall(request(), SCOPE)
    assert insufficient["status"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_provider_error_codes_are_not_a_data_channel():
    result = await RecallService([Source(error=RecallError("SECRET"))], Model()).recall(request(), SCOPE)
    assert result["error_code"] == "source_unavailable"
    assert "SECRET" not in compact_json(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"bot_id": "peer"}, {"user_id": "someone"}, {"source": "tasks"}])
async def test_provider_cannot_broaden_scope(changed):
    source = Source()
    source.rows = [replace(source.rows[0], **changed)]
    model = Model()
    result = await RecallService([source], model).recall(request(), SCOPE)
    assert result["error_code"] == "unauthorized" and not model.calls


@pytest.mark.asyncio
async def test_excluded_caller_fails_before_any_retrieval():
    source, model = Source(), Model()
    result = await RecallService([source], model).recall(request(), replace(SCOPE, bot_id="excluded"))
    assert result["error_code"] == "unauthorized"
    assert not source.calls and not model.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [
    [Evidence("messages", str(i), "reviewer", "operator", "x") for i in range(7)],
    [Evidence("messages", "m1", "reviewer", "operator", "x" * 2401)],
])
async def test_source_budgets_fail_closed(rows):
    model = Model()
    result = await RecallService([Source(rows)], model).recall(request(), SCOPE)
    assert result["error_code"] == "retrieval_budget" and not model.calls


@pytest.mark.asyncio
async def test_input_and_output_budgets_apply_to_complete_packets():
    sources = []
    for kind in ("messages", "memories", "tasks", "tool_evidence"):
        source = Source([Evidence(kind, str(i), "reviewer", "operator", "Evidence " * 250) for i in range(6)])
        source.source = kind
        sources.append(source)
    model = Model()
    result = await RecallService(sources, model).recall(
        RecallRequest(query="Evidence", sources=tuple(s.source for s in sources), output_tokens=400), SCOPE)
    assert result["status"] == "ok"
    assert result["metrics"]["utility_input_tokens"] <= 6000
    assert result["metrics"]["parent_tokens"] == tokens(compact_json(result)) <= 400
    assert result["truncated"]


@pytest.mark.parametrize("kwargs", [
    {"query": " "}, {"query": "x" * 1001}, {"output_tokens": 399}, {"output_tokens": 1601},
    {"bot_id": "peer"}, {"sources": []}, {"sources": ["messages", "messages"]},
    {"task_ids": ["../private"]}, {"task_ids": ["TASK-42"]}, {"output_tokens": True},
])
def test_request_validation(kwargs):
    with pytest.raises(ValidationError):
        RecallRequest(**{"query": "hello", **kwargs})


@pytest.mark.asyncio
async def test_query_token_budget_cannot_be_bypassed_with_unicode():
    model = Model()
    result = await RecallService([Source()], model).recall(RecallRequest(query="💥" * 200), SCOPE)
    assert result["error_code"] == "query_budget" and not model.calls


@pytest.mark.asyncio
async def test_handle_owner_expiry_tampering_and_reauthorization():
    now = [1.0]
    store = HandleStore(ttl=10, clock=lambda: now[0])
    source, model = Source(), Model()
    service = RecallService([source], model, store)
    first = await service.recall(request(), SCOPE)
    followup = RecallRequest(query="Was it deployed?", sources=("messages",), handle=first["handle"])
    other = await service.recall(followup, replace(SCOPE, user_id="other"))
    tampered = await service.recall(followup.model_copy(update={"handle": "x" * 43}), SCOPE)
    now[0] = 11
    expired = await service.recall(followup, SCOPE)
    assert other == expired == tampered
    assert expired["error_code"] == "invalid_handle"
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_followup_refreshes_and_carries_intent_not_source_text():
    store = HandleStore()
    source, model = Source(), Model()
    service = RecallService([source], model, store)
    first = await service.recall(request(), SCOPE)
    source.rows = [replace(source.rows[0], text="Deployment has not happened.")]
    second = await service.recall(RecallRequest(query="deployed?", sources=("messages",), handle=first["handle"]), SCOPE)
    assert second["findings"][0]["quote"] == "Deployment has not happened."
    assert source.calls[-1][0] == "relay background\ndeployed?"
    assert "cool glass" not in repr(store._entries)
    mismatched = await service.recall(RecallRequest(query="again", sources=("memories",), handle=first["handle"]), SCOPE)
    assert mismatched["error_code"] == "handle_scope_mismatch"


@pytest.mark.asyncio
async def test_handle_eviction_and_followup_limit():
    store = HandleStore(capacity=2, per_owner=1)
    service = RecallService([Source()], Model(), store)
    first = await service.recall(request(), SCOPE)
    second = await service.recall(request(), SCOPE)
    assert len(store._entries) == 1
    assert first["handle"] not in store._entries
    for _ in range(3):
        second = await service.recall(request(handle=second["handle"]), SCOPE)
    result = await service.recall(request(handle=second["handle"]), SCOPE)
    assert result["error_code"] == "followup_limit"


@pytest.mark.asyncio
async def test_timeout_returns_no_source_data(monkeypatch):
    import llm_bawt.recall.service as module
    monkeypatch.setattr(module, "MAX_SECONDS", 0.01)
    class Slow(Source):
        async def search(self, *args):
            await asyncio.sleep(10)
    result = await RecallService([Slow()], Model()).recall(request(), SCOPE)
    assert result["error_code"] == "timeout" and not result["findings"]


@pytest.mark.asyncio
async def test_worker_keeps_slot_until_cancelled_call_actually_finishes():
    import threading
    blocker = threading.Event()
    worker = BoundedWorker(1, "recall-test")
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(worker.run(blocker.wait, 1), timeout=0.01)
        with pytest.raises(RecallError, match="busy"):
            await worker.run(lambda: 1)
    finally:
        blocker.set()
        worker._pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_partial_source_failure_preserves_coverage_not_evidence():
    failing = Source(error=RecallError("source_unavailable"))
    failing.source = "memories"
    result = await RecallService([Source(), failing], Model()).recall(RecallRequest(query="relay"), SCOPE)
    assert result["status"] == "error" and result["findings"] == []
    assert result["coverage"]["messages"]["status"] == "ok"
    assert result["coverage"]["memories"]["status"] == "source_unavailable"


@pytest.mark.asyncio
async def test_withheld_task_does_not_abort_other_sources_or_claim_no_match():
    from llm_bawt.recall.contracts import EvidenceBatch
    source = Source(EvidenceBatch(withheld=1, search_mode="semantic"))
    source.source = "tasks"
    service = RecallService([Source(), source], Model())
    result = await service.recall(RecallRequest(query="relay", sources=("messages", "tasks")), SCOPE)
    assert result["status"] == "ok"
    assert result["coverage"]["tasks"]["status"] == "partial"
    assert result["coverage"]["tasks"]["withheld"] == 1
    only_tasks = await service.recall(RecallRequest(query="relay", sources=("tasks",)), SCOPE)
    assert only_tasks["status"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_research_concurrency_rejects_without_queueing():
    gate = asyncio.Event()
    class Blocked(Source):
        async def search(self, *args):
            await gate.wait()
            return []
    service = RecallService([Blocked()], Model())
    pending = [asyncio.create_task(service.recall(request(), SCOPE)) for _ in range(4)]
    await asyncio.sleep(0)
    assert (await service.recall(request(), SCOPE))["error_code"] == "busy"
    gate.set()
    await asyncio.gather(*pending)


def test_handle_ttl_cannot_outlive_capability(monkeypatch):
    import llm_bawt.recall.handles as module
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    store = HandleStore(ttl=900, clock=lambda: 50.0)
    key = store.put(request(), replace(SCOPE, capability_expires_at=1010), ("relay",))
    assert store._entries[key].expires_at == 60.0


@pytest.mark.asyncio
async def test_adapter_missing_authority_does_not_initialize_sources(monkeypatch):
    from llm_bawt.mcp_server import recall_tools
    from llm_bawt.task_turn_context import TaskTurnContextError
    def missing():
        raise TaskTurnContextError("SECRET")
    monkeypatch.setattr(recall_tools, "current_task_turn_context", missing)
    monkeypatch.setattr(recall_tools, "_runtime", lambda: pytest.fail("must not initialize"))
    result = json.loads(await recall_tools.context_recall(request()))
    assert result["error_code"] == "unauthorized" and "SECRET" not in str(result)


@pytest.mark.asyncio
async def test_adapter_returns_single_compact_text_packet(monkeypatch):
    from llm_bawt.mcp_server import recall_tools
    async def scope(_):
        return SCOPE
    monkeypatch.setattr(recall_tools, "current_task_turn_context", lambda: object())
    monkeypatch.setattr(recall_tools, "_runtime", lambda: (SimpleNamespace(scope=scope), RecallService([Source()], Model())))
    result = await recall_tools.context_recall(request())
    assert isinstance(result, str)
    assert tokens(result) == json.loads(result)["metrics"]["parent_tokens"]


def follow_on(id_, hit, at, text="x"):
    return Evidence("messages", id_, "reviewer", "operator", text, at=at, follows=hit)


class ThreadSource(Source):
    max_records = 18


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [
    # Never more than two follow-ons per hit, even inside the row allowance.
    [Evidence("messages", "h", "reviewer", "operator", "x"), *[follow_on(f"f{i}", "h", None) for i in range(3)]],
    [Evidence("messages", str(i), "reviewer", "operator", "x") for i in range(7)],
])
async def test_undeclared_or_extra_hits_fail_closed(rows):
    model = Model()
    result = await RecallService([ThreadSource(rows)], model).recall(request(), SCOPE)
    assert result["error_code"] == "retrieval_budget" and not model.calls


@pytest.mark.asyncio
async def test_follow_on_must_continue_a_hit_from_the_same_batch():
    rows = [Evidence("messages", "h", "reviewer", "operator", "x"), follow_on("f", "elsewhere", None)]
    result = await RecallService([ThreadSource(rows)], Model()).recall(request(), SCOPE)
    assert result["error_code"] == "unauthorized"


@pytest.mark.asyncio
async def test_follow_ons_never_displace_hits_and_evidence_is_oldest_first():
    messages = ThreadSource([
        Evidence("messages", "stale", "reviewer", "operator", "Messages + summaries", at="2026-10-21T10:33:37Z"),
        follow_on("correction", "stale", "2026-10-21T10:33:52Z", "No summaries"),
        follow_on("fixed", "stale", "2026-10-21T10:34:05Z", "Messages only"),
    ])
    memories = Source([Evidence("memories", "fact", "reviewer", "operator", "older fact", at="2026-09-01T00:00:00Z")])
    memories.source = "memories"
    model = Model()
    result = await RecallService([messages, memories], model).recall(
        RecallRequest(query="relay background", sources=("messages", "memories")), SCOPE)
    assert result["status"] == "ok"
    assert [row.id for row in model.calls[0][1]] == ["fact", "stale", "correction", "fixed"]
    assert result["coverage"]["messages"]["records"] == 3


@pytest.mark.asyncio
async def test_follow_ons_fill_only_the_remaining_global_budget():
    big = "Ω" * 2400  # clipped to 512 tokens each: 18 rows cannot all fit 6000
    messages = ThreadSource([
        *[Evidence("messages", f"h{i}", "reviewer", "operator", big) for i in range(6)],
        *[follow_on(f"f{i}", f"h{i}", None, big) for i in range(6)],
    ])
    memories = Source([Evidence("memories", f"m{i}", "reviewer", "operator", big) for i in range(6)])
    memories.source = "memories"
    model = Model()
    result = await RecallService([messages, memories], model).recall(
        RecallRequest(query="relay background", sources=("messages", "memories")), SCOPE)
    assert model.calls, result
    chosen = {row.id for row in model.calls[0][1]}
    # Direct hits from every source are admitted before any follow-on.
    assert {f"h{i}" for i in range(6)} <= chosen and {"m0", "m1", "m2"} <= chosen
    assert not {f"f{i}" for i in range(6)} & chosen
    assert result["truncated"] and result["metrics"]["utility_input_tokens"] <= 6000


@pytest.mark.asyncio
async def test_follow_ons_cannot_push_out_a_later_direct_hit():
    """Interleaved rank order (hit, its follow-ons, next hit…) as MessageSource emits it.

    Filling in that order would spend the token budget on follow-ons before
    later hits; the later direct hits must survive and only follow-ons drop.
    """
    from llm_bawt.recall.completion import input_tokens
    from llm_bawt.recall.contracts import MAX_INPUT_TOKENS
    from llm_bawt.recall.budgets import clip
    big = "Ω" * 2400
    rows = []
    for i in range(6):
        rows += [Evidence("messages", f"h{i}", "reviewer", "operator", big),
                 follow_on(f"f{i}a", f"h{i}", None, big), follow_on(f"f{i}b", f"h{i}", None, big)]
    memories = Source([Evidence("memories", "m0", "reviewer", "operator", big)])
    memories.source = "memories"
    question = "relay background"

    # Counterfactual: single-pass in emitted order loses later hits.
    naive = []
    for row in [*rows, *memories.rows]:
        candidate = replace(row, text=clip(row.text, 512))
        if input_tokens(question, [*naive, candidate]) <= MAX_INPUT_TOKENS:
            naive.append(candidate)
    naive_ids = {row.id for row in naive}
    assert "m0" not in naive_ids and "h5" not in naive_ids

    model = Model()
    result = await RecallService([ThreadSource(rows), memories], model).recall(
        RecallRequest(query=question, sources=("messages", "memories")), SCOPE)
    assert model.calls, result
    chosen = [row.id for row in model.calls[0][1]]
    assert {f"h{i}" for i in range(6)} | {"m0"} <= set(chosen)
    admitted_follow_ons = [row for row in model.calls[0][1] if row.follows]
    assert admitted_follow_ons and len(admitted_follow_ons) < 12
    assert all(row.follows in chosen for row in admitted_follow_ons)
    assert result["truncated"] and result["metrics"]["utility_input_tokens"] <= MAX_INPUT_TOKENS
