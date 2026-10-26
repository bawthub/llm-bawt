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
from llm_bawt.recall.prompt import DEFAULT_RECALL_SYSTEM_PROMPT
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
        self.systems = []

    def system_prompt(self, bot_id):
        return DEFAULT_RECALL_SYSTEM_PROMPT

    async def synthesize(self, system, query, evidence, output_tokens):
        self.systems.append(system)
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
@pytest.mark.parametrize("changed", [{"bot_id": "peer"}, {"user_id": "someone"},
                                     {"source": "tasks"}, {"source_bot": "excluded"}])
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
async def test_partial_source_failure_preserves_healthy_grounded_evidence(caplog):
    failing = Source(error=RecallError("source_unavailable"))
    failing.source = "memories"
    result = await RecallService([Source(), failing], Model()).recall(RecallRequest(query="relay"), SCOPE)
    assert result["status"] == "ok" and result["findings"][0]["id"] == "m1"
    assert result["coverage"]["messages"]["status"] == "ok"
    assert result["coverage"]["memories"]["status"] == "unavailable"
    assert "incomplete" in result["uncertainty"]
    assert "source=memories" in caplog.text and "RecallError" in caplog.text


@pytest.mark.asyncio
async def test_failed_source_and_empty_healthy_source_never_claim_no_match():
    failing = Source(error=RuntimeError("PRIVATE DB EXCEPTION"))
    failing.source = "memories"
    result = await RecallService([Source([]), failing], Model()).recall(RecallRequest(query="relay"), SCOPE)
    assert result["status"] == "insufficient_evidence" and not result["findings"]
    assert result["coverage"]["memories"]["status"] == "unavailable"
    assert "PRIVATE DB EXCEPTION" not in compact_json(result)
    all_failed = await RecallService([failing], Model()).recall(
        RecallRequest(query="relay", sources=("memories",)), SCOPE)
    assert all_failed["status"] == "error" and all_failed["coverage"]["memories"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_answer_exceeding_six_quote_spans_is_not_returned_as_a_prefix():
    answer = "B5: " + (("A" * 205 + ". ") * 11) + "Approval was revoked at the end."
    assert len(answer) < 2400
    raw = compact_json({"findings": [{"source": "messages", "id": "long", "quote": answer}]})
    item = Evidence("messages", "long", "reviewer", "operator", answer)
    result = await RecallService([Source([item])], Model(raw)).recall(request(output_tokens=1600), SCOPE)
    assert result["status"] == "insufficient_evidence" and result["findings"] == []
    assert result["handle"] and "expand" in result["uncertainty"]


@pytest.mark.asyncio
async def test_shortened_answer_with_rejected_selection_is_still_insufficient():
    answer = "B5: " + (("A" * 205 + ". ") * 11) + "Approval was revoked at the end."
    raw = compact_json({"findings": [
        {"source": "messages", "id": "long", "quote": answer},
        {"source": "messages", "id": "long", "quote": "invented"},
    ]})
    item = Evidence("messages", "long", "reviewer", "operator", answer)
    result = await RecallService([Source([item])], Model(raw)).recall(request(output_tokens=1600), SCOPE)
    assert result["status"] == "insufficient_evidence" and result["findings"] == []
    assert result["rejected_findings"] == 1 and result["handle"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["unauthorized", "policy_unavailable", "task_provenance_incomplete"])
async def test_authority_and_provenance_denials_remain_fatal_with_healthy_source(code):
    denied = Source(error=RecallError(code))
    denied.source = "memories"
    result = await RecallService([Source(), denied], Model()).recall(RecallRequest(query="relay"), SCOPE)
    assert result["error_code"] == code and result["findings"] == []


@pytest.mark.asyncio
async def test_empty_synthesis_with_exact_receipt_is_explicitly_insufficient():
    receipt = Evidence("messages", "exact-receipt", "reviewer", "operator",
                       "TASK-1144 app recreate 8ee7d89 succeeded; health checks passed.")
    result = await RecallService([Source([receipt])], Model('{"findings":[]}')).recall(
        RecallRequest(query="TASK-1144 app recreate 8ee7d89", sources=("messages",)), SCOPE)
    assert result["status"] == "insufficient_evidence" and result["coverage"]["messages"]["records"] == 1


@pytest.mark.asyncio
async def test_same_session_reviews_retain_distinct_source_owned_provenance():
    reviews = [Evidence("messages", "m8", "reviewer", "operator", "r8 B5: old review outcome.",
                        at="2026-10-21T10:00:00Z", source_bot="peer", session_id="s1", review_ref="r8"),
               Evidence("messages", "m9", "reviewer", "operator", "r9 B5: corrected review outcome.",
                        at="2026-10-21T11:00:00Z", source_bot="peer", session_id="s1", review_ref="r9")]
    raw = compact_json({"findings": [{"source": "messages", "id": "m9", "quote": reviews[1].text}]})
    result = await RecallService([Source(reviews)], Model(raw)).recall(request(), SCOPE)
    assert result["status"] == "ok" and result["findings"][0]["provenance"] == {
        "source_bot": "peer", "session_id": "s1", "at": "2026-10-21T11:00:00Z", "review_ref": "r9"}


@pytest.mark.asyncio
async def test_long_grounded_answer_is_split_or_marked_insufficient_not_silently_cut():
    answer = "B5: " + ("The complete explanation includes this detail. " * 15) + "Final correction: nothing was shipped."
    item = Evidence("messages", "b5", "reviewer", "operator", answer, start_char=32,
                    source_bot="peer", session_id="s9")
    raw = compact_json({"findings": [{"source": "messages", "id": "b5", "quote": answer}]})
    result = await RecallService([Source([item])], Model(raw)).recall(request(output_tokens=1600), SCOPE)
    assert result["status"] == "ok", result
    spans = result["findings"]
    assert len(spans) >= 2 and "".join(span["quote"] for span in spans) == answer
    assert spans[0]["start_char"] == 32
    assert spans[-1]["quote"].endswith("nothing was shipped.")


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
        if input_tokens(DEFAULT_RECALL_SYSTEM_PROMPT, question, [*naive, candidate]) <= MAX_INPUT_TOKENS:
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


# TASK-1210 live shapes (al, fc0dab70): grok collapses "\n\n" and elides with
# "..." inside quotes; one misquote used to discard every grounded finding.
LIVE_TEXT = ("Option B fits better.\n\n- **What it does:** keeps hits first.\n\n"
             "It gave wrong answers live. Later we fixed it. It hasn't been pushed yet.")


def _live_source():
    return Source([Evidence("messages", "fc0dab70", "reviewer", "operator", LIVE_TEXT)])


@pytest.mark.asyncio
async def test_misquoted_finding_is_dropped_not_fatal():
    raw = compact_json({"findings": [
        {"source": "messages", "id": "fc0dab70", "quote": "Option B fits better."},
        {"source": "messages", "id": "fc0dab70", "quote": "SECRET invented claim"},
        {"source": "messages", "id": "outside", "quote": "SECRET"},
        {"source": "messages", "id": "fc0dab70", "quote": "x" * 401},
        {"source": "messages", "id": "fc0dab70", "quote": "ok", "prose": "SECRET"},
    ]})
    result = await RecallService([_live_source()], Model(raw)).recall(request(), SCOPE)
    assert result["status"] == "ok"
    assert result["findings"] == [{"source": "messages", "id": "fc0dab70", "quote": "Option B fits better."}]
    assert result["rejected_findings"] == 4 and "ungrounded_dropped" not in result["metrics"]
    assert "4 selected excerpt(s) could not be verified" in result["uncertainty"]
    assert "SECRET" not in compact_json(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("quote,expected", [
    ("fits better.\n- **What it does:**", "fits better.\n\n- **What it does:**"),
    ("gave wrong answers live. ... It hasn't been pushed",
     "gave wrong answers live. Later we fixed it. It hasn't been pushed"),
    ("...gave wrong answers live.…It hasn't been", "gave wrong answers live. Later we fixed it. It hasn't been"),
])
async def test_reformatted_quote_returns_exact_source_span(quote, expected):
    raw = compact_json({"findings": [{"source": "messages", "id": "fc0dab70", "quote": quote}]})
    result = await RecallService([_live_source()], Model(raw)).recall(request(), SCOPE)
    assert result["status"] == "ok"
    assert result["findings"][0]["quote"] == expected
    assert expected in LIVE_TEXT and result["rejected_findings"] == 0
    assert "could not be verified" not in result["uncertainty"]


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", [
    "It hasn't been pushed ... gave wrong answers live",  # fragments out of order
    "fits better. ... SECRET",                            # one fragment invented
    "Option B fits\nbetter SECRET",                       # whitespace drift plus an added word
    "fits ... pushed",                                    # short fragments would bridge on common words
    "Option B fits better. ... pushed yet.",              # every elided fragment needs 3 words
])
async def test_tolerant_matching_cannot_ground_reordered_or_invented_text(quote):
    raw = compact_json({"findings": [{"source": "messages", "id": "fc0dab70", "quote": quote}]})
    result = await RecallService([_live_source()], Model(raw)).recall(request(), SCOPE)
    assert result["error_code"] == "ungrounded_synthesis"
    assert "SECRET" not in compact_json(result)


@pytest.mark.asyncio
async def test_dropped_correction_is_surfaced_not_a_silent_stale_answer():
    # Stale-vs-correction shape: the old statement grounds, the later correction
    # is mangled by the model and dropped. The caller must be told.
    stale = Evidence("messages", "old", "reviewer", "operator", "Use messages plus summaries.",
                     at="2026-10-21T15:00:00.000000Z")
    correction = Evidence("messages", "new", "reviewer", "operator",
                          "No summaries. I said message history only.", at="2026-10-21T15:05:00.000000Z")
    raw = compact_json({"findings": [
        {"source": "messages", "id": "old", "quote": "Use messages plus summaries."},
        {"source": "messages", "id": "new", "quote": "No summaries. ... message history only, always"},
    ]})
    result = await RecallService([Source([stale, correction])], Model(raw)).recall(request(), SCOPE)
    assert result["status"] == "ok"
    assert [item["id"] for item in result["findings"]] == ["old"]
    assert result["rejected_findings"] == 1
    assert "a later correction may be missing" in result["uncertainty"]
    assert result["metrics"]["parent_tokens"] == tokens(compact_json(result))


# Al's 8-run stress (TASK-1210): grok often quotes 440-600 chars; the key answer
# ("Only partly. The first version...") was dropped by the length cap.
LONG_ANSWER = ("Only partly. The first version of recall is live. The latest fix isn't.\n\n"
               + " ".join(f"- **Item {n}:** detail sentence number {n} explains the status." for n in range(12)))


@pytest.mark.asyncio
async def test_over_long_grounded_quote_keeps_complete_answer_or_reports_insufficient():
    assert len(LONG_ANSWER) > 600
    rows = [Evidence("messages", "2a5f048f", "reviewer", "operator", LONG_ANSWER)]
    raw = compact_json({"findings": [{"source": "messages", "id": "2a5f048f",
                                      "quote": LONG_ANSWER.replace("\n\n", "\n")}]})
    result = await RecallService([Source(rows)], Model(raw)).recall(request(), SCOPE)
    assert result["rejected_findings"] == 0
    assert result["status"] == "ok"
    assert "".join(row["quote"] for row in result["findings"]) == LONG_ANSWER
    assert len(result["findings"]) >= 2
    assert all(len(row["quote"]) <= 400 and tokens(row["quote"]) <= 100
               for row in result["findings"])


@pytest.mark.asyncio
async def test_unshortened_findings_carry_no_truncation_marker():
    result = await RecallService([Source()], Model()).recall(request(), SCOPE)
    assert "truncated" not in result["findings"][0]
    assert "marked truncated" not in result["uncertainty"]


def test_capped_span_drops_a_trailing_correction_so_it_is_marked():
    # Snark probe shape: the revocation sits past the cap.
    from llm_bawt.recall.grounding import cap_span
    span = "Approved. " + "The plan moves ahead with the rollout as described. " * 8 \
        + "That approval was revoked on Wednesday; nothing shipped."
    assert len(span) > 400
    capped = cap_span(span, 400, 100)
    assert capped != span and span.startswith(capped) and "revoked" not in capped


@pytest.mark.asyncio
async def test_non_object_finding_is_rejected_not_fatal():
    raw = '{"findings":["SECRET", 7, {"source":"messages","id":"m1","quote":"The relay background"}]}'
    result = await RecallService([Source()], Model(raw)).recall(request(), SCOPE)
    assert result["status"] == "ok" and result["rejected_findings"] == 2
    assert "SECRET" not in compact_json(result)


def test_elision_considers_overlapping_first_fragment_starts():
    from llm_bawt.recall.grounding import ground_quote
    assert ground_quote("a a a ... b c d", "a a a a b c d") == "a a a b c d"


def test_successful_chain_on_adversarial_repetition_is_linear(monkeypatch):
    from llm_bawt.recall.grounding import ground_quote
    text = _repetitive_excerpt() + " closing words end"
    scanned = _count_scanned(monkeypatch)
    assert ground_quote("the bot said ... the bot said ... closing words end", text)
    _assert_linear(scanned, text, fragments=3)


@pytest.mark.asyncio
async def test_literal_ellipsis_in_source_is_not_an_elision_point():
    text = "We tried it… and then\n\nthe recall tool returned nothing useful at all."
    rows = [Evidence("messages", "e1", "reviewer", "operator", text)]
    raw = compact_json({"findings": [{"source": "messages", "id": "e1",
                                      "quote": "We tried it… and then\nthe recall tool returned nothing"}]})
    result = await RecallService([Source(rows)], Model(raw)).recall(request(), SCOPE)
    assert result["findings"][0]["quote"] == "We tried it… and then\n\nthe recall tool returned nothing"


# Snark r3 P1/P2: elision grounding must be linear (it runs on the MCP event
# loop, which asyncio.timeout cannot interrupt) and return the tightest span.
def _count_scanned(monkeypatch) -> list[int]:
    """Count the characters grounding's regex searches cover, from each search's
    start to its match end (or the end of the text). Linear work is a few passes
    per fragment; rescanning from every candidate start is one pass per start.
    This measures the work itself, so a slow runner can't fail it."""
    import re
    from llm_bawt.recall import grounding

    scanned = [0]

    class Counting:
        def __init__(self, pattern):
            self.pattern = pattern

        def search(self, text, pos=0):
            found = self.pattern.search(text, pos)
            scanned[0] += (found.end() if found else len(text)) - pos
            return found

    class CountingRe:
        def __getattr__(self, name):
            return getattr(re, name)

        def compile(self, pattern, flags=0):
            return Counting(re.compile(pattern, flags))

    monkeypatch.setattr(grounding, "re", CountingRe())
    return scanned


def _assert_linear(scanned: list[int], text: str, fragments: int) -> None:
    passes = scanned[0] / len(text)
    assert passes <= 2 * fragments, f"{passes:.1f} passes over the text for {fragments} fragments"


def _repetitive_excerpt():
    from llm_bawt.recall.budgets import clip
    return clip("the bot said the bot said the bot " * 400, 511)


@pytest.mark.parametrize("fragments", [3, 4])
def test_elision_grounding_is_linear_on_repetitive_text(fragments, monkeypatch):
    from llm_bawt.recall.grounding import ground_quote
    text = _repetitive_excerpt()
    quote = " ... ".join(["the bot said"] * (fragments - 1) + ["absent words here"])
    scanned = _count_scanned(monkeypatch)
    assert ground_quote(quote, text) is None
    _assert_linear(scanned, text, fragments)


@pytest.mark.parametrize("fragments", [5, 6])
def test_elision_fragment_count_is_capped(fragments):
    from llm_bawt.recall.grounding import MAX_ELIDED_FRAGMENTS, ground_quote
    assert MAX_ELIDED_FRAGMENTS == 4
    text = _repetitive_excerpt()
    assert ground_quote(" ... ".join(["the bot said"] * fragments), text) is None


def test_elision_grounds_to_the_tightest_span_not_the_leftmost():
    from llm_bawt.recall.grounding import ground_quote
    text = ("We discussed the recall tool at length. " + "Filler words go here. " * 10
            + "Later we discussed the recall tool again and it now works fine.")
    span = ground_quote("discussed the recall ... now works fine", text)
    assert span == "discussed the recall tool again and it now works fine"


# Al's last live miss (3a7c87dc): grok dropped bold markers from the quote.
@pytest.mark.parametrize("quote,text,expected", [
    ("Why it fails: I re-ran the probe",
     "Intro.\n\n**Why it fails:** I re-ran the probe twice.", "**Why it fails:** I re-ran the probe"),
    ("set the maintenance_model to grok", "We set the `maintenance_model` to grok today.",
     "set the `maintenance_model` to grok"),
    ("**Running:** the app restarted", "Running: the app restarted at 16:14.", "Running: the app restarted"),
])
def test_markdown_emphasis_markers_are_tolerated_and_source_bytes_returned(quote, text, expected):
    from llm_bawt.recall.grounding import ground_quote
    assert ground_quote(quote, text) == expected


def test_markers_inside_a_word_still_match_literally():
    from llm_bawt.recall.grounding import ground_quote
    assert ground_quote("set the maintenancemodel to grok", "We set the maintenance_model to grok.") is None
    assert ground_quote("set the maintenance*model to grok", "We set the maintenance_model to grok.") is None


# Nick 2026-10-21: NO PROMPTS ARE EVER HARDCODED. The recall instructions are a
# prompt-registry key resolved per call; what is budgeted is what is sent.
class RegistryModel(Model):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.resolved_for = []

    def system_prompt(self, bot_id):
        self.resolved_for.append(bot_id)
        return self.system


@pytest.mark.asyncio
async def test_recall_sends_and_budgets_the_registry_resolved_prompt_once():
    from llm_bawt.recall.completion import input_tokens
    model = RegistryModel("EDITED IN THE PROMPT REGISTRY " * 40)
    result = await RecallService([Source()], model).recall(
        RecallRequest(query="relay background", sources=("messages",)), SCOPE)
    assert result["status"] == "ok", result
    assert model.resolved_for == ["reviewer"]  # caller bot, once per recall
    assert model.systems == [model.system]
    evidence = model.calls[0][1]
    assert result["metrics"]["utility_input_tokens"] == input_tokens(model.system, "relay background", evidence)
    assert input_tokens(model.system, "q", evidence) > input_tokens("short", "q", evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["", "   ", None])
async def test_blank_resolved_prompt_fails_closed_without_a_model_call(system):
    model = RegistryModel(system)
    result = await RecallService([Source()], model).recall(
        RecallRequest(query="relay background", sources=("messages",)), SCOPE)
    assert result["error_code"] == "model_unavailable" and not model.calls


def _resolver(monkeypatch, behaviour):
    import llm_bawt.prompt_registry as registry
    seen = []

    class Resolver:
        def resolve(self, key, scope_type="global", scope_id=None):
            seen.append((key, scope_type, scope_id))
            if isinstance(behaviour, Exception):
                raise behaviour
            return behaviour

    monkeypatch.setattr(registry, "get_prompt_resolver", lambda config=None: Resolver())
    return seen


def test_resolver_uses_bot_scoped_registry_body(monkeypatch):
    from llm_bawt.recall.prompt import RECALL_SYSTEM_PROMPT_KEY, resolve_recall_system_prompt
    seen = _resolver(monkeypatch, SimpleNamespace(body="bot override"))
    assert resolve_recall_system_prompt("reviewer") == "bot override"
    assert seen == [(RECALL_SYSTEM_PROMPT_KEY, "bot", "reviewer")]


@pytest.mark.parametrize("behaviour", [None, SimpleNamespace(body="  "), RuntimeError("db down")])
def test_resolver_falls_back_to_seed_on_blank_or_registry_failure(monkeypatch, behaviour):
    from llm_bawt.recall.prompt import resolve_recall_system_prompt
    _resolver(monkeypatch, behaviour)
    assert resolve_recall_system_prompt("reviewer") == DEFAULT_RECALL_SYSTEM_PROMPT


def test_recall_prompt_is_registered_with_seed_default_and_no_placeholders():
    from llm_bawt.prompt_registry import DEFAULT_PROMPT_DEFINITIONS, PromptResolver
    from llm_bawt.recall.prompt import RECALL_SYSTEM_PROMPT_KEY
    definition = DEFAULT_PROMPT_DEFINITIONS[RECALL_SYSTEM_PROMPT_KEY]
    assert definition.default_body() == DEFAULT_RECALL_SYSTEM_PROMPT
    assert definition.required_vars == ()
    assert "untrusted data" in definition.default_body()
    # Verbatim body (never .format()-ed): its JSON braces must not trip save-time
    # validation in the Prompt Registry editor.
    resolver = PromptResolver.__new__(PromptResolver)  # validate() is pure; skip the DB store
    assert resolver.validate(key=RECALL_SYSTEM_PROMPT_KEY, body=definition.default_body())["valid"]


def test_recall_completion_module_holds_no_prompt_literal():
    import llm_bawt.recall.completion as completion
    assert not hasattr(completion, "SYSTEM")
    assert DEFAULT_RECALL_SYSTEM_PROMPT not in open(completion.__file__, encoding="utf-8").read()
