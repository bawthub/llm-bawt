"""Independent TASK-1210 acceptance checks for the parent-context boundary.

These replay synthetic question/evidence shapes with a deterministic extractor.
They measure packet size and evidence preservation, NOT real-model accuracy,
provider billing savings, or production latency. No service/DB/LLM is contacted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import tiktoken

from llm_bawt.recall.contracts import Evidence, RecallRequest, RecallScope
from llm_bawt.recall.service import RecallService


_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "isolated_recall_cases.json").read_text()
)["cases"]
_SOURCE = {"task": "tasks", "message": "messages", "memory": "memories", "tool": "tool_evidence"}
_NOISE = "BULK_HISTORY_NOT_FOR_PARENT: unrelated build output and archived diagnostics.\n"


def _tokens(value: object) -> int:
    return len(tiktoken.get_encoding("o200k_base").encode(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), disallowed_special=(),
    ))


def _scope(bot_id: str = "vex", user_id: str = "nick") -> RecallScope:
    return RecallScope(bot_id=bot_id, user_id=user_id, session_id="evaluation-session",
                       excluded_bot_ids=frozenset({"protected-fixture-bot"}))


class FixtureProvider:
    def __init__(self, source: str, records: list[Evidence]):
        self.source = source
        self.records = records
        self.calls: list[tuple] = []

    async def search(self, query, scope, task_ids):
        self.calls.append((query, scope, task_ids))
        return list(self.records)


class SelectingSynthesizer:
    def __init__(self, source: str, source_id: str, quote: str):
        self.source = source
        self.source_id = source_id
        self.quote = quote
        self.inputs: list[tuple] = []

    async def synthesize(self, query, evidence, output_tokens):
        self.inputs.append((query, evidence, output_tokens))
        assert any(item.id == self.source_id and self.quote in item.text for item in evidence)
        return json.dumps({"findings": [{
            "source": self.source, "id": self.source_id, "quote": self.quote,
        }]})


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
async def test_large_retrieval_preserves_oracle_evidence_without_parent_dump(case):
    records = [Evidence(source=_SOURCE[row["kind"]], id=row["id"],
                        bot_id="vex", user_id="nick", text=row["text"])
               for row in case["records"]]
    # The baseline is a legacy full-record dump; recall's provider contract
    # instead delivers bounded snippets. A separate test checks that a provider
    # violating this contract fails closed rather than dumping its payload.
    source = records[0].source
    records.append(Evidence(source=source, id="distractor", bot_id="vex", user_id="nick",
                            text=_NOISE * case["noise_repeat"]))
    sources = tuple(dict.fromkeys(row.source for row in records))
    providers = [FixtureProvider(name, [replace(row, text=row.text[:2400])
                                        for row in records if row.source == name])
                 for name in sources]
    target = next(row for row in records if row.id == case["required_source"])
    synth = SelectingSynthesizer(target.source, target.id, case["required_quote"])
    service = RecallService(providers=providers, synthesizer=synth)
    result = await service.recall(RecallRequest(query=case["question"], sources=sources), _scope())

    assert result["status"] == "ok"
    assert any(row["id"] == target.id and row["quote"] == case["required_quote"]
               for row in result["findings"])
    assert len(synth.inputs) == 1
    assert "BULK_HISTORY_NOT_FOR_PARENT" not in json.dumps(result)
    packet_tokens = _tokens(result)
    raw_tokens = _tokens([row.text for row in records])
    assert packet_tokens <= 1000
    assert raw_tokens >= 15000
    assert packet_tokens < raw_tokens / 10
    assert result["handle"]


class UntrustedSynthesizer:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def synthesize(self, query, evidence, output_tokens):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    json.dumps({"findings": [{"source": "messages", "id": "invented", "quote": "approved"}]}),
    json.dumps({"findings": [{"source": "messages", "id": "known", "quote": "deployment approved"}]}),
    _NOISE * 3000,
    RuntimeError("BULK_HISTORY_NOT_FOR_PARENT: provider secret exception contents"),
])
async def test_bad_synthesis_never_falls_back_to_raw_evidence(response):
    evidence = Evidence(source="messages", id="known", bot_id="vex", user_id="nick",
                        text="Deployment was not approved. " + _NOISE * 3)
    synth = UntrustedSynthesizer(response)
    service = RecallService(providers=[FixtureProvider("messages", [evidence])], synthesizer=synth)
    result = await service.recall(RecallRequest(query="Was deployment approved?", sources=("messages",)),
                                 _scope())
    assert synth.calls == 1
    assert result["status"] in {"error", "insufficient_evidence"}
    assert not result.get("findings")
    assert "BULK_HISTORY_NOT_FOR_PARENT" not in json.dumps(result)
    assert _tokens(result) <= 1000


@pytest.mark.asyncio
async def test_wrong_user_and_bot_evidence_never_reaches_synthesizer():
    rows = [Evidence(source="messages", id="wrong-user", bot_id="vex", user_id="someone-else",
                     text="Different user's private result."),
            Evidence(source="messages", id="wrong-bot", bot_id="other-bot", user_id="nick",
                     text="Different bot's private result."),
            Evidence(source="messages", id="protected", bot_id="protected-fixture-bot", user_id="nick",
                     text="Excluded source.")]
    synth = UntrustedSynthesizer('{"findings": []}')
    service = RecallService(providers=[FixtureProvider("messages", rows)], synthesizer=synth)
    result = await service.recall(RecallRequest(query="private result", sources=("messages",)), _scope())
    assert synth.calls == 0
    assert result["status"] in {"error", "no_match", "insufficient_evidence"}
    assert not result.get("findings")
    assert "private result" not in json.dumps(result)


@pytest.mark.asyncio
async def test_followup_handle_cannot_cross_user_bot_or_session():
    item = Evidence(source="messages", id="known", bot_id="vex", user_id="nick",
                    text="The verified setting is disabled.")
    provider = FixtureProvider("messages", [item])
    synth = SelectingSynthesizer("messages", "known", "The verified setting is disabled.")
    service = RecallService(providers=[provider], synthesizer=synth)
    first = await service.recall(RecallRequest(query="What is the setting?", sources=("messages",)), _scope())
    assert first["status"] == "ok"
    calls_before = len(provider.calls)
    scopes = [_scope(user_id="someone-else"), _scope(bot_id="other-bot"),
              RecallScope(bot_id="vex", user_id="nick", session_id="different-session")]
    for scope in scopes:
        result = await service.recall(RecallRequest(query="Tell me more", sources=("messages",),
                                                   handle=first["handle"]), scope)
        assert result["status"] == "error"
        assert not result.get("findings")
        assert "verified setting" not in json.dumps(result)
    assert len(provider.calls) == calls_before
    assert len(synth.inputs) == 1


@pytest.mark.asyncio
async def test_followup_refreshes_evidence_without_importing_parent_transcript():
    provider = FixtureProvider("messages", [Evidence(source="messages", id="old", bot_id="vex",
                               user_id="nick", text="The queue was paused.")])
    synth = SelectingSynthesizer("messages", "old", "The queue was paused.")
    service = RecallService(providers=[provider], synthesizer=synth)
    first = await service.recall(RecallRequest(query="What happened to the queue?", sources=("messages",)),
                                 _scope())
    assert first["status"] == "ok"
    provider.records = [Evidence(source="messages", id="new", bot_id="vex", user_id="nick",
                                 text="The queue is now resumed.")]
    synth.source_id = "new"
    synth.quote = "The queue is now resumed."
    result = await service.recall(RecallRequest(query="What happened afterward?", sources=("messages",),
                                                handle=first["handle"]), _scope())
    assert result["status"] == "ok"
    assert result["findings"][0]["id"] == "new"
    assert len(provider.calls) == 2
    assert "queue" in provider.calls[-1][0].lower()
    assert "afterward" in provider.calls[-1][0].lower()
    assert all(row.id != "old" for row in synth.inputs[-1][1])
    assert "The queue was paused" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [400, 1000, 1600])
async def test_entire_serialized_packet_obeys_requested_budget(budget):
    text = "A verified fact with a useful source. " * 5
    rows = [Evidence(source="messages", id=f"record-{index}", bot_id="vex", user_id="nick", text=text)
            for index in range(6)]
    output = json.dumps({"findings": [{"source": "messages", "id": row.id, "quote": row.text}
                                      for row in rows]})
    service = RecallService(providers=[FixtureProvider("messages", rows)],
                            synthesizer=UntrustedSynthesizer(output))
    result = await service.recall(RecallRequest(query="What is verified?", sources=("messages",),
                                               output_tokens=budget), _scope())
    assert _tokens(result) <= budget
    assert result["status"] in {"ok", "insufficient_evidence", "error"}
    for finding in result.get("findings", []):
        assert finding["quote"] in next(row.text for row in rows if row.id == finding["id"])
