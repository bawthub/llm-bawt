"""TASK-2289: rewrites retain provenance; deletions follow paired receipts."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_bawt.memory.extraction.prompts import MEMORY_UPDATE_PROMPT_TEMPLATE
from llm_bawt.memory.extraction.service import (
    ExtractedFact,
    MeaningAssociation,
    MemoryAction,
    MemoryExtractionService,
)
from llm_bawt.service import background_tasks
from llm_bawt.service.background_tasks import BackgroundTasksMixin, ExtractionBatchFailed
from tests.test_bot_only_extraction_skips import HUMAN, _summary_backend
from tests.test_extraction_failure_retry import _service as raw_service


@pytest.fixture
def fact():
    return ExtractedFact(
        content="User recently moved to Seattle", tags=["fact"], importance=.8,
        source_message_ids=["h1"],
        meaning=MeaningAssociation(intent="share location"),
        profile_attribute={"category": "fact", "key": "location"},
    )


def _parse(facts, actions, targets=None):
    return MemoryExtractionService()._parse_action_response(
        json.dumps({"actions": actions}), facts,
        targets if targets is not None else [{"id": "old"}],
    )


def _write(kind="ADD", index=0, content="User lives in Seattle", target="old"):
    return {"action": kind, "fact_index": index, "target_memory_id": target,
            "fact": {"content": content, "source_message_ids": ["invented"],
                     "tags": ["invented"], "importance": 0,
                     "profile_attribute": {"key": "invented"}}}


def _delete(index=0, target="old"):
    return {"action": "DELETE", "fact_index": index, "target_memory_id": target}


@pytest.mark.parametrize("kind", ["ADD", "UPDATE"])
def test_rewrite_keeps_only_original_provenance_and_metadata(fact, kind):
    action, = _parse([fact], [_write(kind)])
    assert action.fact.content == "User lives in Seattle"
    assert action.fact is not fact and action.source_fact is fact
    assert action.fact.source_message_ids == ["h1"]
    assert action.fact.tags == ["fact"] and action.fact.importance == .8
    assert action.fact.meaning is fact.meaning
    assert action.fact.profile_attribute is fact.profile_attribute
    assert fact.content == "User recently moved to Seattle"


@pytest.mark.parametrize("index", [True, False, -1, 1, 0.0, "0", None, [], {}])
def test_invalid_explicit_index_never_falls_back_to_content(fact, index):
    assert _parse([fact], [_write(index=index, content=fact.content)]) == []


def test_legacy_exact_link_must_be_unique_and_cannot_authorize_a_rewrite(fact):
    action = _write(content=fact.content)
    del action["fact_index"]
    parsed, = _parse([fact], [action])
    assert parsed.fact is fact and parsed.source_fact is fact
    assert _parse([fact, replace(fact)], [action]) == []
    action["fact"]["content"] = "User lives in Seattle"
    assert _parse([fact], [action]) == []
    assert _parse([fact], [{"action": "DELETE", "target_memory_id": "old"}]) == []


@pytest.mark.parametrize("kind", ["UPDATE", "DELETE"])
@pytest.mark.parametrize("target", ["foreign", "ol", " old", None, [], {}])
def test_mutation_targets_must_be_exact_supplied_ids(fact, kind, target):
    action = _write(kind, target=target) if kind == "UPDATE" else _delete(target=target)
    assert _parse([fact], [action]) == []


def test_only_one_write_per_fact_and_no_delete_of_update_target(fact):
    actions = _parse([fact], [_delete(), _write("UPDATE"), _write()])
    assert [action.action for action in actions] == ["UPDATE"]


def test_delete_links_original_but_needs_no_rewritten_fact(fact):
    action, = _parse([fact], [_delete()])
    assert action.fact is None and action.source_fact is fact


def test_request_and_canonical_prompt_include_explicit_indices(fact, monkeypatch):
    from llm_bawt.memory.extraction import service as service_module

    seen = {}

    def prompt(existing, facts):
        seen["facts"] = json.loads(facts)
        return MEMORY_UPDATE_PROMPT_TEMPLATE.format(existing_memories=existing, new_facts=facts)

    def query(**kwargs):
        seen["prompt"] = kwargs["messages"][1].content
        return json.dumps({"actions": [_delete(), _write()]})

    monkeypatch.setattr(service_module, "get_memory_update_prompt", prompt)
    monkeypatch.setattr(service_module, "get_extraction_system_prompt", lambda _: "system")
    result = MemoryExtractionService(SimpleNamespace(query=query)).determine_memory_actions(
        [fact, replace(fact, content="User writes software")], [{"id": "old"}],
    )
    assert [item["fact_index"] for item in seen["facts"]] == [0, 1]
    assert '"fact_index": 0' in seen["prompt"] and '"fact_index": 1' in seen["prompt"]
    assert "DELETE alone cannot remove a memory" in seen["prompt"]
    assert [action.action for action in result] == ["DELETE", "ADD"]


class _Memory:
    def __init__(self, outcome="ok"):
        self.rows = {"old": "User lives in San Francisco"}
        self.calls = []
        self.outcome = outcome

    def list_memories(self, **kwargs):
        return [{"id": key, "content": value} for key, value in self.rows.items()]

    def add_memory(self, **kwargs):
        self.calls.append(("ADD", kwargs))
        if self.outcome == "raise":
            raise RuntimeError("write outage")
        if self.outcome == "none":
            return None
        if self.outcome == "duplicate":
            return "old"
        self.rows["new"] = kwargs["content"]
        return "new"

    def update_memory(self, memory_id, **kwargs):
        self.calls.append(("UPDATE", kwargs))
        if self.outcome == "raise":
            raise RuntimeError("write outage")
        if self.outcome == "none":
            return None
        self.rows[memory_id] = kwargs["content"]
        return {"id": memory_id}

    def delete_memory(self, memory_id):
        self.calls.append(("DELETE", memory_id))
        return self.rows.pop(memory_id, None) is not None


def _consumer(monkeypatch, path, facts, actions, memory, *, profile=False):
    monkeypatch.setattr(background_tasks, "_extraction_attempts", background_tasks._ExtractionAttempts())
    monkeypatch.setattr(MemoryExtractionService, "determine_memory_actions", lambda *a, **kw: actions)
    profiles = []
    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_profile_attributes_from_fact",
                        lambda **kw: profiles.append(kw["fact"]))
    if path == "raw":
        service, backend, task = raw_service(monkeypatch, memory)
        monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", lambda *a, **kw: facts)
        service.config.MEMORY_PROFILE_ATTRIBUTE_ENABLED = profile
        run = lambda: service._process_memory_extraction(task)
    else:
        backend = _summary_backend([HUMAN])
        monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
        monkeypatch.setattr("llm_bawt.memory.summary_extraction_policy.summary_allows_extraction",
                            lambda *a: True)
        monkeypatch.setattr(MemoryExtractionService, "extract_from_summary", lambda *a, **kw: facts)
        service = BackgroundTasksMixin()
        service.config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=profile)
        service._get_background_client = lambda **kw: (object(), "mock")
        service.get_memory_client = lambda *a: memory
        service._mark_summary_extracted = lambda *a: None
        run = lambda: service._extract_from_summaries(
            [{"created": True, "summary_id": "summary", "summary_text": "User moved"}], "test", "u")
    return run, service, profiles


@pytest.mark.parametrize("path", ["raw", "summary"])
def test_prompt_example_deletes_only_after_rewritten_replacement(fact, monkeypatch, path):
    memory = _Memory()
    actions = _parse([fact], [_delete(), _write()])
    run, _service, profiles = _consumer(monkeypatch, path, [fact], actions, memory, profile=True)
    result = run()
    assert result["facts_stored"] == 1
    assert memory.rows == {"new": "User lives in Seattle"}
    assert [call[0] for call in memory.calls] == ["ADD", "DELETE"]
    assert memory.calls[0][1]["source_message_ids"] == ["h1"]
    assert [candidate.content for candidate in profiles] == ["User lives in Seattle"]


@pytest.mark.parametrize("path", ["raw", "summary"])
def test_update_keeps_provenance_and_cannot_delete_its_own_target(fact, monkeypatch, path):
    memory = _Memory()
    actions = _parse([fact], [_delete(), _write("UPDATE")])
    run, _service, profiles = _consumer(monkeypatch, path, [fact], actions, memory, profile=True)
    assert run()["facts_stored"] == 1
    assert memory.rows == {"old": "User lives in Seattle"}
    assert [call[0] for call in memory.calls] == ["UPDATE"]
    assert memory.calls[0][1]["source_message_ids"] == ["h1"]
    assert len(profiles) == 1


@pytest.mark.parametrize("path", ["raw", "summary"])
@pytest.mark.parametrize("outcome", ["raise", "none", "duplicate"])
def test_failed_or_folded_replacement_never_deletes_target(fact, monkeypatch, path, outcome):
    memory = _Memory(outcome)
    run, _service, _profiles = _consumer(
        monkeypatch, path, [fact], _parse([fact], [_delete(), _write()]), memory,
    )
    if path == "raw" and outcome != "duplicate":
        with pytest.raises(ExtractionBatchFailed):
            run()
    else:
        run()
    assert memory.rows == {"old": "User lives in San Francisco"}
    assert [call[0] for call in memory.calls] == ["ADD"]


@pytest.mark.parametrize("path", ["raw", "summary"])
def test_delete_only_cannot_remove_memory(fact, monkeypatch, path):
    memory = _Memory()
    run, _service, _profiles = _consumer(
        monkeypatch, path, [fact], _parse([fact], [_delete()]), memory,
    )
    assert run()["facts_stored"] == 0
    assert memory.calls == []


@pytest.mark.parametrize("path", ["raw", "summary"])
def test_unrelated_success_cannot_release_failed_replacement_delete(fact, monkeypatch, path):
    memory = _Memory("none")
    other = replace(fact, content="User writes software")
    actions = _parse([fact, other], [_delete(), _write("UPDATE"), _write(index=1)])
    # Make the failed update and deletion target distinct, so overlap isn't
    # what prevents deletion; only the successful unrelated ADD must not suffice.
    actions.insert(0, MemoryAction("DELETE", target_memory_id="old", source_fact=fact))
    actions[1].target_memory_id = "other"
    memory.rows["other"] = "Previous work"
    original_add = memory.add_memory
    memory.add_memory = lambda **kw: (setattr(memory, "outcome", "ok") or original_add(**kw))
    run, _service, _profiles = _consumer(monkeypatch, path, [fact, other], actions, memory)
    run()
    assert memory.rows["old"] == "User lives in San Francisco"
    assert not any(call[0] == "DELETE" for call in memory.calls)


@pytest.mark.parametrize("path", ["raw", "summary"])
def test_consumer_rechecks_target_snapshot_even_for_injected_actions(fact, monkeypatch, path):
    memory = _Memory()
    actions = [
        MemoryAction("ADD", fact=replace(fact, content="New"), source_fact=fact),
        MemoryAction("UPDATE", fact=fact, target_memory_id="foreign", source_fact=fact),
        MemoryAction("DELETE", target_memory_id="foreign", source_fact=fact),
    ]
    run, _service, _profiles = _consumer(monkeypatch, path, [fact], actions, memory)
    run()
    assert [call[0] for call in memory.calls] == ["ADD"]


@pytest.mark.parametrize("outcome", ["none", "raise"])
def test_failed_rewritten_update_suppresses_original_profile_on_retry(fact, monkeypatch, outcome):
    memory = _Memory(outcome)
    actions = _parse([fact], [_write("UPDATE")])
    run, _service, profiles = _consumer(monkeypatch, "raw", [fact], actions, memory, profile=True)
    # Existing retry policy offers originals, not rewritten copies, to profiles.
    key = ("test", "u", "h1")
    background_tasks._extraction_attempts.record_failure(key)
    if outcome == "raise":
        with pytest.raises(ExtractionBatchFailed):
            run()
    else:
        run()
    assert profiles == []
