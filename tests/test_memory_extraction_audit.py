"""TASK-2248/2289: preserve extraction containers and statement semantics."""
import json

import pytest

from llm_bawt.memory.extraction.prompts import estimate_importance
from llm_bawt.memory.extraction.service import MemoryExtractionService


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("form", ["array", "wrapped", "fenced-array", "fenced-object", "prose-array"])
def test_complete_outer_json_container_keeps_every_fact(count, form):
    facts = [{"content": f"Synthetic preference {i}", "source_message_ids": ["human-source"]}
             for i in range(count)]
    data = {"facts": facts} if form in {"wrapped", "fenced-object"} else facts
    response = json.dumps(data)
    if form.startswith("fenced"):
        response = f"```json\n{response}\n```"
    elif form == "prose-array":
        response = f"Extracted facts:\n{response}\nDone."
    result = MemoryExtractionService()._parse_extraction_response(
        response, [], eligible_ids={"human-source"}, strict=True)
    assert [fact.content for fact in result] == [fact["content"] for fact in facts]
    assert all(fact.source_message_ids == ["human-source"] for fact in result)


def test_raw_array_keeps_source_validation_and_summary_source_contract():
    service = MemoryExtractionService()
    response = json.dumps([
        {"content": "Valid", "source_message_ids": ["human-source"]},
        {"content": "Invented", "source_message_ids": ["foreign-source"]},
    ])
    scoped = service._parse_extraction_response(response, [], eligible_ids={"human-source"}, strict=True)
    assert [fact.content for fact in scoped] == ["Valid"]
    summary = service._parse_extraction_response(response, ["summary-source"], strict=True)
    assert len(summary) == 2
    assert all(fact.source_message_ids == ["summary-source"] for fact in summary)


@pytest.mark.parametrize("statement", ["I hate peanuts", "I love peanuts", "I have no allergies",
                                       "I work for a nonprofit", "My name is Synthetic"])
def test_heuristics_preserve_the_complete_predicate(statement):
    facts = MemoryExtractionService()._extract_with_heuristics([
        {"id": "human-source", "role": "user", "content": statement},
    ])
    assert facts
    assert any(statement in fact.content for fact in facts)
    assert all(fact.source_message_ids == ["human-source"] for fact in facts)


@pytest.mark.parametrize("text", ["studied", "stubborn", "glove", "soldier", "network", "already"])
def test_importance_does_not_match_inside_other_words(text):
    assert estimate_importance(text) == 0.5


@pytest.mark.parametrize("text,score", [("LOVE!", 0.85), ("never again", 0.85),
                                        ("I prefer tea", 0.6), ("maybe tomorrow", 0.35)])
def test_importance_still_matches_whole_words(text, score):
    assert estimate_importance(text) == score
