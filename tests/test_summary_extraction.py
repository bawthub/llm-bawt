"""Tests for extraction guards applied to generated summaries."""

import pytest

from llm_bawt.memory.extraction.service import EmptyCompletionError, MemoryExtractionService


class _EmptyClient:
    def query(self, **_kwargs):
        return "   "


@pytest.mark.parametrize("call", ["summary", "conversation"])
def test_empty_llm_reply_is_a_failed_call_not_zero_facts(monkeypatch, call) -> None:
    service = MemoryExtractionService(llm_client=_EmptyClient())
    fallback = object()
    monkeypatch.setattr(service, "_extract_with_heuristics", lambda *a, **k: fallback)
    if call == "summary":
        out = service.extract_from_summary(
            summary_text="Summary: Nick adopted a third dog named Pixel.\nKey Details: Pixel is a beagle.",
            session_start=0, session_end=1, summary_id="s1", use_llm=True)
    else:
        out = service.extract_from_conversation(
            [{"role": "user", "content": "I adopted a beagle named Pixel.", "id": "m1"}], use_llm=True)
    assert out is fallback


def test_empty_reply_never_parses_to_an_empty_list() -> None:
    service = MemoryExtractionService(llm_client=_EmptyClient())
    with pytest.raises(EmptyCompletionError):
        service._parse_extraction_response("", ["m1"])
    with pytest.raises(EmptyCompletionError):
        service._parse_action_response("", [])


def test_all_placeholder_summary_is_not_extractable() -> None:
    summary = """[HISTORICAL SUMMARY]
Summary: N/A
Key Details: N/A
Intent: N/A
Tone: N/A
Open Loops: N/A
"""

    assert MemoryExtractionService.summary_has_extractable_content(summary) is False


def test_substantive_summary_is_extractable() -> None:
    summary = """[HISTORICAL SUMMARY]
Summary: User prefers PostgreSQL for durable storage.
Key Details: N/A
Intent: Choose a database.
Tone: Neutral
Open Loops: None
"""

    assert MemoryExtractionService.summary_has_extractable_content(summary) is True


def test_placeholder_summary_does_not_call_llm() -> None:
    class FailingClient:
        def query(self, **_kwargs):
            raise AssertionError("placeholder summary must not call the LLM")

    service = MemoryExtractionService(llm_client=FailingClient())
    facts = service.extract_from_summary(
        summary_text="Summary: N/A\nKey Details: None\nIntent: N/A\nTone: N/A\nOpen Loops: None",
        session_start=0,
        session_end=1,
        summary_id="summary-id",
        use_llm=True,
    )

    assert facts == []
