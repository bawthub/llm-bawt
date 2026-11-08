"""Reset estimates cannot masquerade as measured resident context."""

import pytest

from llm_bawt.context_usage import is_estimated_context


@pytest.mark.parametrize("usage", [
    {"usage_status": "estimated"},
    {"usage_status": "reset"},
    {"resident_source": "session_seed_char_estimate"},
    {"resident_source": "session_seed_and_skills_estimate"},
    {"resident_source": "session_skills_estimate"},
    {"resident_source": "clean_session_reset"},
    {"usage_status": "final", "resident_source": "session_seed_char_estimate"},
])
def test_reset_estimate_provenance(usage):
    assert is_estimated_context(usage)


@pytest.mark.parametrize("usage", [
    {},
    {"resident_source": None},
    {"resident_source": 123},
    {"resident_source": "claude_sdk_context"},
    {"resident_source": "claude_compact_metadata"},
    {"usage_status": "final", "resident_source": "final_iteration_total_input"},
    {"usage_status": "partial", "resident_source": "live_iteration_total_input"},
    {"usage_status": "partial", "resident_source": "interrupted_iteration_total_input"},
])
def test_provider_measurements_are_not_reset_estimates(usage):
    assert not is_estimated_context(usage)
