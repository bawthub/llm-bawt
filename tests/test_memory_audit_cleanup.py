"""Behavior retained while removing unused memory-audit code (TASK-2290)."""

import json

import pytest

from llm_bawt.memory.extraction.service import MemoryExtractionService
from llm_bawt.memory.maintenance import MemoryMaintenance


@pytest.mark.parametrize(
    ("raw_tags", "expected"),
    [
        ([" Fact ", " Custom-Tag ", "FACT", None, 3], ["fact", "custom-tag", "fact"]),
        (" Custom-Tag ", ["custom-tag"]),
        ([], ["misc"]),
    ],
)
def test_extraction_preserves_extensible_tags(raw_tags, expected):
    response = json.dumps({"facts": [{
        "content": "A synthetic extracted fact",
        "tags": raw_tags,
        "importance": 0.6,
        "source_message_ids": ["source"],
    }]})
    facts = MemoryExtractionService()._parse_extraction_response(
        response, ["source"], eligible_ids={"source"}, strict=True,
    )
    assert len(facts) == 1
    assert facts[0].tags == expected
    assert facts[0].source_message_ids == ["source"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_legacy_orphan_cleanup_flag_stays_a_storage_free_noop(dry_run):
    # A bare object has no storage API: this must not read or mutate anything.
    result = MemoryMaintenance(object()).run(
        run_consolidation=False,
        run_recurrence_detection=False,
        run_decay_pruning=False,
        run_orphan_cleanup=True,
        dry_run=dry_run,
    )
    assert result.to_dict() == {
        "consolidation_clusters_merged": 0,
        "consolidation_memories_consolidated": 0,
        "recurrence_themes_detected": 0,
        "recurrence_memories_tagged": 0,
        "decay_memories_archived": 0,
        "orphans_cleaned": 0,
        "errors": [],
        "dry_run": dry_run,
    }
