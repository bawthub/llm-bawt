"""TASK-2248: explicit zero importance must survive database result scoring."""
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
from llm_bawt.memory.postgresql_memories import PostgreSQLSemanticMemoryMixin


@pytest.mark.parametrize("low_importance,expected", [(0.0, 0.0), (0.1, 0.1), (None, 0.5)])
def test_vector_ranking_defaults_only_missing_importance(low_importance, expected):
    def row(identity, importance):
        return SimpleNamespace(id=identity, content="Synthetic fact", tags=["fact"],
            importance=importance, source_message_ids=[], access_count=0,
            created_at=None, last_accessed=None, similarity=1.0,
            meaning_similarity=None, age_days=0, intent=None, stakes=None, emotional_charge=None)

    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
        row("low", low_importance), row("medium", 0.25)]
    backend = SimpleNamespace(engine=engine, _memories_table_name="synthetic_memories",
        config=SimpleNamespace(MEMORY_DECAY_ENABLED=False, MEMORY_DIVERSITY_ENABLED=False,
                               MEMORY_RECENCY_WEIGHT=0.0),
        _ownership=lambda *_args: OPERATOR_MEMORY_AUTHORITY.scope("synthetic"),
        update_memory_access=Mock())

    results = PostgreSQLSemanticMemoryMixin.search_memories_by_embedding(
        backend, [1.0, 0.0], n_results=2, authority=OPERATOR_MEMORY_AUTHORITY)

    by_id = {result["id"]: result for result in results}
    assert by_id["low"]["importance"] == expected
    assert by_id["low"]["effective_score"] == expected
    assert [result["id"] for result in results] == (
        ["medium", "low"] if expected < 0.25 else ["low", "medium"])
    assert backend.update_memory_access.call_count == 2
