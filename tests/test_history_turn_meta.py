"""History pages carry per-reply turn stats from the exact turn_logs join."""

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

from llm_bawt.service.routes import history_pages


def _service_with_rows(rows, captured):
    class _Result:
        def mappings(self):
            return self

        def all(self):
            return rows

    class _Conn:
        def execute(self, statement, params):
            captured.append((str(statement), params))
            return _Result()

    @contextmanager
    def connect():
        yield _Conn()

    return SimpleNamespace(_turn_log_store=SimpleNamespace(engine=SimpleNamespace(connect=connect)))


def test_turn_meta_maps_reply_edge_model_usage_and_completion():
    ended = datetime(2026, 10, 11, 23, 35, 31, tzinfo=timezone.utc)
    captured = []
    service = _service_with_rows([
        {
            "assistant_message_id": "a1",
            "trigger_message_id": "u1",
            "model": "gpt-6-astra@openai-oauth",
            "token_usage_json": '{"input_tokens": 519, "cache_read_tokens": 77824, '
                                '"cache_creation_tokens": 0, "output_tokens": 193}',
            "ended_at": ended,
        },
        {
            "assistant_message_id": "a2",
            "trigger_message_id": None,
            "model": None,
            "token_usage_json": "not-json",
            "ended_at": None,
        },
    ], captured)
    page = [
        {"id": "u1", "role": "user"},
        {"id": "a1", "role": "assistant"},
        {"id": "a2", "role": "assistant"},
    ]

    meta = history_pages._hydrate_turn_meta_for_page(service, "snark", page)

    assert meta["a1"] == {
        "reply_to_message_id": "u1",
        "model": "gpt-6-astra@openai-oauth",
        "token_usage": {
            "input_tokens": 519, "cache_read_tokens": 77824,
            "cache_creation_tokens": 0, "output_tokens": 193,
        },
        "completed_at": ended.timestamp(),
    }
    # In-flight / legacy rows degrade to empty stats instead of failing the page.
    assert meta["a2"] == {
        "reply_to_message_id": None, "model": None, "token_usage": None, "completed_at": None,
    }
    # Only assistant ids on this page are looked up.
    assert captured[0][1] == {"bot_id": "snark", "assistant_ids": ["a1", "a2"]}


def test_turn_meta_skips_query_without_assistant_rows():
    captured = []
    service = _service_with_rows([], captured)
    assert history_pages._hydrate_turn_meta_for_page(service, "snark", [{"id": "u1", "role": "user"}]) == {}
    assert captured == []
