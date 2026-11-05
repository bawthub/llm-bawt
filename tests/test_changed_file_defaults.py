"""Generated command noise stays inspectable; real source and tool edits do not move."""
# Imported pytest fixture is intentionally injected into test arguments.
# ruff: noqa: F811
import pytest

from tests.test_changed_files_store import store, _conversation, _trigger, _mod  # noqa: F401
from tests.test_changed_file_decisions import decide
from llm_bawt.service.changed_file_defaults import default_ignored


@pytest.mark.parametrize("path", [
    "__pycache__/module.pyc", "pkg/__pycache__/data.bin", "pkg/module.pyc",
    ".pytest_cache/v/cache/nodeids", "web/node_modules/library/index.js",
    ".mypy_cache/3.12/mod.json", "pkg/.ruff_cache/0.9/cache",
    "/workspace/project/pkg/__pycache__/module.cpython-312.pyc",
])
def test_only_command_cache_paths_default_to_ignored(path):
    assert default_ignored(provenance="command_observation", path=path)
    assert not default_ignored(provenance="tool", path=path)
    assert not default_ignored(provenance=None, path=path)


@pytest.mark.parametrize("path", [
    "src/cache.py", "tests/test_cache.py", "node_modules.md", "node_modules_extra/a.js",
    ".pytest_cache_config", "docs/__pycache__.md", "data.pyc.txt", "__pycache__",
    ".gitignore", "package-lock.json", "uv.lock",
])
def test_real_paths_are_unaffected(path):
    assert not default_ignored(provenance="command_observation", path=path)


def save(store, turn="cache-turn", after="new"):
    store.save_turn_files(
        turn_id=turn, bot_id="b", user_id="u", trigger_message_id=turn,
        files=[
            _mod("pkg/__pycache__/cache.pyc", "old", after, additions=100, deletions=90,
                 identity={"provenance": "command_observation"}),
            _mod("src/real.py", "old", after, additions=2, deletions=1,
                 identity={"provenance": "command_observation"}),
        ],
    )


def test_defaults_apply_to_history_and_totals_without_changing_capture(store):
    _conversation(store, session_id="s")
    _trigger(store, message_id="cache-turn", session_id="s")
    save(store)
    summary = store.summary_for_turn("cache-turn")
    # Both rows are still present in turn history; only source counts.
    assert len(summary["files"]) == 2
    assert (summary["total_files"], summary["total_additions"], summary["total_deletions"]) == (1, 2, 1)
    cache = next(f for f in summary["files"] if f["path"].endswith(".pyc"))
    assert cache["ignored"] and cache["has_content"]
    _, pending = store.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert [f["path"] for f in pending["files"]] == ["src/real.py"]
    assert len(pending["history_files"]) == 2
    assert pending["history_files"][0]["ignored"]
    # Policy is read-time, not a destructive capture filter or backfill.
    from sqlalchemy import text
    with store.engine.connect() as conn:
        assert conn.execute(text("SELECT ignored_version FROM turn_changed_files LIMIT 1")).scalar() is None


def test_restore_is_durable_for_exact_snapshot_and_new_edits_default_again(store):
    _conversation(store, session_id="s")
    _trigger(store, message_id="cache-turn", session_id="s")
    save(store)
    cache = store.summary_for_turn("cache-turn")["files"][0]
    decide(store, [cache], False)
    assert not store.summary_for_turn("cache-turn")["files"][0]["ignored"]
    _, pending = store.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert pending["total_files"] == 2
    save(store)  # Same capture retry preserves Restore.
    assert not store.summary_for_turn("cache-turn")["files"][0]["ignored"]
    save(store, after="changed")  # New bytes invalidate the exact-snapshot override.
    assert store.summary_for_turn("cache-turn")["files"][0]["ignored"]
    save(store, after="new")  # Old bytes must not resurrect an expired override.
    assert store.summary_for_turn("cache-turn")["files"][0]["ignored"]
    latest = store.summary_for_turn("cache-turn")["files"][0]
    decide(store, [latest], False)
    decide(store, [latest], True)
    assert store.summary_for_turn("cache-turn")["files"][0]["ignored"]
    _trigger(store, message_id="next", session_id="s")
    save(store, turn="next")
    assert store.summary_for_turn("next")["files"][0]["ignored"]
