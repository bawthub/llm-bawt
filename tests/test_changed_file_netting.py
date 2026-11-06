"""Session action scope nets proven reverts without rewriting turn history."""
# Imported fixture is injected by pytest.
# ruff: noqa: F811
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

from llm_bawt.service.changed_files_store import TurnChangedFile
from tests.test_changed_files_store import store, _conversation, _trigger, _mod  # noqa: F401


def save(store, turn, before, after, *, bot="b", user="u", path="same.py",
         repo="repoA", kind="modified", identity=None, **kwargs):
    _trigger(store, message_id=turn, session_id="s")
    file = _mod(path, before, after, additions=1, deletions=1,
                identity=identity or {}, **kwargs)
    file.repo_key = repo
    file.change_kind = kind
    store.save_turn_files(turn_id=turn, bot_id=bot, user_id=user,
                          trigger_message_id=turn, files=[file])


def summary(store, anchor=None):
    return store.uncommitted_summaries_for_session(
        session_id="s", bot_id="b", user_id="u", anchor_turn_id=anchor,
    )[1:]


@pytest.mark.parametrize("provenance", ["tool", "command_observation"])
def test_edit_then_revert_drops_only_aggregate_action_scope(store, provenance):
    _conversation(store, session_id="s")
    save(store, "first", "A", "B", identity={"provenance": "tool"})
    save(store, "last", "B", "A", identity={"provenance": provenance})
    pending, other = summary(store, "last")
    assert (pending["files"], pending["total_files"],
            pending["total_additions"], pending["total_deletions"]) == ([], 0, 0, 0)
    assert pending["turn_ids"] == ["first", "last"]
    assert len(pending["history_files"]) == 2
    assert other["total_files"] == 1
    assert store.summary_for_turn("first")["total_files"] == 1
    assert store.summary_for_turn("last")["total_files"] == 1
    before, after = store.get_file_content(turn_id="last", repo_key="repoA", path="same.py")
    assert (before.text, after.text) == ("B", "A")


def test_create_then_delete_nets_absent_endpoints(store):
    _conversation(store, session_id="s")
    save(store, "create", None, "B", kind="added")
    save(store, "delete", "B", None, kind="deleted")
    pending, _ = summary(store)
    assert pending["total_files"] == 0
    assert len(pending["history_files"]) == 2


def test_delete_then_recreate_same_content_is_net_zero(store):
    _conversation(store, session_id="s")
    save(store, "delete", "A", None, kind="deleted")
    save(store, "create", None, "A", kind="added")
    assert summary(store)[0]["total_files"] == 0


def test_revert_inside_one_turn_is_net_zero(store):
    _conversation(store, session_id="s")
    save(store, "only", "A", "B")
    # One immutable row retains the original before and latest after.
    store.save_turn_files(turn_id="only", bot_id="b", user_id="u", trigger_message_id="only",
                          files=[_mod("same.py", "B", "A")])
    pending, _ = summary(store)
    assert pending["total_files"] == 0
    assert len(pending["history_files"]) == 1


def test_real_change_and_zero_change_are_counted_independently(store):
    _conversation(store, session_id="s")
    save(store, "first", "A", "B")
    save(store, "last", "B", "A")
    save(store, "real", "A", "C", path="real.py")
    pending, _ = summary(store)
    assert [file["path"] for file in pending["files"]] == ["real.py"]
    assert (pending["total_files"], pending["total_additions"], pending["total_deletions"]) == (1, 1, 1)


@pytest.mark.parametrize("middle_before,final", [("B", "C"), ("unobserved", "A")])
def test_real_or_discontinuous_chain_stays_visible(store, middle_before, final):
    _conversation(store, session_id="s")
    save(store, "first", "A", "B")
    save(store, "last", middle_before, final)
    pending, _ = summary(store)
    assert pending["total_files"] == 1
    assert pending["files"][0]["turn_id"] == "last"
    assert pending["total_additions"] == pending["total_deletions"] == 2


@pytest.mark.parametrize("kwargs", [
    {"binary": True}, {"truncated": True}, {"old_path": "renamed.py"},
    {"kind": "renamed"},
])
def test_uncertain_or_rename_snapshots_stay_visible(store, kwargs):
    _conversation(store, session_id="s")
    save(store, "one", "A", "A", **kwargs)
    assert summary(store)[0]["total_files"] == 1


def test_missing_digests_are_not_absent_files(store):
    _conversation(store, session_id="s")
    save(store, "one", None, None)
    assert summary(store)[0]["total_files"] == 1


def test_empty_file_is_distinct_from_absent_file(store):
    _conversation(store, session_id="s")
    save(store, "one", None, "", kind="added")
    assert summary(store)[0]["total_files"] == 1


@pytest.mark.parametrize("foreign_bot,foreign_user", [("other", "u"), ("b", "other"), (None, "u")])
def test_intervening_foreign_owner_prevents_netting(store, foreign_bot, foreign_user):
    _conversation(store, session_id="s")
    save(store, "first", "A", "B")
    save(store, "foreign", "B", "B", bot=foreign_bot, user=foreign_user)
    save(store, "last", "B", "A")
    assert summary(store)[0]["total_files"] == 1


def test_foreign_edit_before_chain_does_not_prevent_netting(store):
    _conversation(store, session_id="s")
    save(store, "foreign", "X", "A", bot="other")
    save(store, "first", "A", "B")
    save(store, "last", "B", "A")
    # Explicit deterministic timestamps, without wall-clock assertions/sleeps.
    with Session(store.engine) as session:
        for index, row in enumerate(session.exec(select(TurnChangedFile).order_by(TurnChangedFile.id))):
            row.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
            session.add(row)
        session.commit()
    assert summary(store)[0]["total_files"] == 0


def test_unrelated_foreign_path_does_not_prevent_netting(store):
    _conversation(store, session_id="s")
    save(store, "first", "A", "B")
    save(store, "foreign", "B", "C", bot="other", path="elsewhere.py")
    save(store, "last", "B", "A")
    assert summary(store)[0]["total_files"] == 0


def test_resolved_workspace_identity_detects_foreign_path_alias(store):
    _conversation(store, session_id="s")
    identity = {"identity_status": "resolved", "workspace_repo_id": "registered",
                "workspace_id": "workspace-one", "repo_relative_path": "same.py"}
    save(store, "first", "A", "B", path="/repo/same.py", identity=identity)
    save(store, "foreign", "B", "B", bot="other", path="~/repo/same.py", identity=identity)
    save(store, "last", "B", "A", path="/repo/same.py", identity=identity)
    assert summary(store)[0]["total_files"] == 1


def test_canonical_aliases_net_only_with_same_workspace(store, monkeypatch):
    from llm_bawt.service import changed_file_commits

    _conversation(store, session_id="s")
    identity = {"identity_status": "resolved", "workspace_repo_id": "registered",
                "workspace_id": "workspace-one", "repo_relative_path": "same.py"}
    save(store, "first", "A", "B", path="/repo/same.py", identity=identity)
    save(store, "last", "B", "A", path="~/repo/same.py", identity=identity)

    def proof(files):
        for file in files:
            file["commit_evidence"] = {"repo_id": "registered", "repo_path": "same.py"}
    monkeypatch.setattr(changed_file_commits, "reconcile_files", proof)
    assert summary(store)[0]["total_files"] == 0
    with Session(store.engine) as session:
        row = session.exec(select(TurnChangedFile).where(TurnChangedFile.turn_id == "last")).one()
        row.workspace_id = "different-worktree"
        session.add(row)
        session.commit()
    assert summary(store)[0]["total_files"] == 1


def test_committed_intermediate_snapshot_is_not_cancelled_by_later_revert(store, monkeypatch):
    from llm_bawt.service import changed_file_commits

    _conversation(store, session_id="s")
    save(store, "committed", "A", "B")
    save(store, "revert", "B", "A")

    def proof(files):
        files[0]["commit_state"] = "committed"
    monkeypatch.setattr(changed_file_commits, "reconcile_files", proof)
    assert summary(store)[0]["total_files"] == 1
