"""Journal I/O failure receipts, without a database or a filled disk."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import errno
import json
import os

import pytest

from llm_bawt import orphan_partitions
from llm_bawt.orphan_partitions import PartitionChild, PartitionManifestExecutor
from llm_bawt.service.routes import settings


@pytest.fixture
def journal_run(monkeypatch, tmp_path):
    children = tuple(PartitionChild(
        parent_oid=11, parent='"public"."messages"', child_oid=oid,
        child=f'"public"."messages_p_review{index}"', bound=f"FOR VALUES IN ('review{index}')",
        storage_id=f"review{index}", relkind="r", persistence="p", row_count=0,
        incoming_fks=0, disposition="proposed", reason=None, action="drop-empty-RESTRICT",
    ) for index, oid in enumerate((20, 21)))
    plan = SimpleNamespace(
        content_digest="reviewed", children=children, database="private_test", database_oid=1,
        cluster_id="private", server_version=160000, max_identifier_length=63,
        parent_oids={"messages": 11}, protected_ids=(),
    )
    rows = [{
        "parent_oid": child.parent_oid, "parent_schema": "public", "parent_name": "messages",
        "child_oid": child.child_oid, "child_schema": "public",
        "child_name": f"messages_p_review{index}", "bound": child.bound,
        "storage_id": child.storage_id, "incoming_fks": 0,
    } for index, child in enumerate(children)]
    planner = orphan_partitions.OrphanPartitionPlanner
    monkeypatch.setattr(planner, "plan", lambda self: plan)
    monkeypatch.setattr(planner, "_identity", staticmethod(lambda conn: {
        "db": "private_test", "db_oid": 1, "cluster_id": "private",
        "server_version": 160000, "max_name": 63,
    }))
    monkeypatch.setattr(planner, "_parents", staticmethod(lambda conn: {
        "messages": {"oid": 11, "nspname": "public"},
    }))
    monkeypatch.setattr(planner, "_children", staticmethod(lambda conn, parents: rows))
    monkeypatch.setattr(planner, "_validate_child", staticmethod(lambda row, parents, max_name: row["storage_id"]))
    monkeypatch.setattr(planner, "_registry", staticmethod(lambda conn: {}))
    # No database: the tag-locked protected set is empty (proven in test_orphan_partitions_pg).
    monkeypatch.setattr(orphan_partitions, "protected_storage_ids", lambda conn: frozenset())

    committed = []

    class Connection:
        def exec_driver_sql(self, sql):
            if sql.startswith("DROP TABLE"):
                self.dropped = sql

        def execute(self, sql):
            return SimpleNamespace(scalar_one=lambda: 0)

    class Engine:
        @contextmanager
        def begin(self):
            conn = Connection()
            yield conn
            committed.append(conn.dropped)

    journal = tmp_path / "journal.jsonl"
    engine = Engine()
    executor = PartitionManifestExecutor(engine)
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(settings, "cleanup_orphaned_bot_data", lambda config, *, dry_run, manifest_digest:
                        executor.execute(plan, manifest_digest, journal_path=journal))
    return journal, committed, engine


@pytest.mark.parametrize("failure,expected_commits,expected_status", [
    ("second_intent", 1, "partial"),
    ("first_postcommit", 1, "partial"),
    ("second_postcommit", 2, "completed"),
    ("close_only", 2, "completed"),
    ("intent_and_close", 1, "partial"),
])
def test_route_preserves_journal_failure_receipt(journal_run, monkeypatch, failure,
                                                  expected_commits, expected_status):
    journal, committed, _engine = journal_run
    real_write, real_open, real_close = os.write, os.open, os.close
    journal_fd = None

    def track_open(path, flags, *args, **kwargs):
        nonlocal journal_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == journal:
            journal_fd = fd
        return fd

    def fail_write(fd, data):
        record = bytes(data)
        if (fd == journal_fd and
                ((b'"child_oid": 21' in record and failure in ("second_intent", "intent_and_close") and
                  b'"status": "pending"' in record) or
                 (failure == "first_postcommit" and b'"child_oid": 20' in record and
                  b'"status": "committed"' in record) or
                 (failure == "second_postcommit" and b'"child_oid": 21' in record and
                  b'"status": "committed"' in record))):
            raise OSError(errno.ENOSPC, "synthetic journal full")
        return real_write(fd, data)

    def fail_close(fd):
        real_close(fd)
        if fd == journal_fd and failure in ("close_only", "intent_and_close"):
            raise OSError(errno.EIO, "synthetic close failure")

    monkeypatch.setattr(orphan_partitions.os, "open", track_open)
    monkeypatch.setattr(orphan_partitions.os, "write", fail_write)
    monkeypatch.setattr(orphan_partitions.os, "close", fail_close)
    response = settings.cleanup_orphans(dry_run=False, manifest_digest="reviewed")
    assert response.status_code == 207
    receipt = json.loads(response.body)
    assert receipt["status"] == expected_status
    assert receipt["committed_count"] == expected_commits == len(committed)
    assert receipt["journal"] == str(journal)
    assert receipt["journal_incomplete"] is True
    assert "synthetic" in receipt["journal_error"]
    assert len(receipt["outcomes"]) == (1 if failure == "first_postcommit" else 2)
    assert len({row["child"] for row in receipt["outcomes"]}) == len(receipt["outcomes"])
    if failure in ("second_intent", "intent_and_close"):
        assert receipt["failed_child"].endswith('"messages_p_review1"')
        assert receipt["outcomes"][-1]["status"] == "halted"
    else:
        assert "failed_child" not in receipt
        assert "error" not in receipt
        assert all(row["status"] == "committed" for row in receipt["outcomes"])
        if "postcommit" in failure:
            assert "synthetic journal full" in receipt["outcomes"][-1]["journal_error"]
        if failure == "close_only":
            assert all("journal_error" not in row for row in receipt["outcomes"])
    if failure == "intent_and_close":
        assert "Journal close failed" in receipt["journal_error"]
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["status"] for row in records[:2]] == (
        ["pending"] if failure == "first_postcommit" else ["pending", "committed"]
    )
    assert sum(row["status"] == "committed" for row in records) == (
        0 if failure == "first_postcommit" else
        1 if failure in ("second_intent", "intent_and_close", "second_postcommit") else 2
    )


def test_ambiguous_commit_is_unknown_without_journal_failure(journal_run, monkeypatch):
    journal, committed, engine = journal_run
    original_begin = engine.begin

    @contextmanager
    def lost_commit_reply():
        with original_begin() as conn:
            yield conn
        raise OSError("synthetic COMMIT reply lost")

    monkeypatch.setattr(engine, "begin", lost_commit_reply)
    response = settings.cleanup_orphans(dry_run=False, manifest_digest="reviewed")
    assert response.status_code == 207
    receipt = json.loads(response.body)
    assert receipt["status"] == "unknown"
    assert receipt["journal_incomplete"] is False
    assert receipt["journal_error"] is None
    assert receipt["committed_count"] == 0
    assert len(receipt["outcomes"]) == 1
    assert receipt["outcomes"][0]["status"] == "unknown"
    assert receipt["failed_child"].endswith('"messages_p_review0"')
    assert len(committed) == 1  # The executor cannot know this from the lost reply.
    assert [row["status"] for row in map(json.loads, journal.read_text().splitlines())] == [
        "pending", "unknown",
    ]


def test_ddl_error_before_commit_is_halted_not_unknown(journal_run, monkeypatch):
    _journal, committed, engine = journal_run
    original_begin = engine.begin

    @contextmanager
    def failed_ddl():
        with original_begin() as conn:
            original_sql = conn.exec_driver_sql

            def reject_drop(sql):
                if sql.startswith("DROP TABLE"):
                    raise OSError("synthetic DDL failure")
                return original_sql(sql)

            conn.exec_driver_sql = reject_drop
            yield conn

    monkeypatch.setattr(engine, "begin", failed_ddl)
    response = settings.cleanup_orphans(dry_run=False, manifest_digest="reviewed")
    receipt = json.loads(response.body)
    assert response.status_code == 207
    assert receipt["status"] == "halted"
    assert receipt["journal_incomplete"] is False
    assert receipt["outcomes"][0]["status"] == "halted"
    assert committed == []


def test_journal_helper_retries_short_raw_writes(tmp_path, monkeypatch):
    journal = tmp_path / "short.jsonl"
    real_write = os.write
    calls = []

    def short_write(fd, data):
        calls.append(len(data))
        return real_write(fd, bytes(data)[:3])

    fd = os.open(journal, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        monkeypatch.setattr(orphan_partitions.os, "write", short_write)
        orphan_partitions._append_journal(fd, {"status": "pending", "child": "synthetic"})
    finally:
        os.close(fd)
    assert len(calls) > 1
    assert [json.loads(line) for line in journal.read_text().splitlines()] == [
        {"status": "pending", "child": "synthetic"},
    ]
