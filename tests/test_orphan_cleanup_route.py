"""Digest-confirmed cleanup route never runs without an exact preview token."""

from types import SimpleNamespace
from pathlib import Path
import json

import pytest
from fastapi import HTTPException

from llm_bawt.orphan_partitions import PartitionExecutionHalted
from llm_bawt.service.routes import settings


def test_cleanup_route_maps_mismatch_to_409(monkeypatch):
    called = []
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))

    def reject(config, *, dry_run, manifest_digest):
        called.append((dry_run, manifest_digest))
        raise ValueError("Manifest digest mismatch; preview again")

    monkeypatch.setattr(settings, "cleanup_orphaned_bot_data", reject)
    with pytest.raises(HTTPException) as exc:
        settings.cleanup_orphans(dry_run=False, manifest_digest="old-preview")
    assert exc.value.status_code == 409
    assert called == [(False, "old-preview")]


def test_cleanup_route_reports_committed_prefix_and_journal(monkeypatch):
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))
    journal = Path("/app/.logs/orphan-partitions/receipt.jsonl")

    def partial(config, *, dry_run, manifest_digest):
        raise PartitionExecutionHalted([
            {"child": "messages_p_empty", "status": "committed", "row_count": 0},
            {"child": "messages_p_spark", "status": "halted", "error": "new registration"},
        ], journal, expected_children=2)

    monkeypatch.setattr(settings, "cleanup_orphaned_bot_data", partial)
    response = settings.cleanup_orphans(dry_run=False, manifest_digest="reviewed")
    assert response.status_code == 207
    receipt = json.loads(response.body)
    assert receipt["status"] == "partial"
    assert receipt["committed_count"] == 1
    assert receipt["journal"] == str(journal)
    assert receipt["failed_child"] == "messages_p_spark"


def test_cleanup_route_passes_exact_digest_and_dry_run(monkeypatch):
    called = []
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))

    def cleanup(config, *, dry_run, manifest_digest):
        called.append((dry_run, manifest_digest))
        return {"dry_run": dry_run, "manifest": {"sha256": "abc"}}

    monkeypatch.setattr(settings, "cleanup_orphaned_bot_data", cleanup)
    assert settings.cleanup_orphans(dry_run=True, manifest_digest=None)["dry_run"]
    assert not settings.cleanup_orphans(dry_run=False, manifest_digest="abc")["dry_run"]
    assert called == [(True, None), (False, "abc")]
