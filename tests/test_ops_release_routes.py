"""Read-only lifecycle API contract consumed by BawtHub Operations."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlmodel import create_engine

from llm_bawt.ops.release_store import ReleaseStore
from llm_bawt.service.routes import ops as routes


def test_release_list_and_detail_are_paginated_and_include_events(monkeypatch, tmp_path):
    store = ReleaseStore(None, engine=create_engine(f"sqlite:///{tmp_path / 'releases.sqlite'}"))
    row = store.create(parent_job_id="a" * 32, release_task="TASK-1030", bump="patch",
                       github_repository="bawthub/bawthub",
                       workflow_path=".github/workflows/release-frontend.yml",
                       canonical_branch="main")
    monkeypatch.setattr(routes, "_service", lambda: SimpleNamespace(releases=store))
    listing = routes.list_releases(limit=1, offset=0)
    assert listing["total"] == listing["limit"] == 1
    assert listing["releases"][0]["id"] == row.id
    assert routes.list_releases(limit=1, offset=1)["releases"] == []
    detail = routes.get_release(row.id)
    assert detail["parent_job_id"] == "a" * 32
    assert detail["events"][0]["event_type"] == "release.created"
    with pytest.raises(HTTPException) as missing:
        routes.get_release("b" * 32)
    assert missing.value.status_code == 404
