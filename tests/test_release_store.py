"""TASK-1030: durable release rows, claims, CAS transitions and bindings (SQLite)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update
from sqlmodel import Session, create_engine

from llm_bawt.ops.release_models import (
    RELEASE_BUILD_COMPLETE,
    RELEASE_BUILDING,
    RELEASE_DEPLOYED,
    RELEASE_DISPATCHING_BUILD,
    RELEASE_PREFLIGHT,
    ReleaseRun,
)
from llm_bawt.ops.release_store import ReleaseStore, ReleaseStoreUnavailable

PLAN = dict(
    release_task="TASK-1030",
    bump="patch",
    llm_bawt_mode="auto",
    github_repository="bawthub/bawthub",
    workflow_path=".github/workflows/release-frontend.yml",
    canonical_branch="main",
    llm_bawt_repository="bawthub/llm-bawt",
    llm_bawt_branch="master",
)
DIGEST = "sha256:" + "d" * 64
SOURCE = "1" * 40


@pytest.fixture
def store(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'release.sqlite'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    return ReleaseStore(None, engine=engine)


def test_create_is_idempotent_per_parent_and_rejects_plan_drift(store):
    first = store.create(parent_job_id="a" * 32, **PLAN)
    again = store.create(parent_job_id="a" * 32, **PLAN)
    assert again.id == first.id
    assert first.release_request_id.startswith("release-")
    assert first.state == RELEASE_PREFLIGHT
    assert [e.event_type for e in store.events(first.id)] == ["release.created"]
    with pytest.raises(ValueError, match="another release plan"):
        store.create(parent_job_id="a" * 32, **{**PLAN, "bump": "minor"})
    other = store.create(parent_job_id="b" * 32, **PLAN)
    assert other.release_request_id != first.release_request_id
    assert store.get_by_request_id(first.release_request_id).id == first.id


def test_store_without_engine_is_unavailable_for_writes():
    empty = ReleaseStore(None)
    assert empty.get("x") is None and empty.active_ids() == []
    with pytest.raises(ReleaseStoreUnavailable):
        empty.create(parent_job_id="a" * 32, **PLAN)


def test_claim_is_exclusive_until_the_lease_expires(store):
    row = store.create(parent_job_id="a" * 32, **PLAN)
    first = store.claim(row.id, lease_seconds=30)
    assert first is not None and first.reconcile_attempts == 1
    assert store.claim(row.id) is None
    with Session(store.engine) as session:
        session.execute(
            update(ReleaseRun)
            .where(ReleaseRun.id == row.id)
            .values(claim_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
        session.commit()
    recovered = store.claim(row.id)
    assert recovered is not None and recovered.claim_token != first.claim_token
    assert recovered.reconcile_attempts == 2
    # The old worker's token is fenced out of every transition.
    assert store.transition(
        row.id,
        from_states=RELEASE_PREFLIGHT,
        to_state=RELEASE_DISPATCHING_BUILD,
        claim_token=first.claim_token,
        event_type="dispatch.started",
    ) is None
    with pytest.raises(ValueError):
        store.claim(row.id, lease_seconds=1)


def test_transition_is_compare_and_swap_with_ordered_events(store):
    row = store.create(parent_job_id="a" * 32, **PLAN)
    token = store.claim(row.id).claim_token
    moved = store.transition(
        row.id,
        from_states=RELEASE_PREFLIGHT,
        to_state=RELEASE_DISPATCHING_BUILD,
        claim_token=token,
        event_type="dispatch.started",
        detail={"source_sha": SOURCE},
        source_sha=SOURCE,
        release_plan_json={"b": 1, "a": 2},
    )
    assert moved.state == RELEASE_DISPATCHING_BUILD and moved.source_sha == SOURCE
    assert moved.release_plan_json == '{"a":2,"b":1}'
    # Stale source state loses the race and records nothing.
    assert store.transition(
        row.id,
        from_states=RELEASE_PREFLIGHT,
        to_state=RELEASE_BUILDING,
        claim_token=token,
        event_type="never",
    ) is None
    noted = store.note(row.id, claim_token=token, event_type="dispatch.poll", detail={"n": 1})
    assert noted.state == RELEASE_DISPATCHING_BUILD
    events = store.events(row.id)
    assert [e.sequence for e in events] == [1, 2, 3]
    assert [e.event_type for e in events] == ["release.created", "dispatch.started", "dispatch.poll"]
    assert json.loads(events[1].detail_json) == {"source_sha": SOURCE}
    with pytest.raises(ValueError, match="unsupported"):
        store.transition(row.id, from_states=RELEASE_DISPATCHING_BUILD, to_state=RELEASE_BUILDING,
                         claim_token=token, event_type="x", claim_token_override="nope")


def test_terminal_transition_sets_finished_and_blocks_claims(store):
    row = store.create(parent_job_id="a" * 32, **PLAN)
    token = store.claim(row.id).claim_token
    done = store.transition(row.id, from_states=RELEASE_PREFLIGHT, to_state=RELEASE_DEPLOYED,
                            claim_token=token, event_type="test.terminal")
    assert done.finished_at is not None
    assert store.active_ids() == []
    store.release_claim(row.id, token)
    assert store.claim(row.id) is None
    assert done.to_api()["terminal"] is True


def _verified(store, *, receipt_over=None, **row_over):
    row = store.create(parent_job_id="c" * 32, **PLAN)
    token = store.claim(row.id).claim_token
    receipt = {
        "schema": "bawthub.release-receipt/v1",
        "status": "complete",
        "repository": row.github_repository,
        "release_request_id": row.release_request_id,
        "workflow_run_id": "123456",
        "workflow_run_attempt": "2",
        "base_sha": "2" * 40,
        "source_sha": SOURCE,
        "version": "0.1.63",
        "tag": "v0.1.63",
        "digest": DIGEST,
        "image_repository": "ghcr.io/bawthub/frontend",
    }
    receipt.update(receipt_over or {})
    values = dict(
        github_run_id="123456",
        github_run_attempt=2,
        source_sha=SOURCE,
        version="0.1.63",
        tag="v0.1.63",
        digest=DIGEST,
        image_repository="ghcr.io/bawthub/frontend",
        receipt_json=receipt,
        receipt_verified_at=datetime.now(timezone.utc),
        deployable=True,
    )
    values.update(row_over)
    store.transition(row.id, from_states=RELEASE_PREFLIGHT, to_state=RELEASE_BUILD_COMPLETE,
                     claim_token=token, event_type="receipt.verified", **values)
    return row.id


def test_verified_binding_normalizes_string_attempts_from_the_receipt(store):
    binding = store.verified_binding(_verified(store))
    assert binding["workflow_run_attempt"] == "2" and binding["workflow_run_id"] == "123456"
    assert binding["image_ref"] == f"ghcr.io/bawthub/frontend@{DIGEST}"
    assert binding["trigger_sha"] == "2" * 40 and binding["verified_at"].endswith("+00:00")


@pytest.mark.parametrize(
    "receipt_over, row_over",
    [
        ({"workflow_run_attempt": "1"}, {}),
        ({"digest": "sha256:" + "e" * 64}, {}),
        ({"release_request_id": "release-other"}, {}),
        ({}, {"deployable": False}),
    ],
)
def test_verified_binding_refuses_mismatch_or_undeployable(store, receipt_over, row_over):
    release_id = _verified(store, receipt_over=receipt_over, **row_over)
    with pytest.raises(ValueError):
        store.verified_binding(release_id)
