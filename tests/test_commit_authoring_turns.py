"""TASK-1462: Git object → signed capability → immutable receipt → review model."""

import asyncio
import base64
import hashlib

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt import task_turn_context as codec
from llm_bawt.service import commit_authoring_turns as records
from llm_bawt.service.routes import providers


def commit(bot="fixture-author", *, algorithm="sha1"):
    raw = (
        f"tree {'0' * 40}\nauthor {bot} <{bot}@bots.bawt.local> 2000000000 +0000\n"
        f"committer shared <shared@example.test> 2000000000 +0000\n\nTASK-1 test\n"
    ).encode()
    sha = hashlib.new(algorithm, b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return sha, base64.b64encode(raw).decode()


@pytest.fixture
def world(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: signer)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from llm_bawt.utils import db
    monkeypatch.setattr(db, "get_shared_engine", lambda config: engine)
    app = FastAPI()
    app.include_router(providers.router)
    values = {
        "session_id": "14620000-0000-4000-8000-000000000001",
        "trigger_message_id": "14620000-0000-4000-8000-000000000002",
        "turn_id": "turn-" + "1" * 32, "bot_id": "fixture-author", "user_id": "fixture-user",
    }
    yield engine, app, values
    engine.dispose()


def post(world, body=None, *, token="signed", **context_overrides):
    _, app, values = world
    if body is None:
        sha, encoded = commit()
        body = {"repo_key": "remote:fixture", "commit_sha": sha, "commit_base64": encoded}
    capability = codec.mint_task_turn_context(**(values | context_overrides)) if token == "signed" else token

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(
                "/v1/git/commits", json=body,
                headers={codec.TASK_TURN_CONTEXT_HEADER: capability} if capability else {},
            )
    return asyncio.run(request())


@pytest.mark.parametrize("algorithm", ["sha1", "sha256"])
def test_receipt_checks_object_and_uses_only_signed_identity(world, algorithm):
    sha, encoded = commit(algorithm=algorithm)
    response = post(world, {"repo_key": "remote:fixture", "commit_sha": sha.upper(), "commit_base64": encoded})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "repo_key": "remote:fixture", "commit_sha": sha,
        "turn_id": world[2]["turn_id"], "bot_id": world[2]["bot_id"],
    }
    assert records.authoring_turns(world[0], sha) == [(world[2]["turn_id"], world[2]["bot_id"])]


def test_idempotent_retry_cannot_reassign_existing_commit(world):
    assert post(world).status_code == 200
    assert post(world).status_code == 200
    response = post(world, turn_id="turn-" + "2" * 32)
    assert response.status_code == 409
    assert records.authoring_turns(world[0], commit()[0]) == [(world[2]["turn_id"], "fixture-author")]


@pytest.mark.parametrize("token", [None, "", "not-a-token"])
def test_no_unsigned_or_forged_commit_mapping(world, token):
    assert post(world, token=token).status_code == 401


@pytest.mark.parametrize("field", ["turn_id", "bot_id", "author_model", "author_provider"])
def test_caller_cannot_supply_identity_or_model_fields(world, field):
    sha, encoded = commit()
    response = post(world, {"repo_key": "remote:fixture", "commit_sha": sha,
                           "commit_base64": encoded, field: "forged"})
    assert response.status_code == 422


@pytest.mark.parametrize("fault", ["hash", "encoding", "author", "message-author", "oversize"])
def test_mismatched_or_malformed_git_proof_is_refused(world, fault):
    sha, encoded = commit()
    if fault == "hash":
        sha = "a" * 40
    elif fault == "encoding":
        encoded = "not base64"
    elif fault == "author":
        sha, encoded = commit(bot="fixture-other")
    elif fault == "message-author":
        raw = b"tree " + b"0" * 40 + b"\n\n" + base64.b64decode(encoded)
        sha = hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        encoded = base64.b64encode(raw).decode()
    else:
        encoded = base64.b64encode(b"x" * (records.MAX_COMMIT_BYTES + 1)).decode()
    assert post(world, {"repo_key": "remote:fixture", "commit_sha": sha, "commit_base64": encoded}).status_code in {400, 422}


def test_prefix_lookup_keeps_ambiguity_instead_of_selecting_a_record(world):
    context = codec.open_task_turn_context(codec.mint_task_turn_context(**world[2]))
    for repo in ("remote:one", "remote:two"):
        records.record_commit(world[0], context, repo, commit()[0])
    assert len(records.authoring_turns(world[0], commit()[0][:10])) == 2


def test_receipt_database_failure_does_not_claim_it_was_recorded(world, monkeypatch):
    def fail(*args):
        raise RuntimeError("private database detail")
    monkeypatch.setattr(records, "record_commit", fail)
    response = post(world)
    assert response.status_code == 500
    assert "private database detail" not in response.text


def test_persisted_signed_receipt_is_the_reviewers_author_model_source(world, monkeypatch):
    from types import SimpleNamespace
    from sqlalchemy import text
    from llm_bawt.mcp_server import review_provenance

    sha, _ = commit()
    assert post(world).status_code == 200
    engine, _, values = world
    reviewer_turn = "turn-" + "3" * 32
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE turn_logs (id TEXT PRIMARY KEY, bot_id TEXT, model TEXT)"))
        conn.execute(text("INSERT INTO turn_logs VALUES (:turn, :bot, :model)"), [
            {"turn": values["turn_id"], "bot": values["bot_id"], "model": "author@access"},
            {"turn": reviewer_turn, "bot": "fixture-reviewer", "model": "reviewer@access"},
        ])
    monkeypatch.setattr(review_provenance, "_engine", lambda: engine)
    monkeypatch.setattr(review_provenance, "provider_of", lambda ref: {
        "author@access": "author-vendor", "reviewer@access": "reviewer-vendor",
    }.get(ref))
    fields = review_provenance.resolve_review_provenance(
        SimpleNamespace(turn_id=reviewer_turn, bot_id="fixture-reviewer"), [], sha,
    )
    assert fields["authorModel"] == "author@access"
    assert fields["authorProvider"] == "author-vendor"
    assert fields["crossProvider"] is True
