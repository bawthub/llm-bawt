"""TASK-1144 identity table and advisory-lock contract (no live database)."""
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from llm_bawt import request_identity as identity


class Result:
    def __init__(self, row=None):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class FakeConnection:
    def __init__(self):
        self.rows = []
        self.locks = []
        self.race = None
        self.insert_attempts = 0

    def execute(self, sql, params):
        query = str(sql)
        if "pg_advisory_xact_lock" in query:
            self.locks.append((params["namespace"], params["hashed"]))
            return Result()
        if "SELECT * FROM inter_bot_deliveries" in query:
            if "WHERE user_message_id" in query:
                return Result(next((r for r in self.rows if r["user_message_id"] == params["message_id"]), None))
            scope = (("author_entity_type", "author_type"), ("author_entity_id", "author_id"),
                     ("sender_bot_id", "sender"), ("target_bot_id", "target"),
                     ("idempotency_key", "idem"))
            return Result(next((r for r in self.rows if r["idempotency_key"] is not None
                                and all(r[column] == params[key] for column, key in scope)), None))
        assert "ON CONFLICT DO NOTHING RETURNING *" in query
        self.insert_attempts += 1
        if self.race is not None:
            self.rows.append(self.race)
            self.race = None
            return Result()
        if any(r["user_message_id"] == params["message_id"] or
               r["idempotency_key"] and r["idempotency_key"] == params["idem"]
               for r in self.rows):
            return Result()
        row = make_row(params)
        self.rows.append(row)
        return Result(row)


def make_row(params):
    now = datetime.now(timezone.utc)
    return {
        "id": params["id"], "sender_bot_id": params["sender"],
        "author_entity_type": params["author_type"], "author_entity_id": params["author_id"],
        "target_bot_id": params["target"], "message": params["message"],
        "payload_json": json.loads(params["payload"]), "metadata_json": json.loads(params["metadata"]),
        "user_message_id": params["message_id"], "turn_id": params["turn_id"],
        "idempotency_key": params["idem"], "project_id": None, "task_id": None,
        "message_kind": None, "status": "QUEUED", "attempt_count": 0,
        "max_attempts": 5, "created_at": now, "updated_at": now, "available_at": now,
        "session_policy": params["session_policy"],
    }


def submit(conn, *, user=None, key=None, author="nick", target="snark",
           producer="bot", message="hello", assistant=None, metadata=None, payload=None, policy="continue"):
    values = {"sender": "loopy", "author_type": "user", "author_id": author,
              "target": target, "message": message.strip(), "idem": key,
              "project": None, "task": None, "kind": None, "max_attempts": 5,
              "session_policy": policy, "reset_reason": None}
    return identity.RequestIdentityResolver(conn).admit(
        values=values, payload=payload or {"messages": [{"role": "user", "content": message}]},
        metadata=metadata or {}, producer=producer, explicit_user=user,
        explicit_assistant=assistant,
    )


def test_all_nine_uuid_key_rows_and_first_keyed_submission():
    u1, u2 = str(uuid.uuid4()), str(uuid.uuid4())
    # 1: neither U nor K; 2: keyed miss; 3: key hit without U.
    conn = FakeConnection()
    plain, duplicate = submit(conn)
    assert not duplicate and plain.user_message_id != u1
    keyed, duplicate = submit(conn, key="k")
    assert not duplicate and keyed.id != plain.id
    assert submit(conn, key="k")[0].id == keyed.id
    # 4: U miss, K absent/miss; 5: U hit, K absent.
    explicit, duplicate = submit(conn, user=u1)
    assert not duplicate and explicit.user_message_id == u1
    assert explicit.id == f"delivery-{uuid.UUID(u1).hex}"
    assert explicit.turn_id == f"turn-delivery-{uuid.UUID(u1).hex}"
    assert submit(conn, user=u1)[0].id == explicit.id
    # 6: U hit, K miss; 7: U miss, K hit.
    with pytest.raises(identity.IdentityConflict):
        submit(conn, user=u1, key="different")
    with pytest.raises(identity.IdentityConflict):
        submit(conn, user=u2, key="k")
    # 8: U and K hit same row; 9: different rows.
    linked, _ = submit(conn, user=u2, key="linked")
    assert submit(conn, user=u2, key="linked")[0].id == linked.id
    assert submit(conn, user=u2)[0].id == linked.id  # U hit, omitted K
    with pytest.raises(identity.IdentityConflict):
        submit(conn, user=u2, key="k")
    assert conn.insert_attempts == 4


@pytest.mark.parametrize("kind", ["same", "different-target", "different-author",
                                  "key-different-uuid", "uuid-other-key", "missing"])
def test_on_conflict_one_winner_pass_never_remints(kind):
    user = str(uuid.uuid4())
    seed = FakeConnection()
    submit(seed, user=user, key="key")
    winner = seed.rows[0].copy()
    if kind == "different-target":
        winner["target_bot_id"] = "other"
    if kind == "different-author":
        winner["author_entity_id"] = "other"
    if kind == "key-different-uuid":
        winner["user_message_id"] = str(uuid.uuid4())
    if kind == "uuid-other-key":
        winner["idempotency_key"] = "another"
    conn = FakeConnection()
    conn.race = winner if kind != "missing" else {**winner, "user_message_id": str(uuid.uuid4()),
                                                   "idempotency_key": "another"}
    if kind == "same":
        assert submit(conn, user=user, key="key")[1] is True
    elif kind == "missing":
        with pytest.raises(identity.AdmissionUnavailable):
            submit(conn, user=user, key="key")
    else:
        with pytest.raises(identity.IdentityConflict):
            submit(conn, user=user, key="key")
    assert conn.insert_attempts == 1


def test_ordinary_insert_race_uses_stored_assistant_on_winner_reread():
    user, assistant = str(uuid.uuid4()), str(uuid.uuid4())
    payload = {"user_message_id": user, "assistant_message_id": assistant,
               "messages": [{"role": "user", "content": "hello"}]}
    seed = FakeConnection()
    submit(seed, user=user, key=f"ordinary:{user}", producer="ordinary",
           assistant=assistant, payload=payload, metadata={"ordinary_chat": {}})
    conn = FakeConnection()
    conn.race = seed.rows[0].copy()
    replay = dict(payload)
    replay.pop("assistant_message_id")
    record, duplicate = submit(conn, user=user, key=f"ordinary:{user}",
                               producer="ordinary", payload=replay,
                               metadata={"ordinary_chat": {}})
    assert duplicate and record.assistant_message_id == assistant
    assert conn.insert_attempts == 1


def test_actual_numeric_lock_pairs_are_sorted_and_deduplicated(monkeypatch):
    user = str(uuid.uuid4())
    conn = FakeConnection()
    monkeypatch.setattr(identity, "identity_hash", lambda tag: 17 if tag.startswith("uuid:") else -3)
    submit(conn, user=user, key="key")
    assert conn.locks == [(identity.IDENTITY_NS, -3), (identity.IDENTITY_NS, 17)]
    conn = FakeConnection()
    monkeypatch.setattr(identity, "identity_hash", lambda tag: 9)
    submit(conn, user=user, key="key")
    assert conn.locks == [(identity.IDENTITY_NS, 9)]


@pytest.mark.parametrize("different", ["target", "author"])
def test_concurrent_same_uuid_cannot_cross_target_or_author(different):
    user = str(uuid.uuid4())
    rows = []
    gate = threading.RLock()
    start = threading.Barrier(2)

    class ConcurrentConnection(FakeConnection):
        def __init__(self):
            super().__init__()
            self.rows = rows
            self.acquired = 0

        def execute(self, sql, params):
            if "pg_advisory_xact_lock" in str(sql):
                gate.acquire()
                self.acquired += 1
            return super().execute(sql, params)

        def commit(self):
            for _ in range(self.acquired):
                gate.release()

    def worker(alternate):
        conn = ConcurrentConnection()
        start.wait(timeout=5)
        try:
            return submit(conn, user=user, key=None,
                          **({different: "another"} if alternate else {}))[0].id
        except identity.IdentityConflict:
            return "conflict"
        finally:
            conn.commit()  # transaction-scoped identity locks end here

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (False, True)))
    assert len(rows) == 1
    assert results.count("conflict") == 1
    assert results.count(rows[0]["id"]) == 1


def test_schedule_two_occurrences_can_enqueue_while_dispatcher_waits_on_job():
    job_lock, target_lock = threading.Lock(), threading.Lock()
    dispatcher_has_target = threading.Event()
    records = []
    conn = FakeConnection()

    def dispatcher():
        with target_lock:
            dispatcher_has_target.set()
            with job_lock:  # claim order: target -> job
                records.append("claimed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with job_lock:
            future = pool.submit(dispatcher)
            assert dispatcher_has_target.wait(timeout=5)
            for occurrence in (1, 2):
                record, duplicate = submit(
                    conn, key=f"schedule:job:occurrence:{occurrence}",
                    producer="schedule", metadata={"prompt_schedule": {"occurrence": occurrence}},
                )
                assert not duplicate
                records.append(record.id)
            # The identity-only leaf never waits for the dispatcher's target lock.
        future.result(timeout=5)
    assert len(set(records[:2])) == 2 and records[2] == "claimed"


def test_retry_identity_leaf_does_not_wait_for_cancel_target():
    original_row, target_lock = threading.Lock(), threading.Lock()
    cancel_has_target = threading.Event()
    conn = FakeConnection()
    original, _ = submit(conn, key="original")
    observations = []

    def cancel():
        with target_lock:  # cancel order: target -> original row
            cancel_has_target.set()
            with original_row:
                observations.append("cancel saw original")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with original_row:  # retry locks FAILED original first
            future = pool.submit(cancel)
            assert cancel_has_target.wait(timeout=5)
            child, duplicate = submit(conn, key=f"manual-retry:{original.id}",
                                      producer="retry", metadata={"manual_retry_of": original.id})
            assert not duplicate and child.id != original.id
            observations.append("retry inserted")
        future.result(timeout=5)
    assert observations == ["retry inserted", "cancel saw original"]


def test_ordinary_legacy_assistant_omission_explicit_and_mutated_execution_payload():
    user, assistant = str(uuid.uuid4()), str(uuid.uuid4())
    incoming = {"bot_id": "snark", "user": "nick", "user_message_id": user,
                "assistant_message_id": assistant, "prefer_steer": False}
    conn = FakeConnection()
    first, _ = submit(conn, user=user, key="ordinary:" + user, producer="ordinary",
                      assistant=assistant, metadata={"ordinary_chat": {}}, payload=incoming)
    row = conn.rows[0]
    assert row["metadata_json"]["ordinary_chat"]["assistant_supplied"] is True
    row["metadata_json"]["ordinary_chat"].pop("assistant_supplied")  # historical row
    row["metadata_json"].pop("_request_identity_producer")
    # Reset and overflow rewrite execution state, not immutable comparison evidence.
    row["payload_json"].update(session_id="rotated", inter_bot_session_policy="reset_retain_history",
                               inter_bot_bridge_request_id="req_recovery_1")
    omitted = {k: v for k, v in incoming.items() if k != "assistant_message_id"}
    replay, duplicate = submit(conn, user=user, key="ordinary:" + user,
                               producer="ordinary", metadata={"ordinary_chat": {}}, payload=omitted)
    assert duplicate and replay.assistant_message_id == assistant == first.assistant_message_id
    with pytest.raises(identity.IdentityConflict):
        submit(conn, user=user, key="ordinary:" + user, producer="ordinary",
               assistant=str(uuid.uuid4()), payload={**incoming, "assistant_message_id": str(uuid.uuid4())})
    row["metadata_json"]["ordinary_chat"].pop("digest")
    with pytest.raises(identity.ReplayUnverifiable):
        submit(conn, user=user, key="ordinary:" + user, producer="ordinary", payload=omitted)


def test_maintenance_normalization_parent_and_scope_conflicts():
    user = str(uuid.uuid4())
    conn = FakeConnection()
    record, _ = submit(conn, user=user, key=f"maintenance-user:parent:{user}",
                       producer="maintenance", metadata={"maintenance_user_queue": True},
                       message="  hello  ")
    assert submit(conn, user=user, key=f"maintenance-user:parent:{user}",
                  producer="maintenance", metadata={"maintenance_user_queue": True},
                  message="hello")[0].id == record.id
    for changes in ({"key": f"maintenance-user:other-parent:{user}"},
                    {"target": "other"}, {"author": "other"}, {"message": "changed"}):
        with pytest.raises(identity.IdentityConflict):
            submit(conn, producer="maintenance", metadata={"maintenance_user_queue": True},
                   **{"user": user, "key": f"maintenance-user:parent:{user}", **changes})


def test_manual_retry_replaces_copied_identity_and_reattaches_once():
    conn = FakeConnection()
    parent, _ = submit(conn, key="parent")
    copied = dict(conn.rows[0]["payload_json"])
    key = f"manual-retry:{parent.id}"
    child, duplicate = submit(conn, key=key, producer="retry", payload=copied,
                              metadata={"manual_retry_of": parent.id})
    assert not duplicate and child.id != parent.id
    assert child.user_message_id != parent.user_message_id
    assert child.assistant_message_id != parent.assistant_message_id
    again, duplicate = submit(conn, key=key, producer="retry", payload=copied,
                              metadata={"manual_retry_of": parent.id})
    assert duplicate and again.id == child.id


def test_schedule_policy_and_producer_cannot_be_downgraded():
    conn = FakeConnection()
    first, _ = submit(conn, key="schedule:job:occurrence:1", producer="schedule",
                      metadata={"prompt_schedule": {"job": "1"}})
    assert submit(conn, key="schedule:job:occurrence:1", producer="schedule",
                  metadata={"prompt_schedule": {"job": "1"}})[0].id == first.id
    with pytest.raises(identity.IdentityConflict):
        submit(conn, key="schedule:job:occurrence:1", producer="schedule", policy="reset_without_history")
    with pytest.raises(identity.IdentityConflict):
        submit(conn, key="schedule:job:occurrence:1", producer="bot", metadata={})
    with pytest.raises(identity.InvalidIdentity):
        submit(conn, user="not-a-uuid")
    assert identity.canonical_uuid(str(uuid.UUID(first.user_message_id)).upper(), "user_message_id") == first.user_message_id
    forged = FakeConnection()
    user = str(uuid.uuid4())
    submit(forged, user=user, key="ordinary:" + user, producer="ordinary",
           metadata={"ordinary_chat": {}, "_request_identity_producer": "bot"},
           payload={"user_message_id": user, "assistant_message_id": str(uuid.uuid4())})
    assert forged.rows[0]["metadata_json"]["_request_identity_producer"] == "ordinary"
    with pytest.raises(identity.IdentityConflict):
        submit(forged, user=user, key="ordinary:" + user, producer="bot", metadata={})


def test_bot_pinned_uuid_replay_must_match_its_request():
    """A service caller's U is one request: a changed replay is a conflict."""
    user = str(uuid.uuid4())
    conn = FakeConnection()
    first, duplicate = submit(conn, user=user, key=user, message="dispatch TASK-1")
    assert not duplicate
    assert submit(conn, user=user, key=user, message="dispatch TASK-1")[0].id == first.id
    with pytest.raises(identity.IdentityConflict, match="different queued request"):
        submit(conn, user=user, key=user, message="dispatch TASK-2")
    with pytest.raises(identity.IdentityConflict, match="different queued request"):
        submit(conn, user=user, key=user, message="dispatch TASK-1",
               payload={"messages": [{"role": "user", "content": "dispatch TASK-1"}],
                        "session_id": "other"})
    # Unpinned bot sends keep key-only replay semantics.
    keyed, _ = submit(conn, key="plain", message="one")
    assert submit(conn, key="plain", message="two")[0].id == keyed.id


def test_bot_request_digest_is_server_owned():
    user = str(uuid.uuid4())
    conn = FakeConnection()
    record, _ = submit(conn, key="k", metadata={"_request_digest": "forged"})
    assert "_request_digest" not in conn.rows[0]["metadata_json"]
    assert submit(conn, key="k", message="changed")[0].id == record.id
    submit(conn, user=user, metadata={"_request_digest": "forged"})
    assert conn.rows[1]["metadata_json"]["_request_digest"] != "forged"
