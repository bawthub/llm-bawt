"""Dormant gate claim on disposable PG, ordered by observed real lock waits.

All bots/users/catalog rows are synthetic. No executor, live DB or dispatcher
is used. Holding the winning transaction invokes the same _claim as the public
internal wrapper, not a second implementation of authorization.
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, delete, event, text
from sqlmodel import Session

from llm_bawt import task_turn_context as codec
from llm_bawt.bot_tags import catalog
from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.ops.gate_claim import _claim, gate_claim_snapshot
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.gate_history import GATE_HISTORY_INDEX, plan_key
from llm_bawt.ops.models import JOB_TERMINAL_STATES, OpsJob
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import OpsStore
from llm_bawt.ops.validation import canonical_json
from llm_bawt.profiles import EntityProfile, EntityType
from llm_bawt.runtime_settings import BotProfileStore
from tests.pg_lock_wait import wait_blocked_by
from tests.test_ops_service import FakeExecutor, _op_data

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]
HANG_GUARD = 20


@pytest.fixture
def rig(private_app_pg, monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    first = OpsStore(private_app_pg)
    second_engine = create_engine(first.engine.url, pool_pre_ping=True)
    second = OpsStore(private_app_pg, engine=second_engine)
    BotProfileStore(private_app_pg)
    engine = first.engine
    bot = "fixture-gate-" + uuid.uuid4().hex[:12]
    user = "fixture-user-" + uuid.uuid4().hex[:12]
    role = "fixture_gate_" + uuid.uuid4().hex[:12]
    slugs = []
    token_values = {"session_id": str(uuid.uuid4()), "turn_id": "turn-" + uuid.uuid4().hex,
                    "trigger_message_id": str(uuid.uuid4()), "bot_id": bot, "user_id": user}
    token = codec.mint_task_turn_context(**token_values)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                agent_backend, created_at, updated_at)
            VALUES (:bot, :bot, '', '', false, false, false, false, false, 'agent',
                'claude-code', now(), now())
        """), {"bot": bot})
        conn.execute(text("INSERT INTO bot_tag_catalog (slug, name, kind) VALUES (:tag,:tag,'role')"),
                     {"tag": role})
        catalog.lock_bot(conn, bot)
        catalog.assign(conn, bot, [role], "fixture")
    EntityProfile.__table__.create(engine, checkfirst=True)
    with Session(engine) as session:
        session.add(EntityProfile(entity_type=EntityType.USER, entity_id=user))
        session.commit()

    def sql(statement, **params):
        with engine.begin() as conn:
            return conn.execute(text(statement), params)

    def operation(**changes):
        op = first.create_operation(_op_data(**{"slug": "test.gate-" + uuid.uuid4().hex,
            "execution_tag": role, **changes}))
        slugs.append(op.slug)
        return op

    default_op = operation()
    plan = GatePlan("llm-bawt", uuid.uuid4().hex + "a" * 8)

    peers = []

    def peer():
        slug = "fixture-peer-" + uuid.uuid4().hex[:12]
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                    voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                    agent_backend, created_at, updated_at)
                VALUES (:bot, :bot, '', '', false, false, false, false, false, 'agent',
                    'claude-code', now(), now())
            """), {"bot": slug})
            catalog.lock_bot(conn, slug)
            catalog.assign(conn, slug, [role], "fixture")
        peers.append(slug)
        return codec.mint_task_turn_context(**{**token_values, "bot_id": slug})

    def job(*, op=None, use_plan=None, args=None, snapshot=None, key=None, caller_over=None, caller_token=token):
        op = op or default_op
        use_plan = use_plan or plan
        snapshot = gate_claim_snapshot(op, use_plan, turn_token=caller_token) if snapshot is None else snapshot
        context = codec.open_task_turn_context(caller_token)
        caller = {"caller_bot_id": context.bot_id, "caller_user_id": context.user_id, "caller_turn_id": context.turn_id}
        caller.update(caller_over or {})
        return first.create_job(operation=op, args_json=canonical_json(use_plan.to_dict() if args is None else args),
            display_args_json="{}", invocation_snapshot_json=canonical_json(snapshot),
            idempotency_key=key or uuid.uuid4().hex, **caller)

    def claim(job_id, *, store=first, use_token=token):
        return store.claim_gate_job(job_id, turn_token=use_token)

    def events(job_id):
        with engine.connect() as conn:
            return conn.execute(text("SELECT state FROM ops_jobs WHERE id=:id"), {"id": job_id}).all(), conn.execute(
                text("SELECT event_key FROM system_events WHERE subject_id=:id ORDER BY id"), {"id": job_id}).all()

    yield SimpleNamespace(store=first, other=second, engine=engine, bot=bot, user=user, role=role,
        token=token, caller=codec.open_task_turn_context(token), values=token_values, op=default_op,
        plan=plan, job=job, claim=claim, sql=sql, operation=operation, events=events, peer=peer)
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM system_events WHERE user_id=:u"), {"u": user})
            for slug in slugs:
                conn.execute(text("DELETE FROM ops_jobs WHERE operation_slug=:s"), {"s": slug})
                conn.execute(text("DELETE FROM ops_operation_revisions WHERE operation_slug=:s"), {"s": slug})
                conn.execute(text("DELETE FROM ops_operations WHERE slug=:s"), {"s": slug})
            for slug in [bot, *peers]:
                conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug=:b"), {"b": slug})
                conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})
            conn.execute(text("DELETE FROM bot_tag_catalog WHERE slug=:t"), {"t": role})
            conn.execute(delete(EntityProfile).where(EntityProfile.entity_type == EntityType.USER,
                                                      EntityProfile.entity_id == user))
    finally:
        second_engine.dispose()


def _unchanged(rig, job, *, use_token=None):
    before = rig.store.get_job(job.id).model_dump()
    events = rig.events(job.id)
    assert rig.claim(job.id, **({"use_token": use_token} if use_token is not None else {})) is False
    assert rig.store.get_job(job.id).model_dump() == before
    assert rig.events(job.id) == events


def test_normal_additive_migration_on_private_pg_is_nullable_and_preserves_existing_rows(rig):
    op = rig.store.update_operation(rig.op.id, {"execution_tag": None})
    before = op.to_api()
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE ops_operations DROP COLUMN execution_tag"))
        rig.store._migrate_add_columns(conn)
        rig.store._migrate_add_columns(conn)
    assert rig.store.get_operation(op.id).to_api() == before
    _unchanged(rig, rig.job(op=rig.store.get_operation(op.id)))


def test_first_claim_is_durable_idempotent_and_never_dispatches_or_attaches_callback(rig):
    job = rig.job(key="fixture-" + uuid.uuid4().hex)
    assert rig.job(key=job.idempotency_key).id == job.id  # Existing ledger idempotency.
    executor = FakeExecutor()
    service = OpsService(rig.store, executor=executor)
    service._dispatch_queued(job)
    assert rig.store.claim_job(job.id) is False  # No generic authorization bypass.
    assert executor.dispatch_calls == []
    assert rig.claim(job.id)
    current = rig.store.get_job(job.id)
    assert current.state == "dispatching" and current.dispatched_at is not None
    assert current.worker_slot_at is None and current.host_unit_name is None
    assert current.receipt_callback_json is None
    _unchanged(rig, current)
    assert sum("dispatching" in row[0] for row in rig.events(job.id)[1]) == 1


@pytest.mark.parametrize("change", [{"execution_tag": None}, {"execution_tag": "fixture_unknown"},
    {"enabled": False}])
def test_unconfigured_unknown_or_disabled_operation_refuses_without_mutation(rig, change):
    _unchanged(rig, rig.job(op=rig.operation(**change)))


@pytest.mark.parametrize("mutation", ["missing_tag", "missing_profile", "private", "inactive"])
def test_missing_or_restricted_current_caller_refuses(rig, mutation):
    job = rig.job()
    if mutation == "missing_tag":
        BotTagStore(rig.engine).unassign(rig.role, rig.bot)
    elif mutation == "missing_profile":
        rig.sql("DELETE FROM bot_profiles WHERE slug=:b", b=rig.bot)
    else:
        with rig.engine.begin() as conn:
            catalog.lock_bot(conn, rig.bot)
            catalog.assign(conn, rig.bot, [mutation], "fixture")
    _unchanged(rig, job)


@pytest.mark.parametrize("field,value", [("bot_id", "fixture-spoof"), ("user_id", "fixture-spoof"),
    ("turn_id", "turn-" + "b" * 32), ("session_id", "00000000-0000-0000-0000-000000000001")])
def test_signed_other_caller_cannot_claim_the_recorded_invocation(rig, field, value):
    token = codec.mint_task_turn_context(**{**rig.values, field: value})
    _unchanged(rig, rig.job(), use_token=token)


@pytest.mark.parametrize("token", ["forged", "", "fixture-gate"])
def test_raw_bot_id_or_forged_token_is_not_caller_authority(rig, token):
    _unchanged(rig, rig.job(), use_token=token)


@pytest.mark.parametrize("field,value", [("caller_bot_id", "fixture-spoof"),
    ("caller_user_id", "fixture-spoof"), ("caller_turn_id", None)])
def test_spoofed_or_missing_job_provenance_is_refused(rig, field, value):
    _unchanged(rig, rig.job(caller_over={field: value}))


@pytest.mark.parametrize("change", [{"repository": "bawthub"}, {"source_sha": "HEAD"},
    {"source_sha": "b" * 40}, {"target": "gate"}, {"required_tag": "fixture_gate"}, {"attempt": 2}])
def test_bad_or_drifted_stored_plan_and_tag_override_refuse(rig, change):
    _unchanged(rig, rig.job(args={**rig.plan.to_dict(), **change}))


def test_snapshot_tag_revision_hash_and_extra_field_drift_are_refused(rig):
    for change in ("execution_tag", "version", "bool_version", "snapshot_hash", "extra"):
        snapshot = gate_claim_snapshot(rig.op, rig.plan, turn_token=rig.token)
        if change == "extra":
            snapshot["required_tag"] = rig.role
        elif change == "snapshot_hash":
            snapshot[change] = "0" * 64
        elif change == "bool_version":
            snapshot["operation"]["version"] = True  # True == 1 is not canonical content equality.
        else:
            snapshot["operation"][change] = "fixture_other" if change == "execution_tag" else rig.op.version + 1
        _unchanged(rig, rig.job(snapshot=snapshot))


def test_current_grant_edit_even_without_version_bump_invalidates_snapshot(rig):
    job = rig.job()
    rig.sql("UPDATE ops_operations SET execution_tag='fixture_other' WHERE id=:id", id=rig.op.id)
    _unchanged(rig, job)


def test_normal_catalog_revision_edit_invalidates_gate_snapshot(rig):
    job = rig.job()
    rig.store.update_operation(rig.op.id, {"title": "edited"})
    _unchanged(rig, job)


def test_ordinary_invocation_snapshot_is_not_a_gate_authorization(rig):
    service = OpsService(rig.store, executor=FakeExecutor())
    _unchanged(rig, rig.job(snapshot=service.prepare_invocation(rig.op.slug, {"target": "a"})))


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_terminal_state_never_reopens_this_job_or_admits_attempt_two(rig, state):
    job = rig.job()
    assert rig.claim(job.id)
    rig.store.mark_terminal(job.id, state=state)
    _unchanged(rig, job)
    _unchanged(rig, rig.job())  # New key does NOT prove positive runner retirement.


@pytest.mark.parametrize("state", ["lost", "timed_out"])
def test_uncertain_terminal_without_dispatch_timestamp_is_not_retirement(rig, state):
    first = rig.job()
    rig.store.mark_terminal(first.id, state=state)
    _unchanged(rig, rig.job())


def test_rolled_back_claim_leaves_no_attempt_or_transition(rig):
    job = rig.job()
    before = rig.store.get_job(job.id).model_dump()
    events = rig.events(job.id)
    with rig.engine.connect() as conn:
        conn.begin()
        assert _claim(conn, job.id, rig.caller, events=rig.store.events)
        conn.rollback()
    assert rig.store.get_job(job.id).model_dump() == before
    assert rig.events(job.id) == events
    assert rig.claim(job.id)


def test_gate_claim_respects_existing_operation_slot_cap(rig):
    op = rig.operation(max_concurrent=1)
    assert rig.claim(rig.job(op=op).id)
    second = rig.job(op=op, use_plan=GatePlan("llm-bawt", uuid.uuid4().hex + "b" * 8))
    _unchanged(rig, second)


def test_prior_uncertain_attempt_does_not_block_a_different_sha(rig):
    first = rig.job()
    rig.store.mark_terminal(first.id, state="lost")
    assert rig.claim(rig.job(use_plan=GatePlan("llm-bawt", uuid.uuid4().hex + "b" * 8)).id)


@pytest.mark.parametrize("change", ["revoke", "grant_edit"])
def test_change_committing_before_claim_denies_after_observed_lock_wait(rig, change):
    job = rig.job()
    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        if change == "revoke":
            catalog.lock_bot(holder, rig.bot)
            holder.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug=:b AND tag_slug=:t"),
                           {"b": rig.bot, "t": rig.role})
        else:
            holder.execute(text("UPDATE ops_operations SET execution_tag=NULL, version=version+1 WHERE id=:id"),
                           {"id": rig.op.id})
        claim = pool.submit(rig.claim, job.id)
        try:
            wait_blocked_by(rig.engine, pid, claim)
        finally:
            holder.commit()
        assert claim.result(timeout=HANG_GUARD) is False
    current = rig.store.get_job(job.id)
    assert current.state == "queued" and current.dispatched_at is None
    assert len(rig.events(job.id)[1]) == 1  # Only creation, no denied-claim event.


@pytest.mark.parametrize("change", ["revoke", "grant_edit"])
def test_change_after_winning_claim_waits_then_is_prospective(rig, change):
    job = rig.job()
    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        assert _claim(holder, job.id, rig.caller, events=rig.store.events)
        if change == "revoke":
            edit = pool.submit(BotTagStore(rig.engine).unassign, rig.role, rig.bot)
        else:
            edit = pool.submit(rig.store.update_operation, rig.op.id, {"execution_tag": None})
        try:
            wait_blocked_by(rig.engine, pid, edit)
        finally:
            holder.commit()
        assert edit.result(timeout=HANG_GUARD) is not None
    assert rig.store.get_job(job.id).state == "dispatching"
    _unchanged(rig, job)  # No cancellation or second claim after revocation.
    if change == "grant_edit":
        assert rig.store.get_operation(rig.op.id).execution_tag is None
    else:
        assert rig.sql("SELECT 1 FROM bot_profile_tags WHERE bot_slug=:b AND tag_slug=:t",
                       b=rig.bot, t=rig.role).first() is None


def test_two_concurrent_claims_on_one_job_have_exactly_one_winner(rig):
    job = rig.job()
    with ThreadPoolExecutor(2) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        holder.execute(text("SELECT 1 FROM ops_operations WHERE id=:id FOR UPDATE"), {"id": rig.op.id})
        claims = [pool.submit(rig.claim, job.id, store=store) for store in (rig.store, rig.other)]
        try:
            for claim in claims:
                wait_blocked_by(rig.engine, pid, claim)
        finally:
            holder.commit()
        assert sorted(claim.result(timeout=HANG_GUARD) for claim in claims) == [False, True]
    assert sum("dispatching" in row[0] for row in rig.events(job.id)[1]) == 1


def test_repo_sha_claim_lock_prevents_second_job_even_across_operations(rig):
    first = rig.job()
    peer_token = rig.peer()
    other = rig.job(op=rig.operation(), caller_token=peer_token)
    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        assert _claim(holder, first.id, rig.caller, events=rig.store.events)
        claim = pool.submit(rig.claim, other.id, store=rig.other, use_token=peer_token)
        try:
            wait_blocked_by(rig.engine, pid, claim)
        finally:
            holder.commit()
        assert claim.result(timeout=HANG_GUARD) is False
    assert rig.store.get_job(other.id).state == "queued"


@pytest.mark.parametrize("key", [None, "caller-supplied", "different-plan"])
def test_missing_or_drifted_internal_history_key_refuses_without_mutation(rig, key):
    job = rig.job()
    rig.sql("UPDATE ops_jobs SET gate_plan_key=:key WHERE id=:id", key=key, id=job.id)
    _unchanged(rig, job)


@pytest.mark.parametrize("state", ["queued", *sorted(JOB_TERMINAL_STATES)])
def test_legacy_migration_recovers_plan_identity_without_rewriting_evidence(rig, state):
    import json

    prior = rig.job()
    if state != "queued":
        rig.store.mark_terminal(prior.id, state=state)
    # Old formatting and escaped JSON must not evade the same-plan fence.
    raw = json.dumps(dict(reversed(list(rig.plan.to_dict().items()))), indent=2).replace("llm", "\\u006clm")
    rig.sql("UPDATE ops_jobs SET args_json=:raw WHERE id=:id", raw=raw, id=prior.id)
    before = rig.store.get_job(prior.id).model_dump()
    evidence = rig.events(prior.id)
    with rig.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX {GATE_HISTORY_INDEX}"))
        conn.execute(text("ALTER TABLE ops_jobs DROP COLUMN gate_plan_key"))
        rig.store._migrate_add_columns(conn)
        rig.store._migrate_add_columns(conn)
    assert rig.store.get_job(prior.id).model_dump() == before
    assert rig.events(prior.id) == evidence
    candidate = rig.job()
    if state == "queued":
        assert rig.claim(candidate.id)  # A previous unclaimed queued row is not an attempt.
    else:
        _unchanged(rig, candidate)  # Includes never-dispatched cancelled/lost/timed_out.


def test_prior_attempt_query_uses_partial_index_and_never_decodes_unrelated_history(rig):
    prior = rig.job()
    rig.store.mark_terminal(prior.id, state="lost")
    candidate = rig.job()
    # Non-gate poison args and many queued peers are absent from the partial
    # attempt index. Raw fixture inserts deliberately bypass the store decoder.
    rows = [{"id": uuid.uuid4().hex, "operation_slug": rig.op.slug,
             "operation_version": rig.op.version, "operation_script_hash": rig.op.script_hash,
             "display_args_json": "{}", "submitted_at": prior.submitted_at,
             "args_json": "not-json" if n < 3000 else canonical_json(rig.plan.to_dict()),
             "gate_plan_key": None if n < 3000 else plan_key(rig.plan),
             "idempotency_key": uuid.uuid4().hex, "state": "lost" if n < 3000 else "queued"}
            for n in range(4000)]
    with rig.engine.begin() as conn:
        conn.execute(OpsJob.__table__.insert(), rows)
        conn.execute(text("ANALYZE ops_jobs"))
    queries = []

    def capture(conn, cursor, statement, parameters, context, many):
        if statement.startswith("SELECT 1 FROM ops_jobs WHERE gate_plan_key="):
            queries.append((statement, parameters))

    event.listen(rig.engine, "before_cursor_execute", capture)
    try:
        _unchanged(rig, candidate)
    finally:
        event.remove(rig.engine, "before_cursor_execute", capture)
    assert len(queries) == 1
    statement, parameters = queries[0]
    assert "LIMIT 1" in statement and "args_json" not in statement
    with rig.engine.connect() as conn:
        result = conn.exec_driver_sql("EXPLAIN (ANALYZE, FORMAT JSON) " + statement, parameters).scalar_one()

    def nodes(node):
        yield node
        for child in node.get("Plans", []):
            yield from nodes(child)

    plan_nodes = list(nodes(result[0]["Plan"]))
    assert any(n.get("Index Name") == GATE_HISTORY_INDEX for n in plan_nodes), result
    assert result[0]["Plan"]["Actual Rows"] == 1
    assert not any(n["Node Type"] == "Seq Scan" and n.get("Relation Name") == "ops_jobs"
                   for n in plan_nodes), result


def test_existing_history_schema_bootstrap_does_not_wait_on_open_job_writer(rig):
    job = rig.job()
    with rig.engine.connect() as writer, rig.other.engine.begin() as bootstrap:
        writer.begin()
        try:
            writer.execute(text("UPDATE ops_jobs SET error_text='fixture open writer' WHERE id=:id"),
                           {"id": job.id})
            # Hang guard, not a performance assertion. Repeated CREATE INDEX
            # takes SHARE and conflicts with the writer's ROW EXCLUSIVE lock.
            bootstrap.execute(text("SET LOCAL lock_timeout = '1s'"))
            rig.store._migrate_add_columns(bootstrap)
        finally:
            writer.rollback()
    assert rig.store.get_job(job.id).error_text is None


def test_failed_pg_history_migration_rolls_back_column_and_partial_keys_together(rig, monkeypatch):
    from llm_bawt.ops import gate_history

    job = rig.job()
    before = rig.store.get_job(job.id).model_dump()
    real_decoder = gate_history.gate_plan_key
    with rig.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX {GATE_HISTORY_INDEX}"))
        conn.execute(text("ALTER TABLE ops_jobs DROP COLUMN gate_plan_key"))

    def fail(_raw):
        raise RuntimeError("fixture migration interruption")

    try:
        monkeypatch.setattr(gate_history, "gate_plan_key", fail)
        with pytest.raises(RuntimeError, match="fixture migration interruption"), rig.engine.begin() as conn:
            rig.store._migrate_add_columns(conn)
        from sqlalchemy import inspect
        with rig.engine.connect() as conn:
            assert "gate_plan_key" not in {c["name"] for c in inspect(conn).get_columns("ops_jobs")}
    finally:
        monkeypatch.setattr(gate_history, "gate_plan_key", real_decoder)
        with rig.engine.begin() as conn:
            rig.store._migrate_add_columns(conn)
    assert rig.store.get_job(job.id).model_dump() == before
    assert rig.claim(job.id)
